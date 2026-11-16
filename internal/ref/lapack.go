package ref

import "math"

// Ports of the reference LAPACK 3.12.0 routines (Reference-LAPACK/lapack, tag
// v3.12.0, SRC) dgetrf.f, dgetrf2.f, dgetrs.f, dlaswp.f, dpotrf.f, dpotrf2.f,
// dpotrs.f and their single precision twins sgetrf.f, sgetrf2.f, sgetrs.f,
// slaswp.f, spotrf.f, spotrf2.f, spotrs.f, written once, generic over float32 and
// float64. Same rules as the BLAS ports of this package: the loops and the
// recursion are the reference's, 0-based indices into column-major slices, no
// fused multiply-add (the BLAS calls are this package's), the arguments already
// checked, and the reference's quick returns (m = 0 or n = 0) the caller's. A
// call of the reference that is itself a BLAS quick return (a GEMM or SYRK with
// k = 0 and beta = 1) is skipped here, since this package's BLAS ports leave
// those returns to the caller. The pivots are 1-based, as in LAPACK.

// SafeMin is DLAMCH('S') / SLAMCH('S'): the smallest normal number, 2^-1022 for
// float64 and 2^-126 for float32. The reference takes the larger of that and
// 1/huge, which is smaller.
func SafeMin[T Float]() T {
	if bits64[T]() {
		return T(math.Ldexp(1, -1022))
	}
	return T(math.Ldexp(1, -126))
}

// NB is the block size ILAENV answers for xGETRF and xPOTRF (ilaenv.f, tag
// v3.12.0: NB = 64 for C2 = 'GE' and 'PO', C3 = 'TRF', single and double).
const NB = 64

// Laswp is xLASWP (dlaswp.f): a series of row interchanges on the matrix a with n
// columns, rows k1 to k2 (1-based), row i swapped with row ipiv[k1+(i-k1)*|incx|],
// in reverse order for a negative incx. incx = 0 does nothing.
func Laswp[T Float](n int, a []T, lda, k1, k2 int, ipiv []int, incx int) {
	var ix0, i1, i2, inc int
	switch {
	case incx > 0:
		ix0, i1, i2, inc = k1, k1, k2, 1
	case incx < 0:
		ix0, i1, i2, inc = k1+(k1-k2)*incx, k2, k1, -1
	default:
		return
	}
	n32 := n / 32 * 32
	if n32 != 0 {
		for j := 1; j <= n32; j += 32 {
			ix := ix0
			for i := i1; (inc > 0 && i <= i2) || (inc < 0 && i >= i2); i += inc {
				if ip := ipiv[ix-1]; ip != i {
					for k := j; k <= j+31; k++ {
						a[i-1+(k-1)*lda], a[ip-1+(k-1)*lda] = a[ip-1+(k-1)*lda], a[i-1+(k-1)*lda]
					}
				}
				ix += incx
			}
		}
	}
	if n32 != n {
		n32++
		ix := ix0
		for i := i1; (inc > 0 && i <= i2) || (inc < 0 && i >= i2); i += inc {
			if ip := ipiv[ix-1]; ip != i {
				for k := n32; k <= n; k++ {
					a[i-1+(k-1)*lda], a[ip-1+(k-1)*lda] = a[ip-1+(k-1)*lda], a[i-1+(k-1)*lda]
				}
			}
			ix += incx
		}
	}
}

// Getrf2 is xGETRF2 (dgetrf2.f): LU factorization with partial pivoting by the
// recursive algorithm of Toledo, of the m x n matrix a, m, n >= 1. It returns
// INFO: 0, or i > 0 if U(i,i) is exactly zero.
func Getrf2[T Float](m, n int, a []T, lda int, ipiv []int) (info int) {
	if m == 1 {
		// One row: just the pivot and INFO.
		ipiv[0] = 1
		if a[0] == 0 {
			info = 1
		}
		return info
	}
	if n == 1 {
		// One column: find the pivot, test for singularity, apply the
		// interchange, compute elements 2:M of the column.
		sfmin := SafeMin[T]()
		i := Iamax(m, a, 1)
		ipiv[0] = i
		if a[i-1] == 0 {
			return 1
		}
		if i != 1 {
			a[0], a[i-1] = a[i-1], a[0]
		}
		if abs(a[0]) >= sfmin {
			Scal(m-1, 1/a[0], a[1:], 1)
		} else {
			for i := 1; i < m; i++ {
				a[i] = a[i] / a[0]
			}
		}
		return 0
	}
	// Recursive code: factor [A11; A21], apply the interchanges to [A12; A22],
	// solve A12, update A22, factor A22, adjust INFO and the pivots, apply the
	// interchanges to A21.
	mn := min(m, n)
	n1 := mn / 2
	n2 := n - n1
	iinfo := Getrf2(m, n1, a, lda, ipiv)
	if info == 0 && iinfo > 0 {
		info = iinfo
	}
	Laswp(n2, a[n1*lda:], lda, 1, n1, ipiv, 1)
	Trsm(true, false, true, false, n1, n2, 1, a, lda, a[n1*lda:], lda)
	Gemm(true, true, m-n1, n2, n1, -1, a[n1:], lda, a[n1*lda:], lda, 1, a[n1+n1*lda:], lda)
	iinfo = Getrf2(m-n1, n2, a[n1+n1*lda:], lda, ipiv[n1:])
	if info == 0 && iinfo > 0 {
		info = iinfo + n1
	}
	for i := n1; i < mn; i++ {
		ipiv[i] += n1
	}
	Laswp(n1, a, lda, n1+1, mn, ipiv, 1)
	return info
}

// Getrf is xGETRF (dgetrf.f): the blocked LU factorization with partial pivoting
// of the m x n matrix a, m, n >= 1, blocks of nb columns factored by Getrf2 (nb =
// NB is what the reference uses; nb <= 1 or nb >= min(m, n) is the unblocked
// code). It returns INFO: 0, or i > 0 if U(i,i) is exactly zero.
func Getrf[T Float](nb, m, n int, a []T, lda int, ipiv []int) (info int) {
	mn := min(m, n)
	if nb <= 1 || nb >= mn {
		return Getrf2(m, n, a, lda, ipiv)
	}
	for j := 1; j <= mn; j += nb {
		jb := min(mn-j+1, nb)
		// Factor the diagonal and subdiagonal blocks and test for exact
		// singularity.
		iinfo := Getrf2(m-j+1, jb, a[j-1+(j-1)*lda:], lda, ipiv[j-1:])
		// Adjust INFO and the pivot indices.
		if info == 0 && iinfo > 0 {
			info = iinfo + j - 1
		}
		for i := j; i <= min(m, j+jb-1); i++ {
			ipiv[i-1] += j - 1
		}
		// Apply the interchanges to columns 1:J-1.
		Laswp(j-1, a, lda, j, j+jb-1, ipiv, 1)
		if j+jb <= n {
			// Apply the interchanges to columns J+JB:N, compute the block row
			// of U, update the trailing submatrix.
			Laswp(n-j-jb+1, a[(j+jb-1)*lda:], lda, j, j+jb-1, ipiv, 1)
			Trsm(true, false, true, false, jb, n-j-jb+1, 1, a[j-1+(j-1)*lda:], lda, a[j-1+(j+jb-1)*lda:], lda)
			if j+jb <= m {
				Gemm(true, true, m-j-jb+1, n-j-jb+1, jb, -1, a[j+jb-1+(j-1)*lda:], lda, a[j-1+(j+jb-1)*lda:], lda, 1, a[j+jb-1+(j+jb-1)*lda:], lda)
			}
		}
	}
	return info
}

// Getrs is xGETRS (dgetrs.f): solves A*X = B (notran) or A**T*X = B with the LU
// factors Getrf computed, n, nrhs >= 1.
func Getrs[T Float](notran bool, n, nrhs int, a []T, lda int, ipiv []int, b []T, ldb int) {
	if notran {
		// Apply the interchanges to the right-hand sides, solve L*X = B then
		// U*X = B.
		Laswp(nrhs, b, ldb, 1, n, ipiv, 1)
		Trsm(true, false, true, false, n, nrhs, 1, a, lda, b, ldb)
		Trsm(true, true, true, true, n, nrhs, 1, a, lda, b, ldb)
		return
	}
	// Solve U**T*X = B then L**T*X = B, apply the interchanges to the solution.
	Trsm(true, true, false, true, n, nrhs, 1, a, lda, b, ldb)
	Trsm(true, false, false, false, n, nrhs, 1, a, lda, b, ldb)
	Laswp(nrhs, b, ldb, 1, n, ipiv, -1)
}

// Potrf2 is xPOTRF2 (dpotrf2.f): the Cholesky factorization of the n x n
// symmetric positive definite matrix a, n >= 1, by recursion on its halves,
// A = U**T*U (upper) or A = L*L**T. It returns INFO: 0, or i > 0 if the leading
// minor of order i is not positive definite (a NaN counts as not positive).
func Potrf2[T Float](upper bool, n int, a []T, lda int) (info int) {
	if n == 1 {
		// Test for non-positive-definiteness, then factor.
		if a[0] <= 0 || math.IsNaN(float64(a[0])) {
			return 1
		}
		a[0] = sqrt(a[0])
		return 0
	}
	n1 := n / 2
	n2 := n - n1
	// Factor A11.
	if iinfo := Potrf2(upper, n1, a, lda); iinfo != 0 {
		return iinfo
	}
	if upper {
		// Update and scale A12, update and factor A22.
		Trsm(true, true, false, true, n1, n2, 1, a, lda, a[n1*lda:], lda)
		Syrk(true, false, n2, n1, -1, a[n1*lda:], lda, 1, a[n1+n1*lda:], lda)
	} else {
		// Update and scale A21, update and factor A22.
		Trsm(false, false, false, true, n2, n1, 1, a, lda, a[n1:], lda)
		Syrk(false, true, n2, n1, -1, a[n1:], lda, 1, a[n1+n1*lda:], lda)
	}
	if iinfo := Potrf2(upper, n2, a[n1+n1*lda:], lda); iinfo != 0 {
		return iinfo + n1
	}
	return 0
}

// Potrf is xPOTRF (dpotrf.f): the blocked Cholesky factorization of the n x n
// matrix a, n >= 1, diagonal blocks of nb rows factored by Potrf2 (nb = NB is
// what the reference uses; nb <= 1 or nb >= n is the unblocked code). It returns
// INFO as Potrf2 does, the index counted from the first row of a.
func Potrf[T Float](nb int, upper bool, n int, a []T, lda int) (info int) {
	if nb <= 1 || nb >= n {
		return Potrf2(upper, n, a, lda)
	}
	for j := 1; j <= n; j += nb {
		// Update and factorize the current diagonal block and test for
		// non-positive-definiteness; compute the current block row or column.
		jb := min(nb, n-j+1)
		jj := j - 1 + (j-1)*lda
		if upper {
			if j > 1 {
				Syrk(true, false, jb, j-1, -1, a[(j-1)*lda:], lda, 1, a[jj:], lda)
			}
			if info = Potrf2(true, jb, a[jj:], lda); info != 0 {
				return info + j - 1
			}
			if j+jb <= n {
				if j > 1 {
					Gemm(false, true, jb, n-j-jb+1, j-1, -1, a[(j-1)*lda:], lda, a[(j+jb-1)*lda:], lda, 1, a[j-1+(j+jb-1)*lda:], lda)
				}
				Trsm(true, true, false, true, jb, n-j-jb+1, 1, a[jj:], lda, a[j-1+(j+jb-1)*lda:], lda)
			}
			continue
		}
		if j > 1 {
			Syrk(false, true, jb, j-1, -1, a[j-1:], lda, 1, a[jj:], lda)
		}
		if info = Potrf2(false, jb, a[jj:], lda); info != 0 {
			return info + j - 1
		}
		if j+jb <= n {
			if j > 1 {
				Gemm(true, false, n-j-jb+1, jb, j-1, -1, a[j+jb-1:], lda, a[j-1:], lda, 1, a[j+jb-1+(j-1)*lda:], lda)
			}
			Trsm(false, false, false, true, n-j-jb+1, jb, 1, a[jj:], lda, a[j+jb-1+(j-1)*lda:], lda)
		}
	}
	return 0
}

// Potrs is xPOTRS (dpotrs.f): solves A*X = B with the Cholesky factor Potrf
// computed, n, nrhs >= 1.
func Potrs[T Float](upper bool, n, nrhs int, a []T, lda int, b []T, ldb int) {
	if upper {
		// A = U**T*U: solve U**T*X = B, then U*X = B.
		Trsm(true, true, false, true, n, nrhs, 1, a, lda, b, ldb)
		Trsm(true, true, true, true, n, nrhs, 1, a, lda, b, ldb)
		return
	}
	// A = L*L**T: solve L*X = B, then L**T*X = B.
	Trsm(true, false, true, true, n, nrhs, 1, a, lda, b, ldb)
	Trsm(true, false, false, true, n, nrhs, 1, a, lda, b, ldb)
}
