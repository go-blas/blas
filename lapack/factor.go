package lapack

import (
	"math"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/ref"
)

// leaf is the number of columns (getrf: min(m, n); potrf: n) at or below which
// the factorization is plain loops, with no recursion and no BLAS call: above
// it the problem is cut in halves, and the work of a cut is a Trsm and a Gemm
// (Syrk). A variable only so that the tests can make small matrices span many
// levels of the recursion.
var leaf = 16

// Gemm, Trsm and Syrk call the BLAS and refuse to go on if it refuses: the
// arguments are views of matrices already checked.
func (o *ops[T]) Gemm(tA, tB blas.Transpose, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {
	must(o.gemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
}

func (o *ops[T]) Trsm(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int) {
	must(o.trsm(side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb))
}

func (o *ops[T]) Syrk(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha T, a []T, lda int, beta T, c []T, ldc int) {
	must(o.syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc))
}

// laswp applies the row interchanges ipiv[k1-1:k2] (rows 1-based) to the n
// columns of a, in order or, with forward false, in reverse: xLASWP with
// incx = 1 or -1, one column at a time.
func laswp[T real](n int, a []T, lda, k1, k2 int, ipiv []int, forward bool) {
	for c := 0; c < n; c++ {
		col := a[c*lda:]
		if forward {
			for i := k1; i <= k2; i++ {
				if ip := ipiv[i-1]; ip != i {
					col[i-1], col[ip-1] = col[ip-1], col[i-1]
				}
			}
			continue
		}
		for i := k2; i >= k1; i-- {
			if ip := ipiv[i-1]; ip != i {
				col[i-1], col[ip-1] = col[ip-1], col[i-1]
			}
		}
	}
}

// getrfRec factors the m x n matrix a, m, n >= 1, by the recursion of xGETRF2:
// the columns are cut in halves, the left half factored, the interchanges applied
// to the right half, a Trsm for the block row of U, a Gemm for the trailing
// matrix, the right half factored, the interchanges applied to the left. ipiv
// holds 1-based rows relative to this matrix. It returns INFO.
func getrfRec[T real](o *ops[T], m, n int, a []T, lda int, ipiv []int) (info int) {
	mn := min(m, n)
	if mn <= leaf {
		return getf2(m, n, a, lda, ipiv)
	}
	n1 := mn / 2
	n2 := n - n1
	info = getrfRec(o, m, n1, a, lda, ipiv)
	laswp(n2, a[n1*lda:], lda, 1, n1, ipiv, true)
	o.Trsm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit, n1, n2, 1, a, lda, a[n1*lda:], lda)
	o.Gemm(blas.NoTrans, blas.NoTrans, m-n1, n2, n1, -1, a[n1:], lda, a[n1*lda:], lda, 1, a[n1+n1*lda:], lda)
	iinfo := getrfRec(o, m-n1, n2, a[n1+n1*lda:], lda, ipiv[n1:])
	if info == 0 && iinfo > 0 {
		info = iinfo + n1
	}
	for i := n1; i < mn; i++ {
		ipiv[i] += n1
	}
	laswp(n1, a, lda, n1+1, mn, ipiv, true)
	return info
}

// getf2 is the right-looking LU factorization with partial pivoting of the m x n
// matrix a, m, n >= 1, in plain loops: the pivot of column j is the first entry
// of largest magnitude (as IDAMAX: a NaN never replaces it), the row interchange
// is applied to the whole row, the column is scaled by the reciprocal of the
// pivot (by division when the pivot is below the smallest normal number, as the
// reference), and the trailing columns get the rank-1 update. A zero pivot is
// INFO (the first one), nothing else is done for the column, and the
// factorization goes on.
func getf2[T real](m, n int, a []T, lda int, ipiv []int) (info int) {
	sfmin := ref.SafeMin[T]()
	for j := 0; j < min(m, n); j++ {
		col := a[j*lda : j*lda+m]
		p := j + ref.Iamax(m-j, col[j:], 1) - 1
		ipiv[j] = p + 1
		d := col[p]
		if d == 0 {
			if info == 0 {
				info = j + 1
			}
			continue
		}
		if p != j {
			for c := 0; c < n; c++ {
				a[j+c*lda], a[p+c*lda] = a[p+c*lda], a[j+c*lda]
			}
		}
		below := col[j+1:]
		if math.Abs(float64(d)) >= float64(sfmin) {
			r := 1 / d
			for i := range below {
				below[i] *= r
			}
		} else {
			for i := range below {
				below[i] /= d
			}
		}
		for c := j + 1; c < n; c++ {
			u := a[j+c*lda]
			cc := a[c*lda+j+1 : c*lda+m]
			for i, l := range below {
				cc[i] -= l * u
			}
		}
	}
	return info
}

// getrsRec solves with the factors, n, nrhs >= 1, pivots already checked.
func getrsRec[T real](o *ops[T], notran bool, n, nrhs int, a []T, lda int, ipiv []int, b []T, ldb int) {
	if notran {
		laswp(nrhs, b, ldb, 1, n, ipiv, true)
		o.Trsm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit, n, nrhs, 1, a, lda, b, ldb)
		o.Trsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
		return
	}
	o.Trsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
	o.Trsm(blas.Left, blas.Lower, blas.Trans, blas.Unit, n, nrhs, 1, a, lda, b, ldb)
	laswp(nrhs, b, ldb, 1, n, ipiv, false)
}

// potrfRec factors the n x n matrix a, n >= 1, by the recursion of xPOTRF2:
// the order is cut in halves, A11 factored, the off-diagonal block solved (Trsm)
// and the trailing matrix updated (Syrk), A22 factored. It returns INFO.
func potrfRec[T real](o *ops[T], upper bool, n int, a []T, lda int) int {
	if n <= leaf {
		return potf2(upper, n, a, lda)
	}
	n1 := n / 2
	n2 := n - n1
	if iinfo := potrfRec(o, upper, n1, a, lda); iinfo != 0 {
		return iinfo
	}
	if upper {
		o.Trsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, n1, n2, 1, a, lda, a[n1*lda:], lda)
		o.Syrk(blas.Upper, blas.Trans, n2, n1, -1, a[n1*lda:], lda, 1, a[n1+n1*lda:], lda)
	} else {
		o.Trsm(blas.Right, blas.Lower, blas.Trans, blas.NonUnit, n2, n1, 1, a, lda, a[n1:], lda)
		o.Syrk(blas.Lower, blas.NoTrans, n2, n1, -1, a[n1:], lda, 1, a[n1+n1*lda:], lda)
	}
	if iinfo := potrfRec(o, upper, n2, a[n1+n1*lda:], lda); iinfo != 0 {
		return iinfo + n1
	}
	return 0
}

// potf2 is the Cholesky factorization of the n x n matrix a, n >= 1, in plain
// loops: by columns of U with inner products for Upper (every operand is a
// contiguous column), by columns of L with a rank-1 update for Lower. A diagonal
// that is not positive, or is NaN, is INFO (its index, 1-based); the diagonal
// element then holds the value that failed (Upper) or is left (Lower).
func potf2[T real](upper bool, n int, a []T, lda int) int {
	if upper {
		for j := 0; j < n; j++ {
			cj := a[j*lda : j*lda+j]
			d := a[j+j*lda] - dot(cj, cj)
			if !(d > 0) {
				a[j+j*lda] = d
				return j + 1
			}
			d = T(math.Sqrt(float64(d)))
			a[j+j*lda] = d
			for k := j + 1; k < n; k++ {
				a[j+k*lda] = (a[j+k*lda] - dot(cj, a[k*lda:k*lda+j])) / d
			}
		}
		return 0
	}
	for j := 0; j < n; j++ {
		d := a[j+j*lda]
		if !(d > 0) {
			return j + 1
		}
		d = T(math.Sqrt(float64(d)))
		a[j+j*lda] = d
		col := a[j*lda : j*lda+n]
		for i := j + 1; i < n; i++ {
			col[i] /= d
		}
		for k := j + 1; k < n; k++ {
			l := col[k]
			ck := a[k*lda : k*lda+n]
			for i := k; i < n; i++ {
				ck[i] -= col[i] * l
			}
		}
	}
	return 0
}

// dot is the inner product of two vectors of the same length.
func dot[T real](x, y []T) T {
	var s T
	for i, v := range x {
		s += v * y[i]
	}
	return s
}

// potrsRec solves with the factor, n, nrhs >= 1.
func potrsRec[T real](o *ops[T], upper bool, n, nrhs int, a []T, lda int, b []T, ldb int) {
	if upper {
		o.Trsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
		o.Trsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
		return
	}
	o.Trsm(blas.Left, blas.Lower, blas.NoTrans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
	o.Trsm(blas.Left, blas.Lower, blas.Trans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
}
