package ref

// The level 3 routines beyond Gemm: dtrsm.f, dtrmm.f, dsyrk.f, dsyr2k.f and
// dsymm.f of Reference-LAPACK v3.12.0 (BLAS/SRC), and their single precision
// twins, in the reference's loop order. Each is the Fortran after its argument
// tests and its quick return, which the caller has done: a Fortran line
// `B(I,J) = B(I,J) - B(K,J)*A(I,K)` is `b[i+j*ldb] = b[i+j*ldb] - T(b[k+j*ldb]*a[i+k*lda])`,
// the product wrapped in a conversion so that it is rounded and never fused. The
// reference's `IF (X.NE.ZERO)` skips are kept: they decide which products are
// formed, and so how a NaN or an infinity travels.

// Trsm is xTRSM: B := alpha*inv(op(A))*B (left) or B := alpha*B*inv(op(A))
// (right), A triangular, m x m for left and n x n for right, B m x n. upper says
// which triangle of A is referenced, notA that op(A) = A, nounit that the
// diagonal of A is used and not taken to be one. alpha = 0 sets B to zero, and
// does not read A.
func Trsm[T Float](left, upper, notA, nounit bool, m, n int, alpha T, a []T, lda int, b []T, ldb int) {
	// And when alpha.eq.zero.
	if alpha == 0 {
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				b[i+j*ldb] = 0
			}
		}
		return
	}
	switch {
	case left && notA && upper:
		// Form B := alpha*inv(A)*B.
		for j := 0; j < n; j++ {
			if alpha != 1 {
				for i := 0; i < m; i++ {
					b[i+j*ldb] = T(alpha * b[i+j*ldb])
				}
			}
			for k := m - 1; k >= 0; k-- {
				if b[k+j*ldb] != 0 {
					if nounit {
						b[k+j*ldb] = b[k+j*ldb] / a[k+k*lda]
					}
					for i := 0; i < k; i++ {
						b[i+j*ldb] = b[i+j*ldb] - T(b[k+j*ldb]*a[i+k*lda])
					}
				}
			}
		}
	case left && notA:
		for j := 0; j < n; j++ {
			if alpha != 1 {
				for i := 0; i < m; i++ {
					b[i+j*ldb] = T(alpha * b[i+j*ldb])
				}
			}
			for k := 0; k < m; k++ {
				if b[k+j*ldb] != 0 {
					if nounit {
						b[k+j*ldb] = b[k+j*ldb] / a[k+k*lda]
					}
					for i := k + 1; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] - T(b[k+j*ldb]*a[i+k*lda])
					}
				}
			}
		}
	case left && upper:
		// Form B := alpha*inv(A**T)*B.
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				temp := T(alpha * b[i+j*ldb])
				for k := 0; k < i; k++ {
					temp = temp - T(a[k+i*lda]*b[k+j*ldb])
				}
				if nounit {
					temp = temp / a[i+i*lda]
				}
				b[i+j*ldb] = temp
			}
		}
	case left:
		for j := 0; j < n; j++ {
			for i := m - 1; i >= 0; i-- {
				temp := T(alpha * b[i+j*ldb])
				for k := i + 1; k < m; k++ {
					temp = temp - T(a[k+i*lda]*b[k+j*ldb])
				}
				if nounit {
					temp = temp / a[i+i*lda]
				}
				b[i+j*ldb] = temp
			}
		}
	case notA && upper:
		// Form B := alpha*B*inv(A).
		for j := 0; j < n; j++ {
			if alpha != 1 {
				for i := 0; i < m; i++ {
					b[i+j*ldb] = T(alpha * b[i+j*ldb])
				}
			}
			for k := 0; k < j; k++ {
				if a[k+j*lda] != 0 {
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] - T(a[k+j*lda]*b[i+k*ldb])
					}
				}
			}
			if nounit {
				temp := T(1) / a[j+j*lda]
				for i := 0; i < m; i++ {
					b[i+j*ldb] = T(temp * b[i+j*ldb])
				}
			}
		}
	case notA:
		for j := n - 1; j >= 0; j-- {
			if alpha != 1 {
				for i := 0; i < m; i++ {
					b[i+j*ldb] = T(alpha * b[i+j*ldb])
				}
			}
			for k := j + 1; k < n; k++ {
				if a[k+j*lda] != 0 {
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] - T(a[k+j*lda]*b[i+k*ldb])
					}
				}
			}
			if nounit {
				temp := T(1) / a[j+j*lda]
				for i := 0; i < m; i++ {
					b[i+j*ldb] = T(temp * b[i+j*ldb])
				}
			}
		}
	case upper:
		// Form B := alpha*B*inv(A**T).
		for k := n - 1; k >= 0; k-- {
			if nounit {
				temp := T(1) / a[k+k*lda]
				for i := 0; i < m; i++ {
					b[i+k*ldb] = T(temp * b[i+k*ldb])
				}
			}
			for j := 0; j < k; j++ {
				if a[j+k*lda] != 0 {
					temp := a[j+k*lda]
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] - T(temp*b[i+k*ldb])
					}
				}
			}
			if alpha != 1 {
				for i := 0; i < m; i++ {
					b[i+k*ldb] = T(alpha * b[i+k*ldb])
				}
			}
		}
	default:
		for k := 0; k < n; k++ {
			if nounit {
				temp := T(1) / a[k+k*lda]
				for i := 0; i < m; i++ {
					b[i+k*ldb] = T(temp * b[i+k*ldb])
				}
			}
			for j := k + 1; j < n; j++ {
				if a[j+k*lda] != 0 {
					temp := a[j+k*lda]
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] - T(temp*b[i+k*ldb])
					}
				}
			}
			if alpha != 1 {
				for i := 0; i < m; i++ {
					b[i+k*ldb] = T(alpha * b[i+k*ldb])
				}
			}
		}
	}
}

// Trmm is xTRMM: B := alpha*op(A)*B (left) or B := alpha*B*op(A) (right), with
// the arguments of Trsm. alpha = 0 sets B to zero, and does not read A.
func Trmm[T Float](left, upper, notA, nounit bool, m, n int, alpha T, a []T, lda int, b []T, ldb int) {
	// And when alpha.eq.zero.
	if alpha == 0 {
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				b[i+j*ldb] = 0
			}
		}
		return
	}
	switch {
	case left && notA && upper:
		// Form B := alpha*A*B.
		for j := 0; j < n; j++ {
			for k := 0; k < m; k++ {
				if b[k+j*ldb] != 0 {
					temp := T(alpha * b[k+j*ldb])
					for i := 0; i < k; i++ {
						b[i+j*ldb] = b[i+j*ldb] + T(temp*a[i+k*lda])
					}
					if nounit {
						temp = T(temp * a[k+k*lda])
					}
					b[k+j*ldb] = temp
				}
			}
		}
	case left && notA:
		for j := 0; j < n; j++ {
			for k := m - 1; k >= 0; k-- {
				if b[k+j*ldb] != 0 {
					temp := T(alpha * b[k+j*ldb])
					b[k+j*ldb] = temp
					if nounit {
						b[k+j*ldb] = T(b[k+j*ldb] * a[k+k*lda])
					}
					for i := k + 1; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] + T(temp*a[i+k*lda])
					}
				}
			}
		}
	case left && upper:
		// Form B := alpha*A**T*B.
		for j := 0; j < n; j++ {
			for i := m - 1; i >= 0; i-- {
				temp := b[i+j*ldb]
				if nounit {
					temp = T(temp * a[i+i*lda])
				}
				for k := 0; k < i; k++ {
					temp = temp + T(a[k+i*lda]*b[k+j*ldb])
				}
				b[i+j*ldb] = T(alpha * temp)
			}
		}
	case left:
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				temp := b[i+j*ldb]
				if nounit {
					temp = T(temp * a[i+i*lda])
				}
				for k := i + 1; k < m; k++ {
					temp = temp + T(a[k+i*lda]*b[k+j*ldb])
				}
				b[i+j*ldb] = T(alpha * temp)
			}
		}
	case notA && upper:
		// Form B := alpha*B*A.
		for j := n - 1; j >= 0; j-- {
			temp := alpha
			if nounit {
				temp = T(temp * a[j+j*lda])
			}
			for i := 0; i < m; i++ {
				b[i+j*ldb] = T(temp * b[i+j*ldb])
			}
			for k := 0; k < j; k++ {
				if a[k+j*lda] != 0 {
					temp = T(alpha * a[k+j*lda])
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] + T(temp*b[i+k*ldb])
					}
				}
			}
		}
	case notA:
		for j := 0; j < n; j++ {
			temp := alpha
			if nounit {
				temp = T(temp * a[j+j*lda])
			}
			for i := 0; i < m; i++ {
				b[i+j*ldb] = T(temp * b[i+j*ldb])
			}
			for k := j + 1; k < n; k++ {
				if a[k+j*lda] != 0 {
					temp = T(alpha * a[k+j*lda])
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] + T(temp*b[i+k*ldb])
					}
				}
			}
		}
	case upper:
		// Form B := alpha*B*A**T.
		for k := 0; k < n; k++ {
			for j := 0; j < k; j++ {
				if a[j+k*lda] != 0 {
					temp := T(alpha * a[j+k*lda])
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] + T(temp*b[i+k*ldb])
					}
				}
			}
			temp := alpha
			if nounit {
				temp = T(temp * a[k+k*lda])
			}
			if temp != 1 {
				for i := 0; i < m; i++ {
					b[i+k*ldb] = T(temp * b[i+k*ldb])
				}
			}
		}
	default:
		for k := n - 1; k >= 0; k-- {
			for j := k + 1; j < n; j++ {
				if a[j+k*lda] != 0 {
					temp := T(alpha * a[j+k*lda])
					for i := 0; i < m; i++ {
						b[i+j*ldb] = b[i+j*ldb] + T(temp*b[i+k*ldb])
					}
				}
			}
			temp := alpha
			if nounit {
				temp = T(temp * a[k+k*lda])
			}
			if temp != 1 {
				for i := 0; i < m; i++ {
					b[i+k*ldb] = T(temp * b[i+k*ldb])
				}
			}
		}
	}
}

// triRange is the rows of column j of an n x n triangle: 0..j for the upper
// triangle, j..n-1 for the lower one, as the half-open [lo, hi).
func triRange(upper bool, j, n int) (lo, hi int) {
	if upper {
		return 0, j + 1
	}
	return j, n
}

// Syrk is xSYRK: C := alpha*A*A**T + beta*C (notA) or C := alpha*A**T*A + beta*C,
// C n x n symmetric of which only the triangle upper selects is referenced and
// written, A n x k (notA) or k x n. alpha = 0 does not read A, beta = 0 does not
// read C.
func Syrk[T Float](upper, notA bool, n, k int, alpha T, a []T, lda int, beta T, c []T, ldc int) {
	// And when alpha.eq.zero.
	if alpha == 0 {
		for j := 0; j < n; j++ {
			lo, hi := triRange(upper, j, n)
			for i := lo; i < hi; i++ {
				if beta == 0 {
					c[i+j*ldc] = 0
				} else {
					c[i+j*ldc] = T(beta * c[i+j*ldc])
				}
			}
		}
		return
	}
	if notA {
		// Form C := alpha*A*A**T + beta*C.
		for j := 0; j < n; j++ {
			lo, hi := triRange(upper, j, n)
			if beta == 0 {
				for i := lo; i < hi; i++ {
					c[i+j*ldc] = 0
				}
			} else if beta != 1 {
				for i := lo; i < hi; i++ {
					c[i+j*ldc] = T(beta * c[i+j*ldc])
				}
			}
			for l := 0; l < k; l++ {
				if a[j+l*lda] != 0 {
					temp := T(alpha * a[j+l*lda])
					for i := lo; i < hi; i++ {
						c[i+j*ldc] = c[i+j*ldc] + T(temp*a[i+l*lda])
					}
				}
			}
		}
		return
	}
	// Form C := alpha*A**T*A + beta*C.
	for j := 0; j < n; j++ {
		lo, hi := triRange(upper, j, n)
		for i := lo; i < hi; i++ {
			var temp T
			for l := 0; l < k; l++ {
				temp = temp + T(a[l+i*lda]*a[l+j*lda])
			}
			if beta == 0 {
				c[i+j*ldc] = T(alpha * temp)
			} else {
				c[i+j*ldc] = T(alpha*temp) + T(beta*c[i+j*ldc])
			}
		}
	}
}

// Syr2k is xSYR2K: C := alpha*A*B**T + alpha*B*A**T + beta*C (notA) or C :=
// alpha*A**T*B + alpha*B**T*A + beta*C, with the arguments of Syrk, B of the
// shape of A. alpha = 0 does not read A or B, beta = 0 does not read C.
func Syr2k[T Float](upper, notA bool, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {
	// And when alpha.eq.zero.
	if alpha == 0 {
		for j := 0; j < n; j++ {
			lo, hi := triRange(upper, j, n)
			for i := lo; i < hi; i++ {
				if beta == 0 {
					c[i+j*ldc] = 0
				} else {
					c[i+j*ldc] = T(beta * c[i+j*ldc])
				}
			}
		}
		return
	}
	if notA {
		// Form C := alpha*A*B**T + alpha*B*A**T + C.
		for j := 0; j < n; j++ {
			lo, hi := triRange(upper, j, n)
			if beta == 0 {
				for i := lo; i < hi; i++ {
					c[i+j*ldc] = 0
				}
			} else if beta != 1 {
				for i := lo; i < hi; i++ {
					c[i+j*ldc] = T(beta * c[i+j*ldc])
				}
			}
			for l := 0; l < k; l++ {
				if a[j+l*lda] != 0 || b[j+l*ldb] != 0 {
					temp1 := T(alpha * b[j+l*ldb])
					temp2 := T(alpha * a[j+l*lda])
					for i := lo; i < hi; i++ {
						c[i+j*ldc] = c[i+j*ldc] + T(a[i+l*lda]*temp1) + T(b[i+l*ldb]*temp2)
					}
				}
			}
		}
		return
	}
	// Form C := alpha*A**T*B + alpha*B**T*A + C.
	for j := 0; j < n; j++ {
		lo, hi := triRange(upper, j, n)
		for i := lo; i < hi; i++ {
			var temp1, temp2 T
			for l := 0; l < k; l++ {
				temp1 = temp1 + T(a[l+i*lda]*b[l+j*ldb])
				temp2 = temp2 + T(b[l+i*ldb]*a[l+j*lda])
			}
			if beta == 0 {
				c[i+j*ldc] = T(alpha*temp1) + T(alpha*temp2)
			} else {
				c[i+j*ldc] = T(beta*c[i+j*ldc]) + T(alpha*temp1) + T(alpha*temp2)
			}
		}
	}
}

// Symm is xSYMM: C := alpha*A*B + beta*C (left) or C := alpha*B*A + beta*C, A
// symmetric, m x m for left and n x n for right, of which only the triangle
// upper selects is referenced; B and C m x n. alpha = 0 does not read A or B,
// beta = 0 does not read C.
func Symm[T Float](left, upper bool, m, n int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {
	// And when alpha.eq.zero.
	if alpha == 0 {
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				if beta == 0 {
					c[i+j*ldc] = 0
				} else {
					c[i+j*ldc] = T(beta * c[i+j*ldc])
				}
			}
		}
		return
	}
	if left {
		// Form C := alpha*A*B + beta*C.
		for j := 0; j < n; j++ {
			for step := 0; step < m; step++ {
				i := step
				if !upper {
					i = m - 1 - step
				}
				temp1 := T(alpha * b[i+j*ldb])
				var temp2 T
				lo, hi := 0, i // the rows of column i of A on the side of the diagonal that is stored
				if !upper {
					lo, hi = i+1, m
				}
				for k := lo; k < hi; k++ {
					c[k+j*ldc] = c[k+j*ldc] + T(temp1*a[k+i*lda])
					temp2 = temp2 + T(b[k+j*ldb]*a[k+i*lda])
				}
				if beta == 0 {
					c[i+j*ldc] = T(temp1*a[i+i*lda]) + T(alpha*temp2)
				} else {
					c[i+j*ldc] = T(beta*c[i+j*ldc]) + T(temp1*a[i+i*lda]) + T(alpha*temp2)
				}
			}
		}
		return
	}
	// Form C := alpha*B*A + beta*C.
	for j := 0; j < n; j++ {
		temp1 := T(alpha * a[j+j*lda])
		if beta == 0 {
			for i := 0; i < m; i++ {
				c[i+j*ldc] = T(temp1 * b[i+j*ldb])
			}
		} else {
			for i := 0; i < m; i++ {
				c[i+j*ldc] = T(beta*c[i+j*ldc]) + T(temp1*b[i+j*ldb])
			}
		}
		for k := 0; k < j; k++ {
			if upper {
				temp1 = T(alpha * a[k+j*lda])
			} else {
				temp1 = T(alpha * a[j+k*lda])
			}
			for i := 0; i < m; i++ {
				c[i+j*ldc] = c[i+j*ldc] + T(temp1*b[i+k*ldb])
			}
		}
		for k := j + 1; k < n; k++ {
			if upper {
				temp1 = T(alpha * a[j+k*lda])
			} else {
				temp1 = T(alpha * a[k+j*lda])
			}
			for i := 0; i < m; i++ {
				c[i+j*ldc] = c[i+j*ldc] + T(temp1*b[i+k*ldb])
			}
		}
	}
}
