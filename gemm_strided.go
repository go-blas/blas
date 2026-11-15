package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/kernels"
)

// DgemmStrided computes C := alpha*A*B + beta*C for an m x k matrix A, a k x n
// matrix B and an m x n matrix C, each given with a row stride and a column
// stride: element (i, j) of A is a[i*rsa+j*csa], and likewise B and C. It is
// the general-stride form of BLIS's bli_dgemm (see docs/design.md), without the
// transpose flags, which a stride swap expresses: a transposed operand is the
// same data with rs and cs exchanged. A row-major matrix has csa = 1, a
// column-major one rsa = 1, a sliced or decimated view anything else, and none
// of them is copied: the packing of the kernel reads the operands through
// their strides, as Dgemm's does through lda.
//
// The parameters are numbered as in the signature, from 1: m, n, k, alpha (4), a
// (5), rsa (6), csa (7), b (8), rsb (9), csb (10), beta (11), c (12), rsc (13),
// csc (14). Every stride is at least 1. C must be row-major (csc = 1, rsc >=
// n) or column-major (rsc = 1, csc >= m): a C with both strides above 1 has no
// kernel form and is refused as parameter 13, and a column-major C whose csc is
// below m as 14. An invalid argument returns an *ArgError with Routine
// "DGEMMSTRIDED"; a slice is too short when it holds less than
// 1+(rows-1)*rs+(cols-1)*cs elements, reported against the slice's own number
// (5, 8, 12), the length being computed without overflow.
//
// Beta, alpha, the quick returns, the error bound and the aliasing rule (C must
// not overlap A or B) are those of Dgemm; the product runs on the same packed,
// multicore kernel. The elements of the slices that are not elements of a
// matrix are never read or written.
func DgemmStrided(m, n, k int, alpha float64, a []float64, rsa, csa int, b []float64, rsb, csb int, beta float64, c []float64, rsc, csc int) error {
	return gemmStrided("DGEMMSTRIDED", m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, dHW)
}

// SgemmStrided is DgemmStrided in single precision (SGEMMSTRIDED). Without a
// float32 kernel on the running CPU it runs plain loops.
func SgemmStrided(m, n, k int, alpha float32, a []float32, rsa, csa int, b []float32, rsb, csb int, beta float32, c []float32, rsc, csc int) error {
	return gemmStrided("SGEMMSTRIDED", m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, sHW)
}

func gemmStrided[T real](name string, m, n, k int, alpha T, a []T, rsa, csa int, b []T, rsb, csb int, beta T, c []T, rsc, csc int, h *hw[T]) error {
	if p := check.GemmStrided(m, n, k, rsa, csa, rsb, csb, rsc, csc); p != 0 { // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 || ((alpha == 0 || k == 0) && beta == 1) { // guard
		return nil
	}
	if p := check.GemmStridedLens(m, n, k, rsa, csa, rsb, csb, rsc, csc, len(a), len(b), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	// Nothing to multiply, or no kernel: plain loops, which also carry the
	// alpha = 0 and k = 0 rules.
	if alpha == 0 || k == 0 || h.mul == nil { // guard
		stridedRef(m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
		return nil
	}
	// C := beta*C, without reading C when beta is 0 (it may hold NaN).
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			switch beta {
			case 0:
				c[i*rsc+j*csc] = 0
			case 1:
			default:
				c[i*rsc+j*csc] *= beta
			}
		}
	}
	if alpha == 1 {
		// With both strides 1, C is a single row (m = 1) or column (n = 1):
		// row-major when it has the width for it, else column-major.
		stridedProduct(h, c, csc == 1 && rsc >= n, max(rsc, csc), a, rsa, csa, b, rsb, csb, m, n, k)
		return nil
	}
	// The kernel has no alpha: form the product in a scratch matrix, row-major,
	// then add alpha times it.
	t := getScratch[T](&h.scratch, m*n)
	stridedProduct(h, t, true, n, a, rsa, csa, b, rsb, csb, m, n, k)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			c[i*rsc+j*csc] += alpha * t[i*n+j]
		}
	}
	h.scratch.Put(&t)
	return nil
}

// stridedProduct adds a*b to dst, row-major with rows ld apart when rowMajor,
// else column-major with columns ld apart (the transpose identity: the kernel
// computes B**T * A**T into the row-major C**T).
func stridedProduct[T real](h *hw[T], dst []T, rowMajor bool, ld int, a []T, rsa, csa int, b []T, rsb, csb int, m, n, k int) {
	if rowMajor {
		h.mulS(dst, ld, kernels.MatOf[T]{Data: a, RS: rsa, CS: csa}, kernels.MatOf[T]{Data: b, RS: rsb, CS: csb}, m, k, n)
		return
	}
	h.mulS(dst, ld, kernels.MatOf[T]{Data: b, RS: csb, CS: rsb}, kernels.MatOf[T]{Data: a, RS: csa, CS: rsa}, n, k, m)
}

// stridedRef is the reference loops of DGEMM over strided operands, which also
// carry the alpha = 0 and k = 0 rules.
func stridedRef[T real](m, n, k int, alpha T, a []T, rsa, csa int, b []T, rsb, csb int, beta T, c []T, rsc, csc int) {
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			switch beta {
			case 0:
				c[i*rsc+j*csc] = 0
			case 1:
			default:
				c[i*rsc+j*csc] *= beta
			}
		}
		if alpha == 0 {
			continue
		}
		for l := 0; l < k; l++ {
			t := alpha * b[l*rsb+j*csb]
			for i := 0; i < m; i++ {
				c[i*rsc+j*csc] += t * a[i*rsa+l*csa]
			}
		}
	}
}
