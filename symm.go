package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// Dsymm multiplies by a symmetric matrix: C := alpha*A*B + beta*C for side Left,
// C := alpha*B*A + beta*C for side Right. A is symmetric, m x m for Left and n x n
// for Right, of which only the triangle uplo names is referenced; B and C are
// m x n. All matrices are column-major. It is DSYMM of the reference BLAS,
// parameter for parameter: side 1, uplo 2, m 3, n 4, alpha 5, a 6, lda 7, b 8,
// ldb 9, beta 10, c 11, ldc 12.
//
// With beta = 0, C is not read (it may hold NaN); with alpha = 0, A and B are not
// read (they may be empty); m = 0 or n = 0, and alpha = 0 with beta = 1, return at
// once. An invalid argument returns a *ArgError numbered like xerbla's, among
// them a slice too short for the call: a is checked against lda*(s-1)+s, s the
// order of A, b and c against ldb*(n-1)+m and ldc*(n-1)+m (parameters 6, 8 and 11).
//
// The referenced triangle is copied, exactly, into a dense s x s scratch matrix
// (at most the size of A) and the product is one GEMM of the packed kernel, so the
// result is within the bound of Dgemm, with k = s,
//
//	|C^ - C| <= gamma_{s+2} (|alpha| |A| |B| + |beta| |C0|)   elementwise (|B| |A| for Right)
//
// of the exact result, not bit-identical to netlib.Dsymm. It is not safe to call
// with c overlapping a or b.
func Dsymm(side Side, uplo Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) error {
	return symm("DSYMM", side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, dHW)
}

// Ssymm is Dsymm in single precision (SSYMM). Without a float32 kernel on the
// running CPU it runs the reference loops of netlib.Ssymm.
func Ssymm(side Side, uplo Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	return symm("SSYMM", side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, sHW)
}

func symm[T real](name string, side Side, uplo Uplo, m, n int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int, h *hw[T]) error {
	side, uplo = Side(check.Upper(byte(side))), Uplo(check.Upper(byte(uplo))) // LSAME ignores case
	if p := check.Symm(byte(side), byte(uplo), m, n, lda, ldb, ldc); p != 0 { // guard
		return argErr(name, p)
	}
	// Quick return if possible.
	if m == 0 || n == 0 || (alpha == 0 && beta == 1) { // guard
		return nil
	}
	if p := check.SymmLens(byte(side), m, n, lda, ldb, ldc, len(a), len(b), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	left, upper := side == Left, uplo == Upper
	// Nothing to multiply, or no kernel: the reference loops, which also carry the
	// alpha = 0 rule.
	if alpha == 0 || h.mul == nil { // guard
		ref.Symm(left, upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
		return nil
	}
	s := n
	if left {
		s = m
	}
	// The dense symmetric matrix: element (i, j) is the stored one of (i, j) and
	// (j, i). Its size is s*s <= lda*(s-1)+s <= len(a), so it fits.
	sym := getScratch[T](&h.scratch, s*s)
	for j := 0; j < s; j++ {
		for i := 0; i < s; i++ {
			if (i <= j) == upper {
				sym[i+j*s] = a[i+j*lda]
			} else {
				sym[i+j*s] = a[j+i*lda]
			}
		}
	}
	if left {
		gemmCore(NoTrans, NoTrans, m, n, s, alpha, sym, s, b, ldb, beta, c, ldc, h)
	} else {
		gemmCore(NoTrans, NoTrans, m, n, s, alpha, b, ldb, sym, s, beta, c, ldc, h)
	}
	h.scratch.Put(&sym)
	return nil
}
