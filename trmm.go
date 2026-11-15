package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// Dtrmm multiplies by a triangular matrix, in place: B := alpha*op(A)*B for side
// Left, B := alpha*B*op(A) for side Right, where op(A) is A, or A**T for Trans
// and ConjTrans, and B is m x n. A is triangular, m x m for Left and n x n for
// Right, of which only the triangle uplo names is referenced, and with diag Unit
// its diagonal is not read and taken to be one. All matrices are column-major.
// It is DTRMM of the reference BLAS, parameter for parameter, with the
// parameters, the quick returns and the slice checks of Dtrsm: side 1, uplo 2,
// transA 3, diag 4, m 5, n 6, alpha 7, a 8, lda 9, b 10, ldb 11. With alpha = 0, B
// is set to zero and A is not read.
//
// The product is blocked: GEMM updates on the packed kernel, and the reference's
// loops on diagonal blocks of blockSize rows. With s the order of A, it is within
//
//	|B^ - alpha op(A) B| <= gamma_{s+3} |alpha| |op(A)| |B|   elementwise (B op(A) for Right)
//
// of the exact result, not bit-identical to netlib.Dtrmm. It is not safe to call
// with b overlapping a.
func Dtrmm(side Side, uplo Uplo, tA Transpose, diag Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) error {
	return trmm("DTRMM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb, dHW)
}

// Strmm is Dtrmm in single precision (STRMM). Without a float32 kernel on the
// running CPU it runs the reference loops of netlib.Strmm.
func Strmm(side Side, uplo Uplo, tA Transpose, diag Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) error {
	return trmm("STRMM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb, sHW)
}

func trmm[T real](name string, side Side, uplo Uplo, tA Transpose, diag Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int, h *hw[T]) error {
	side, uplo, tA, diag = Side(check.Upper(byte(side))), Uplo(check.Upper(byte(uplo))), Transpose(check.Upper(byte(tA))), Diag(check.Upper(byte(diag))) // LSAME ignores case
	if p := check.Tr(byte(side), byte(uplo), byte(tA), byte(diag), m, n, lda, ldb); p != 0 {                                                             // guard
		return argErr(name, p)
	}
	// Quick return if possible.
	if m == 0 || n == 0 { // guard
		return nil
	}
	if p := check.TrLens(byte(side), m, n, lda, ldb, len(a), len(b), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	left, upper, notA, nounit := side == Left, uplo == Upper, tA == NoTrans, diag == NonUnit
	// Nothing to multiply, or no kernel: the reference loops, which also carry the
	// alpha = 0 rule.
	if alpha == 0 || h.mul == nil { // guard
		ref.Trmm(left, upper, notA, nounit, m, n, alpha, a, lda, b, ldb)
		return nil
	}
	nb := blockSize
	// A block of the result is the diagonal block times the block itself, in
	// place, plus the GEMM of the other blocks of op(A) in its row (Left) or
	// column (Right) with blocks of B that must still be the original ones, which
	// fixes the order: Left goes from the first block row to the last when op(A)
	// is upper (the rows below, still original, are the ones it reads), from the
	// last to the first when it is lower; Right goes from the last block column
	// to the first when op(A) is upper, from the first to the last when it is
	// lower.
	if left {
		if upper == notA { // op(A) upper
			for i0 := 0; i0 < m; i0 += nb {
				ib := min(nb, m-i0)
				ref.Trmm(true, upper, notA, nounit, ib, n, alpha, a[i0+i0*lda:], lda, b[i0:], ldb)
				if i1 := i0 + ib; i1 < m {
					gemmCore(tA, NoTrans, ib, n, m-i1, alpha, a[opOff(tA, lda, i0, i1):], lda, b[i1:], ldb, 1, b[i0:], ldb, h)
				}
			}
			return nil
		}
		for i1 := m; i1 > 0; i1 -= nb {
			i0 := max(0, i1-nb)
			ib := i1 - i0
			ref.Trmm(true, upper, notA, nounit, ib, n, alpha, a[i0+i0*lda:], lda, b[i0:], ldb)
			if i0 > 0 {
				gemmCore(tA, NoTrans, ib, n, i0, alpha, a[opOff(tA, lda, i0, 0):], lda, b, ldb, 1, b[i0:], ldb, h)
			}
		}
		return nil
	}
	if upper == notA { // op(A) upper
		for j1 := n; j1 > 0; j1 -= nb {
			j0 := max(0, j1-nb)
			jb := j1 - j0
			ref.Trmm(false, upper, notA, nounit, m, jb, alpha, a[j0+j0*lda:], lda, b[j0*ldb:], ldb)
			if j0 > 0 {
				gemmCore(NoTrans, tA, m, jb, j0, alpha, b, ldb, a[opOff(tA, lda, 0, j0):], lda, 1, b[j0*ldb:], ldb, h)
			}
		}
		return nil
	}
	for j0 := 0; j0 < n; j0 += nb {
		jb := min(nb, n-j0)
		ref.Trmm(false, upper, notA, nounit, m, jb, alpha, a[j0+j0*lda:], lda, b[j0*ldb:], ldb)
		if j1 := j0 + jb; j1 < n {
			gemmCore(NoTrans, tA, m, jb, n-j1, alpha, b[j1*ldb:], ldb, a[opOff(tA, lda, j1, j0):], lda, 1, b[j0*ldb:], ldb, h)
		}
	}
	return nil
}
