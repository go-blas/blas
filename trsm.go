package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// Dtrsm solves a triangular system with m x n right-hand sides B, in place:
// B := alpha*inv(op(A))*B for side Left, B := alpha*B*inv(op(A)) for side Right,
// where op(A) is A, or A**T for Trans and ConjTrans. A is triangular, m x m for
// Left and n x n for Right, of which only the triangle uplo names is referenced,
// and with diag Unit its diagonal is not read and taken to be one. All matrices
// are column-major. It is DTRSM of the reference BLAS, parameter for parameter:
// side 1, uplo 2, transA 3, diag 4, m 5, n 6, alpha 7, a 8, lda 9, b 10, ldb 11.
//
// With alpha = 0, B is set to zero and A is not read (it may be empty); m = 0 or
// n = 0 returns at once. As in the reference there is no singularity test: a zero
// on the diagonal of A gives infinities and NaNs in B. An invalid argument
// returns a *ArgError numbered like xerbla's, among them a slice too short for
// the call: a is checked against lda*(s-1)+s, s the order of A, and b against
// ldb*(n-1)+m (parameters 8 and 10).
//
// The solve is blocked: GEMM updates on the packed kernel, and the reference's
// substitution on diagonal blocks of blockSize rows. Its contract is the
// backward error of substitution, of Higham's Theorem 8.5 (see docs/design.md),
// stated as a residual: with s the order of A and X the computed solution,
//
//	|op(A) X - alpha B| <= gamma_{s+4} (|op(A)| |X| + |alpha B|)   elementwise (Left; X op(A) for Right)
//
// It is not bit-identical to netlib.Dtrsm. It is not safe to call with b
// overlapping a.
func Dtrsm(side Side, uplo Uplo, tA Transpose, diag Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) error {
	return trsm("DTRSM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb, dHW)
}

// Strsm is Dtrsm in single precision (STRSM). Without a float32 kernel on the
// running CPU it runs the reference loops of netlib.Strsm.
func Strsm(side Side, uplo Uplo, tA Transpose, diag Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) error {
	return trsm("STRSM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb, sHW)
}

func trsm[T real](name string, side Side, uplo Uplo, tA Transpose, diag Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int, h *hw[T]) error {
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
	// Nothing to solve, or no kernel: the reference loops, which also carry the
	// alpha = 0 rule.
	if alpha == 0 || h.mul == nil { // guard
		ref.Trsm(left, upper, notA, nounit, m, n, alpha, a, lda, b, ldb)
		return nil
	}
	// B := alpha*B once, then solve with alpha = 1: the reference scales in the
	// same way, column by column, before it substitutes.
	if alpha != 1 {
		scaleCols(m, n, alpha, b, ldb)
	}
	nb := blockSize
	// op(A) is lower triangular for (Lower, NoTrans) and (Upper, Trans), upper for
	// (Upper, NoTrans) and (Lower, Trans). Left solves op(A) X = B block row by
	// block row: from the first to the last when op(A) is lower, from the last to
	// the first when it is upper. Right solves X op(A) = B block column by block
	// column: from the first to the last when op(A) is upper, from the last to the
	// first when it is lower. A block is B minus the GEMM of the blocks already
	// solved, then the reference's substitution on the diagonal block.
	if left {
		if upper != notA { // op(A) lower: forward over the rows of B
			for i0 := 0; i0 < m; i0 += nb {
				ib := min(nb, m-i0)
				if i0 > 0 {
					gemmCore(tA, NoTrans, ib, n, i0, -1, a[opOff(tA, lda, i0, 0):], lda, b, ldb, 1, b[i0:], ldb, h)
				}
				ref.Trsm(true, upper, notA, nounit, ib, n, 1, a[i0+i0*lda:], lda, b[i0:], ldb)
			}
			return nil
		}
		for i1 := m; i1 > 0; i1 -= nb {
			i0 := max(0, i1-nb)
			ib := i1 - i0
			if i1 < m {
				gemmCore(tA, NoTrans, ib, n, m-i1, -1, a[opOff(tA, lda, i0, i1):], lda, b[i1:], ldb, 1, b[i0:], ldb, h)
			}
			ref.Trsm(true, upper, notA, nounit, ib, n, 1, a[i0+i0*lda:], lda, b[i0:], ldb)
		}
		return nil
	}
	if upper == notA { // op(A) upper: forward over the columns of B
		for j0 := 0; j0 < n; j0 += nb {
			jb := min(nb, n-j0)
			if j0 > 0 {
				gemmCore(NoTrans, tA, m, jb, j0, -1, b, ldb, a[opOff(tA, lda, 0, j0):], lda, 1, b[j0*ldb:], ldb, h)
			}
			ref.Trsm(false, upper, notA, nounit, m, jb, 1, a[j0+j0*lda:], lda, b[j0*ldb:], ldb)
		}
		return nil
	}
	for j1 := n; j1 > 0; j1 -= nb {
		j0 := max(0, j1-nb)
		jb := j1 - j0
		if j1 < n {
			gemmCore(NoTrans, tA, m, jb, n-j1, -1, b[j1*ldb:], ldb, a[opOff(tA, lda, j1, j0):], lda, 1, b[j0*ldb:], ldb, h)
		}
		ref.Trsm(false, upper, notA, nounit, m, jb, 1, a[j0+j0*lda:], lda, b[j0*ldb:], ldb)
	}
	return nil
}
