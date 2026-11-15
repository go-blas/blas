package blas

// The level 3 routines other than Gemm are blocked, as in GotoBLAS and BLIS: the
// triangular (or symmetric) operand is cut into diagonal blocks of blockSize rows
// and columns; the bulk of the work, which is rectangular, is a call of the
// packed GEMM on the blocks off the diagonal, and what stays on the diagonal is
// one small triangular problem per block, run by the reference loops of package
// ref. There is one GEMM, and no kernel of its own: the blocks are views into
// the caller's matrices, so nothing is copied but the symmetric operand of Symm
// and the one diagonal tile of Syrk and Syr2k.

// blockSize is the order of a diagonal block. A block of 64 keeps the reference
// loops, which run at scalar speed, to a small share of the flops (blockSize/m
// of a solve, for the order m of the triangle), and leaves GEMM updates with a
// contraction long enough for the packed kernel. The tests lower it to make
// small matrices span several blocks.
var blockSize = 64

// opOff is the offset in the stored matrix of element (r, c) of op(X), with
// leading dimension ld: r+c*ld if t is NoTrans, c+r*ld otherwise (t is upper
// case; ConjTrans is Trans for the real types).
func opOff(t Transpose, ld, r, c int) int {
	if t == NoTrans {
		return r + c*ld
	}
	return c + r*ld
}

// scaleCols sets the m x n matrix b to alpha*b, column-major with leading
// dimension ldb.
func scaleCols[T real](m, n int, alpha T, b []T, ldb int) {
	for j := 0; j < n; j++ {
		col := b[j*ldb : j*ldb+m]
		for i := range col {
			col[i] *= alpha
		}
	}
}
