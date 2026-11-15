package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// Dsyrk is the symmetric rank-k update C := alpha*A*A**T + beta*C for trans
// NoTrans, C := alpha*A**T*A + beta*C for Trans and ConjTrans (the same for real
// types). C is n x n symmetric and only the triangle uplo names is referenced and
// written (the other one is left as it was); A is n x k for NoTrans, k x n
// otherwise. All matrices are column-major. It is DSYRK of the reference BLAS,
// parameter for parameter: uplo 1, trans 2, n 3, k 4, alpha 5, a 6, lda 7, beta
// 8, c 9, ldc 10.
//
// With beta = 0, C is not read (it may hold NaN); with alpha = 0, A is not read
// (it may be empty); n = 0, and alpha = 0 or k = 0 with beta = 1, return at once.
// An invalid argument returns a *ArgError numbered like xerbla's, among them a
// slice too short for the call: a is checked against lda*(cols-1)+rows, and c
// against ldc*(n-1)+n (parameters 6 and 9).
//
// The triangle is cut into blocks of blockSize: GEMM on the blocks off the
// diagonal, and on each diagonal block a GEMM into a small scratch tile of which
// the triangle is merged into C. The result is within
//
//	|C^ - C| <= gamma_{k+2} (|alpha| |op(A)| |op(A)**T| + |beta| |C0|)   elementwise, on the triangle
//
// of the exact result, the bound of Dgemm, not bit-identical to netlib.Dsyrk. It
// is not safe to call with c overlapping a.
func Dsyrk(uplo Uplo, trans Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) error {
	return syrk("DSYRK", uplo, trans, n, k, alpha, a, lda, beta, c, ldc, dHW)
}

// Ssyrk is Dsyrk in single precision (SSYRK). Without a float32 kernel on the
// running CPU it runs the reference loops of netlib.Ssyrk.
func Ssyrk(uplo Uplo, trans Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) error {
	return syrk("SSYRK", uplo, trans, n, k, alpha, a, lda, beta, c, ldc, sHW)
}

func syrk[T real](name string, uplo Uplo, trans Transpose, n, k int, alpha T, a []T, lda int, beta T, c []T, ldc int, h *hw[T]) error {
	uplo, trans = Uplo(check.Upper(byte(uplo))), Transpose(check.Upper(byte(trans))) // LSAME ignores case
	if p := check.Syrk(byte(uplo), byte(trans), n, k, lda, ldc); p != 0 {            // guard
		return argErr(name, p)
	}
	// Quick return if possible.
	if n == 0 || ((alpha == 0 || k == 0) && beta == 1) { // guard
		return nil
	}
	if p := check.SyrkLens(byte(trans), n, k, lda, ldc, len(a), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	upper, notA := uplo == Upper, trans == NoTrans
	// Nothing to multiply, or no kernel: the reference loops, which also carry the
	// alpha = 0 and k = 0 rules.
	if alpha == 0 || k == 0 || h.mul == nil { // guard
		ref.Syrk(upper, notA, n, k, alpha, a, lda, beta, c, ldc)
		return nil
	}
	// C(I,J) = op(A)(I,:) * op(A)(J,:)**T, so a block row of C takes the block
	// rows of op(A) as its left operand and the transposes of the block columns as
	// its right one: A(I,:) and A(J,:)**T for NoTrans, A(:,I)**T and A(:,J) for
	// Trans.
	tA, tB := NoTrans, Trans
	if !notA {
		tA, tB = Trans, NoTrans
	}
	blk := func(i0 int) []T {
		if notA {
			return a[i0:]
		}
		return a[i0*lda:]
	}
	nb := blockSize
	for j0 := 0; j0 < n; j0 += nb {
		jb := min(nb, n-j0)
		// The blocks of the triangle in this block column: above and on the
		// diagonal for the upper triangle, on and below it for the lower one.
		i0, i1 := 0, j0+jb
		if !upper {
			i0, i1 = j0, n
		}
		for ; i0 < i1; i0 += nb {
			ib := min(nb, n-i0)
			if i0 != j0 {
				gemmCore(tA, tB, ib, jb, k, alpha, blk(i0), lda, blk(j0), lda, beta, c[i0+j0*ldc:], ldc, h)
				continue
			}
			// The diagonal block: the whole tile product, of which C takes the triangle.
			t := getScratch[T](&h.scratch, jb*jb)
			gemmCore(tA, tB, jb, jb, k, 1, blk(j0), lda, blk(j0), lda, 0, t, jb, h)
			mergeTile(upper, jb, alpha, t, beta, c[j0+j0*ldc:], ldc, false)
			h.scratch.Put(&t)
		}
	}
	return nil
}

// mergeTile sets the triangle of the nb x nb block c, column-major with leading
// dimension ldc, to alpha*t + beta*c, where t is the nb x nb tile product,
// column-major with leading dimension nb, without reading c when beta is zero. With
// sym, the product is t + t**T (the tile of Syr2k is A B**T, and the result
// A B**T + B A**T).
func mergeTile[T real](upper bool, nb int, alpha T, t []T, beta T, c []T, ldc int, sym bool) {
	for j := 0; j < nb; j++ {
		lo, hi := 0, j+1
		if !upper {
			lo, hi = j, nb
		}
		for i := lo; i < hi; i++ {
			v := t[i+j*nb]
			if sym {
				v += t[j+i*nb]
			}
			if beta == 0 {
				c[i+j*ldc] = alpha * v
			} else {
				c[i+j*ldc] = alpha*v + beta*c[i+j*ldc]
			}
		}
	}
}

// Dsyr2k is the symmetric rank-2k update C := alpha*A*B**T + alpha*B*A**T +
// beta*C for trans NoTrans, C := alpha*A**T*B + alpha*B**T*A + beta*C for Trans
// and ConjTrans. C is n x n symmetric and only the triangle uplo names is
// referenced and written; A and B are n x k for NoTrans, k x n otherwise. All
// matrices are column-major. It is DSYR2K of the reference BLAS, parameter for
// parameter: uplo 1, trans 2, n 3, k 4, alpha 5, a 6, lda 7, b 8, ldb 9, beta 10,
// c 11, ldc 12.
//
// The scalar rules, the quick returns and the slice checks are those of Dsyrk,
// with b checked like a (parameter 8). Blocked like Dsyrk, with two GEMMs per
// block off the diagonal, and on the diagonal one tile product A B**T of which
// the triangle of the sum with its transpose is merged into C. The result is
// within
//
//	|C^ - C| <= gamma_{2k+2} (|alpha| (|op(A)| |op(B)**T| + |op(B)| |op(A)**T|) + |beta| |C0|)   elementwise, on the triangle
//
// of the exact result, not bit-identical to netlib.Dsyr2k. It is not safe to call
// with c overlapping a or b.
func Dsyr2k(uplo Uplo, trans Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) error {
	return syr2k("DSYR2K", uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dHW)
}

// Ssyr2k is Dsyr2k in single precision (SSYR2K). Without a float32 kernel on
// the running CPU it runs the reference loops of netlib.Ssyr2k.
func Ssyr2k(uplo Uplo, trans Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	return syr2k("SSYR2K", uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, sHW)
}

func syr2k[T real](name string, uplo Uplo, trans Transpose, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int, h *hw[T]) error {
	uplo, trans = Uplo(check.Upper(byte(uplo))), Transpose(check.Upper(byte(trans))) // LSAME ignores case
	if p := check.Syr2k(byte(uplo), byte(trans), n, k, lda, ldb, ldc); p != 0 {      // guard
		return argErr(name, p)
	}
	// Quick return if possible.
	if n == 0 || ((alpha == 0 || k == 0) && beta == 1) { // guard
		return nil
	}
	if p := check.Syr2kLens(byte(trans), n, k, lda, ldb, ldc, len(a), len(b), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	upper, notA := uplo == Upper, trans == NoTrans
	if alpha == 0 || k == 0 || h.mul == nil { // guard
		ref.Syr2k(upper, notA, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return nil
	}
	tA, tB := NoTrans, Trans
	if !notA {
		tA, tB = Trans, NoTrans
	}
	blk := func(x []T, ld, i0 int) []T {
		if notA {
			return x[i0:]
		}
		return x[i0*ld:]
	}
	nb := blockSize
	for j0 := 0; j0 < n; j0 += nb {
		jb := min(nb, n-j0)
		i0, i1 := 0, j0+jb
		if !upper {
			i0, i1 = j0, n
		}
		for ; i0 < i1; i0 += nb {
			ib := min(nb, n-i0)
			if i0 != j0 {
				cb := c[i0+j0*ldc:]
				gemmCore(tA, tB, ib, jb, k, alpha, blk(a, lda, i0), lda, blk(b, ldb, j0), ldb, beta, cb, ldc, h)
				gemmCore(tA, tB, ib, jb, k, alpha, blk(b, ldb, i0), ldb, blk(a, lda, j0), lda, 1, cb, ldc, h)
				continue
			}
			t := getScratch[T](&h.scratch, jb*jb)
			gemmCore(tA, tB, jb, jb, k, 1, blk(a, lda, j0), lda, blk(b, ldb, j0), ldb, 0, t, jb, h)
			mergeTile(upper, jb, alpha, t, beta, c[j0+j0*ldc:], ldc, true)
			h.scratch.Put(&t)
		}
	}
	return nil
}
