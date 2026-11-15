package blas

import (
	"sync"

	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/kernels"
	"github.com/go-blas/blas/internal/ref"
)

// real is the element types of the routines of this release.
type real interface{ float32 | float64 }

// Dgemm computes C := alpha*op(A)*op(B) + beta*C, where op(X) is X, or X**T
// for Trans and ConjTrans. op(A) is m x k, op(B) is k x n and C is m x n, all
// stored column-major: element (i, j) of A is a[i+j*lda]. It is DGEMM of the
// reference BLAS, parameter for parameter.
//
// With beta = 0, C is not read (it may hold NaN); with alpha = 0, A and B are
// not read (they may be empty); m = 0 or n = 0, and alpha = 0 or k = 0 with
// beta = 1, return at once. An invalid argument returns a *ArgError numbered
// like xerbla's, among them a slice too short for the call: a is checked
// against lda*(cols-1)+rows, and likewise b and c (parameters 7, 9 and 12).
//
// The product runs on a packed, cache-blocked, multicore SIMD kernel: it is
// within gamma_{k+2}(|alpha||op(A)||op(B)| + |beta||C|) elementwise of the
// exact result (see the package comment), not bit-identical to netlib.Dgemm.
// The beta term sits inside the k+2 because C is scaled first and the product
// accumulated into it, as in netlib's own loops and in every blocked BLAS. It is not safe to call with c overlapping a or b.
func Dgemm(tA, tB Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) error {
	return gemm("DGEMM", tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dHW)
}

// Sgemm is Dgemm in single precision (SGEMM). Without a float32 kernel on the
// running CPU it runs the reference loops of netlib.Sgemm.
func Sgemm(tA, tB Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	return gemm("SGEMM", tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, sHW)
}

// HasSgemmKernel reports whether Sgemm runs on a SIMD kernel on this CPU. When
// it is false Sgemm runs the reference loops, which a caller with a faster
// fallback (widening to float64, say) may prefer to avoid.
func HasSgemmKernel() bool { return sHW.mul != nil }

// hw is one element type's kernel and its scratch buffers. mul is nil when the
// running CPU has no kernel for the type.
type hw[T real] struct {
	mul     func(dst []T, ldc int, tA, tB Transpose, m, n, k int, a []T, lda int, b []T, ldb int)
	scratch sync.Pool
	// mulS is the same kernel behind the general-stride entry (gemm_strided.go):
	// dst += a*b, dst row-major with rows ld apart. It is nil exactly when mul is.
	mulS func(dst []T, ld int, a, b kernels.MatOf[T], m, k, n int)
}

var dHW = &hw[float64]{mul: dMul, mulS: kernels.MatMulLDStridedP}

// sHW has no kernel where the CPU has no float32 one (see kernels.HasGemm32).
var sHW = &hw[float32]{mul: kernelIf(kernels.HasGemm32(), sMul), mulS: kernelIf(kernels.HasGemm32(), kernels.MatMul32LDStridedP)}

// kernelIf returns f when ok, and the zero value, nil for a func, otherwise.
func kernelIf[F any](ok bool, f F) (none F) {
	if ok {
		return f
	}
	return none
}

// The kernel computes the row-major product dst += A'*B'. A column-major C is
// the row-major C**T (n x m, rows ldc apart), and C**T = op(B)**T * op(A)**T:
// A' is op(B)**T, an n x k view of b, and B' is op(A)**T, a k x m view of a.
// For an operand that is not transposed the view has a unit column stride, so
// the kernel reads the column-major data in place.

func dMul(dst []float64, ldc int, tA, tB Transpose, m, n, k int, a []float64, lda int, b []float64, ldb int) {
	kernels.MatMulLDStridedP(dst, ldc, view(b, ldb, tB), view(a, lda, tA), n, k, m)
}

func sMul(dst []float32, ldc int, tA, tB Transpose, m, n, k int, a []float32, lda int, b []float32, ldb int) {
	kernels.MatMul32LDStridedP(dst, ldc, view32(b, ldb, tB), view32(a, lda, tA), n, k, m)
}

// view is the transpose of op(X) as a row-major view: element (r, c) of the
// view is op(X)(c, r), at x[c+r*ld] if X is not transposed, else at x[r+c*ld].
func view(x []float64, ld int, t Transpose) kernels.Mat {
	if t == NoTrans {
		return kernels.Mat{Data: x, RS: ld, CS: 1}
	}
	return kernels.Mat{Data: x, RS: 1, CS: ld}
}

func view32(x []float32, ld int, t Transpose) kernels.Mat32 {
	if t == NoTrans {
		return kernels.Mat32{Data: x, RS: ld, CS: 1}
	}
	return kernels.Mat32{Data: x, RS: 1, CS: ld}
}

func gemm[T real](name string, tA, tB Transpose, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int, h *hw[T]) error {
	tA, tB = Transpose(check.Upper(byte(tA))), Transpose(check.Upper(byte(tB))) // LSAME ignores case
	if p := check.Gemm(byte(tA), byte(tB), m, n, k, lda, ldb, ldc); p != 0 {    // guard
		return argErr(name, p)
	}
	// Quick return if possible.
	if m == 0 || n == 0 || ((alpha == 0 || k == 0) && beta == 1) { // guard
		return nil
	}
	if p := check.GemmLens(byte(tA), byte(tB), m, n, k, lda, ldb, ldc, len(a), len(b), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	gemmCore(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, h)
	return nil
}

// gemmCore is gemm after the argument tests and the quick return, which the
// caller has done: tA and tB are upper case, the slices are long enough, m and n
// are positive. The level 3 routines of this package (trsm.go, trmm.go, syrk.go)
// call it on the blocks of their operands.
func gemmCore[T real](tA, tB Transpose, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int, h *hw[T]) {
	// Nothing to multiply, or no kernel: the reference loops, which also carry
	// the alpha = 0 and k = 0 rules.
	if alpha == 0 || k == 0 || h.mul == nil { // guard
		ref.Gemm(tA == NoTrans, tB == NoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	// C := beta*C, without reading C when beta is 0 (it may hold NaN).
	for j := 0; j < n; j++ {
		col := c[j*ldc : j*ldc+m]
		switch beta {
		case 0:
			clear(col)
		case 1:
		default:
			for i := range col {
				col[i] *= beta
			}
		}
	}
	if alpha == 1 {
		h.mul(c, ldc, tA, tB, m, n, k, a, lda, b, ldb)
		return
	}
	// The kernel has no alpha: form the product in a scratch matrix, then add
	// alpha times it.
	t := getScratch[T](&h.scratch, m*n)
	h.mul(t, m, tA, tB, m, n, k, a, lda, b, ldb)
	for j := 0; j < n; j++ {
		col, tc := c[j*ldc:j*ldc+m], t[j*m:j*m+m]
		for i, v := range tc {
			col[i] += alpha * v
		}
	}
	h.scratch.Put(&t)
}

// getScratch returns a zeroed slice of n elements from p, or a new one.
func getScratch[T real](p *sync.Pool, n int) []T {
	if t, ok := p.Get().(*[]T); ok && cap(*t) >= n {
		s := (*t)[:n]
		clear(s)
		return s
	}
	return make([]T, n)
}
