package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
	"github.com/go-simd/floats"
)

// Dgemv computes y := alpha*op(A)*x + beta*y, where A is m x n column-major and
// op(A) is A, or A**T for Trans and ConjTrans. x has n elements for NoTrans and
// m otherwise, y the other; both are spaced by their increment, which may be
// negative (the vector is then traversed from its far end) but not zero. It is
// DGEMV of the reference BLAS, parameter for parameter, with the same rules for
// beta = 0 (y is not read), alpha = 0 (A and x are not read) and the quick
// returns; a slice too short for the call is an *ArgError on its own parameter
// (5, 7 and 10).
//
// With unit increments each column is one go-simd/floats Axpy or Dot, so the
// result is within gamma_{r+2}(|alpha||op(A)||x| + |beta||y|) of the exact
// one, r being the length of the sums.
func Dgemv(t Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incx int, beta float64, y []float64, incy int) error {
	return gemv("DGEMV", t, m, n, alpha, a, lda, x, incx, beta, y, incy, floats.Axpy, floats.Dot)
}

// Sgemv is Dgemv in single precision (SGEMV).
func Sgemv(t Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incx int, beta float32, y []float32, incy int) error {
	return gemv("SGEMV", t, m, n, alpha, a, lda, x, incx, beta, y, incy, nil, floats.Float32Dot)
}

// gemv runs the checks and the reference loops, with axpy and dot as the
// unit-increment shortcuts of the two forms (axpy is nil where the element
// type has no vector kernel).
func gemv[T real](name string, t Transpose, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int,
	axpy func(alpha T, x, y []T), dot func(x, y []T) T) error {
	if p := check.Gemv(byte(t), m, n, lda, incx, incy); p != 0 { // guard
		return argErr(name, p)
	}
	// Quick return if possible.
	if m == 0 || n == 0 || (alpha == 0 && beta == 1) { // guard
		return nil
	}
	if p := check.GemvLens(byte(t), m, n, lda, incx, incy, len(a), len(x), len(y), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	switch {
	case alpha != 0 && t == NoTrans && incy == 1 && axpy != nil:
		// y := beta*y, then one Axpy per column: y += (alpha*x[j]) * A[:, j].
		scaleVec(y[:m], beta)
		jx := 0
		if incx < 0 {
			jx = -(n - 1) * incx
		}
		for j := 0; j < n; j++ {
			axpy(alpha*x[jx], a[j*lda:j*lda+m], y[:m])
			jx += incx
		}
	case alpha != 0 && t != NoTrans && incx == 1:
		// y := beta*y, then y[j] += alpha * (A[:, j] . x).
		jy := 0
		if incy < 0 {
			jy = -(n - 1) * incy
		}
		for j := 0; j < n; j++ {
			d := alpha * dot(a[j*lda:j*lda+m], x[:m])
			switch {
			case beta == 0:
				y[jy] = d
			case beta == 1:
				y[jy] += d
			default:
				y[jy] = beta*y[jy] + d
			}
			jy += incy
		}
	default:
		ref.Gemv(t == NoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy)
	}
	return nil
}

// scaleVec is y := beta*y without reading y when beta is 0.
func scaleVec[T real](y []T, beta T) {
	switch beta {
	case 0:
		clear(y)
	case 1:
	default:
		for i := range y {
			y[i] *= beta
		}
	}
}
