package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
	"github.com/go-simd/floats"
)

// The level 1 routines take n elements of each vector, spaced by its increment
// in a slice. A negative increment traverses the vector from its far end (the
// element for i = 0 is x[(n-1)*|incx|]), as in the reference; the routines that
// the reference makes quick returns for a non-positive increment (Scal, Nrm2's
// siblings Asum and Iamax) do the same here. Like the reference, a non-positive
// n is not an error. A slice too short for n and its increment is an *ArgError
// on the slice's parameter number: 2 and 4 for Dot, 3 and 5 for Axpy, 3 for
// Scal, 2 for Nrm2, Asum and Iamax. They are checked only when the call reads
// or writes them.

// Ddot returns the dot product of x and y (DDOT). With unit increments it is
// go-simd/floats.Dot, within gamma_n|x|.|y| of the exact value; otherwise the
// reference loop.
func Ddot(n int, x []float64, incx int, y []float64, incy int) (float64, error) {
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("DDOT", p)
	}
	if p := check.Vec1(4, n, incy, len(y)); p != 0 {
		return 0, argErr("DDOT", p)
	}
	if n > 0 && incx == 1 && incy == 1 {
		return floats.Dot(x[:n], y[:n]), nil
	}
	return ref.Dot(n, x, incx, y, incy), nil
}

// Sdot is Ddot in single precision (SDOT).
func Sdot(n int, x []float32, incx int, y []float32, incy int) (float32, error) {
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("SDOT", p)
	}
	if p := check.Vec1(4, n, incy, len(y)); p != 0 {
		return 0, argErr("SDOT", p)
	}
	if n > 0 && incx == 1 && incy == 1 {
		return floats.Float32Dot(x[:n], y[:n]), nil
	}
	return ref.Dot(n, x, incx, y, incy), nil
}

// Daxpy computes y := alpha*x + y (DAXPY). With unit increments it is
// go-simd/floats.Axpy, one rounding per element. n <= 0 and alpha = 0 return at
// once, without reading x or y.
func Daxpy(n int, alpha float64, x []float64, incx int, y []float64, incy int) error {
	if n <= 0 || alpha == 0 {
		return nil
	}
	if p := check.Vec1(3, n, incx, len(x)); p != 0 {
		return argErr("DAXPY", p)
	}
	if p := check.Vec1(5, n, incy, len(y)); p != 0 {
		return argErr("DAXPY", p)
	}
	if incx == 1 && incy == 1 {
		floats.Axpy(alpha, x[:n], y[:n])
		return nil
	}
	ref.Axpy(n, alpha, x, incx, y, incy)
	return nil
}

// Saxpy is Daxpy in single precision (SAXPY).
func Saxpy(n int, alpha float32, x []float32, incx int, y []float32, incy int) error {
	if n <= 0 || alpha == 0 {
		return nil
	}
	if p := check.Vec1(3, n, incx, len(x)); p != 0 {
		return argErr("SAXPY", p)
	}
	if p := check.Vec1(5, n, incy, len(y)); p != 0 {
		return argErr("SAXPY", p)
	}
	ref.Axpy(n, alpha, x, incx, y, incy)
	return nil
}

// Dscal computes x := alpha*x (DSCAL). n <= 0, incx <= 0 and alpha = 1 return
// at once; alpha = 0 stores 0*x, so a NaN stays one, as in the reference.
func Dscal(n int, alpha float64, x []float64, incx int) error {
	if n <= 0 || incx <= 0 || alpha == 1 {
		return nil
	}
	if p := check.Vec1(3, n, incx, len(x)); p != 0 {
		return argErr("DSCAL", p)
	}
	if incx == 1 {
		floats.Scale(alpha, x[:n])
		return nil
	}
	ref.Scal(n, alpha, x, incx)
	return nil
}

// Sscal is Dscal in single precision (SSCAL).
func Sscal(n int, alpha float32, x []float32, incx int) error {
	if n <= 0 || incx <= 0 || alpha == 1 {
		return nil
	}
	if p := check.Vec1(3, n, incx, len(x)); p != 0 {
		return argErr("SSCAL", p)
	}
	ref.Scal(n, alpha, x, incx)
	return nil
}

// Dnrm2 returns the Euclidean norm of x (DNRM2), by the 3.10 reference
// algorithm that neither overflows nor underflows when the result is
// representable, within gamma_{n+2} of the exact norm. The increment may be
// negative; n <= 0 returns 0.
func Dnrm2(n int, x []float64, incx int) (float64, error) {
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("DNRM2", p)
	}
	return ref.Nrm2(n, x, incx), nil
}

// Snrm2 is Dnrm2 in single precision (SNRM2).
func Snrm2(n int, x []float32, incx int) (float32, error) {
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("SNRM2", p)
	}
	return ref.Nrm2(n, x, incx), nil
}

// Dasum returns the sum of the absolute values of x (DASUM), 0 for n <= 0 or
// incx <= 0.
func Dasum(n int, x []float64, incx int) (float64, error) {
	if n <= 0 || incx <= 0 {
		return 0, nil
	}
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("DASUM", p)
	}
	return ref.Asum(n, x, incx), nil
}

// Sasum is Dasum in single precision (SASUM).
func Sasum(n int, x []float32, incx int) (float32, error) {
	if n <= 0 || incx <= 0 {
		return 0, nil
	}
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("SASUM", p)
	}
	return ref.Asum(n, x, incx), nil
}

// Idamax returns the Fortran, 1-based, index of the first element of largest
// absolute value (IDAMAX), and 0 for n < 1 or incx <= 0. A NaN never compares
// larger, so it is skipped unless it comes first.
func Idamax(n int, x []float64, incx int) (int, error) {
	if n < 1 || incx <= 0 {
		return 0, nil
	}
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("IDAMAX", p)
	}
	return ref.Iamax(n, x, incx), nil
}

// Isamax is Idamax in single precision (ISAMAX).
func Isamax(n int, x []float32, incx int) (int, error) {
	if n < 1 || incx <= 0 {
		return 0, nil
	}
	if p := check.Vec1(2, n, incx, len(x)); p != 0 {
		return 0, argErr("ISAMAX", p)
	}
	return ref.Iamax(n, x, incx), nil
}
