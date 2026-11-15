// Package netlib is the exact option: plain-Go ports of the reference BLAS 3.12
// routines (Reference-LAPACK/lapack, tag v3.12.0, BLAS/SRC: dgemm.f, sgemm.f,
// dgemv.f, sgemv.f, ddot.f, sdot.f, daxpy.f, saxpy.f, dscal.f, sscal.f,
// dnrm2.f90, snrm2.f90, dasum.f, sasum.f, idamax.f, isamax.f), in the
// reference's loop order and without fused multiply-add. The results are the
// same on every architecture, and the same as the Fortran reference compiled
// without FMA contraction; they are the equality oracle the tests of package
// blas use while porting.
//
// The signatures, argument checks and errors are those of package blas, which
// it shares its types with. The routines are not fast: use package blas unless
// the exact reference rounding is the point.
package netlib

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

func argErr(routine string, p int) error {
	return &blas.ArgError{Routine: routine, Param: p}
}

// Dgemm is DGEMM (dgemm.f): C := alpha*op(A)*op(B) + beta*C, column-major. See
// blas.Dgemm for the arguments and the errors.
func Dgemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) error {
	return gemm("DGEMM", tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Sgemm is SGEMM (sgemm.f).
func Sgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	return gemm("SGEMM", tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func gemm[T ref.Float](name string, tA, tB blas.Transpose, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) error {
	if p := check.Gemm(byte(tA), byte(tB), m, n, k, lda, ldb, ldc); p != 0 { // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 || ((alpha == 0 || k == 0) && beta == 1) { // guard
		return nil
	}
	if p := check.GemmLens(byte(tA), byte(tB), m, n, k, lda, ldb, ldc, len(a), len(b), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Gemm(tA == blas.NoTrans, tB == blas.NoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	return nil
}

// Dgemv is DGEMV (dgemv.f): y := alpha*op(A)*x + beta*y. See blas.Dgemv.
func Dgemv(t blas.Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incx int, beta float64, y []float64, incy int) error {
	return gemv("DGEMV", t, m, n, alpha, a, lda, x, incx, beta, y, incy)
}

// Sgemv is SGEMV (sgemv.f).
func Sgemv(t blas.Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incx int, beta float32, y []float32, incy int) error {
	return gemv("SGEMV", t, m, n, alpha, a, lda, x, incx, beta, y, incy)
}

func gemv[T ref.Float](name string, t blas.Transpose, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) error {
	if p := check.Gemv(byte(t), m, n, lda, incx, incy); p != 0 { // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 || (alpha == 0 && beta == 1) { // guard
		return nil
	}
	if p := check.GemvLens(byte(t), m, n, lda, incx, incy, len(a), len(x), len(y), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Gemv(t == blas.NoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy)
	return nil
}

// Ddot is DDOT (ddot.f): the dot product, accumulated in order.
func Ddot(n int, x []float64, incx int, y []float64, incy int) (float64, error) {
	return dot("DDOT", n, x, incx, y, incy)
}

// Sdot is SDOT (sdot.f).
func Sdot(n int, x []float32, incx int, y []float32, incy int) (float32, error) {
	return dot("SDOT", n, x, incx, y, incy)
}

func dot[T ref.Float](name string, n int, x []T, incx int, y []T, incy int) (T, error) {
	if p := check.Vec1(2, n, incx, len(x)); p != 0 { // guard
		return 0, argErr(name, p)
	}
	if p := check.Vec1(4, n, incy, len(y)); p != 0 { // guard
		return 0, argErr(name, p)
	}
	return ref.Dot(n, x, incx, y, incy), nil
}

// Daxpy is DAXPY (daxpy.f): y := alpha*x + y.
func Daxpy(n int, alpha float64, x []float64, incx int, y []float64, incy int) error {
	return axpy("DAXPY", n, alpha, x, incx, y, incy)
}

// Saxpy is SAXPY (saxpy.f).
func Saxpy(n int, alpha float32, x []float32, incx int, y []float32, incy int) error {
	return axpy("SAXPY", n, alpha, x, incx, y, incy)
}

func axpy[T ref.Float](name string, n int, alpha T, x []T, incx int, y []T, incy int) error {
	if n <= 0 || alpha == 0 { // guard
		return nil
	}
	if p := check.Vec1(3, n, incx, len(x)); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.Vec1(5, n, incy, len(y)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Axpy(n, alpha, x, incx, y, incy)
	return nil
}

// Dscal is DSCAL (dscal.f): x := alpha*x.
func Dscal(n int, alpha float64, x []float64, incx int) error {
	return scal("DSCAL", n, alpha, x, incx)
}

// Sscal is SSCAL (sscal.f).
func Sscal(n int, alpha float32, x []float32, incx int) error {
	return scal("SSCAL", n, alpha, x, incx)
}

func scal[T ref.Float](name string, n int, alpha T, x []T, incx int) error {
	if n <= 0 || incx <= 0 || alpha == 1 { // guard
		return nil
	}
	if p := check.Vec1(3, n, incx, len(x)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Scal(n, alpha, x, incx)
	return nil
}

// Dnrm2 is DNRM2 (dnrm2.f90): the Euclidean norm by the 3.10 algorithm.
func Dnrm2(n int, x []float64, incx int) (float64, error) { return nrm2("DNRM2", n, x, incx) }

// Snrm2 is SNRM2 (snrm2.f90).
func Snrm2(n int, x []float32, incx int) (float32, error) { return nrm2("SNRM2", n, x, incx) }

func nrm2[T ref.Float](name string, n int, x []T, incx int) (T, error) {
	if p := check.Vec1(2, n, incx, len(x)); p != 0 { // guard
		return 0, argErr(name, p)
	}
	return ref.Nrm2(n, x, incx), nil
}

// Dasum is DASUM (dasum.f): the sum of absolute values.
func Dasum(n int, x []float64, incx int) (float64, error) { return asum("DASUM", n, x, incx) }

// Sasum is SASUM (sasum.f).
func Sasum(n int, x []float32, incx int) (float32, error) { return asum("SASUM", n, x, incx) }

func asum[T ref.Float](name string, n int, x []T, incx int) (T, error) {
	if n <= 0 || incx <= 0 { // guard
		return 0, nil
	}
	if p := check.Vec1(2, n, incx, len(x)); p != 0 { // guard
		return 0, argErr(name, p)
	}
	return ref.Asum(n, x, incx), nil
}

// Idamax is IDAMAX (idamax.f): the 1-based index of the first element of
// largest absolute value.
func Idamax(n int, x []float64, incx int) (int, error) { return iamax("IDAMAX", n, x, incx) }

// Isamax is ISAMAX (isamax.f).
func Isamax(n int, x []float32, incx int) (int, error) { return iamax("ISAMAX", n, x, incx) }

func iamax[T ref.Float](name string, n int, x []T, incx int) (int, error) {
	if n < 1 || incx <= 0 { // guard
		return 0, nil
	}
	if p := check.Vec1(2, n, incx, len(x)); p != 0 { // guard
		return 0, argErr(name, p)
	}
	return ref.Iamax(n, x, incx), nil
}
