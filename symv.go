package blas

import (
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// Dsymv computes y := alpha*A*x + beta*y for the n x n symmetric column-major
// matrix A, of which only the triangle uplo names is referenced (DSYMV, parameter
// for parameter: uplo 1, n 2, alpha 3, a 4, lda 5, x 6, incx 7, beta 8, y 9, incy
// 10). The increments may be negative but not zero. The quick returns are the
// reference's (n = 0, or alpha = 0 and beta = 1); with alpha = 0, a and x are not
// read, with beta = 0, y is not read. It is the reference loop, within
// gamma_{n+2}(|alpha||A||x| + |beta||y|) of the exact result.
func Dsymv(uplo Uplo, n int, alpha float64, a []float64, lda int, x []float64, incx int, beta float64, y []float64, incy int) error {
	return symv("DSYMV", uplo, n, alpha, a, lda, x, incx, beta, y, incy)
}

// Ssymv is Dsymv in single precision (SSYMV).
func Ssymv(uplo Uplo, n int, alpha float32, a []float32, lda int, x []float32, incx int, beta float32, y []float32, incy int) error {
	return symv("SSYMV", uplo, n, alpha, a, lda, x, incx, beta, y, incy)
}

func symv[T real](name string, uplo Uplo, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) error {
	uplo = Uplo(check.Upper(byte(uplo)))                         // LSAME ignores case
	if p := check.Symv(byte(uplo), n, lda, incx, incy); p != 0 { // guard
		return argErr(name, p)
	}
	if n == 0 || (alpha == 0 && beta == 1) { // guard
		return nil
	}
	if p := check.SymvLens(n, lda, incx, incy, len(a), len(x), len(y), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Symv(uplo == Upper, n, alpha, a, lda, x, incx, beta, y, incy)
	return nil
}

// Dsyr2 computes the symmetric rank-2 update A := alpha*x*y**T + alpha*y*x**T + A
// on the triangle uplo names of the n x n matrix A (DSYR2: uplo 1, n 2, alpha 3, x
// 4, incx 5, y 6, incy 7, a 8, lda 9). The increments may be negative but not
// zero. n = 0 or alpha = 0 return at once. It is the reference loop, within
// gamma_4 (|A| + |alpha|(|x||y|^T + |y||x|^T)) of the exact result per element.
func Dsyr2(uplo Uplo, n int, alpha float64, x []float64, incx int, y []float64, incy int, a []float64, lda int) error {
	return syr2("DSYR2", uplo, n, alpha, x, incx, y, incy, a, lda)
}

// Ssyr2 is Dsyr2 in single precision (SSYR2).
func Ssyr2(uplo Uplo, n int, alpha float32, x []float32, incx int, y []float32, incy int, a []float32, lda int) error {
	return syr2("SSYR2", uplo, n, alpha, x, incx, y, incy, a, lda)
}

func syr2[T real](name string, uplo Uplo, n int, alpha T, x []T, incx int, y []T, incy int, a []T, lda int) error {
	uplo = Uplo(check.Upper(byte(uplo)))                         // LSAME ignores case
	if p := check.Syr2(byte(uplo), n, incx, incy, lda); p != 0 { // guard
		return argErr(name, p)
	}
	if n == 0 || alpha == 0 { // guard
		return nil
	}
	if p := check.Syr2Lens(n, incx, incy, lda, len(x), len(y), len(a)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Syr2(uplo == Upper, n, alpha, x, incx, y, incy, a, lda)
	return nil
}

// Drot applies the plane rotation x := c*x + s*y, y := c*y - s*x to the n
// elements of x and y (DROT). n <= 0 returns at once; a slice too short is an
// *ArgError on parameter 2 (x) or 4 (y). Increments may be negative, as in the
// reference; they are not tested for zero (the reference does not either).
func Drot(n int, x []float64, incx int, y []float64, incy int, c, s float64) error {
	return rot("DROT", n, x, incx, y, incy, c, s)
}

// Srot is Drot in single precision (SROT).
func Srot(n int, x []float32, incx int, y []float32, incy int, c, s float32) error {
	return rot("SROT", n, x, incx, y, incy, c, s)
}

func rot[T real](name string, n int, x []T, incx int, y []T, incy int, c, s T) error {
	// A non-positive n passes both checks (Vec1 reads nothing) and Rot's loop
	// does not run: no quick return is needed.
	if p := check.Vec1(2, n, incx, len(x)); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.Vec1(4, n, incy, len(y)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Rot(n, x, incx, y, incy, c, s)
	return nil
}

// Dswap exchanges the n elements of x and y (DSWAP). n <= 0 returns at once; a
// slice too short is an *ArgError on parameter 2 (x) or 4 (y).
func Dswap(n int, x []float64, incx int, y []float64, incy int) error {
	return swap("DSWAP", n, x, incx, y, incy)
}

// Sswap is Dswap in single precision (SSWAP).
func Sswap(n int, x []float32, incx int, y []float32, incy int) error {
	return swap("SSWAP", n, x, incx, y, incy)
}

func swap[T real](name string, n int, x []T, incx int, y []T, incy int) error {
	// As in rot, a non-positive n needs no quick return.
	if p := check.Vec1(2, n, incx, len(x)); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.Vec1(4, n, incy, len(y)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Swap(n, x, incx, y, incy)
	return nil
}

// Dcopy copies the n elements of x to y (DCOPY). n <= 0 returns at once; a slice
// too short is an *ArgError on parameter 2 (x) or 4 (y).
func Dcopy(n int, x []float64, incx int, y []float64, incy int) error {
	return copyVec("DCOPY", n, x, incx, y, incy)
}

// Scopy is Dcopy in single precision (SCOPY).
func Scopy(n int, x []float32, incx int, y []float32, incy int) error {
	return copyVec("SCOPY", n, x, incx, y, incy)
}

func copyVec[T real](name string, n int, x []T, incx int, y []T, incy int) error {
	// As in rot, a non-positive n needs no quick return.
	if p := check.Vec1(2, n, incx, len(x)); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.Vec1(4, n, incy, len(y)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Copy(n, x, incx, y, incy)
	return nil
}
