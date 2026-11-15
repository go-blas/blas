package blas_test

import (
	"math"
	"math/big"
	"math/rand/v2"
	"runtime"
	"strconv"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
	"github.com/go-blas/blas/netlib"
)

type float interface{ float32 | float64 }

// impl is one implementation of the routines of one element type: package
// blas, or its exact option netlib. Every test runs on both.
type impl[T float] struct {
	name  string
	gemm  func(tA, tB blas.Transpose, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) error
	gemv  func(t blas.Transpose, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) error
	dot   func(n int, x []T, incx int, y []T, incy int) (T, error)
	axpy  func(n int, alpha T, x []T, incx int, y []T, incy int) error
	scal  func(n int, alpha T, x []T, incx int) error
	nrm2  func(n int, x []T, incx int) (T, error)
	asum  func(n int, x []T, incx int) (T, error)
	iamax func(n int, x []T, incx int) (int, error)
	trsm  func(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int) error
	trmm  func(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int) error
	syrk  func(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha T, a []T, lda int, beta T, c []T, ldc int) error
	syr2k func(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) error
	symm  func(side blas.Side, uplo blas.Uplo, m, n int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) error
}

var impls64 = []impl[float64]{
	{"blas", blas.Dgemm, blas.Dgemv, blas.Ddot, blas.Daxpy, blas.Dscal, blas.Dnrm2, blas.Dasum, blas.Idamax, blas.Dtrsm, blas.Dtrmm, blas.Dsyrk, blas.Dsyr2k, blas.Dsymm},
	{"netlib", netlib.Dgemm, netlib.Dgemv, netlib.Ddot, netlib.Daxpy, netlib.Dscal, netlib.Dnrm2, netlib.Dasum, netlib.Idamax, netlib.Dtrsm, netlib.Dtrmm, netlib.Dsyrk, netlib.Dsyr2k, netlib.Dsymm},
}

var impls32 = []impl[float32]{
	{"blas", blas.Sgemm, blas.Sgemv, blas.Sdot, blas.Saxpy, blas.Sscal, blas.Snrm2, blas.Sasum, blas.Isamax, blas.Strsm, blas.Strmm, blas.Ssyrk, blas.Ssyr2k, blas.Ssymm},
	{"netlib", netlib.Sgemm, netlib.Sgemv, netlib.Sdot, netlib.Saxpy, netlib.Sscal, netlib.Snrm2, netlib.Sasum, netlib.Isamax, netlib.Strsm, netlib.Strmm, netlib.Ssyrk, netlib.Ssyr2k, netlib.Ssymm},
}

// each runs a test for both element types and both implementations.
func each(t *testing.T, f64 func(t *testing.T, im impl[float64]), f32 func(t *testing.T, im impl[float32])) {
	t.Helper()
	for _, im := range impls64 {
		t.Run("float64/"+im.name, func(t *testing.T) { defer noPanic(t); f64(t, im) })
	}
	for _, im := range impls32 {
		t.Run("float32/"+im.name, func(t *testing.T) { defer noPanic(t); f32(t, im) })
	}
}

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)) }

// within asserts |got - exact| <= gp*P + gq*Q for element i of r.
func within[T float](t *testing.T, what string, got T, r oracle.Result, i int, gp, gq *big.Float) {
	t.Helper()
	if !oracle.Within(got, r, i, gp, gq) {
		exact, _ := r.Exact[i].Float64()
		t.Errorf("%s[%d] = %v, exact %v: outside the bound (gp*P = %v, gq*Q = %v)", what, i, got, exact,
			scaled(gp, r.P[i]), scaled(gq, r.Q[i]))
	}
}

func scaled(g, x *big.Float) float64 {
	f, _ := new(big.Float).Mul(g, x).Float64()
	return f
}

func transName(t blas.Transpose) string { return string(rune(t)) }

// sentinel marks the elements a routine must not touch.
const sentinel = 12345.5

// matrix returns a column-major rows x cols matrix with leading dimension ld,
// whose padding rows hold the sentinel.
func matrix[T float](rng *rand.Rand, rows, cols, ld int) []T {
	n := ld*(cols-1) + rows
	if rows == 0 || cols == 0 {
		return nil
	}
	v := make([]T, n)
	for j := 0; j < cols; j++ {
		copy(v[j*ld:j*ld+rows], oracle.Fill[T](rng, rows))
		for i := rows; i < ld && j*ld+i < n; i++ {
			v[j*ld+i] = sentinel
		}
	}
	return v
}

// padUntouched asserts the padding of a matrix still holds the sentinel.
func padUntouched[T float](t *testing.T, what string, v []T, rows, ld int) {
	t.Helper()
	for p := 0; p < len(v); p++ {
		if p%ld >= rows && v[p] != sentinel {
			t.Fatalf("%s: padding element %d was written", what, p)
		}
	}
}

// vector returns n elements spaced |inc| apart, the gaps holding the sentinel.
func vector[T float](rng *rand.Rand, n, inc int) []T {
	if n <= 0 {
		return nil
	}
	a := max(inc, -inc)
	v := make([]T, 1+(n-1)*a)
	for i := range v {
		v[i] = sentinel
	}
	for i, x := range oracle.Fill[T](rng, n) {
		v[i*a] = x
	}
	return v
}

// gapsUntouched asserts the elements between those a vector's increment
// addresses still hold the sentinel.
func gapsUntouched[T float](t *testing.T, what string, v []T, inc int) {
	t.Helper()
	a := max(inc, -inc)
	for p := range v {
		if a > 1 && p%a != 0 && v[p] != sentinel {
			t.Fatalf("%s: gap element %d was written", what, p)
		}
	}
}

func same[T float](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] && !(math.IsNaN(float64(a[i])) && math.IsNaN(float64(b[i]))) {
			return false
		}
	}
	return true
}

func clone[T float](v []T) []T { return append([]T(nil), v...) }

func itoa(n int) string { return strconv.Itoa(n) }

func ftoa(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }

// setProcs sets GOMAXPROCS for the rest of the test.
func setProcs(t *testing.T, n int) {
	t.Helper()
	old := runtime.GOMAXPROCS(n)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })
}

// noPanic, deferred, turns a panic of the routine under test into a failed
// assertion: no routine may panic on its arguments, and a missing guard must be
// reported as a failure of the test, not as the death of the test binary.
func noPanic(t *testing.T) {
	t.Helper()
	if r := recover(); r != nil {
		t.Fatalf("panicked: %v", r)
	}
}
