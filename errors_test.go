package blas_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-blas/blas"
)

const maxInt = math.MaxInt

// errCase is a call that must fail on one parameter, the routine's own.
type errCase struct {
	name  string
	param int
	call  func() error
}

func runErrCases(t *testing.T, routine string, cases []errCase) {
	t.Helper()
	for _, c := range cases {
		err := callRecovered(t, routine+" "+c.name, c.call)
		var ae *blas.ArgError
		if !errors.As(err, &ae) {
			t.Errorf("%s %s: error = %v, want *ArgError", routine, c.name, err)
			continue
		}
		if ae.Routine != routine || ae.Param != c.param {
			t.Errorf("%s %s: got %s param %d, want %s param %d", routine, c.name, ae.Routine, ae.Param, routine, c.param)
		}
		if ae.Info() != -c.param {
			t.Errorf("%s %s: Info() = %d, want %d", routine, c.name, ae.Info(), -c.param)
		}
		if !strings.Contains(ae.Error(), routine) || !strings.Contains(ae.Error(), itoa(c.param)) {
			t.Errorf("%s %s: Error() = %q", routine, c.name, ae.Error())
		}
	}
}

func testGemmErrors[T float](t *testing.T, im impl[T], routine string) {
	v := make([]T, 100)
	N, T_ := blas.NoTrans, blas.Trans
	g := func(tA, tB blas.Transpose, m, n, k int, a []T, lda int, b []T, ldb int, c []T, ldc int) func() error {
		return func() error { return im.gemm(tA, tB, m, n, k, 1, a, lda, b, ldb, 1, c, ldc) }
	}
	runErrCases(t, routine, []errCase{
		{"transA", 1, g('X', N, 3, 3, 3, v, 3, v, 3, v, 3)},
		{"transB", 2, g(N, 'x', 3, 3, 3, v, 3, v, 3, v, 3)},
		{"m", 3, g(N, N, -1, 3, 3, v, 3, v, 3, v, 3)},
		{"n", 4, g(N, N, 3, -1, 3, v, 3, v, 3, v, 3)},
		{"k", 5, g(N, N, 3, 3, -1, v, 3, v, 3, v, 3)},
		{"lda notrans", 8, g(N, N, 3, 3, 3, v, 2, v, 3, v, 3)},
		{"lda trans", 8, g(T_, N, 3, 3, 5, v, 4, v, 5, v, 3)},
		{"lda zero", 8, g(N, N, 0, 3, 3, v, 0, v, 3, v, 1)},
		{"ldb notrans", 10, g(N, N, 3, 3, 3, v, 3, v, 2, v, 3)},
		{"ldb trans", 10, g(N, T_, 3, 5, 3, v, 3, v, 4, v, 3)},
		{"ldc", 13, g(N, N, 3, 3, 3, v, 3, v, 3, v, 2)},
		{"a short", 7, g(N, N, 3, 3, 3, v[:8], 3, v, 3, v, 3)},
		{"a short padded", 7, g(N, N, 3, 3, 3, v[:12], 5, v, 3, v, 3)},
		{"a short trans", 7, g(T_, N, 3, 3, 4, v[:11], 4, v, 4, v, 3)},
		{"b short", 9, g(N, N, 3, 3, 3, v, 3, v[:8], 3, v, 3)},
		{"b short trans", 9, g(N, T_, 3, 3, 4, v, 3, v[:11], 3, v, 3)},
		{"c short", 12, g(N, N, 3, 3, 3, v, 3, v, 3, v[:8], 3)},
		{"c short padded", 12, g(N, N, 3, 3, 3, v, 3, v, 3, v[:12], 5)},
		{"a size overflows int", 7, g(N, N, 3, 3, 3, v, maxInt, v, 3, v, 3)},
		{"b size overflows int", 9, g(N, N, 3, 3, 3, v, 3, v, maxInt, v, 3)},
		{"c size overflows int", 12, g(N, N, 3, 3, 3, v, 3, v, 3, v, maxInt)},
		{"a size overflows by one", 7, g(N, N, 1, 1, 2, v, maxInt, v, 2, v, 1)},
	})
	// The first bad parameter wins, in the reference's order.
	runErrCases(t, routine, []errCase{
		{"transA before m", 1, g('X', N, -1, 3, 3, nil, 0, nil, 0, nil, 0)},
		{"m before lda", 3, g(N, N, -1, 3, 3, nil, 0, nil, 0, nil, 0)},
		{"lda before slices", 8, g(N, N, 3, 3, 3, nil, 1, nil, 3, nil, 3)},
		{"a before b before c", 7, g(N, N, 3, 3, 3, nil, 3, nil, 3, nil, 3)},
	})
	// Not errors: a slice a call does not read, or an empty product.
	ok := []func() error{
		func() error { return im.gemm(N, N, 3, 3, 3, 0, nil, 3, nil, 3, 1.5, v, 3) }, // alpha = 0 reads no A, B
		func() error { return im.gemm(N, N, 3, 3, 3, 1, v, 3, v, 3, 1, v[:9], 3) },
		func() error { return im.gemm(N, N, 0, 3, 3, 1, nil, 1, nil, 3, 1, nil, 1) },
	}
	for i, f := range ok {
		if err := f(); err != nil {
			t.Errorf("%s valid call %d: %v", routine, i, err)
		}
	}
}

func TestGemmErrors(t *testing.T) {
	each(t, func(t *testing.T, im impl[float64]) { testGemmErrors(t, im, "DGEMM") },
		func(t *testing.T, im impl[float32]) { testGemmErrors(t, im, "SGEMM") })
}

func testGemvErrors[T float](t *testing.T, im impl[T], routine string) {
	v := make([]T, 100)
	N, T_ := blas.NoTrans, blas.Trans
	g := func(tr blas.Transpose, m, n int, a []T, lda int, x []T, incx int, y []T, incy int) func() error {
		return func() error { return im.gemv(tr, m, n, 1, a, lda, x, incx, 1, y, incy) }
	}
	runErrCases(t, routine, []errCase{
		{"trans", 1, g('Q', 3, 3, v, 3, v, 1, v, 1)},
		{"m", 2, g(N, -1, 3, v, 3, v, 1, v, 1)},
		{"n", 3, g(N, 3, -1, v, 3, v, 1, v, 1)},
		{"lda", 6, g(N, 3, 3, v, 2, v, 1, v, 1)},
		{"lda trans", 6, g(T_, 3, 3, v, 2, v, 1, v, 1)},
		{"incx", 8, g(N, 3, 3, v, 3, v, 0, v, 1)},
		{"incy", 11, g(N, 3, 3, v, 3, v, 1, v, 0)},
		{"a short", 5, g(N, 3, 3, v[:8], 3, v, 1, v, 1)},
		{"a short padded", 5, g(N, 3, 3, v[:10], 4, v, 1, v, 1)},
		{"x short", 7, g(N, 3, 4, v, 3, v[:3], 1, v, 1)},
		{"x short trans", 7, g(T_, 4, 3, v, 4, v[:3], 1, v, 1)},
		{"x short strided", 7, g(N, 3, 3, v, 3, v[:4], 2, v, 1)},
		{"x short negative inc", 7, g(N, 3, 3, v, 3, v[:4], -2, v, 1)},
		{"y short", 10, g(N, 4, 3, v, 4, v, 1, v[:3], 1)},
		{"y short trans", 10, g(T_, 3, 4, v, 3, v, 1, v[:3], 1)},
		{"y short strided", 10, g(N, 3, 3, v, 3, v, 1, v[:4], 2)},
		{"a size overflows int", 5, g(N, 3, 3, v, maxInt, v, 1, v, 1)},
		{"x size overflows int", 7, g(N, 3, 3, v, 3, v, maxInt, v, 1)},
		{"y size overflows int", 10, g(N, 3, 3, v, 3, v, 1, v, math.MinInt)},
	})
	ok := []func() error{
		func() error { return im.gemv(N, 3, 3, 0, nil, 3, nil, 1, 2, v, 1) }, // alpha = 0 reads no A, x
		func() error { return im.gemv(N, 3, 3, 1, v, 3, v[:3], 1, 1, v[:3], 1) },
		func() error { return im.gemv(N, 0, 3, 1, nil, 1, nil, 1, 1, nil, 1) },
	}
	for i, f := range ok {
		if err := f(); err != nil {
			t.Errorf("%s valid call %d: %v", routine, i, err)
		}
	}
}

func TestGemvErrors(t *testing.T) {
	each(t, func(t *testing.T, im impl[float64]) { testGemvErrors(t, im, "DGEMV") },
		func(t *testing.T, im impl[float32]) { testGemvErrors(t, im, "SGEMV") })
}

func testLevel1Errors[T float](t *testing.T, im impl[T], p string) {
	v := make([]T, 20)
	short := v[:4]
	runErrCases(t, p+"DOT", []errCase{
		{"x short", 2, func() error { _, e := im.dot(5, short, 1, v, 1); return e }},
		{"y short", 4, func() error { _, e := im.dot(5, v, 1, short, 1); return e }},
		{"x strided short", 2, func() error { _, e := im.dot(3, v[:4], 2, v, 1); return e }},
		{"x negative short", 2, func() error { _, e := im.dot(3, v[:4], -2, v, 1); return e }},
		{"x overflows", 2, func() error { _, e := im.dot(3, v, maxInt, v, 1); return e }},
		{"x most negative inc", 2, func() error { _, e := im.dot(3, v, math.MinInt, v, 1); return e }},
	})
	runErrCases(t, p+"AXPY", []errCase{
		{"x short", 3, func() error { return im.axpy(5, 2, short, 1, v, 1) }},
		{"y short", 5, func() error { return im.axpy(5, 2, v, 1, short, 1) }},
	})
	runErrCases(t, p+"SCAL", []errCase{{"x short", 3, func() error { return im.scal(5, 2, short, 1) }}})
	runErrCases(t, p+"NRM2", []errCase{{"x short", 2, func() error { _, e := im.nrm2(5, short, 1); return e }}})
	runErrCases(t, p+"ASUM", []errCase{{"x short", 2, func() error { _, e := im.asum(5, short, 1); return e }}})
	runErrCases(t, "I"+p+"AMAX", []errCase{{"x short", 2, func() error { _, e := im.iamax(5, short, 1); return e }}})

	// Not errors: nothing is read or written for n <= 0, alpha = 0, a
	// non-positive increment where the reference returns, and n = 1 with any
	// increment.
	if e := im.axpy(0, 1, nil, 1, nil, 1); e != nil {
		t.Error(e)
	}
	if e := im.axpy(5, 0, nil, 1, nil, 1); e != nil {
		t.Error(e)
	}
	if e := im.scal(5, 2, nil, 0); e != nil {
		t.Error(e)
	}
	if e := im.scal(5, 1, nil, 1); e != nil {
		t.Error(e)
	}
	if _, e := im.asum(5, nil, -1); e != nil {
		t.Error(e)
	}
	if _, e := im.iamax(0, nil, 1); e != nil {
		t.Error(e)
	}
	if _, e := im.dot(1, v[:1], maxInt, v[:1], math.MinInt); e != nil {
		t.Error(e)
	}
}

func TestLevel1Errors(t *testing.T) {
	for _, im := range impls64 {
		t.Run("float64/"+im.name, func(t *testing.T) { defer noPanic(t); testLevel1Errors(t, im, "D") })
	}
	for _, im := range impls32 {
		t.Run("float32/"+im.name, func(t *testing.T) { defer noPanic(t); testLevel1Errors(t, im, "S") })
	}
}

// callRecovered runs call and returns its error, or fails the test, and returns
// nil, if it panicked.
func callRecovered(t *testing.T, what string, call func() error) (err error) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s: panicked: %v", what, r)
			err = nil
		}
	}()
	return call()
}
