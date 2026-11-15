package blas_test

import (
	"math"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

var gemvShapes = [][2]int{{0, 3}, {3, 0}, {1, 1}, {5, 7}, {37, 19}, {64, 64}, {130, 3}}

// incs are pairs (incx, incy): the unit ones that take the vector shortcuts, a
// negative one for each vector, and strided ones.
var incPairs = [][2]int{{1, 1}, {2, -1}, {-3, 2}, {1, -2}, {-1, 1}, {1, 3}, {2, 1}}

func testGemvBound[T float](t *testing.T, im impl[T]) {
	rng := newRNG(11)
	for _, sh := range gemvShapes {
		m, n := sh[0], sh[1]
		for _, tr := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
			lenx, leny := n, m
			if tr != blas.NoTrans {
				lenx, leny = m, n
			}
			for _, ic := range incPairs {
				incx, incy := ic[0], ic[1]
				for _, pad := range []int{0, 2} {
					for _, ab := range alphaBeta {
						alpha, beta := T(ab[0]), T(ab[1])
						lda := max(1, m) + pad
						a := matrix[T](rng, m, n, lda)
						x, y := vector[T](rng, lenx, incx), vector[T](rng, leny, incy)
						x0, y0, a0 := clone(x), clone(y), clone(a)
						if beta == 0 && len(y) > 0 {
							y[0] = T(math.NaN()) // y must not be read
						}
						r := oracle.Gemv(byte(tr), m, n, alpha, a, lda, x, incx, beta, y0, incy)
						y1 := clone(y)
						if err := im.gemv(tr, m, n, alpha, a, lda, x, incx, beta, y, incy); err != nil {
							t.Fatal(err)
						}
						if m == 0 || n == 0 { // a quick return: y is not even scaled
							if !same(y, y1) {
								t.Fatal("quick return wrote y")
							}
							continue
						}
						what := "gemv " + transName(tr) + " " + itoa(m) + "x" + itoa(n) + " inc " + itoa(incx) + "," + itoa(incy) + " ab " + ftoa(ab[0]) + "," + ftoa(ab[1])
						gp := oracle.Gamma[T](lenx + 2)
						for i := 0; i < leny; i++ {
							within(t, what, y[oracle.Idx(leny, incy, i)], r, i, gp, gp)
						}
						gapsUntouched(t, what+" y", y, incy)
						if !same(x, x0) || !same(a, a0) {
							t.Fatalf("%s: an input was modified", what)
						}
					}
				}
			}
		}
	}
}

func TestGemvBound(t *testing.T) {
	each(t, testGemvBound[float64], testGemvBound[float32])
}

// alpha = 0 reads neither A nor x; the quick returns read nothing.
func testGemvScalarRules[T float](t *testing.T, im impl[T]) {
	rng := newRNG(12)
	nan := T(math.NaN())
	y := vector[T](rng, 4, 1)
	want := make([]T, 4)
	for i := range y {
		want[i] = 2 * y[i]
	}
	if err := im.gemv(blas.NoTrans, 4, 3, 0, nil, 4, nil, 1, 2, y, 1); err != nil {
		t.Fatal(err)
	}
	if !same(y, want) {
		t.Fatal("alpha=0: y != beta*y")
	}
	for i := range y {
		y[i] = nan
	}
	if err := im.gemv(blas.Trans, 3, 4, 0, nil, 3, nil, 1, 0, y, 1); err != nil {
		t.Fatal(err)
	}
	for _, v := range y {
		if v != 0 {
			t.Fatal("alpha=0, beta=0: y is not zero")
		}
	}
	y = []T{nan}
	for _, call := range []func() error{
		func() error { return im.gemv(blas.NoTrans, 0, 3, 1, nil, 1, nil, 1, 0, y, 1) },
		func() error { return im.gemv(blas.NoTrans, 3, 0, 1, nil, 3, nil, 1, 0, y, 1) },
		func() error { return im.gemv(blas.NoTrans, 3, 3, 0, nil, 3, nil, 1, 1, y, 1) },
	} {
		if err := call(); err != nil {
			t.Fatal(err)
		}
		if y[0] == y[0] {
			t.Fatal("a quick return wrote y")
		}
	}
}

func TestGemvScalarRules(t *testing.T) {
	each(t, testGemvScalarRules[float64], testGemvScalarRules[float32])
}
