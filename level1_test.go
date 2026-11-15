package blas_test

import (
	"math"
	"testing"

	"github.com/go-blas/blas/internal/oracle"
)

var l1Sizes = []int{-1, 0, 1, 2, 5, 6, 33, 1000}

var l1Incs = [][2]int{{1, 1}, {2, 3}, {-1, 1}, {1, -2}, {-2, -2}, {0, 1}, {1, 0}}

func testDot[T float](t *testing.T, im impl[T]) {
	rng := newRNG(21)
	for _, n := range l1Sizes {
		for _, ic := range l1Incs {
			x, y := vector[T](rng, n, ic[0]), vector[T](rng, n, ic[1])
			r := oracle.Dot(n, x, ic[0], y, ic[1])
			got, err := im.dot(n, x, ic[0], y, ic[1])
			if err != nil {
				t.Fatal(err)
			}
			g := oracle.Gamma[T](n)
			within(t, "dot n="+itoa(n)+" inc="+itoa(ic[0])+","+itoa(ic[1]), got, r, 0, g, g)
		}
	}
}

func TestDot(t *testing.T) { each(t, testDot[float64], testDot[float32]) }

func testAxpy[T float](t *testing.T, im impl[T]) {
	rng := newRNG(22)
	for _, n := range l1Sizes {
		for _, ic := range l1Incs {
			for _, alpha := range []T{0, 1, -2.5} {
				x, y := vector[T](rng, n, ic[0]), vector[T](rng, n, ic[1])
				x0, y0 := clone(x), clone(y)
				r := oracle.Axpy(n, alpha, x, ic[0], y0, ic[1])
				if err := im.axpy(n, alpha, x, ic[0], y, ic[1]); err != nil {
					t.Fatal(err)
				}
				what := "axpy n=" + itoa(n) + " inc=" + itoa(ic[0]) + "," + itoa(ic[1])
				g := oracle.Gamma[T](2)
				for i := 0; i < n; i++ {
					got := y[oracle.Idx(n, ic[1], i)]
					if alpha == 0 {
						if got != y0[oracle.Idx(n, ic[1], i)] {
							t.Fatalf("%s: alpha=0 changed y", what)
						}
						continue
					}
					if ic[1] == 0 {
						continue // every element is the one cell y[0]: accumulated n times
					}
					within(t, what, got, r, i, g, g)
				}
				if !same(x, x0) {
					t.Fatalf("%s: x was modified", what)
				}
				gapsUntouched(t, what, y, ic[1])
			}
		}
	}
}

func TestAxpy(t *testing.T) { each(t, testAxpy[float64], testAxpy[float32]) }

func testScal[T float](t *testing.T, im impl[T]) {
	rng := newRNG(23)
	for _, n := range l1Sizes {
		for _, inc := range []int{1, 2, 3, 0, -1} {
			for _, alpha := range []T{0, 1, -2.5} {
				x := vector[T](rng, n, inc)
				x0 := clone(x)
				var r oracle.Result
				if inc > 0 {
					r = oracle.Scal(n, alpha, x, inc)
				}
				if err := im.scal(n, alpha, x, inc); err != nil {
					t.Fatal(err)
				}
				what := "scal n=" + itoa(n) + " inc=" + itoa(inc)
				if inc <= 0 || alpha == 1 {
					if !same(x, x0) {
						t.Fatalf("%s: quick return wrote x", what)
					}
					continue
				}
				g := oracle.Gamma[T](1)
				for i := 0; i < n; i++ {
					within(t, what, x[i*inc], r, i, g, g)
				}
				gapsUntouched(t, what, x, inc)
			}
		}
	}
	// 0*NaN is NaN, as in the reference.
	x := []T{T(math.NaN()), 1}
	if err := im.scal(2, 0, x, 1); err != nil {
		t.Fatal(err)
	}
	if x[0] == x[0] || x[1] != 0 {
		t.Fatalf("scal by 0 = %v, want [NaN 0]", x)
	}
}

func TestScal(t *testing.T) { each(t, testScal[float64], testScal[float32]) }

func testNrm2[T float](t *testing.T, im impl[T]) {
	rng := newRNG(24)
	check := func(what string, n int, x []T, inc int) {
		t.Helper()
		got, err := im.nrm2(n, x, inc)
		if err != nil {
			t.Fatal(err)
		}
		r := oracle.Nrm2(n, x, inc)
		g := oracle.Gamma[T](n + 2)
		within(t, what, got, r, 0, g, g)
	}
	for _, n := range l1Sizes {
		for _, inc := range []int{1, 2, -1, -3} {
			check("nrm2 n="+itoa(n)+" inc="+itoa(inc), n, vector[T](rng, n, inc), inc)
		}
	}
	// Values whose squares would overflow or underflow: the three bins.
	var huge, tiny T = 1e30, 1e-30
	if _, ok := any(huge).(float64); ok {
		h, l := 1e200, 1e-200
		huge, tiny = T(h), T(l)
	}
	// Ten values just below the small bin's limit (2^-511, 2^-63) and one at it:
	// the small bin outweighs the medium one.
	e := -63
	if _, ok := any(huge).(float64); ok {
		e = -511
	}
	edge := T(math.Ldexp(1, e))
	below := T(math.Ldexp(0.99, e))
	var mixed []T
	for range 10 {
		mixed = append(mixed, below)
	}
	mixed = append(mixed, edge)
	check("nrm2 small bin outweighs medium", len(mixed), mixed, 1)
	for _, v := range [][]T{
		{huge, huge, 1}, {tiny, tiny, tiny}, {huge, tiny, 1}, {tiny, 1}, {tiny, huge}, {huge}, {tiny},
		{huge, huge, huge, 1, 1}, {tiny, tiny, 1, 1, huge},
	} {
		check("nrm2 range", len(v), v, 1)
	}
}

func TestNrm2(t *testing.T) { each(t, testNrm2[float64], testNrm2[float32]) }

func testAsum[T float](t *testing.T, im impl[T]) {
	rng := newRNG(25)
	for _, n := range l1Sizes {
		for _, inc := range []int{1, 2, 3, 0, -1} {
			x := vector[T](rng, n, inc)
			got, err := im.asum(n, x, inc)
			if err != nil {
				t.Fatal(err)
			}
			if inc <= 0 {
				if got != 0 {
					t.Fatalf("asum with inc %d = %v, want 0", inc, got)
				}
				continue
			}
			g := oracle.Gamma[T](max(n, 1))
			within(t, "asum n="+itoa(n)+" inc="+itoa(inc), got, oracle.Asum(n, x, inc), 0, g, g)
		}
	}
}

func TestAsum(t *testing.T) { each(t, testAsum[float64], testAsum[float32]) }

func testIamax[T float](t *testing.T, im impl[T]) {
	rng := newRNG(26)
	for _, n := range l1Sizes {
		for _, inc := range []int{1, 2, 3, 0, -1} {
			x := vector[T](rng, n, inc)
			got, err := im.iamax(n, x, inc)
			if err != nil {
				t.Fatal(err)
			}
			want := 0
			if n >= 1 && inc > 0 {
				want = 1
				for i := 1; i < n; i++ {
					if math.Abs(float64(x[i*inc])) > math.Abs(float64(x[(want-1)*inc])) {
						want = i + 1
					}
				}
			}
			if got != want {
				t.Fatalf("iamax n=%d inc=%d = %d, want %d", n, inc, got, want)
			}
		}
	}
	// Ties go to the first; a NaN is never larger, and nothing is larger than a
	// NaN that comes first.
	nan := T(math.NaN())
	for _, c := range []struct {
		x    []T
		want int
	}{{[]T{1, -3, 3, 2}, 2}, {[]T{nan, 1, 2}, 1}, {[]T{1, nan, 0}, 1}, {[]T{0, 0, 0}, 1}} {
		if got, _ := im.iamax(len(c.x), c.x, 1); got != c.want {
			t.Fatalf("iamax(%v) = %d, want %d", c.x, got, c.want)
		}
	}
}

func TestIamax(t *testing.T) { each(t, testIamax[float64], testIamax[float32]) }
