package blas_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

// simpl is one element type's Symv, Syr2, Rot, Swap and Copy.
type simpl[T float] struct {
	symv func(uplo blas.Uplo, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) error
	syr2 func(uplo blas.Uplo, n int, alpha T, x []T, incx int, y []T, incy int, a []T, lda int) error
	rot  func(n int, x []T, incx int, y []T, incy int, c, s T) error
	swap func(n int, x []T, incx int, y []T, incy int) error
	cp   func(n int, x []T, incx int, y []T, incy int) error
}

var simpl64 = simpl[float64]{blas.Dsymv, blas.Dsyr2, blas.Drot, blas.Dswap, blas.Dcopy}
var simpl32 = simpl[float32]{blas.Ssymv, blas.Ssyr2, blas.Srot, blas.Sswap, blas.Scopy}

func eachS(t *testing.T, f64 func(*testing.T, simpl[float64]), f32 func(*testing.T, simpl[float32])) {
	t.Helper()
	t.Run("float64", func(t *testing.T) { defer noPanic(t); f64(t, simpl64) })
	t.Run("float32", func(t *testing.T) { defer noPanic(t); f32(t, simpl32) })
}

var symShapes = []int{0, 1, 2, 5, 37, 64, 130}

// symStore returns an n x n matrix with the triangle up names random, the other
// triangle NaN (it must not be read), the padding the sentinel; and the full
// symmetric matrix the triangle stands for.
func symStore[T float](n, lda int, up bool, seed uint64) (a, full []T) {
	rng := newRNG(seed)
	tri := matrix[T](rng, n, n, lda)
	a = clone(tri)
	full = make([]T, n*n)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			in := i == j || (up && i < j) || (!up && i > j)
			if in {
				full[i+j*n] = tri[i+j*lda]
				full[j+i*n] = tri[i+j*lda]
			} else {
				a[i+j*lda] = T(math.NaN())
			}
		}
	}
	return a, full
}

func upName(up bool) string {
	if up {
		return "U"
	}
	return "L"
}

func testSymvBound[T float](t *testing.T, im simpl[T]) {
	rng := newRNG(21)
	for _, n := range symShapes {
		for _, up := range []bool{true, false} {
			for _, ic := range incPairs {
				for _, pad := range []int{0, 2} {
					for _, ab := range alphaBeta {
						alpha, beta := T(ab[0]), T(ab[1])
						lda := max(1, n) + pad
						a, full := symStore[T](n, lda, up, uint64(n*7+pad))
						x, y := vector[T](rng, n, ic[0]), vector[T](rng, n, ic[1])
						x0, y0, a0 := clone(x), clone(y), clone(a)
						if beta == 0 && len(y) > 0 {
							y[0] = T(math.NaN())
						}
						r := oracle.Gemv(byte('N'), n, n, alpha, full, max(1, n), x, ic[0], beta, y0, ic[1])
						if err := im.symv(uploOf(up), n, alpha, a, lda, x, ic[0], beta, y, ic[1]); err != nil {
							t.Fatal(err)
						}
						if n == 0 {
							continue
						}
						what := "symv " + upName(up) + " n" + itoa(n) + " inc " + itoa(ic[0]) + "," + itoa(ic[1]) + " ab " + ftoa(ab[0]) + "," + ftoa(ab[1])
						g := oracle.Gamma[T](n + 2)
						for i := 0; i < n; i++ {
							within(t, what, y[oracle.Idx(n, ic[1], i)], r, i, g, g)
						}
						gapsUntouched(t, what+" y", y, ic[1])
						if !same(x, x0) || !sameNaN(a, a0) {
							t.Fatalf("%s: an input was modified", what)
						}
					}
				}
			}
		}
	}
}

func uploOf(up bool) blas.Uplo {
	if up {
		return blas.Upper
	}
	return blas.Lower
}

// sameNaN is same, with NaN equal to NaN.
func sameNaN[T float](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] && !(a[i] != a[i] && b[i] != b[i]) {
			return false
		}
	}
	return true
}

func TestSymvBound(t *testing.T) { eachS(t, testSymvBound[float64], testSymvBound[float32]) }

func testSymvScalarRules[T float](t *testing.T, im simpl[T]) {
	nan := T(math.NaN())
	// alpha = 0 reads neither A nor x; beta = 0 does not read y.
	y := []T{1, 2, 3}
	if err := im.symv(blas.Lower, 3, 0, nil, 3, nil, 1, 2, y, 1); err != nil || !same(y, []T{2, 4, 6}) {
		t.Fatalf("alpha=0: y %v, err %v", y, err)
	}
	y = []T{nan, nan, nan}
	if err := im.symv(blas.Upper, 3, 0, nil, 3, nil, 1, 0, y, 1); err != nil || !same(y, []T{0, 0, 0}) {
		t.Fatalf("alpha=0, beta=0: y %v, err %v", y, err)
	}
	// Quick returns: n = 0, alpha = 0 and beta = 1 write nothing, read no slice.
	y = []T{nan}
	for _, c := range []func() error{
		func() error { return im.symv(blas.Upper, 0, 1, nil, 1, nil, 1, 0, y, 1) },
		func() error { return im.symv(blas.Upper, 3, 0, nil, 3, nil, 1, 1, nil, 1) },
	} {
		if err := c(); err != nil || y[0] == y[0] {
			t.Fatalf("quick return: err %v, y %v", err, y)
		}
	}
	// Lower case flags.
	a := []T{2}
	y = []T{1}
	if err := im.symv('l', 1, 3, a, 1, []T{5}, 1, 1, y, 1); err != nil || y[0] != 31 {
		t.Fatalf("symv n=1: y %v, err %v", y, err)
	}
}

func TestSymvScalarRules(t *testing.T) {
	eachS(t, testSymvScalarRules[float64], testSymvScalarRules[float32])
}

func testSyr2Bound[T float](t *testing.T, im simpl[T]) {
	rng := newRNG(22)
	for _, n := range symShapes {
		for _, up := range []bool{true, false} {
			for _, ic := range incPairs {
				for _, alpha := range []T{1, -0.75, 2.5} {
					lda := max(1, n) + 1
					a, _ := symStore[T](n, lda, up, uint64(n*5+1))
					x, y := vector[T](rng, n, ic[0]), vector[T](rng, n, ic[1])
					x0, y0, a0 := clone(x), clone(y), clone(a)
					if err := im.syr2(uploOf(up), n, alpha, x, ic[0], y, ic[1], a, lda); err != nil {
						t.Fatal(err)
					}
					what := "syr2 " + upName(up) + " n" + itoa(n) + " inc " + itoa(ic[0]) + "," + itoa(ic[1])
					// temp := alpha*y(j); x(i)*temp (2 roundings) + y(i)*temp2
					// (2), summed (3), added to a (4).
					g := oracle.Gamma[T](4)
					for j := 0; j < n; j++ {
						for i := 0; i < n; i++ {
							in := i == j || (up && i < j) || (!up && i > j)
							got, was := a[i+j*lda], a0[i+j*lda]
							if !in {
								if !(got == was || (got != got && was != was)) {
									t.Fatalf("%s: (%d,%d) outside the triangle was written", what, i, j)
								}
								continue
							}
							xi, yi := float64(x[oracle.Idx(n, ic[0], i)]), float64(y[oracle.Idx(n, ic[1], i)])
							xj, yj := float64(x[oracle.Idx(n, ic[0], j)]), float64(y[oracle.Idx(n, ic[1], j)])
							bg := func(v float64) *big.Float { return new(big.Float).SetPrec(256).SetFloat64(v) }
							p := new(big.Float).SetPrec(256).Add(
								new(big.Float).SetPrec(256).Mul(bg(xi), bg(yj)),
								new(big.Float).SetPrec(256).Mul(bg(yi), bg(xj)))
							p.Mul(p, bg(float64(alpha)))
							exact := new(big.Float).SetPrec(256).Add(p, bg(float64(was)))
							mag := new(big.Float).SetPrec(256).Add(new(big.Float).Abs(bg(float64(was))),
								new(big.Float).Abs(new(big.Float).SetPrec(256).Mul(bg(float64(alpha)),
									new(big.Float).SetPrec(256).Add(
										new(big.Float).Abs(new(big.Float).SetPrec(256).Mul(bg(xi), bg(yj))),
										new(big.Float).Abs(new(big.Float).SetPrec(256).Mul(bg(yi), bg(xj)))))))
							diff := new(big.Float).Abs(new(big.Float).SetPrec(256).Sub(bg(float64(got)), exact))
							lim := new(big.Float).SetPrec(256).Mul(g, mag)
							if diff.Cmp(lim) > 0 {
								t.Fatalf("%s: (%d,%d) = %v, exact %v, outside gamma_4 (|a|+|alpha||xy|)", what, i, j, got, exact)
							}
						}
					}
					gapsUntouched(t, what+" x", x, ic[0])
					if !same(x, x0) || !same(y, y0) {
						t.Fatalf("%s: x or y was modified", what)
					}
				}
			}
		}
	}
}

func TestSyr2Bound(t *testing.T) { eachS(t, testSyr2Bound[float64], testSyr2Bound[float32]) }

func testRotSwapCopy[T float](t *testing.T, im simpl[T]) {
	rng := newRNG(23)
	for _, n := range []int{0, 1, 2, 5, 37, 100} {
		for _, ic := range incPairs {
			cs := [][2]T{{1, 0}, {0, 1}, {0.6, 0.8}, {-0.28, 0.96}}
			for _, c := range cs {
				x, y := vector[T](rng, n, ic[0]), vector[T](rng, n, ic[1])
				x0, y0 := clone(x), clone(y)
				if err := im.rot(n, x, ic[0], y, ic[1], c[0], c[1]); err != nil {
					t.Fatal(err)
				}
				what := "rot n" + itoa(n) + " inc " + itoa(ic[0]) + "," + itoa(ic[1])
				g := oracle.Gamma[T](2)
				for i := 0; i < n; i++ {
					px, py := oracle.Idx(n, ic[0], i), oracle.Idx(n, ic[1], i)
					for k := 0; k < 2; k++ {
						bg := func(v T) *big.Float { return oracle.B(v) }
						var exact, mag *big.Float
						var got T
						if k == 0 { // x := c*x + s*y
							exact = new(big.Float).SetPrec(256).Add(new(big.Float).SetPrec(256).Mul(bg(c[0]), bg(x0[px])), new(big.Float).SetPrec(256).Mul(bg(c[1]), bg(y0[py])))
							got = x[px]
						} else { // y := c*y - s*x
							exact = new(big.Float).SetPrec(256).Sub(new(big.Float).SetPrec(256).Mul(bg(c[0]), bg(y0[py])), new(big.Float).SetPrec(256).Mul(bg(c[1]), bg(x0[px])))
							got = y[py]
						}
						mag = new(big.Float).SetPrec(256).Add(
							new(big.Float).Abs(new(big.Float).SetPrec(256).Mul(bg(c[0]), bg(x0[px]))),
							new(big.Float).Abs(new(big.Float).SetPrec(256).Mul(bg(c[1]), bg(y0[py]))))
						if k == 1 {
							mag = new(big.Float).SetPrec(256).Add(
								new(big.Float).Abs(new(big.Float).SetPrec(256).Mul(bg(c[0]), bg(y0[py]))),
								new(big.Float).Abs(new(big.Float).SetPrec(256).Mul(bg(c[1]), bg(x0[px]))))
						}
						diff := new(big.Float).Abs(new(big.Float).SetPrec(256).Sub(bg(got), exact))
						if diff.Cmp(new(big.Float).SetPrec(256).Mul(g, mag)) > 0 {
							t.Fatalf("%s: element %d of %c outside gamma_2", what, i, "xy"[k])
						}
					}
				}
				gapsUntouched(t, what+" x", x, ic[0])
				gapsUntouched(t, what+" y", y, ic[1])
			}
			// Swap and copy are exact.
			x, y := vector[T](rng, n, ic[0]), vector[T](rng, n, ic[1])
			x0, y0 := clone(x), clone(y)
			if err := im.swap(n, x, ic[0], y, ic[1]); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < n; i++ {
				px, py := oracle.Idx(n, ic[0], i), oracle.Idx(n, ic[1], i)
				if x[px] != y0[py] || y[py] != x0[px] {
					t.Fatalf("swap n%d inc %d,%d: element %d", n, ic[0], ic[1], i)
				}
			}
			gapsUntouched(t, "swap x", x, ic[0])
			gapsUntouched(t, "swap y", y, ic[1])
			x, y = clone(x0), clone(y0)
			if err := im.cp(n, x, ic[0], y, ic[1]); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < n; i++ {
				px, py := oracle.Idx(n, ic[0], i), oracle.Idx(n, ic[1], i)
				if y[py] != x0[px] || x[px] != x0[px] {
					t.Fatalf("copy n%d inc %d,%d: element %d", n, ic[0], ic[1], i)
				}
			}
			gapsUntouched(t, "copy y", y, ic[1])
		}
	}
	// n <= 0 returns at once, whatever the slices.
	for _, n := range []int{0, -3} {
		if im.rot(n, nil, 1, nil, 1, 1, 0) != nil || im.swap(n, nil, 1, nil, 1) != nil || im.cp(n, nil, 1, nil, 1) != nil {
			t.Fatalf("n=%d: not a quick return", n)
		}
	}
}

func TestRotSwapCopy(t *testing.T) {
	eachS(t, testRotSwapCopy[float64], testRotSwapCopy[float32])
}

func testSymErrors[T float](t *testing.T, im simpl[T], p string) {
	a9, a8 := make([]T, 9), make([]T, 8)
	v3, v2 := make([]T, 3), make([]T, 2)
	sv, sr := "DSYMV", "DSYR2"
	rt, sw, cp := "DROT", "DSWAP", "DCOPY"
	if p == "S" {
		sv, sr, rt, sw, cp = "SSYMV", "SSYR2", "SROT", "SSWAP", "SCOPY"
	}
	runErrCases(t, sv, []errCase{
		{"uplo", 1, func() error { return im.symv('X', 3, 1, a9, 3, v3, 1, 1, v3, 1) }},
		{"n", 2, func() error { return im.symv('L', -1, 1, a9, 3, v3, 1, 1, v3, 1) }},
		{"lda", 5, func() error { return im.symv('L', 3, 1, a9, 2, v3, 1, 1, v3, 1) }},
		{"incx", 7, func() error { return im.symv('L', 3, 1, a9, 3, v3, 0, 1, v3, 1) }},
		{"incy", 10, func() error { return im.symv('L', 3, 1, a9, 3, v3, 1, 1, v3, 0) }},
		{"a short", 4, func() error { return im.symv('U', 3, 1, a8, 3, v3, 1, 1, v3, 1) }},
		{"x short", 6, func() error { return im.symv('U', 3, 1, a9, 3, v2, 1, 1, v3, 1) }},
		{"x short inc", 6, func() error { return im.symv('U', 3, 1, a9, 3, v3, -2, 1, v3, 1) }},
		{"y short", 9, func() error { return im.symv('U', 3, 0, a9, 3, v3, 1, 2, v2, 1) }},
		{"huge", 9, func() error { return im.symv('U', 3, 1, a9, 3, v3, 1, 1, v3, math.MaxInt) }},
	})
	runErrCases(t, sr, []errCase{
		{"uplo", 1, func() error { return im.syr2('X', 3, 1, v3, 1, v3, 1, a9, 3) }},
		{"n", 2, func() error { return im.syr2('L', -1, 1, v3, 1, v3, 1, a9, 3) }},
		{"incx", 5, func() error { return im.syr2('L', 3, 1, v3, 0, v3, 1, a9, 3) }},
		{"incy", 7, func() error { return im.syr2('L', 3, 1, v3, 1, v3, 0, a9, 3) }},
		{"lda", 9, func() error { return im.syr2('L', 3, 1, v3, 1, v3, 1, a9, 2) }},
		{"x short", 4, func() error { return im.syr2('U', 3, 1, v2, 1, v3, 1, a9, 3) }},
		{"y short", 6, func() error { return im.syr2('U', 3, 1, v3, 1, v2, 1, a9, 3) }},
		{"a short", 8, func() error { return im.syr2('U', 3, 1, v3, 1, v3, 1, a8, 3) }},
	})
	for name, call := range map[string]func(n int, x []T, incx int, y []T, incy int) error{
		rt: func(n int, x []T, incx int, y []T, incy int) error { return im.rot(n, x, incx, y, incy, 1, 0) },
		sw: im.swap, cp: im.cp,
	} {
		runErrCases(t, name, []errCase{
			{"x short", 2, func() error { return call(3, v2, 1, v3, 1) }},
			{"x short inc", 2, func() error { return call(3, v3, 2, v3, 1) }},
			{"y short", 4, func() error { return call(3, v3, 1, v2, 1) }},
			{"y short neg", 4, func() error { return call(3, v3, 1, v3, -2) }},
			{"huge", 4, func() error { return call(3, v3, 1, v3, math.MaxInt) }},
		})
	}
}

func TestSymErrors(t *testing.T) {
	t.Run("float64", func(t *testing.T) { testSymErrors(t, simpl64, "D") })
	t.Run("float32", func(t *testing.T) { testSymErrors(t, simpl32, "S") })
}
