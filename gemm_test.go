package blas_test

import (
	"math"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

type shape struct{ m, n, k int }

// shapes cover the empty products, the unit one, odd sizes, the serial and the
// parallel kernel path (m*n >= 6000), and a contraction longer than the kernel's
// KC block of 256, which sums in several partial products.
var gemmShapes = []shape{
	{0, 5, 3}, {4, 0, 3}, {4, 5, 0}, {1, 1, 1}, {7, 5, 3}, {13, 17, 19}, {37, 41, 43},
	{5, 100, 7}, {100, 5, 7}, {6, 8, 300}, {80, 80, 30},
}

// alphaBeta are the scalar pairs: the identity cases the reference branches on
// (0 and 1) and general ones.
var alphaBeta = [][2]float64{{1, 0}, {1, 1}, {2.5, 0}, {2.5, 1}, {1, -1.5}, {0.75, 2}, {0, 0}, {0, 3}, {-1, 0.5}}

func testGemmBound[T float](t *testing.T, im impl[T]) {
	rng := newRNG(1)
	for _, sh := range gemmShapes {
		big := sh.m*sh.n*sh.k > 50000
		for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			for _, tB := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, pad := range []int{0, 3} {
					if big && pad > 0 {
						continue
					}
					for ia, ab := range alphaBeta {
						// The largest products cost the 256-bit oracle the most: three scalar
						// pairs, and one under -short (the emulated CI lanes).
						if big && (ia >= 3 || (testing.Short() && ia != 3)) {
							continue
						}
						alpha, beta := T(ab[0]), T(ab[1])
						rowsA, colsA := sh.m, sh.k
						if tA == blas.Trans {
							rowsA, colsA = sh.k, sh.m
						}
						rowsB, colsB := sh.k, sh.n
						if tB == blas.Trans {
							rowsB, colsB = sh.n, sh.k
						}
						lda, ldb, ldc := max(1, rowsA)+pad, max(1, rowsB)+pad, max(1, sh.m)+pad
						a, b, c := matrix[T](rng, rowsA, colsA, lda), matrix[T](rng, rowsB, colsB, ldb), matrix[T](rng, sh.m, sh.n, ldc)
						a0, b0, c0 := clone(a), clone(b), clone(c)
						if beta == 0 && len(c) > 0 {
							c[0] = T(math.NaN()) // C must not be read
						}
						// The oracle reads C only where beta != 0.
						r := oracle.Gemm(byte(tA), byte(tB), sh.m, sh.n, sh.k, alpha, a, lda, b, ldb, beta, c0, ldc)
						if err := im.gemm(tA, tB, sh.m, sh.n, sh.k, alpha, a, lda, b, ldb, beta, c, ldc); err != nil {
							t.Fatalf("%v: %v", sh, err)
						}
						what := transName(tA) + transName(tB) + " " + sprint(sh, ab, pad)
						gp := oracle.Gamma[T](sh.k + 2)
						for j := 0; j < sh.n; j++ {
							for i := 0; i < sh.m; i++ {
								within(t, what, c[i+j*ldc], r, i+j*sh.m, gp, gp)
							}
						}
						padUntouched(t, what+" C", c, sh.m, ldc)
						if !same(a, a0) || !same(b, b0) {
							t.Fatalf("%s: an input was modified", what)
						}
					}
				}
			}
		}
	}
}

func sprint(sh shape, ab [2]float64, pad int) string {
	return "m,n,k=" + itoa(sh.m) + "," + itoa(sh.n) + "," + itoa(sh.k) + " alpha,beta=" + ftoa(ab[0]) + "," + ftoa(ab[1]) + " pad=" + itoa(pad)
}

func TestGemmBound(t *testing.T) {
	each(t, testGemmBound[float64], testGemmBound[float32])
}

// ConjTrans is Trans for real types, bit for bit.
func testGemmConjTrans[T float](t *testing.T, im impl[T]) {
	rng := newRNG(2)
	m, n, k := 9, 11, 13
	a, b := matrix[T](rng, k, m, k), matrix[T](rng, n, k, n)
	c1, c2 := matrix[T](rng, m, n, m), matrix[T](rng, m, n, m)
	copy(c2, c1)
	if err := im.gemm(blas.Trans, blas.Trans, m, n, k, 1.5, a, k, b, n, 0.5, c1, m); err != nil {
		t.Fatal(err)
	}
	if err := im.gemm(blas.ConjTrans, blas.ConjTrans, m, n, k, 1.5, a, k, b, n, 0.5, c2, m); err != nil {
		t.Fatal(err)
	}
	if !same(c1, c2) {
		t.Fatal("ConjTrans differs from Trans for a real type")
	}
}

func TestGemmConjTrans(t *testing.T) {
	each(t, testGemmConjTrans[float64], testGemmConjTrans[float32])
}

// beta = 0 does not read C, alpha = 0 does not read A and B, and the quick
// returns touch nothing.
func testGemmScalarRules[T float](t *testing.T, im impl[T]) {
	rng := newRNG(3)
	nan := T(math.NaN())
	m, n, k := 5, 4, 3
	a, b := matrix[T](rng, m, k, m), matrix[T](rng, k, n, k)

	// beta = 0 with C full of NaN, on the kernel path (alpha 1 and not 1).
	for _, alpha := range []T{1, 2} {
		c := make([]T, m*n)
		for i := range c {
			c[i] = nan
		}
		if err := im.gemm(blas.NoTrans, blas.NoTrans, m, n, k, alpha, a, m, b, k, 0, c, m); err != nil {
			t.Fatal(err)
		}
		for i, v := range c {
			if v != v {
				t.Fatalf("alpha=%v: beta=0 read C: element %d is NaN", alpha, i)
			}
		}
	}

	// alpha = 0: A and B may be empty; C := beta*C, or zero for beta = 0.
	c := matrix[T](rng, m, n, m)
	want := make([]T, len(c))
	for i := range c {
		want[i] = 2 * c[i]
	}
	if err := im.gemm(blas.NoTrans, blas.NoTrans, m, n, k, 0, nil, m, nil, k, 2, c, m); err != nil {
		t.Fatal(err)
	}
	if !same(c, want) {
		t.Fatal("alpha=0, beta=2: C != 2*C")
	}
	for i := range c {
		c[i] = nan
	}
	if err := im.gemm(blas.NoTrans, blas.NoTrans, m, n, k, 0, nil, m, nil, k, 0, c, m); err != nil {
		t.Fatal(err)
	}
	for _, v := range c {
		if v != 0 {
			t.Fatal("alpha=0, beta=0: C is not zero")
		}
	}

	// k = 0: C := beta*C, whatever the transposes.
	for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
		c = matrix[T](rng, m, n, m)
		for i := range want {
			want[i] = 3 * c[i]
		}
		if err := im.gemm(tA, blas.NoTrans, m, n, 0, 5, nil, max(1, 0)+m, nil, 1, 3, c, m); err != nil {
			t.Fatal(err)
		}
		if !same(c, want) {
			t.Fatalf("k=0 %s: C != beta*C", transName(tA))
		}
	}

	// Quick returns leave C alone, even when C is too short or NaN.
	c = []T{nan}
	for _, call := range []func() error{
		func() error { return im.gemm(blas.NoTrans, blas.NoTrans, 0, n, k, 1, a, 1, b, k, 0, c, 1) },
		func() error { return im.gemm(blas.NoTrans, blas.NoTrans, m, 0, k, 1, a, m, b, k, 0, c, m) },
		func() error { return im.gemm(blas.NoTrans, blas.NoTrans, m, n, k, 0, a, m, b, k, 1, c, m) },
		func() error { return im.gemm(blas.NoTrans, blas.NoTrans, m, n, 0, 1, a, m, b, 1, 1, c, m) },
	} {
		if err := call(); err != nil {
			t.Fatal(err)
		}
		if c[0] == c[0] {
			t.Fatal("a quick return wrote C")
		}
	}
}

func TestGemmScalarRules(t *testing.T) {
	each(t, testGemmScalarRules[float64], testGemmScalarRules[float32])
}

// Both implementations agree on the result of netlib's order when every
// product and sum is exact: integer data below 2^11 are exact in float32 and
// float64 for the sizes used.
func testGemmExactIntegers[T float](t *testing.T, im impl[T]) {
	rng := newRNG(4)
	for _, sh := range []shape{{7, 9, 11}, {64, 64, 64}, {90, 90, 20}, {5, 5, 300}} {
		a, b, c := make([]T, sh.m*sh.k), make([]T, sh.k*sh.n), make([]T, sh.m*sh.n)
		for i := range a {
			a[i] = T(rng.IntN(9) - 4)
		}
		for i := range b {
			b[i] = T(rng.IntN(9) - 4)
		}
		for i := range c {
			c[i] = T(rng.IntN(9) - 4)
		}
		want := make([]T, len(c))
		for j := 0; j < sh.n; j++ {
			for i := 0; i < sh.m; i++ {
				s := T(0)
				for l := 0; l < sh.k; l++ {
					s += a[i+l*sh.m] * b[l+j*sh.k]
				}
				want[i+j*sh.m] = 2*s + 3*c[i+j*sh.m]
			}
		}
		if err := im.gemm(blas.NoTrans, blas.NoTrans, sh.m, sh.n, sh.k, 2, a, sh.m, b, sh.k, 3, c, sh.m); err != nil {
			t.Fatal(err)
		}
		if !same(c, want) {
			t.Fatalf("%v: integer product differs from the exact one", sh)
		}
	}
}

func TestGemmExactIntegers(t *testing.T) {
	each(t, testGemmExactIntegers[float64], testGemmExactIntegers[float32])
}

// netlib's routines are the reference loops without fused multiply-add: on
// operands whose product needs more bits than the format has, an unfused sum
// and a fused one differ, and netlib must give the unfused one on every
// architecture (arm64's compiler fuses x*y+z by default).
func TestNetlibIsNotFused(t *testing.T) {
	t.Run("float64", func(t *testing.T) {
		x := 1 + math.Ldexp(1, -27) // x*x = 1 + 2^-26 + 2^-54, the last term lost when rounded
		c := []float64{-(1 + math.Ldexp(1, -26))}
		// C := 1*A*B + 1*C with A = [x], B = [x]: the unfused sum is 0, the fused 2^-54.
		if err := impls64[1].gemm(blas.NoTrans, blas.NoTrans, 1, 1, 1, 1, []float64{x}, 1, []float64{x}, 1, 1, c, 1); err != nil {
			t.Fatal(err)
		}
		if c[0] != 0 {
			t.Fatalf("netlib.Dgemm = %v, want the unfused 0", c[0])
		}
		y := []float64{-(1 + math.Ldexp(1, -26))}
		if err := impls64[1].axpy(1, x, []float64{x}, 1, y, 1); err != nil {
			t.Fatal(err)
		}
		if y[0] != 0 {
			t.Fatalf("netlib.Daxpy = %v, want the unfused 0", y[0])
		}
	})
	t.Run("float32", func(t *testing.T) {
		x := float32(1 + math.Ldexp(1, -13)) // x*x = 1 + 2^-12 + 2^-26
		c := []float32{-float32(1 + math.Ldexp(1, -12))}
		if err := impls32[1].gemm(blas.NoTrans, blas.NoTrans, 1, 1, 1, 1, []float32{x}, 1, []float32{x}, 1, 1, c, 1); err != nil {
			t.Fatal(err)
		}
		if c[0] != 0 {
			t.Fatalf("netlib.Sgemm = %v, want the unfused 0", c[0])
		}
	})
}

// The packed kernel is allowed to fuse and to reorder; where it does it must
// still be inside the bound, which the other tests check. This one pins that
// the shapes of the kernel's multi-worker path give the same bits as the
// single-worker path of the same call (the kernel's own contract).
func TestGemmSameAcrossWorkers(t *testing.T) {
	rng := newRNG(5)
	m, n, k := 130, 90, 70
	a, b := matrix[float64](rng, m, k, m), matrix[float64](rng, k, n, k)
	c1, c2 := make([]float64, m*n), make([]float64, m*n)
	run := func(c []float64) {
		if err := blas.Dgemm(blas.NoTrans, blas.NoTrans, m, n, k, 1, a, m, b, k, 0, c, m); err != nil {
			t.Fatal(err)
		}
	}
	run(c1)
	setProcs(t, 1)
	run(c2)
	if !same(c1, c2) {
		t.Fatal("the product depends on GOMAXPROCS")
	}
}

// LSAME ignores case: lowercase flags run exactly like uppercase ones, for gemm
// and gemv, and a letter that is no flag is still parameter 1 (or 2).
func testLowercaseFlags[T float](t *testing.T, im impl[T]) {
	rng := newRNG(6)
	m, n, k := 7, 6, 5
	a, b := matrix[T](rng, k, m, k), matrix[T](rng, n, k, n)
	x, y0 := vector[T](rng, m, 1), vector[T](rng, n, 1)
	c0 := matrix[T](rng, m, n, m)
	pairs := [][2]blas.Transpose{{'T', 'T'}, {'t', 't'}, {'C', 'c'}, {'t', 'C'}}
	for _, p := range pairs {
		up := [2]blas.Transpose{blas.Transpose(upper(byte(p[0]))), blas.Transpose(upper(byte(p[1])))}
		c1, c2 := clone(c0), clone(c0)
		if err := im.gemm(p[0], p[1], m, n, k, 1.5, a, k, b, n, 0.5, c1, m); err != nil {
			t.Fatalf("%c%c: %v", p[0], p[1], err)
		}
		if err := im.gemm(up[0], up[1], m, n, k, 1.5, a, k, b, n, 0.5, c2, m); err != nil {
			t.Fatal(err)
		}
		if !same(c1, c2) {
			t.Fatalf("gemm %c%c differs from %c%c", p[0], p[1], up[0], up[1])
		}
	}
	// NoTrans, lower case, on both operands.
	an, bn := matrix[T](rng, m, k, m), matrix[T](rng, k, n, k)
	c1, c2 := clone(c0), clone(c0)
	if err := im.gemm('n', 'n', m, n, k, 1, an, m, bn, k, 0, c1, m); err != nil {
		t.Fatal(err)
	}
	if err := im.gemm('N', 'N', m, n, k, 1, an, m, bn, k, 0, c2, m); err != nil || !same(c1, c2) {
		t.Fatalf("gemm nn differs from NN (%v)", err)
	}
	// gemv: 't' (x of length m, y of length n), then 'n'.
	for _, tr := range []blas.Transpose{'t', 'c'} {
		y1, y2 := clone(y0), clone(y0)
		if err := im.gemv(tr, m, n, 2, matrix[T](newRNG(9), m, n, m), m, x, 1, 1, y1, 1); err != nil {
			t.Fatal(err)
		}
		if err := im.gemv(blas.Transpose(upper(byte(tr))), m, n, 2, matrix[T](newRNG(9), m, n, m), m, x, 1, 1, y2, 1); err != nil || !same(y1, y2) {
			t.Fatalf("gemv %c differs from upper case (%v)", tr, err)
		}
	}
	if err := im.gemm('x', 'N', 1, 1, 1, 1, a, 1, a, 1, 1, c1, 1); err == nil {
		t.Fatal("an unknown flag was accepted")
	}
}

func upper(c byte) byte {
	if c >= 'a' && c <= 'z' {
		return c - 32
	}
	return c
}

func TestLowercaseFlags(t *testing.T) {
	each(t, testLowercaseFlags[float64], testLowercaseFlags[float32])
}
