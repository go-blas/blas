package ref

import "testing"

// failKern is a Kern none of whose functions may be called: the methods around
// them carry the quick returns of the reference BLAS.
func failKern(t *testing.T) *Kern[float64] {
	t.Helper()
	return &Kern[float64]{
		Gemm: func(bool, bool, int, int, int, float64, []float64, int, []float64, int, float64, []float64, int) {
			t.Fatal("Gemm called")
		},
		Trmm: func(bool, bool, bool, int, int, float64, []float64, int, []float64, int) { t.Fatal("Trmm called") },
		Gemv: func(bool, int, int, float64, []float64, int, []float64, int, float64, []float64, int) {
			t.Fatal("Gemv called")
		},
		Ger: func(int, int, float64, []float64, []float64, []float64, int) { t.Fatal("Ger called") },
	}
}

func TestKernQuickReturns(t *testing.T) {
	k := failKern(t)
	var s []float64
	k.gemm(true, true, 0, 3, 3, 1, s, 1, s, 1, 0, s, 1) // m = 0
	k.gemm(true, true, 3, 0, 3, 1, s, 1, s, 1, 0, s, 1) // n = 0
	k.gemm(true, true, 3, 3, 0, 1, s, 1, s, 1, 1, s, 1) // k = 0, beta = 1
	k.gemm(true, true, 3, 3, 3, 0, s, 1, s, 1, 1, s, 1) // alpha = 0, beta = 1
	k.trmm(true, true, true, 0, 3, 1, s, 1, s, 1)       // m = 0
	k.trmm(true, true, true, 3, 0, 1, s, 1, s, 1)       // n = 0
	k.gemv(true, 0, 3, 1, s, 1, s, 0, s)                // m = 0
	k.gemv(true, 3, 0, 1, s, 1, s, 0, s)                // n = 0
	k.gemv(true, 3, 3, 0, s, 1, s, 1, s)                // alpha = 0, beta = 1
	k.ger(0, 3, 1, s, s, s, 1)                          // m = 0
	k.ger(3, 0, 1, s, s, s, 1)                          // n = 0
	k.ger(3, 3, 0, s, s, s, 1)                          // alpha = 0
}

func TestIladlcIladlr(t *testing.T) {
	zero := make([]float64, 12) // 4 x 3
	if g := Iladlc(4, 3, zero, 4); g != 0 {
		t.Fatalf("Iladlc of zero = %d", g)
	}
	if g := Iladlr(4, 3, zero, 4); g != 0 {
		t.Fatalf("Iladlr of zero = %d", g)
	}
	if g := Iladlc[float64](4, 0, nil, 4); g != 0 {
		t.Fatalf("Iladlc with n = 0: %d", g)
	}
	if g := Iladlr[float64](0, 3, nil, 1); g != 0 {
		t.Fatalf("Iladlr with m = 0: %d", g)
	}
	one := make([]float64, 12)
	one[1+1*4] = 5 // element (2, 2) only
	if g := Iladlc(4, 3, one, 4); g != 2 {
		t.Fatalf("Iladlc = %d, want 2", g)
	}
	if g := Iladlr(4, 3, one, 4); g != 2 {
		t.Fatalf("Iladlr = %d, want 2", g)
	}
	one[3+2*4] = 1 // the corner
	if g := Iladlc(4, 3, one, 4); g != 3 {
		t.Fatalf("Iladlc = %d, want 3", g)
	}
	if g := Iladlr(4, 3, one, 4); g != 4 {
		t.Fatalf("Iladlr = %d, want 4", g)
	}
}

// A reflector whose v is all zero is the identity whatever tau: Larf must leave
// c alone (and not scan an empty range); Larfb with an empty c does nothing.
func TestLarfZeroV(t *testing.T) {
	k := RefKern[float64]()
	c := []float64{1, 2, 3, 4, 5, 6}
	for _, left := range []bool{true, false} {
		m, n := 3, 2
		work := make([]float64, 3)
		Larf(k, left, m, n, make([]float64, 3), 1, c, 3, work)
		for i, v := range c {
			if v != float64(i+1) {
				t.Fatalf("left %v: c[%d] = %g", left, i, v)
			}
		}
	}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Larfb of an empty c touched its arguments: %v", r)
		}
	}()
	Larfb(k, true, true, 0, 3, 1, nil, 1, nil, 1, nil, 1, nil, 1)
	Larfb(k, true, true, 3, 0, 1, nil, 1, nil, 1, nil, 1, nil, 1)
}

// The machine constants of dlarfg.f and the overflow threshold, per precision.
func TestPrecisionConstants(t *testing.T) {
	if got := larfgSafmin[float64](); got != 0x1p-969 {
		t.Fatalf("float64 safmin = %g", got)
	}
	if got := larfgSafmin[float32](); got != 0x1p-102 {
		t.Fatalf("float32 safmin = %g", got)
	}
	if got := huge[float64](); got != 0x1.fffffffffffffp1023 {
		t.Fatalf("float64 huge = %g", got)
	}
	if got := huge[float32](); got != 0x1.fffffep127 {
		t.Fatalf("float32 huge = %g", got)
	}
}

// Ormqr uses the unblocked code when nb < 2 or nb >= k, and the blocked code
// (Trmm of Larfb) otherwise; both give the same C up to rounding.
func TestOrmqrPathChoice(t *testing.T) {
	const m, k = 5, 4
	a := make([]float64, m*k)
	for i := range a {
		a[i] = 0.3 + 0.1*float64(i%7)
	}
	tau := []float64{1.1, 0.9, 1.3, 0.7}
	for _, c := range []struct {
		nb      int
		blocked bool
	}{{1, false}, {2, true}, {3, true}, {4, false}, {9, false}} {
		trmm := 0
		kn := RefKern[float64]()
		inner := kn.Trmm
		kn.Trmm = func(u, na, nu bool, m, n int, al float64, a []float64, lda int, b []float64, ldb int) {
			trmm++
			inner(u, na, nu, m, n, al, a, lda, b, ldb)
		}
		cm := make([]float64, m*3)
		for i := range cm {
			cm[i] = float64(i + 1)
		}
		Ormqr(kn, c.nb, true, true, m, 3, k, a, m, tau, cm, m)
		if (trmm > 0) != c.blocked {
			t.Fatalf("nb = %d: %d Trmm calls, blocked = %v", c.nb, trmm, c.blocked)
		}
	}
}
