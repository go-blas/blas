package ref

import (
	"math"
	"testing"
)

func TestLartgEdges(t *testing.T) {
	if c, s, r := Lartg(3.0, 0); c != 1 || s != 0 || r != 3 {
		t.Fatalf("g=0: %v %v %v", c, s, r)
	}
	if c, s, r := Lartg(0.0, -2); c != 0 || s != -1 || r != 2 {
		t.Fatalf("f=0: %v %v %v", c, s, r)
	}
	for _, fg := range [][2]float64{{3, 4}, {-3, 4}, {1e300, 1e300}, {1e-300, 3e-300}, {5e-324, 1}, {1e300, 1e-300}} {
		c, s, r := Lartg(fg[0], fg[1])
		if math.Abs(c*c+s*s-1) > 1e-15 {
			t.Fatalf("%v: c^2+s^2 = %v", fg, c*c+s*s)
		}
		if math.Abs(c*fg[0]+s*fg[1]-r) > 1e-14*math.Abs(r) || math.Abs(-s*fg[0]+c*fg[1]) > 1e-14*math.Abs(r) {
			t.Fatalf("%v: rotation does not map (f,g) to (r,0): %v %v %v", fg, c, s, r)
		}
	}
}

func TestLasclEdges(t *testing.T) {
	Lascl('G', 1, 2, 0, 3, []float64{}, 1)    // m = 0: nothing, not even a read
	Lascl('G', 1, 2, 3, 0, []float64(nil), 1) // n = 0
	a := []float64{1.5, -2}
	Lascl('G', 2, 2, 2, 1, a, 2) // factor 1: unchanged
	if a[0] != 1.5 || a[1] != -2 {
		t.Fatalf("factor 1 changed a: %v", a)
	}
	for _, c := range []struct{ from, to, x float64 }{
		{1e300, 1e-10, 1e300},   // cfrom * smlnum > cto: two steps down
		{1e-300, 1e300, 1e-300}, // cto / bignum > cfrom: two steps up
		{1, 0, 7},               // cto = 0: the factor is cto itself
		{1, math.Inf(1), 7},
		{math.Inf(1), 1, 7}, // cfrom infinite
		{3, 5, 1.25},
	} {
		a := []float64{c.x}
		Lascl('G', c.from, c.to, 1, 1, a, 1)
		want := c.x * (c.to / c.from)
		switch {
		case c.from == 1e300:
			want = 1e300 * 1e-10 / 1e300
		case c.from == 1e-300:
			want = 1e300
		}
		if !(a[0] == want || math.Abs(a[0]-want) <= 4e-16*math.Abs(want)) {
			t.Fatalf("Lascl(%v, %v) of %v = %v, want %v", c.from, c.to, c.x, a[0], want)
		}
	}
	// 'L' and 'U' touch one triangle.
	m := []float64{1, 2, 3, 4}
	Lascl('L', 1, 2, 2, 2, m, 2)
	if m[0] != 2 || m[1] != 4 || m[2] != 3 || m[3] != 8 {
		t.Fatalf("L: %v", m)
	}
	m = []float64{1, 2, 3, 4}
	Lascl('U', 1, 2, 2, 2, m, 2)
	if m[0] != 2 || m[1] != 2 || m[2] != 6 || m[3] != 8 {
		t.Fatalf("U: %v", m)
	}
}

func TestLaev2Edges(t *testing.T) {
	// [1 0; 0 1]: a multiple of the identity, any eigenvector will do; the
	// reference (ab = 0, sgn1 = sgn2) ends with cs1 = -0, sn1 = 1.
	rt1, rt2, cs1, sn1 := Laev2(1.0, 0, 1)
	if rt1 != 1 || rt2 != 1 || cs1 != 0 || sn1 != 1 {
		t.Fatalf("%v %v %v %v", rt1, rt2, cs1, sn1)
	}
	if rt1, rt2 := Lae2(0.0, 0, 0); rt1 != 0 || rt2 != 0 {
		t.Fatalf("zero matrix: %v %v", rt1, rt2)
	}
}

func TestLasrtEdges(t *testing.T) {
	Lasrt(0, []float64(nil))
	d := []float64{2}
	Lasrt(1, d)
	if d[0] != 2 {
		t.Fatalf("n=1 changed d: %v", d)
	}
	d = []float64{3, 1, 2}
	Lasrt(3, d)
	if d[0] != 1 || d[1] != 2 || d[2] != 3 {
		t.Fatalf("%v", d)
	}
}
