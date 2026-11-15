package blas_test

import (
	"errors"
	"math"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

// stridedFn is DgemmStrided or SgemmStrided.
type stridedFn[T float] func(m, n, k int, alpha T, a []T, rsa, csa int, b []T, rsb, csb int, beta T, c []T, rsc, csc int) error

// eachStrided runs a test for both element types.
func eachStrided(t *testing.T, f64 func(t *testing.T, name string, g stridedFn[float64]), f32 func(t *testing.T, name string, g stridedFn[float32])) {
	t.Helper()
	t.Run("float64", func(t *testing.T) { defer noPanic(t); f64(t, "DGEMMSTRIDED", blas.DgemmStrided) })
	t.Run("float32", func(t *testing.T) { defer noPanic(t); f32(t, "SGEMMSTRIDED", blas.SgemmStrided) })
}

// layout returns the strides of a rows x cols matrix in one of four storages,
// each with pad unused elements between rows or columns: 0 row-major, 1
// column-major, 2 row-major of every other column (a[:, ::2]), 3 column-major of
// every other row.
func layout(kind, rows, cols, pad int) (rs, cs int) {
	switch kind {
	case 0:
		return cols + pad, 1
	case 1:
		return 1, rows + pad
	case 2:
		return 2*cols + pad, 2
	}
	return 2, 2*rows + pad
}

// stridedBuf returns the buffer of a strided matrix: its elements random, every
// other element the sentinel.
func stridedBuf[T float](rng interface{ Float64() float64 }, rows, cols, rs, cs int) []T {
	if rows == 0 || cols == 0 {
		return nil
	}
	v := make([]T, 1+(rows-1)*rs+(cols-1)*cs)
	for i := range v {
		v[i] = sentinel
	}
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			v[i*rs+j*cs] = T(rng.Float64()*4 - 2)
		}
	}
	return v
}

// dense copies a strided matrix to column-major, leading dimension rows.
func dense[T float](v []T, rows, cols, rs, cs int) []T {
	d := make([]T, rows*cols)
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			d[i+j*rows] = v[i*rs+j*cs]
		}
	}
	return d
}

// outsideUntouched asserts every element of v that is not an element of the
// rows x cols matrix (rs, cs) still holds the sentinel.
func outsideUntouched[T float](t *testing.T, what string, v []T, rows, cols, rs, cs int) {
	t.Helper()
	in := make([]bool, len(v))
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			in[i*rs+j*cs] = true
		}
	}
	for p := range v {
		if !in[p] && v[p] != sentinel {
			t.Fatalf("%s: element %d outside the matrix was written", what, p)
		}
	}
}

// The storage of A, B and C in each run of the bound test: every kind on every
// operand, C only row- or column-major.
var stridedLayouts = [][3]int{{0, 0, 0}, {1, 1, 1}, {2, 3, 0}, {3, 2, 1}, {0, 2, 1}, {1, 3, 0}, {2, 0, 1}, {3, 1, 0}}

func testStridedBound[T float](t *testing.T, name string, gemm stridedFn[T]) {
	rng := newRNG(21)
	for _, sh := range gemmShapes {
		if sh.m == 0 || sh.n == 0 || sh.k == 0 {
			continue
		}
		big := sh.m*sh.n*sh.k > 50000
		for _, lay := range stridedLayouts {
			for _, pad := range []int{0, 3} {
				if big && pad > 0 {
					continue
				}
				for ia, ab := range alphaBeta {
					if big && (ia >= 3 || (testing.Short() && ia != 3)) {
						continue
					}
					alpha, beta := T(ab[0]), T(ab[1])
					rsa, csa := layout(lay[0], sh.m, sh.k, pad)
					rsb, csb := layout(lay[1], sh.k, sh.n, pad)
					rsc, csc := layout(lay[2], sh.m, sh.n, pad)
					a, b, c := stridedBuf[T](rng, sh.m, sh.k, rsa, csa), stridedBuf[T](rng, sh.k, sh.n, rsb, csb), stridedBuf[T](rng, sh.m, sh.n, rsc, csc)
					a0, b0, c0 := clone(a), clone(b), clone(c)
					da, db, dc := dense(a, sh.m, sh.k, rsa, csa), dense(b, sh.k, sh.n, rsb, csb), dense(c, sh.m, sh.n, rsc, csc)
					if beta == 0 {
						c[0] = T(math.NaN()) // C must not be read
					}
					r := oracle.Gemm('N', 'N', sh.m, sh.n, sh.k, alpha, da, sh.m, db, sh.k, beta, dc, sh.m)
					if err := gemm(sh.m, sh.n, sh.k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc); err != nil {
						t.Fatalf("%v: %v", sh, err)
					}
					what := name + " " + sprint(sh, ab, pad) + " layouts " + itoa(lay[0]) + itoa(lay[1]) + itoa(lay[2])
					gp := oracle.Gamma[T](sh.k + 2)
					for j := 0; j < sh.n; j++ {
						for i := 0; i < sh.m; i++ {
							within(t, what, c[i*rsc+j*csc], r, i+j*sh.m, gp, gp)
						}
					}
					outsideUntouched(t, what+" C", c, sh.m, sh.n, rsc, csc)
					if !same(a, a0) || !same(b, b0) {
						t.Fatalf("%s: an input was modified", what)
					}
					_ = c0
				}
			}
		}
	}
}

func TestGemmStridedBound(t *testing.T) {
	eachStrided(t, testStridedBound[float64], testStridedBound[float32])
}

// The strided entry agrees, bit for bit, with the column-major one on the same
// data: it is the same kernel. Only float64 has a kernel on every target; the
// float32 loops without one (loong64) sum in another order, within the bound
// the bound test asserts.
func TestGemmStridedSameAsGemm(t *testing.T) {
	rng := newRNG(22)
	m, n, k := 37, 41, 43
	a, b := matrix[float64](rng, m, k, m), matrix[float64](rng, k, n, k)
	for _, ab := range [][2]float64{{1, 0}, {2.5, 1}} {
		want, got := matrix[float64](rng, m, n, m), make([]float64, m*n)
		copy(got, want)
		if err := blas.Dgemm(blas.NoTrans, blas.NoTrans, m, n, k, ab[0], a, m, b, k, ab[1], want, m); err != nil {
			t.Fatal(err)
		}
		if err := blas.DgemmStrided(m, n, k, ab[0], a, 1, m, b, 1, k, ab[1], got, 1, m); err != nil {
			t.Fatal(err)
		}
		if !same(want, got) {
			t.Fatalf("differs from Dgemm for alpha, beta = %v", ab)
		}
	}
}

func testStridedScalarRules[T float](t *testing.T, name string, gemm stridedFn[T]) {
	nan := T(math.NaN())
	full := func(n int, v T) []T {
		s := make([]T, n)
		for i := range s {
			s[i] = v
		}
		return s
	}
	// alpha = 0: A and B are not read, whatever they hold, or are empty.
	for _, beta := range []T{0, 0.5, 2} {
		c := full(6, 3)
		if err := gemm(2, 3, 4, 0, full(8, nan), 4, 1, full(12, nan), 3, 1, beta, c, 3, 1); err != nil {
			t.Fatal(err)
		}
		for i, v := range c {
			if v != beta*3 {
				t.Fatalf("%s: alpha = 0, beta = %v: c[%d] = %v, want %v", name, beta, i, v, beta*3)
			}
		}
		if err := gemm(2, 3, 4, 0, nil, 4, 1, nil, 3, 1, beta, c, 3, 1); err != nil {
			t.Fatalf("%s: alpha = 0 with empty A and B: %v", name, err)
		}
	}
	// beta = 0: C is not read.
	c := full(6, nan)
	if err := gemm(2, 3, 1, 1, full(2, 1), 1, 1, full(3, 1), 3, 1, 0, c, 3, 1); err != nil {
		t.Fatal(err)
	}
	for i, v := range c {
		if v != 1 {
			t.Fatalf("%s: beta = 0 read C: c[%d] = %v", name, i, v)
		}
	}
	// k = 0: C := beta*C, with no A or B.
	c = full(6, 4)
	if err := gemm(2, 3, 0, 1, nil, 1, 1, nil, 1, 1, 0.5, c, 3, 1); err != nil {
		t.Fatal(err)
	}
	for i, v := range c {
		if v != 2 {
			t.Fatalf("%s: k = 0: c[%d] = %v, want 2", name, i, v)
		}
	}
	// Quick returns touch nothing and check no slice: alpha = 0 or k = 0 with
	// beta = 1 and C empty, m = 0 or n = 0 with every slice empty.
	if err := gemm(2, 3, 4, 0, nil, 4, 1, nil, 3, 1, 1, nil, 3, 1); err != nil {
		t.Fatalf("%s: alpha = 0, beta = 1: %v", name, err)
	}
	if err := gemm(2, 3, 0, 1, nil, 1, 1, nil, 1, 1, 1, nil, 3, 1); err != nil {
		t.Fatalf("%s: k = 0, beta = 1: %v", name, err)
	}
	if err := gemm(0, 3, 4, 1, nil, 1, 1, nil, 3, 1, 0, nil, 3, 1); err != nil {
		t.Fatalf("%s: m = 0: %v", name, err)
	}
	if err := gemm(2, 0, 4, 1, nil, 4, 1, nil, 1, 1, 0, nil, 1, 1); err != nil {
		t.Fatalf("%s: n = 0: %v", name, err)
	}
}

func TestGemmStridedScalarRules(t *testing.T) {
	eachStrided(t, testStridedScalarRules[float64], testStridedScalarRules[float32])
}

// Each invalid argument is refused with its own parameter number, leaves C as
// it was, and the same call with the argument repaired succeeds.
func testStridedErrors[T float](t *testing.T, name string, gemm stridedFn[T]) {
	const maxInt = int(^uint(0) >> 1)
	type call struct {
		m, n, k                      int
		rsa, csa, rsb, csb, rsc, csc int
		la, lb, lc                   int // slice lengths
	}
	base := call{5, 4, 3, 3, 1, 4, 1, 4, 1, 15, 12, 20}
	cases := []struct {
		what  string
		param int
		edit  func(c *call)
	}{
		{"m < 0", 1, func(c *call) { c.m = -1 }},
		{"n < 0", 2, func(c *call) { c.n = -1 }},
		{"k < 0", 3, func(c *call) { c.k = -1 }},
		{"rsa = 0", 6, func(c *call) { c.rsa = 0 }},
		{"rsa < 0", 6, func(c *call) { c.rsa = -3 }},
		{"csa = 0", 7, func(c *call) { c.csa = 0 }},
		{"rsb = 0", 9, func(c *call) { c.rsb = 0 }},
		{"csb = 0", 10, func(c *call) { c.csb = 0 }},
		{"rsc below n", 13, func(c *call) { c.rsc = 3 }},
		{"rsc = 0", 13, func(c *call) { c.rsc = 0 }},
		{"neither stride of C is 1", 13, func(c *call) { c.rsc, c.csc, c.lc = 5, 2, 100 }},
		{"csc = 0", 14, func(c *call) { c.rsc, c.csc = 5, 0 }},
		{"column-major C, csc below m", 14, func(c *call) { c.rsc, c.csc = 1, 4 }},
		{"A one short", 5, func(c *call) { c.la-- }},
		{"B one short", 8, func(c *call) { c.lb-- }},
		{"C one short", 12, func(c *call) { c.lc-- }},
		{"A length overflows", 5, func(c *call) { c.rsa, c.csa = maxInt/2+1, maxInt/2+1 }},
		{"B length overflows", 8, func(c *call) { c.rsb, c.csb = maxInt/2+1, maxInt/2+1 }},
		{"C length overflows", 12, func(c *call) { c.rsc, c.csc = maxInt/2+1, 1 }},
	}
	for _, cs := range cases {
		c := base
		cs.edit(&c)
		a, b, out := make([]T, max(c.la, 0)), make([]T, max(c.lb, 0)), make([]T, max(c.lc, 0))
		for i := range out {
			out[i] = 7
		}
		err := gemm(c.m, c.n, c.k, 1, a, c.rsa, c.csa, b, c.rsb, c.csb, 1, out, c.rsc, c.csc)
		var ae *blas.ArgError
		if !errors.As(err, &ae) || ae.Param != cs.param || ae.Routine != name {
			t.Fatalf("%s: %s: got %v, want parameter %d", name, cs.what, err, cs.param)
		}
		for _, v := range out {
			if v != 7 {
				t.Fatalf("%s: %s: C was written", name, cs.what)
			}
		}
	}
	// The unedited call is valid, and so is a column-major C and a transposed A.
	a, b, out := make([]T, 15), make([]T, 12), make([]T, 20)
	if err := gemm(5, 4, 3, 1, a, 3, 1, b, 4, 1, 1, out, 4, 1); err != nil {
		t.Fatal(err)
	}
	if err := gemm(5, 4, 3, 1, a, 1, 5, b, 1, 3, 1, out, 1, 5); err != nil {
		t.Fatal(err)
	}
}

func TestGemmStridedErrors(t *testing.T) {
	eachStrided(t, testStridedErrors[float64], testStridedErrors[float32])
}

func TestGemmStridedSameAcrossWorkers(t *testing.T) {
	rng := newRNG(23)
	m, n, k := 130, 90, 70
	rsa, csa := layout(2, m, k, 1)
	rsb, csb := layout(3, k, n, 0)
	a, b := stridedBuf[float64](rng, m, k, rsa, csa), stridedBuf[float64](rng, k, n, rsb, csb)
	run := func(procs int) []float64 {
		setProcs(t, procs)
		c := make([]float64, m*n)
		if err := blas.DgemmStrided(m, n, k, 1.5, a, rsa, csa, b, rsb, csb, 0, c, n, 1); err != nil {
			t.Fatal(err)
		}
		return c
	}
	if one, many := run(1), run(4); !same(one, many) {
		t.Fatal("the product depends on the worker count")
	}
}
