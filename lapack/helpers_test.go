package lapack_test

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
	"github.com/go-blas/blas/lapack"
	"github.com/go-blas/blas/netlib"
)

type float interface{ float32 | float64 }

// impl is one implementation of the routines of one element type: package
// lapack, or its exact option netlib. Every test runs on both.
type impl[T float] struct {
	name  string
	getrf func(m, n int, a []T, lda int, ipiv []int) (int, error)
	getrs func(trans blas.Transpose, n, nrhs int, a []T, lda int, ipiv []int, b []T, ldb int) error
	potrf func(uplo blas.Uplo, n int, a []T, lda int) (int, error)
	potrs func(uplo blas.Uplo, n, nrhs int, a []T, lda int, b []T, ldb int) error
}

var impls64 = []impl[float64]{
	{"lapack", lapack.Dgetrf, lapack.Dgetrs, lapack.Dpotrf, lapack.Dpotrs},
	{"netlib", netlib.Dgetrf, netlib.Dgetrs, netlib.Dpotrf, netlib.Dpotrs},
}

var impls32 = []impl[float32]{
	{"lapack", lapack.Sgetrf, lapack.Sgetrs, lapack.Spotrf, lapack.Spotrs},
	{"netlib", netlib.Sgetrf, netlib.Sgetrs, netlib.Spotrf, netlib.Spotrs},
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

// leaves are the panel widths the lapack implementation is run with: 1 and 2
// make every cut of the recursion happen, 3 an odd one, the default none below
// 16. netlib has no panel width, and runs once (0).
func leaves(name string) []int {
	if name == "netlib" {
		return []int{0}
	}
	return []int{1, 2, 3, lapack.Leaf()}
}

func setLeaf(t *testing.T, l int) {
	t.Helper()
	if l > 0 {
		lapack.SetLeaf(t, l)
	}
}

func noPanic(t *testing.T) {
	t.Helper()
	if r := recover(); r != nil {
		t.Fatalf("panicked: %v", r)
	}
}

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)) }

func upper(c byte) byte {
	if c >= 'a' && c <= 'z' {
		return c - 32
	}
	return c
}

const sentinel = 12345.5

func nan[T float]() T { return T(math.NaN()) }

func clone[T float](v []T) []T { return append([]T(nil), v...) }

// same compares two slices bit for bit, NaN equal to NaN.
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

func matLen(rows, cols, ld int) int {
	if rows == 0 || cols == 0 {
		return 0
	}
	return ld*(cols-1) + rows
}

// general returns a column-major rows x cols matrix, leading dimension ld, of
// mixed signs and magnitudes (oracle.Fill), the padding holding the sentinel.
func general[T float](rng *rand.Rand, rows, cols, ld int) []T {
	v := make([]T, matLen(rows, cols, ld))
	for p := range v {
		v[p] = sentinel
	}
	for j := 0; j < cols && rows > 0; j++ {
		copy(v[j*ld:j*ld+rows], oracle.Fill[T](rng, rows))
	}
	return v
}

// spd returns a symmetric positive definite (diagonally dominant) n x n matrix:
// the triangle named holds the matrix, the other triangle NaN (it must not be
// read), the padding the sentinel.
func spd[T float](rng *rand.Rand, n, ld int, up bool) []T {
	v := make([]T, matLen(n, n, ld))
	for p := range v {
		v[p] = sentinel
	}
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			switch {
			case i == j:
				v[i+j*ld] = T(float64(n) + 1 + rng.Float64())
			case (i < j) == up:
				v[i+j*ld] = T(2*rng.Float64() - 1)
			default:
				v[i+j*ld] = nan[T]()
			}
		}
	}
	return v
}

func padUntouched[T float](t *testing.T, what string, v []T, rows, ld int) {
	t.Helper()
	for p := range v {
		if p%ld >= rows && v[p] != sentinel {
			t.Fatalf("%s: padding element %d was written", what, p)
		}
	}
}

// ---- exact arithmetic ----

func bf() *big.Float { return new(big.Float).SetPrec(oracle.Prec) }

type bm struct {
	r, c int
	v    []*big.Float
}

func newBM(r, c int) bm {
	m := bm{r, c, make([]*big.Float, r*c)}
	for i := range m.v {
		m.v[i] = bf()
	}
	return m
}

func (m bm) at(i, j int) *big.Float { return m.v[i+j*m.r] }

func toBM[T float](a []T, rows, cols, ld int) bm {
	m := newBM(rows, cols)
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			m.at(i, j).Set(oracle.B(a[i+j*ld]))
		}
	}
	return m
}

func (m bm) abs() bm {
	o := newBM(m.r, m.c)
	for i, x := range m.v {
		o.v[i].Abs(x)
	}
	return o
}

func (m bm) T() bm {
	o := newBM(m.c, m.r)
	for j := 0; j < m.c; j++ {
		for i := 0; i < m.r; i++ {
			o.at(j, i).Set(m.at(i, j))
		}
	}
	return o
}

func mul(x, y bm) bm {
	o := newBM(x.r, y.c)
	t := bf()
	for j := 0; j < y.c; j++ {
		for k := 0; k < x.c; k++ {
			if y.at(k, j).Sign() == 0 {
				continue
			}
			for i := 0; i < x.r; i++ {
				o.at(i, j).Add(o.at(i, j), t.Mul(x.at(i, k), y.at(k, j)))
			}
		}
	}
	return o
}

func sub(x, y bm) bm {
	o := newBM(x.r, x.c)
	for i := range o.v {
		o.v[i].Sub(x.v[i], y.v[i])
	}
	return o
}

// swapRows applies the interchanges ipiv[0:k] (1-based) to the rows of m in
// order, or, with forward false, in reverse.
func swapRows(m bm, ipiv []int, k int, forward bool) bm {
	o := newBM(m.r, m.c)
	copy(o.v, m.v)
	step := func(i int) {
		if p := ipiv[i] - 1; p != i {
			for j := 0; j < m.c; j++ {
				o.v[i+j*m.r], o.v[p+j*m.r] = o.v[p+j*m.r], o.v[i+j*m.r]
			}
		}
	}
	for i := 0; i < k; i++ {
		if forward {
			step(i)
		} else {
			step(k - 1 - i)
		}
	}
	return o
}

// bound asserts |res| <= g*scale elementwise, on the entries keep accepts.
func bound(t *testing.T, what string, res, scale bm, g *big.Float, keep func(i, j int) bool) {
	t.Helper()
	for j := 0; j < res.c; j++ {
		for i := 0; i < res.r; i++ {
			if keep != nil && !keep(i, j) {
				continue
			}
			lim := bf().Mul(g, scale.at(i, j))
			if bf().Abs(res.at(i, j)).Cmp(lim) > 0 {
				d, _ := res.at(i, j).Float64()
				l, _ := lim.Float64()
				t.Fatalf("%s(%d,%d): |residual| = %g exceeds the bound %g", what, i, j, math.Abs(d), l)
			}
		}
	}
}

// luFactors splits the factors in a (m x n) into the unit lower trapezoidal L
// (m x r) and the upper trapezoidal U (r x n), r = min(m, n).
func luFactors[T float](a []T, m, n, lda int) (l, u bm) {
	r := min(m, n)
	l, u = newBM(m, r), newBM(r, n)
	for k := 0; k < r; k++ {
		for i := k; i < m; i++ {
			if i == k {
				l.at(i, k).SetInt64(1)
			} else {
				l.at(i, k).Set(oracle.B(a[i+k*lda]))
			}
		}
		for j := k; j < n; j++ {
			u.at(k, j).Set(oracle.B(a[k+j*lda]))
		}
	}
	return l, u
}

func checkPivots(t *testing.T, ipiv []int, m, n int) {
	t.Helper()
	for i := 0; i < min(m, n); i++ {
		if ipiv[i] < i+1 || ipiv[i] > m {
			t.Fatalf("ipiv[%d] = %d, outside %d..%d", i, ipiv[i], i+1, m)
		}
	}
}

// getrfBound asserts |P A - L U| <= gamma_{3r+2} |L||U|.
func getrfBound[T float](t *testing.T, what string, a0, a []T, ipiv []int, m, n, lda int) {
	t.Helper()
	checkPivots(t, ipiv, m, n)
	r := min(m, n)
	l, u := luFactors(a, m, n, lda)
	pa := swapRows(toBM(a0, m, n, lda), ipiv, r, true)
	bound(t, what+": P A - L U", sub(pa, mul(l, u)), mul(l.abs(), u.abs()), oracle.Gamma[T](3*r+2), nil)
}

// triFactor returns the triangle of the leading n x n block of a as dense.
func triFactor[T float](a []T, n, lda int, up bool) bm {
	c := newBM(n, n)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			if (i <= j) == up || i == j {
				c.at(i, j).Set(oracle.B(a[i+j*lda]))
			}
		}
	}
	return c
}

// potrfBound asserts |A - C^T C| <= gamma_{3n+3} |C^T||C| (Upper) or
// |A - C C^T| (Lower) on the referenced triangle of the leading block of order k.
func potrfBound[T float](t *testing.T, what string, a0, a []T, k, lda int, up bool) {
	t.Helper()
	if k == 0 {
		return
	}
	c := triFactor(a, k, lda, up)
	n := triFactor(a0, k, lda, up) // the other triangle of A is NaN and is not read
	var prod, scale bm
	if up {
		prod, scale = mul(c.T(), c), mul(c.T().abs(), c.abs())
	} else {
		prod, scale = mul(c, c.T()), mul(c.abs(), c.T().abs())
	}
	bound(t, what+": A - C'C", sub(n, prod), scale, oracle.Gamma[T](3*k+3), func(i, j int) bool { return (i <= j) == up || i == j })
}

// getrsBound asserts the residual bound of Getrs for the factors in a.
func getrsBound[T float](t *testing.T, what string, trans byte, a []T, ipiv []int, n, nrhs, lda int, b0, x []T, ldb int) {
	t.Helper()
	l, u := luFactors(a, n, n, lda)
	b, xb := toBM(b0, n, nrhs, ldb), toBM(x, n, nrhs, ldb)
	g := oracle.Gamma[T](2*n + 8)
	if trans == 'N' {
		pb := swapRows(b, ipiv, n, true)
		bound(t, what+": P B - L U X", sub(pb, mul(l, mul(u, xb))), mul(l.abs(), mul(u.abs(), xb.abs())), g, nil)
		return
	}
	// P is the operator of the forward interchanges (P X: the interchanges of ipiv
	// applied in order to the rows of X, swapRows with forward true; DLASWP with
	// incx = 1), so that P A = L U, and A^T = U^T L^T P (LAPACK's permutation
	// matrix is P^T, A = P^T L U). Then A^T x = b is U^T L^T (P x) = b: with
	// w = P x, U^T L^T w = b, and getrs recovers x = P^-1 w by applying the
	// interchanges in reverse. The code below forms z = P x.
	z := swapRows(xb, ipiv, n, true)
	bound(t, what+": B - U^T L^T P X", sub(b, mul(u.T(), mul(l.T(), z))), mul(u.T().abs(), mul(l.T().abs(), z.abs())), g, nil)
}

// potrsBound asserts the residual bound of Potrs for the factor in a.
func potrsBound[T float](t *testing.T, what string, a []T, n, nrhs, lda int, b0, x []T, ldb int, up bool) {
	t.Helper()
	c := triFactor(a, n, lda, up)
	b, xb := toBM(b0, n, nrhs, ldb), toBM(x, n, nrhs, ldb)
	g := oracle.Gamma[T](2*n + 8)
	if up {
		bound(t, what+": B - C^T C X", sub(b, mul(c.T(), mul(c, xb))), mul(c.T().abs(), mul(c.abs(), xb.abs())), g, nil)
		return
	}
	bound(t, what+": B - C C^T X", sub(b, mul(c, mul(c.T(), xb))), mul(c.abs(), mul(c.T().abs(), xb.abs())), g, nil)
}
