package lapack_test

import (
	"math"
	"math/big"
	"strconv"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
	"github.com/go-blas/blas/lapack"
	"github.com/go-blas/blas/netlib"
)

// qimpl is one implementation of the QR routines of one element type.
type qimpl[T float] struct {
	name  string
	geqrf func(m, n int, a []T, lda int, tau []T) error
	ormqr func(side blas.Side, trans blas.Transpose, m, n, k int, a []T, lda int, tau []T, c []T, ldc int) error
	orgqr func(m, n, k int, a []T, lda int, tau []T) error
}

var qimpls64 = []qimpl[float64]{
	{"lapack", lapack.Dgeqrf, lapack.Dormqr, lapack.Dorgqr},
	{"netlib", netlib.Dgeqrf, netlib.Dormqr, netlib.Dorgqr},
}

var qimpls32 = []qimpl[float32]{
	{"lapack", lapack.Sgeqrf, lapack.Sormqr, lapack.Sorgqr},
	{"netlib", netlib.Sgeqrf, netlib.Sormqr, netlib.Sorgqr},
}

// qrConfig is a block size, a crossover and an Ormqr block size. The small ones
// make matrices of a few dozen columns span many panels, with and without the
// unblocked tail; the default is the reference's.
type qrConfig struct{ nb, nx, onb int }

func qrConfigs(name string) []qrConfig {
	if name == "netlib" {
		return []qrConfig{{}} // netlib has the reference's block sizes only
	}
	return []qrConfig{{2, 0, 2}, {3, 2, 3}, {4, 5, 4}, {8, 0, 7}, {}}
}

func setQR(t *testing.T, c qrConfig) {
	t.Helper()
	if c != (qrConfig{}) {
		lapack.SetQR(t, c.nb, c.nx, c.onb)
	}
}

// eachQ runs a test for both element types, both implementations and every
// block configuration.
func eachQ(t *testing.T, f64 func(t *testing.T, im qimpl[float64]), f32 func(t *testing.T, im qimpl[float32])) {
	t.Helper()
	for _, im := range qimpls64 {
		for _, c := range qrConfigs(im.name) {
			t.Run("float64/"+im.name+cfgName(c), func(t *testing.T) { defer noPanic(t); setQR(t, c); f64(t, im) })
		}
	}
	for _, im := range qimpls32 {
		for _, c := range qrConfigs(im.name) {
			t.Run("float32/"+im.name+cfgName(c), func(t *testing.T) { defer noPanic(t); setQR(t, c); f32(t, im) })
		}
	}
}

func cfgName(c qrConfig) string {
	if c == (qrConfig{}) {
		return ""
	}
	return "/nb" + itoa(c.nb) + "nx" + itoa(c.nx) + "onb" + itoa(c.onb)
}

func itoa(n int) string { return strconv.Itoa(n) }

// ---- exact arithmetic for QR ----

// exactQ is the product H(1) ... H(k) of the reflectors H(i) = I - tau(i) v v**T,
// v(1:i-1) = 0, v(i) = 1, v(i+1:nq) = a(i+1:nq, i), as an nq x nq matrix in
// exact arithmetic: computed from the stored reflectors, not from the
// algorithm under test.
func exactQ[T float](a []T, nq, k, lda int, tau []T) bm {
	q := newBM(nq, nq)
	for i := 0; i < nq; i++ {
		q.at(i, i).SetInt64(1)
	}
	t, w := bf(), bf()
	for i := k - 1; i >= 0; i-- {
		v := make([]*big.Float, nq)
		for r := range v {
			switch {
			case r < i:
				v[r] = bf()
			case r == i:
				v[r] = bf().SetInt64(1)
			default:
				v[r] = oracle.B(a[r+i*lda])
			}
		}
		tb := oracle.B(tau[i])
		for j := 0; j < nq; j++ {
			w.SetInt64(0)
			for r := i; r < nq; r++ {
				w.Add(w, t.Mul(v[r], q.at(r, j)))
			}
			w.Mul(w, tb)
			for r := i; r < nq; r++ {
				q.at(r, j).Sub(q.at(r, j), t.Mul(v[r], w))
			}
		}
	}
	return q
}

func sqrt(x *big.Float) *big.Float { return bf().Sqrt(x) }

// colNorm is the 2-norm of column j.
func colNorm(m bm, j int) *big.Float {
	s := bf()
	t := bf()
	for i := 0; i < m.r; i++ {
		s.Add(s, t.Mul(m.at(i, j), m.at(i, j)))
	}
	return sqrt(s)
}

// fro is the Frobenius norm.
func fro(m bm) *big.Float {
	s := bf()
	t := bf()
	for _, x := range m.v {
		s.Add(s, t.Mul(x, x))
	}
	return sqrt(s)
}

// colBound asserts ||res(:, j)||_2 <= g ||scale(:, j)||_2 for every column.
func colBound(t *testing.T, what string, res, scale bm, g *big.Float) {
	t.Helper()
	for j := 0; j < res.c; j++ {
		if r, l := colNorm(res, j), bf().Mul(g, colNorm(scale, j)); r.Cmp(l) > 0 {
			rf, _ := r.Float64()
			lf, _ := l.Float64()
			t.Fatalf("%s: column %d, ||residual|| = %g exceeds the bound %g", what, j, rf, lf)
		}
	}
}

func froBound(t *testing.T, what string, res bm, scale, g *big.Float) {
	t.Helper()
	if r, l := fro(res), bf().Mul(g, scale); r.Cmp(l) > 0 {
		rf, _ := r.Float64()
		lf, _ := l.Float64()
		t.Fatalf("%s: ||residual||_F = %g exceeds the bound %g", what, rf, lf)
	}
}

// orthBound asserts ||Q**T Q - I||_F <= g for the m x n matrix q.
func orthBound(t *testing.T, what string, q bm, g *big.Float) {
	t.Helper()
	p := mul(q.T(), q)
	for i := 0; i < p.r; i++ {
		p.at(i, i).Sub(p.at(i, i), bf().SetInt64(1))
	}
	froBound(t, what+": Q'Q - I", p, bf().SetInt64(1), g)
}

// qrGamma is the constant of the bounds, gamma_{8(m+n)+16}. LAWN 22 (Demmel and
// Higham, Stability of Block Algorithms with Fast Level 3 BLAS), section 3,
// states (30) that a WY update is an exact orthogonal update of a perturbation
// of norm c n (c1(r,n,n) + c1(n,r,n)) ||B|| + O(u^2), and concludes that the
// standard backward error results of Householder algorithms (Wilkinson, cited
// there) remain valid for the blocked ones with the constants changed: a
// normwise, first-order, per-column form with a constant that grows with the
// dimensions. The constant asserted here is a tested one, linear in m + n, not a
// proved one.
func qrGamma[T float](m, n int) *big.Float { return oracle.Gamma[T](8*(m+n) + 16) }

var qrShapes = [][2]int{
	{1, 1}, {1, 5}, {5, 1}, {2, 2}, {3, 3}, {7, 4}, {4, 7}, {12, 12}, {20, 9}, {9, 20},
	{33, 17}, {17, 33}, {40, 40}, {65, 30},
}

// rFactor returns R, the upper trapezoid of a, as the m x n matrix.
func rFactor[T float](a []T, m, n, lda int) bm {
	r := newBM(m, n)
	for j := 0; j < n; j++ {
		for i := 0; i <= j && i < m; i++ {
			r.at(i, j).Set(oracle.B(a[i+j*lda]))
		}
	}
	return r
}

func TestGeqrfBound(t *testing.T) { eachQ(t, geqrfBoundT[float64], geqrfBoundT[float32]) }

func geqrfBoundT[T float](t *testing.T, im qimpl[T]) {
	for si, sh := range qrShapes {
		m, n := sh[0], sh[1]
		lda := m + si%3
		a0 := general[T](newRNG(uint64(100+si)), m, n, lda)
		a, tau := clone(a0), make([]T, min(m, n))
		for i := range tau {
			tau[i] = sentinel
		}
		if err := im.geqrf(m, n, a, lda, tau); err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		padUntouched(t, "geqrf", a, m, lda)
		q := exactQ(a, m, min(m, n), lda, tau)
		g := qrGamma[T](m, n)
		b0 := toBM(a0, m, n, lda)
		colBound(t, "A - Q R", sub(b0, mul(q, rFactor(a, m, n, lda))), b0, g)
		orthBound(t, "Q", q, g)
	}
}

func TestOrgqrBound(t *testing.T) { eachQ(t, orgqrBoundT[float64], orgqrBoundT[float32]) }

func orgqrBoundT[T float](t *testing.T, im qimpl[T]) {
	for si, sh := range qrShapes {
		m, n := sh[0], sh[1]
		if m < n {
			m, n = n, m
		}
		for _, k := range []int{0, 1, n / 2, n} {
			if k > n || (k == 0 && si%2 == 1) {
				continue
			}
			lda := m + si%3
			// The reflectors are those of a factorization of the first k columns; the
			// columns k..n-1 of the argument are NaN: they are not referenced.
			f := general[T](newRNG(uint64(200+si)), m, k, lda)
			tau := make([]T, k)
			if k > 0 {
				if err := im.geqrf(m, k, f, lda, tau); err != nil {
					t.Fatal(err)
				}
			}
			a := general[T](newRNG(1), m, n, lda)
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					if j < k {
						a[i+j*lda] = f[i+j*lda]
					} else {
						a[i+j*lda] = nan[T]()
					}
				}
			}
			if err := im.orgqr(m, n, k, a, lda, tau); err != nil {
				t.Fatal(err)
			}
			padUntouched(t, "orgqr", a, m, lda)
			ex := exactQ(f, m, k, lda, tau)
			got := toBM(a, m, n, lda)
			want := newBM(m, n)
			copy(want.v, ex.v[:m*n])
			g := qrGamma[T](m, n)
			colBound(t, "Q - Q exact", sub(got, want), want, g)
			orthBound(t, "Q", got, g)
		}
	}
}

func TestOrmqrBound(t *testing.T) { eachQ(t, ormqrBoundT[float64], ormqrBoundT[float32]) }

func ormqrBoundT[T float](t *testing.T, im qimpl[T]) {
	cases := [][3]int{{1, 1, 1}, {5, 3, 3}, {3, 5, 2}, {9, 9, 9}, {20, 7, 5}, {7, 20, 7}, {33, 40, 17}, {40, 33, 31}}
	for ci, cs := range cases {
		for _, sd := range []blas.Side{blas.Left, blas.Right} {
			for _, tr := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				m, n, k := cs[0], cs[1], cs[2]
				nq := m
				if sd == blas.Right {
					nq = n
				}
				k = min(k, nq)
				lda, ldc := nq+ci%3, m+ci%2
				a := general[T](newRNG(uint64(300+ci)), nq, k, lda)
				tau := make([]T, k)
				if err := im.geqrf(nq, k, a, lda, tau); err != nil {
					t.Fatal(err)
				}
				a0 := clone(a)
				c0 := general[T](newRNG(uint64(400+ci)), m, n, ldc)
				c := clone(c0)
				if err := im.ormqr(sd, tr, m, n, k, a, lda, tau, c, ldc); err != nil {
					t.Fatal(err)
				}
				if !same(a, a0) {
					t.Fatal("ormqr did not restore A")
				}
				padUntouched(t, "ormqr", c, m, ldc)
				q := exactQ(a, nq, k, lda, tau)
				if tr == blas.Trans {
					q = q.T()
				}
				cb := toBM(c0, m, n, ldc)
				var want bm
				if sd == blas.Left {
					want = mul(q, cb)
				} else {
					want = mul(cb, q)
				}
				froBound(t, "C op(Q)", sub(toBM(c, m, n, ldc), want), fro(cb), qrGamma[T](nq, k))
			}
		}
	}
}

// Q**T (Q C) is C again, to rounding, through the two routines: the one is the
// inverse of the other.
func TestOrmqrInverse(t *testing.T) { eachQ(t, ormqrInvT[float64], ormqrInvT[float32]) }

func ormqrInvT[T float](t *testing.T, im qimpl[T]) {
	m, n := 45, 11
	a := general[T](newRNG(7), m, n, m)
	tau := make([]T, n)
	if err := im.geqrf(m, n, a, m, tau); err != nil {
		t.Fatal(err)
	}
	c0 := general[T](newRNG(8), m, 6, m)
	c := clone(c0)
	if err := im.ormqr(blas.Left, blas.NoTrans, m, 6, n, a, m, tau, c, m); err != nil {
		t.Fatal(err)
	}
	if err := im.ormqr(blas.Left, blas.Trans, m, 6, n, a, m, tau, c, m); err != nil {
		t.Fatal(err)
	}
	cb := toBM(c0, m, 6, m)
	froBound(t, "Q'(Q C) - C", sub(toBM(c, m, 6, m), cb), fro(cb), qrGamma[T](m, n))
}

// ---- arguments ----

func TestQRArgErrors(t *testing.T) { eachQ(t, qrArgT[float64], qrArgT[float32]) }

func qrArgT[T float](t *testing.T, im qimpl[T]) {
	a9, a8 := make([]T, 9), make([]T, 8)
	t3, t2 := make([]T, 3), make([]T, 2)
	c9, c8 := make([]T, 9), make([]T, 8)

	gq := func(what string, p, m, n int, a []T, lda int, tau []T) {
		t.Helper()
		wantArg(t, "geqrf "+what, im.geqrf(m, n, a, lda, tau), p, nil)
	}
	gq("m", 1, -1, 3, a9, 3, t3)
	gq("n", 2, 3, -1, a9, 3, t3)
	gq("lda", 4, 3, 3, a9, 2, t3)
	gq("lda zero", 4, 0, 3, a9, 0, t3)
	gq("a short", 3, 3, 3, a8, 3, t3)
	gq("tau short", 5, 3, 3, a9, 3, t2)
	for _, c := range [][2]int{{0, 3}, {3, 0}, {0, 0}} { // quick returns look at nothing else
		wantNone(t, "geqrf quick", im.geqrf(c[0], c[1], nil, max(1, c[0]), nil), 0)
	}

	oq := func(what string, p int, sd blas.Side, tr blas.Transpose, m, n, k int, a []T, lda int, tau []T, c []T, ldc int) {
		t.Helper()
		wantArg(t, "ormqr "+what, im.ormqr(sd, tr, m, n, k, a, lda, tau, c, ldc), p, nil)
	}
	oq("side", 1, 'X', 'N', 3, 3, 3, a9, 3, t3, c9, 3)
	oq("trans", 2, 'L', 'X', 3, 3, 3, a9, 3, t3, c9, 3)
	oq("trans C", 2, 'L', 'C', 3, 3, 3, a9, 3, t3, c9, 3)
	oq("m", 3, 'L', 'N', -1, 3, 0, a9, 3, t3, c9, 3)
	oq("n", 4, 'L', 'N', 3, -1, 0, a9, 3, t3, c9, 3)
	oq("k negative", 5, 'L', 'N', 3, 3, -1, a9, 3, t3, c9, 3)
	oq("k above nq left", 5, 'L', 'N', 3, 3, 4, a9, 3, t3, c9, 3)
	oq("k above nq right", 5, 'R', 'N', 5, 3, 4, a9, 3, t3, c9, 5)
	oq("k above nq m zero", 5, 'L', 'N', 0, 3, 1, a9, 1, t3, c9, 1) // dimensions before the quick return
	oq("a short", 6, 'L', 'N', 3, 3, 3, a8, 3, t3, c9, 3)
	oq("lda", 7, 'L', 'N', 3, 3, 3, a9, 2, t3, c9, 3)
	oq("lda right", 7, 'R', 'N', 3, 3, 3, a9, 2, t3, c9, 3)
	oq("tau short", 8, 'L', 'N', 3, 3, 3, a9, 3, t2, c9, 3)
	oq("c short", 9, 'L', 'N', 3, 3, 3, a9, 3, t3, c8, 3)
	oq("ldc", 10, 'L', 'N', 3, 3, 3, a9, 3, t3, c9, 2)
	for _, q := range [][3]int{{0, 3, 3}, {3, 0, 3}, {3, 3, 0}} {
		wantNone(t, "ormqr quick", im.ormqr('L', 'N', q[0], q[1], min(q[2], q[0]), nil, max(1, q[0]), nil, nil, max(1, q[0])), 0)
	}
	wantNone(t, "ormqr quick right", im.ormqr('R', 'T', 3, 0, 0, nil, 1, nil, nil, 3), 0)

	og := func(what string, p, m, n, k int, a []T, lda int, tau []T) {
		t.Helper()
		wantArg(t, "orgqr "+what, im.orgqr(m, n, k, a, lda, tau), p, nil)
	}
	og("m", 1, -1, 0, 0, a9, 3, t3)
	og("n negative", 2, 3, -1, 0, a9, 3, t3)
	og("n above m", 2, 2, 3, 0, a9, 3, t3)
	og("k negative", 3, 3, 3, -1, a9, 3, t3)
	og("k above n", 3, 3, 2, 3, a9, 3, t3)
	og("lda", 5, 3, 3, 3, a9, 2, t3)
	og("a short", 4, 3, 3, 3, a8, 3, t3)
	og("tau short", 6, 3, 3, 3, a9, 3, t2)
	wantNone(t, "orgqr quick", im.orgqr(3, 0, 0, nil, 3, nil), 0)
	wantNone(t, "orgqr quick m zero", im.orgqr(0, 0, 0, nil, 1, nil), 0)
}

// Flags are accepted in lower case, as LSAME does.
func TestQRFlagsLowerCase(t *testing.T) { eachQ(t, qrFlagsT[float64], qrFlagsT[float32]) }

func qrFlagsT[T float](t *testing.T, im qimpl[T]) {
	a := general[T](newRNG(3), 4, 3, 4)
	tau := make([]T, 3)
	if err := im.geqrf(4, 3, a, 4, tau); err != nil {
		t.Fatal(err)
	}
	for _, sd := range []blas.Side{'L', 'l'} {
		for _, tr := range []blas.Transpose{'N', 'n', 'T', 't'} {
			c := general[T](newRNG(4), 4, 4, 4)
			want := clone(c)
			if err := im.ormqr(upperSide(sd), upperTrans(tr), 4, 4, 3, a, 4, tau, want, 4); err != nil {
				t.Fatal(err)
			}
			if err := im.ormqr(sd, tr, 4, 4, 3, a, 4, tau, c, 4); err != nil {
				t.Fatalf("%c %c: %v", sd, tr, err)
			}
			if !same(c, want) {
				t.Fatalf("%c %c differs from its upper case", sd, tr)
			}
		}
	}
}

func upperSide(s blas.Side) blas.Side { return blas.Side(upper(byte(s))) }

func upperTrans(s blas.Transpose) blas.Transpose { return blas.Transpose(upper(byte(s))) }

// An error leaves every output as it was.
func TestQRErrorLeavesData(t *testing.T) { eachQ(t, qrErrDataT[float64], qrErrDataT[float32]) }

func qrErrDataT[T float](t *testing.T, im qimpl[T]) {
	a := general[T](newRNG(5), 4, 4, 4)
	a0 := clone(a)
	tau := make([]T, 1)
	tau[0] = sentinel
	if im.geqrf(4, 4, a, 4, tau) == nil || !same(a, a0) || tau[0] != sentinel {
		t.Fatal("geqrf with a short tau changed its arguments or did not fail")
	}
	c := clone(a0)
	if im.ormqr(blas.Left, blas.NoTrans, 4, 4, 4, a, 4, tau, c, 4) == nil || !same(a, a0) || !same(c, a0) {
		t.Fatal("ormqr with a short tau changed its arguments or did not fail")
	}
	if im.orgqr(4, 4, 4, a, 4, tau) == nil || !same(a, a0) {
		t.Fatal("orgqr with a short tau changed its arguments or did not fail")
	}
}

// ---- special values ----

// Zero and degenerate matrices: a zero matrix has tau = 0 (H = I), R = 0, and Q
// is the identity; a zero column inside a matrix gives H = I for that column.
func TestQRZero(t *testing.T) { eachQ(t, qrZeroT[float64], qrZeroT[float32]) }

func qrZeroT[T float](t *testing.T, im qimpl[T]) {
	m, n := 9, 5
	a := make([]T, m*n)
	tau := make([]T, n)
	for i := range tau {
		tau[i] = sentinel
	}
	if err := im.geqrf(m, n, a, m, tau); err != nil {
		t.Fatal(err)
	}
	for i, v := range tau {
		if v != 0 {
			t.Fatalf("tau[%d] = %g, want 0", i, v)
		}
	}
	if err := im.orgqr(m, n, n, a, m, tau); err != nil {
		t.Fatal(err)
	}
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			want := T(0)
			if i == j {
				want = 1
			}
			if a[i+j*m] != want {
				t.Fatalf("Q(%d,%d) = %g, want %g", i, j, a[i+j*m], want)
			}
		}
	}
}

// The quick returns of the reference come before the slice-length checks: an
// empty matrix needs no slice, whatever the leading dimension.
func TestQRQuickReturnNoSlices(t *testing.T) { eachQ(t, qrQuickT[float64], qrQuickT[float32]) }

func qrQuickT[T float](t *testing.T, im qimpl[T]) {
	for _, d := range [][2]int{{0, 3}, {3, 0}, {0, 0}} {
		if err := im.geqrf(d[0], d[1], nil, 3, nil); err != nil {
			t.Fatalf("geqrf(%d, %d, nil): %v", d[0], d[1], err)
		}
	}
	if err := im.orgqr(3, 0, 0, nil, 3, nil); err != nil {
		t.Fatalf("orgqr n = 0 with nil slices: %v", err)
	}
}

// With no reflectors (k = 0) Orgqr gives the first n columns of the identity.
func TestOrgqrIdentity(t *testing.T) { eachQ(t, orgqrIdT[float64], orgqrIdT[float32]) }

func orgqrIdT[T float](t *testing.T, im qimpl[T]) {
	m, n := 6, 4
	a := general[T](newRNG(9), m, n, m+1)
	if err := im.orgqr(m, n, 0, a, m+1, nil); err != nil {
		t.Fatal(err)
	}
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			want := T(0)
			if i == j {
				want = 1
			}
			if a[i+j*(m+1)] != want {
				t.Fatalf("Q(%d,%d) = %g", i, j, a[i+j*(m+1)])
			}
		}
	}
}

// Scaling: a column of a magnitude whose square underflows (or overflows) still
// factors, as DLARFG rescales when beta is below safmin; tau stays finite and
// the bound holds relative to the column.
func TestGeqrfExtremeScale(t *testing.T) { eachQ(t, extremeT[float64], extremeT[float32]) }

func extremeT[T float](t *testing.T, im qimpl[T]) {
	lo, hi := 1e-36, math.MaxFloat32/1024 // below safmin = 2^-102, normal; squares overflow
	if _, ok := any(T(0)).(float64); ok {
		lo, hi = 1e-300, 1e300 // below safmin = 2^-969
	}
	tiny, huge := T(lo), T(hi)
	for _, s := range []T{tiny, huge} {
		m, n := 6, 3
		a := general[T](newRNG(11), m, n, m)
		for i := range a {
			a[i] = T(float64(i%5)-2.5) * s / 4
		}
		a0 := clone(a)
		tau := make([]T, n)
		if err := im.geqrf(m, n, a, m, tau); err != nil {
			t.Fatal(err)
		}
		for _, v := range append(clone(a), tau...) {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("scale %g: a non finite value %g in the result", s, v)
			}
		}
		q := exactQ(a, m, n, m, tau)
		b0 := toBM(a0, m, n, m)
		colBound(t, "A - Q R", sub(b0, mul(q, rFactor(a, m, n, m))), b0, qrGamma[T](m, n))
	}
}

// NaN and Inf in the data travel as in the reference: lapack and netlib return
// results of the same class at every element (NaN with NaN, the same infinity),
// with no panic and no error. The block sizes are the default ones in both (a
// NaN reaches different elements through different blocking, and netlib has no
// setter): the shapes below include one that runs the blocked path.
func TestQRNaNInf(t *testing.T) {
	t.Run("float64", func(t *testing.T) { nanInfQ(t, qimpls64) })
	t.Run("float32", func(t *testing.T) { nanInfQ(t, qimpls32) })
}

func nanInfQ[T float](t *testing.T, ims []qimpl[T]) {
	inf := T(math.Inf(1))
	for _, c := range []qrConfig{{}} {
		t.Run(cfgName(c), func(t *testing.T) {
			setQR(t, c)
			for si, sh := range [][2]int{{5, 3}, {9, 9}, {20, 7}, {7, 20}, {40, 33}, {200, 150}} {
				m, n := sh[0], sh[1]
				for _, bad := range []T{nan[T](), inf, -inf} {
					for _, pos := range []int{0, (m*n - 1) / 2, m*n - 1} {
						a0 := general[T](newRNG(uint64(500+si)), m, n, m)
						a0[pos] = bad
						var as [2][]T
						var ts [2][]T
						for i, im := range ims {
							as[i], ts[i] = clone(a0), make([]T, min(m, n))
							if err := im.geqrf(m, n, as[i], m, ts[i]); err != nil {
								t.Fatal(err)
							}
						}
						classSame(t, "geqrf a", as[0], as[1])
						classSame(t, "geqrf tau", ts[0], ts[1])
					}
				}
				// Reflectors with a NaN in tau or in v.
				if m >= n {
					f := general[T](newRNG(uint64(600+si)), m, n, m)
					tau := make([]T, n)
					if err := ims[1].geqrf(m, n, f, m, tau); err != nil {
						t.Fatal(err)
					}
					f2, tau2 := clone(f), clone(tau)
					f2[(m*n-1)/2] = nan[T]()
					tau2[n-1] = inf
					c0 := general[T](newRNG(7), m, n, m)
					var gs [2][]T
					for i, im := range ims {
						gs[i] = clone(f2)
						if err := im.orgqr(m, n, n, gs[i], m, tau2); err != nil {
							t.Fatal(err)
						}
					}
					classSame(t, "orgqr", gs[0], gs[1])
					for _, tr := range []blas.Transpose{blas.NoTrans, blas.Trans} {
						var cs [2][]T
						for i, im := range ims {
							cs[i] = clone(c0)
							if err := im.ormqr(blas.Left, tr, m, n, n, clone(f2), m, tau2, cs[i], m); err != nil {
								t.Fatal(err)
							}
						}
						classSame(t, "ormqr", cs[0], cs[1])
					}
				}
			}
		})
	}
}

func classSame[T float](t *testing.T, what string, a, b []T) {
	t.Helper()
	for i := range a {
		if !classEq(a[i], b[i]) {
			t.Fatalf("%s: element %d is %g in lapack and %g in netlib", what, i, a[i], b[i])
		}
	}
}
