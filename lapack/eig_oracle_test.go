package lapack_test

import (
	"math"
	"math/big"
	"math/rand/v2"
	"sort"

	"github.com/go-blas/blas/internal/oracle"
)

// The oracles of the symmetric eigensolver tests, all in 256-bit math/big:
//
//	O-res    the residual and orthogonality of the computed eigenpairs, formed exactly;
//	O-sturm  the eigenvalues of a symmetric tridiagonal matrix, by bisection on the
//	         Sturm sequence of the exact matrix;
//	O-jac    the eigenvalues of a dense symmetric matrix, by cyclic Jacobi.
//
// The inputs are the matrices as stored (float32 or float64 values, exactly), so
// the oracles carry no error of the generator.

// colNorm2 is the 2-norm of column j of m.
func colNorm2(m bm, j int) *big.Float {
	s, t := bf(), bf()
	for i := 0; i < m.r; i++ {
		s.Add(s, t.Mul(m.at(i, j), m.at(i, j)))
	}
	return bf().Sqrt(s)
}

func identityBM(n int) bm {
	m := newBM(n, n)
	for i := 0; i < n; i++ {
		m.at(i, i).SetInt64(1)
	}
	return m
}

// symFull returns the symmetric matrix whose upper (up) or lower triangle is
// stored in a.
func symFull[T float](a []T, n, lda int, up bool) bm {
	m := newBM(n, n)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			ii, jj := i, j
			if (up && i > j) || (!up && i < j) {
				ii, jj = j, i
			}
			m.at(i, j).Set(oracle.B(a[ii+jj*lda]))
		}
	}
	return m
}

// tridBM returns the symmetric tridiagonal matrix of diagonal d and off-diagonal e.
func tridBM[T float](d, e []T, n int) bm {
	m := newBM(n, n)
	for i := 0; i < n; i++ {
		m.at(i, i).Set(oracle.B(d[i]))
		if i < n-1 {
			m.at(i+1, i).Set(oracle.B(e[i]))
			m.at(i, i+1).Set(oracle.B(e[i]))
		}
	}
	return m
}

// diagBM is Z*diag(w), the columns of z scaled by w.
func scaleCols[T float](z []T, n, ldz int, w []T) bm {
	m := toBM(z, n, n, ldz)
	for j := 0; j < n; j++ {
		wj := oracle.B(w[j])
		for i := 0; i < n; i++ {
			m.at(i, j).Mul(m.at(i, j), wj)
		}
	}
	return m
}

// orthDefect is ||Z**T Z - I||_F of the n x n matrix in z.
func orthDefect[T float](z []T, n, ldz int) *big.Float {
	b := toBM(z, n, n, ldz)
	return fro(sub(mul(b.T(), b), identityBM(n)))
}

// sturmCount is the number of eigenvalues of the tridiagonal matrix (d, e) below x.
func sturmCount(d, e []*big.Float, x, tiny *big.Float) int {
	cnt := 0
	q, prev, t := bf(), bf(), bf()
	for i := range d {
		q.Sub(d[i], x)
		if i > 0 {
			p := prev
			if p.Sign() == 0 {
				p = tiny
			}
			t.Mul(e[i-1], e[i-1])
			t.Quo(t, p)
			q.Sub(q, t)
		}
		if q.Sign() == 0 {
			q.Set(tiny)
		}
		if q.Sign() < 0 {
			cnt++
		}
		prev.Set(q)
	}
	return cnt
}

// sturmEigs returns all the eigenvalues of the symmetric tridiagonal matrix (d,
// e), ascending, each to within 2^-100 of the largest Gershgorin bound.
func sturmEigs[T float](d, e []T, n int) []*big.Float {
	bd, be := make([]*big.Float, n), make([]*big.Float, max(n-1, 0))
	r := bf()
	lo, hi := bf().SetInt64(0), bf().SetInt64(0)
	for i := range bd {
		bd[i] = oracle.B(d[i])
		rad := bf()
		if i > 0 {
			rad.Add(rad, bf().Abs(oracle.B(e[i-1])))
		}
		if i < n-1 {
			rad.Add(rad, bf().Abs(oracle.B(e[i])))
		}
		if v := r.Sub(bd[i], rad); v.Cmp(lo) < 0 {
			lo.Set(v)
		}
		if v := r.Add(bd[i], rad); v.Cmp(hi) > 0 {
			hi.Set(v)
		}
	}
	for i := range be {
		be[i] = oracle.B(e[i])
	}
	width := bf().Sub(hi, lo)
	// Widen a hair so that the extreme eigenvalues are inside.
	pad := bf().Mul(width, bf().SetFloat64(1e-6))
	if pad.Sign() == 0 {
		pad.SetFloat64(1e-300)
	}
	lo.Sub(lo, pad)
	hi.Add(hi, pad)
	tiny := bf().Mul(bf().Sub(hi, lo), bf().SetMantExp(bf().SetInt64(1), -230))
	if tiny.Sign() == 0 {
		tiny.SetMantExp(bf().SetInt64(1), -1000)
	}
	out := make([]*big.Float, n)
	half := bf().SetFloat64(0.5)
	for k := 0; k < n; k++ {
		a, b := bf().Set(lo), bf().Set(hi)
		for it := 0; it < 110; it++ {
			mid := bf().Mul(bf().Add(a, b), half)
			if sturmCount(bd, be, mid, tiny) >= k+1 {
				b = mid
			} else {
				a = mid
			}
		}
		out[k] = bf().Mul(bf().Add(a, b), half)
	}
	return out
}

// jacobiEigs returns the eigenvalues of the symmetric matrix m, ascending, by
// cyclic Jacobi sweeps until the off-diagonal part is below 2^-200 of the whole.
func jacobiEigs(m bm) []*big.Float {
	n := m.r
	a := newBM(n, n)
	for i := range a.v {
		a.v[i].Set(m.v[i])
	}
	tol := bf().SetMantExp(bf().SetInt64(1), -200)
	one := bf().SetInt64(1)
	for sweep := 0; sweep < 40; sweep++ {
		off, all, t := bf(), bf(), bf()
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				t.Mul(a.at(i, j), a.at(i, j))
				all.Add(all, t)
				if i != j {
					off.Add(off, t)
				}
			}
		}
		if off.Cmp(bf().Mul(all, tol)) <= 0 {
			break
		}
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := a.at(p, q)
				if apq.Sign() == 0 {
					continue
				}
				theta := bf().Sub(a.at(q, q), a.at(p, p))
				theta.Quo(theta, bf().Mul(bf().SetInt64(2), apq))
				r := bf().Sqrt(bf().Add(bf().Mul(theta, theta), one))
				tt := bf().Quo(one, bf().Add(bf().Abs(theta), r))
				if theta.Sign() < 0 {
					tt.Neg(tt)
				}
				c := bf().Quo(one, bf().Sqrt(bf().Add(bf().Mul(tt, tt), one)))
				s := bf().Mul(tt, c)
				for k := 0; k < n; k++ { // columns p, q
					akp, akq := bf().Set(a.at(k, p)), bf().Set(a.at(k, q))
					a.at(k, p).Sub(bf().Mul(c, akp), bf().Mul(s, akq))
					a.at(k, q).Add(bf().Mul(s, akp), bf().Mul(c, akq))
				}
				for k := 0; k < n; k++ { // rows p, q
					apk, aqk := bf().Set(a.at(p, k)), bf().Set(a.at(q, k))
					a.at(p, k).Sub(bf().Mul(c, apk), bf().Mul(s, aqk))
					a.at(q, k).Add(bf().Mul(s, apk), bf().Mul(c, aqk))
				}
			}
		}
	}
	out := make([]*big.Float, n)
	for i := range out {
		out[i] = bf().Set(a.at(i, i))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Cmp(out[j]) < 0 })
	return out
}

// ---- the generators ----

// tridKinds are the tridiagonal families: the diagonal and off-diagonal as
// float64, before scaling.
var tridKinds = []string{"random", "clustered", "repeated", "graded", "tinyoff", "wilkinson", "clement", "zerooff"}

func tridiag(kind string, n int, rng *rand.Rand) (d, e []float64) {
	d, e = make([]float64, n), make([]float64, max(n-1, 0))
	u := func() float64 { return 2*rng.Float64() - 1 }
	switch kind {
	case "random":
		for i := range d {
			d[i] = u()
		}
		for i := range e {
			e[i] = u()
		}
	case "clustered": // a cluster at 1 of width 1e-9, a few outliers
		for i := range d {
			d[i] = 1 + 1e-9*u()
			if i%7 == 3 {
				d[i] = 5 * u()
			}
		}
		for i := range e {
			e[i] = 1e-5 * u()
		}
	case "repeated": // three values, blocks coupled by exact zeros
		vals := []float64{-1, 0.5, 2}
		for i := range d {
			d[i] = vals[i%3]
		}
		for i := range e {
			e[i] = 0
			if i%5 == 4 {
				e[i] = 0.25 * u()
			}
		}
	case "graded": // 1 down to 1e-13
		for i := range d {
			d[i] = math.Pow(10, -13*float64(i)/float64(max(n-1, 1))) * (1 + 0.1*u())
		}
		for i := range e {
			e[i] = 0.3 * math.Pow(10, -13*float64(i)/float64(max(n-1, 1))) * u()
		}
	case "tinyoff": // off-diagonals down to 1e-300 and exact zeros
		for i := range d {
			d[i] = u()
		}
		for i := range e {
			switch i % 4 {
			case 0:
				e[i] = 0
			case 1:
				e[i] = 1e-300 * u()
			default:
				e[i] = 1e-12 * u()
			}
		}
	case "wilkinson": // W_n+: diag |i - (n-1)/2|, off-diagonal 1
		for i := range d {
			d[i] = math.Abs(float64(i) - float64(n-1)/2)
		}
		for i := range e {
			e[i] = 1
		}
	case "clement": // zero diagonal, off-diagonal sqrt(i(n-i))
		for i := range e {
			e[i] = math.Sqrt(float64((i + 1) * (n - 1 - i)))
		}
	case "zerooff":
		for i := range d {
			d[i] = u()
		}
	}
	return d, e
}

// spectra are the dense families: a spectrum, spread by a few random reflectors.
var spectraKinds = []string{"random", "clustered", "repeated", "graded"}

func spectrum(kind string, n int, rng *rand.Rand) []float64 {
	l := make([]float64, n)
	u := func() float64 { return 2*rng.Float64() - 1 }
	switch kind {
	case "random":
		for i := range l {
			l[i] = u()
		}
	case "clustered":
		for i := range l {
			l[i] = 1 + 1e-9*u()
			if i%9 == 4 {
				l[i] = -3 * rng.Float64()
			}
		}
	case "repeated":
		for i := range l {
			l[i] = []float64{-2, 0, 0, 1.5}[i%4]
		}
	case "graded":
		for i := range l {
			l[i] = math.Pow(10, -13*float64(i)/float64(max(n-1, 1)))
			if i%2 == 1 {
				l[i] = -l[i]
			}
		}
	}
	return l
}

// reflect spreads diag(lambda) with k random Householder reflections applied on
// both sides, into the dense symmetric n x n matrix (column-major, float64).
func spread(lambda []float64, rng *rand.Rand) []float64 {
	n := len(lambda)
	m := make([]float64, n*n)
	for i, l := range lambda {
		m[i+i*n] = l
	}
	v, w := make([]float64, n), make([]float64, n)
	for r := 0; r < min(n, 6); r++ {
		var vv float64
		for i := range v {
			v[i] = 2*rng.Float64() - 1
			vv += v[i] * v[i]
		}
		if vv == 0 {
			continue
		}
		// M := H M H with H = I - 2 v v**T / (v**T v).
		hm := func(left bool) {
			for c := 0; c < n; c++ {
				var dot float64
				for i := 0; i < n; i++ {
					if left {
						dot += v[i] * m[i+c*n]
					} else {
						dot += v[i] * m[c+i*n]
					}
				}
				w[c] = 2 * dot / vv
			}
			for c := 0; c < n; c++ {
				for i := 0; i < n; i++ {
					if left {
						m[i+c*n] -= v[i] * w[c]
					} else {
						m[c+i*n] -= w[c] * v[i]
					}
				}
			}
		}
		hm(true)
		hm(false)
	}
	return m
}
