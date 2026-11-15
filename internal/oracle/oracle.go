// Package oracle computes BLAS results exactly, in math/big floating point of
// 256 bits, together with the magnitudes the error bounds of package blas are
// stated against. It is test support: the 256 bits are far beyond any product
// or sum of float64 these tests form, so the "exact" value carries no error of
// its own that matters.
package oracle

import (
	"math"
	"math/big"
)

// Prec is the precision of the oracle in bits.
const Prec = 256

// Float is the element types of the real BLAS.
type Float interface{ float32 | float64 }

// U returns the unit roundoff of T: 2^-53 for float64, 2^-24 for float32.
func U[T Float]() float64 {
	var z T
	if _, ok := any(z).(float32); ok {
		return math.Ldexp(1, -24)
	}
	return math.Ldexp(1, -53)
}

func bf() *big.Float { return new(big.Float).SetPrec(Prec) }

// B converts v exactly.
func B[T Float](v T) *big.Float { return bf().SetFloat64(float64(v)) }

// Gamma returns gamma_k = k*u/(1-k*u) for T, as an exact big value.
func Gamma[T Float](k int) *big.Float {
	ku := bf().Mul(bf().SetInt64(int64(k)), bf().SetFloat64(U[T]()))
	return bf().Quo(ku, bf().Sub(bf().SetInt64(1), ku))
}

// Idx is the physical index of logical element i of an n-element vector with
// increment inc, as the reference BLAS addresses it.
func Idx(n, inc, i int) int {
	if inc > 0 {
		return i * inc
	}
	return (n - 1 - i) * -inc
}

// Result is an exact result with, per element, the two magnitudes of the
// bounds: P multiplies the length-dependent gamma, Q the gamma_2 of the beta
// (or y) term.
type Result struct{ Exact, P, Q []*big.Float }

func newResult(n int) Result {
	r := Result{make([]*big.Float, n), make([]*big.Float, n), make([]*big.Float, n)}
	for i := range r.Exact {
		r.Exact[i], r.P[i], r.Q[i] = bf(), bf(), bf()
	}
	return r
}

// Gemm computes alpha*op(A)*op(B) + beta*C exactly, column-major, element
// (i, j) at index i+j*m of the result. C is not read when beta is 0.
func Gemm[T Float](tA, tB byte, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) Result {
	r := newResult(m * n)
	opA := func(i, l int) *big.Float {
		if tA == 'N' {
			return B(a[i+l*lda])
		}
		return B(a[l+i*lda])
	}
	opB := func(l, j int) *big.Float {
		if tB == 'N' {
			return B(b[l+j*ldb])
		}
		return B(b[j+l*ldb])
	}
	al, be := B(alpha), B(beta)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			s, p := bf(), bf()
			for l := 0; l < k && alpha != 0; l++ { // alpha = 0 reads neither A nor B
				x := bf().Mul(opA(i, l), opB(l, j))
				s.Add(s, x)
				p.Add(p, bf().Abs(x))
			}
			e, q := bf().Mul(al, s), bf()
			if beta != 0 {
				cv := bf().Mul(be, B(c[i+j*ldc]))
				e.Add(e, cv)
				q.Abs(cv)
			}
			x := i + j*m
			r.Exact[x], r.P[x], r.Q[x] = e, p.Mul(p, bf().Abs(al)), q
		}
	}
	return r
}

// Gemv computes alpha*op(A)*x + beta*y exactly; element i is the i-th logical
// element of y. y is not read when beta is 0.
func Gemv[T Float](t byte, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) Result {
	lenx, leny := n, m
	if t != 'N' {
		lenx, leny = m, n
	}
	r := newResult(leny)
	al, be := B(alpha), B(beta)
	for i := 0; i < leny; i++ {
		s, p := bf(), bf()
		for l := 0; l < lenx && alpha != 0; l++ { // alpha = 0 reads neither A nor x
			var av T
			if t == 'N' {
				av = a[i+l*lda]
			} else {
				av = a[l+i*lda]
			}
			xv := x[Idx(lenx, incx, l)]
			pr := bf().Mul(B(av), B(xv))
			s.Add(s, pr)
			p.Add(p, bf().Abs(pr))
		}
		e, q := bf().Mul(al, s), bf()
		if beta != 0 {
			yv := bf().Mul(be, B(y[Idx(leny, incy, i)]))
			e.Add(e, yv)
			q.Abs(yv)
		}
		r.Exact[i], r.P[i], r.Q[i] = e, p.Mul(p, bf().Abs(al)), q
	}
	return r
}

// Dot computes the sum of x[i]*y[i] exactly; P is the sum of |x[i]*y[i]|.
func Dot[T Float](n int, x []T, incx int, y []T, incy int) Result {
	r := newResult(1)
	for i := 0; i < n; i++ {
		t := bf().Mul(B(x[Idx(n, incx, i)]), B(y[Idx(n, incy, i)]))
		r.Exact[0].Add(r.Exact[0], t)
		r.P[0].Add(r.P[0], bf().Abs(t))
	}
	return r
}

// Axpy computes alpha*x[i] + y[i] exactly; P is |alpha*x[i]|, Q is |y[i]|.
func Axpy[T Float](n int, alpha T, x []T, incx int, y []T, incy int) Result {
	r := newResult(n)
	for i := 0; i < n; i++ {
		ax := bf().Mul(B(alpha), B(x[Idx(n, incx, i)]))
		yv := B(y[Idx(n, incy, i)])
		r.Exact[i].Add(ax, yv)
		r.P[i].Abs(ax)
		r.Q[i].Abs(yv)
	}
	return r
}

// Scal computes alpha*x[i] exactly; P is its magnitude.
func Scal[T Float](n int, alpha T, x []T, incx int) Result {
	r := newResult(n)
	for i := 0; i < n; i++ {
		r.Exact[i].Mul(B(alpha), B(x[i*incx]))
		r.P[i].Abs(r.Exact[i])
	}
	return r
}

// Nrm2 computes the Euclidean norm of x to 256 bits (P is the norm itself).
func Nrm2[T Float](n int, x []T, incx int) Result {
	r := newResult(1)
	s := bf()
	for i := 0; i < n; i++ {
		v := B(x[Idx(n, incx, i)])
		s.Add(s, bf().Mul(v, v))
	}
	r.Exact[0].Sqrt(s)
	r.P[0].Set(r.Exact[0])
	return r
}

// Asum computes the sum of |x[i]|; P is the same sum.
func Asum[T Float](n int, x []T, incx int) Result {
	r := newResult(1)
	for i := 0; i < n; i++ {
		r.Exact[0].Add(r.Exact[0], bf().Abs(B(x[i*incx])))
	}
	r.P[0].Set(r.Exact[0])
	return r
}

// Within reports whether |got - exact| <= gp*p + gq*q, the bound of the
// package comment, where gp and gq are gamma values. When the bound is zero the
// result must be exact.
func Within[T Float](got T, r Result, i int, gp, gq *big.Float) bool {
	bound := bf().Add(bf().Mul(gp, r.P[i]), bf().Mul(gq, r.Q[i]))
	diff := bf().Abs(bf().Sub(B(got), r.Exact[i]))
	return diff.Cmp(bound) <= 0
}

// Fill returns n values of mixed sign and magnitude, sign*(1+f)*2^e with
// e in [-6, 6], the kind of data on which a wrong sum, a lost term or a
// misplaced element changes the result by far more than any rounding bound.
func Fill[T Float](rng interface {
	Float64() float64
	IntN(int) int
}, n int) []T {
	v := make([]T, n)
	for i := range v {
		x := (1 + rng.Float64()) * math.Ldexp(1, rng.IntN(13)-6)
		if rng.IntN(2) == 0 {
			x = -x
		}
		v[i] = T(x)
	}
	return v
}
