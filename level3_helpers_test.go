package blas_test

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

// The tests of the level 3 routines (trsm, trmm, syrk, syr2k, symm) share these
// helpers. A triangular or symmetric operand is built with NaN in every element
// the routine must not read (the other triangle, and the diagonal of a unit
// triangle), so a routine that reads one fails by its result; the oracle sees a
// dense copy of what the reference reads.

// blockSizes are the diagonal block orders a test runs package blas with: the
// default, and two small ones that make 40-row matrices span many blocks, one
// of them not dividing the sizes. netlib has no blocks.
func blockSizes(impl string) []int {
	if impl == "blas" {
		return []int{64, 5}
	}
	return []int{64}
}

func nan[T float]() T { return T(math.NaN()) }

// triMatrix returns a column-major s x s matrix with leading dimension ld whose
// triangle upper selects holds data, the other triangle NaN, the diagonal NaN
// for a unit matrix, and the padding the sentinel. With dominant the data are
// those of a well-conditioned triangle: a diagonal of magnitude in [1, 2) and an
// off-diagonal scaled down by 256*s, so that a solve does not grow, in single
// precision either.
func triMatrix[T float](rng *rand.Rand, s, ld int, upper, unit, dominant bool) []T {
	if s == 0 {
		return nil
	}
	v := matrix[T](rng, s, s, ld)
	for j := 0; j < s; j++ {
		for i := 0; i < s; i++ {
			p := i + j*ld
			switch {
			case i == j && unit:
				v[p] = nan[T]()
			case i == j && dominant:
				v[p] = T(math.Copysign(1+math.Abs(float64(v[p]))/256, float64(v[p])))
			case (i < j) != upper && i != j:
				v[p] = nan[T]()
			case dominant:
				v[p] = T(float64(v[p]) / float64(256*s))
			}
		}
	}
	return v
}

// symMatrix returns a column-major s x s matrix whose triangle upper selects
// (diagonal included) holds data, the other triangle NaN.
func symMatrix[T float](rng *rand.Rand, s, ld int, upper bool) []T {
	if s == 0 {
		return nil
	}
	v := matrix[T](rng, s, s, ld)
	for j := 0; j < s; j++ {
		for i := 0; i < s; i++ {
			if i != j && (i < j) != upper {
				v[i+j*ld] = nan[T]()
			}
		}
	}
	return v
}

// garbage is what a routine that writes one triangle of C must leave in the
// other one; it is finite, so that the oracle, which reads all of C, accepts it.
const garbage = 777.25

// cMatrix returns a column-major n x n C with leading dimension ld, the
// triangle upper selects holding data, the other one garbage, the padding the
// sentinel.
func cMatrix[T float](rng *rand.Rand, n, ld int, upper bool) []T {
	if n == 0 {
		return nil
	}
	v := matrix[T](rng, n, n, ld)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			if i != j && (i < j) != upper {
				v[i+j*ld] = garbage
			}
		}
	}
	return v
}

// otherTriangleIntact asserts that the elements of C outside the triangle still
// hold what they held.
func otherTriangleIntact[T float](t *testing.T, what string, got, want []T, n, ld int, upper bool) {
	t.Helper()
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			if i != j && (i < j) != upper && got[i+j*ld] != want[i+j*ld] {
				t.Fatalf("%s: element (%d,%d) of the other triangle was written", what, i, j)
			}
		}
	}
}

// triDense is the s x s dense matrix of the triangle of a that the reference
// reads: zero outside it, one on the diagonal for a unit triangle.
func triDense[T float](a []T, lda, s int, upper, unit bool) []T {
	d := make([]T, s*s)
	for j := 0; j < s; j++ {
		for i := 0; i < s; i++ {
			switch {
			case i == j && unit:
				d[i+j*s] = 1
			case i == j || (i < j) == upper:
				d[i+j*s] = a[i+j*lda]
			}
		}
	}
	return d
}

// symDense is the s x s dense symmetric matrix of the triangle of a that the
// reference reads.
func symDense[T float](a []T, lda, s int, upper bool) []T {
	d := make([]T, s*s)
	for j := 0; j < s; j++ {
		for i := 0; i < s; i++ {
			if (i <= j) == upper {
				d[i+j*s] = a[i+j*lda]
			} else {
				d[i+j*s] = a[j+i*lda]
			}
		}
	}
	return d
}

// trsmResidual is the residual op(A) X - alpha B0 (side Left; X op(A) - alpha B0
// for Right) of the computed solution x, exactly, with P the matching sum
// |op(A)||X| + |alpha B0| of the bound; element (i, j) at i+j*m.
func trsmResidual[T float](side, uplo, tA, diag byte, m, n int, alpha T, a []T, lda int, b0, x []T, ldb int) oracle.Result {
	left := side == 'L'
	s := n
	if left {
		s = m
	}
	td := triDense(a, lda, s, uplo == 'U', diag == 'U')
	var r oracle.Result
	if left {
		r = oracle.Gemm(tA, 'N', m, n, m, T(1), td, s, x, ldb, T(0), nil, 0)
	} else {
		r = oracle.Gemm('N', tA, m, n, n, T(1), x, ldb, td, s, T(0), nil, 0)
	}
	al := oracle.B(alpha)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			ab := new(big.Float).SetPrec(oracle.Prec).Mul(al, oracle.B(b0[i+j*ldb]))
			e := i + j*m
			r.Exact[e].Sub(r.Exact[e], ab)
			r.P[e].Add(r.P[e], ab.Abs(ab))
		}
	}
	return r
}

// trsmWithin asserts that the computed solution x of a TRSM is within the
// residual bound gamma_{s+4}(|op(A)||X| + |alpha B0|).
func trsmWithin[T float](t *testing.T, what string, side, uplo, tA, diag byte, m, n int, alpha T, a []T, lda int, b0, x []T, ldb int) {
	t.Helper()
	s := n
	if side == 'L' {
		s = m
	}
	r := trsmResidual(side, uplo, tA, diag, m, n, alpha, a, lda, b0, x, ldb)
	g := oracle.Gamma[T](s + 4)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			// The solution is the computed one, so the residual itself is compared with zero.
			if !oracle.Within(T(0), r, i+j*m, g, g) {
				t.Errorf("%s: the residual of X[%d,%d] = %v is outside the bound", what, i, j, x[i+j*ldb])
				return
			}
		}
	}
}

// syr2kExact is alpha*op(A)*op(B)**T + alpha*op(B)*op(A)**T + beta*C0, exactly,
// with the magnitudes of its bound.
func syr2kExact[T float](trans byte, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c0 []T, ldc int) oracle.Result {
	tA, tB := byte('N'), byte('T')
	if trans != 'N' {
		tA, tB = 'T', 'N'
	}
	r1 := oracle.Gemm(tA, tB, n, n, k, alpha, a, lda, b, ldb, beta, c0, ldc)
	r2 := oracle.Gemm(tA, tB, n, n, k, alpha, b, ldb, a, lda, T(0), nil, 0)
	for i := range r1.Exact {
		r1.Exact[i].Add(r1.Exact[i], r2.Exact[i])
		r1.P[i].Add(r1.P[i], r2.P[i])
	}
	return r1
}

// syrkExact is alpha*op(A)*op(A)**T + beta*C0, exactly.
func syrkExact[T float](trans byte, n, k int, alpha T, a []T, lda int, beta T, c0 []T, ldc int) oracle.Result {
	if trans == 'N' {
		return oracle.Gemm('N', 'T', n, n, k, alpha, a, lda, a, lda, beta, c0, ldc)
	}
	return oracle.Gemm('T', 'N', n, n, k, alpha, a, lda, a, lda, beta, c0, ldc)
}

// symmExact is alpha*A*B + beta*C0 (Left) or alpha*B*A + beta*C0, exactly, from
// the triangle of a that the reference reads.
func symmExact[T float](side, uplo byte, m, n int, alpha T, a []T, lda int, b []T, ldb int, beta T, c0 []T, ldc int) oracle.Result {
	if side == 'L' {
		return oracle.Gemm('N', 'N', m, n, m, alpha, symDense(a, lda, m, uplo == 'U'), m, b, ldb, beta, c0, ldc)
	}
	return oracle.Gemm('N', 'N', m, n, n, alpha, b, ldb, symDense(a, lda, n, uplo == 'U'), n, beta, c0, ldc)
}

// triWithin asserts that the triangle of c is within gamma_k of r, elementwise,
// the matrix being n x n with leading dimension ld and r laid out with ld n.
func triWithin[T float](t *testing.T, what string, c []T, ld, n int, upper bool, r oracle.Result, k int) {
	t.Helper()
	g := oracle.Gamma[T](k)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			if i == j || (i < j) == upper {
				within(t, what, c[i+j*ld], r, i+j*n, g, g)
			}
		}
	}
}

var (
	sides  = []blas.Side{blas.Left, blas.Right}
	uplos  = []blas.Uplo{blas.Upper, blas.Lower}
	diags  = []blas.Diag{blas.NonUnit, blas.Unit}
	trans2 = []blas.Transpose{blas.NoTrans, blas.Trans}
)

func flagName(fs ...byte) string { return string(fs) }
