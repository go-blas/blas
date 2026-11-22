package netlib

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// The QR factorization routines, as plain-Go ports of Reference-LAPACK 3.12.0
// (tag v3.12.0, SRC): dgeqrf.f, dgeqr2.f, dormqr.f, dorm2r.f, dorgqr.f, dorg2r.f
// and what they call (dlarfg.f, dlarf.f, dlarft.f, dlarfb.f, dlapy2.f, iladlc.f,
// iladlr.f, and the level 2 and 3 BLAS of this package), and the single precision
// twins. They have the signatures and the conventions of package lapack, which
// documents them. The block sizes are the reference's (ilaenv.f): 32 for all
// three, with the crossover 128 for Geqrf and Orgqr.

// Dgeqrf is DGEQRF: the QR factorization A = Q*R of the m x n column-major
// matrix a. See lapack.Dgeqrf.
func Dgeqrf(m, n int, a []float64, lda int, tau []float64) error {
	return geqrf("DGEQRF", m, n, a, lda, tau)
}

// Sgeqrf is SGEQRF (sgeqrf.f), Dgeqrf in single precision.
func Sgeqrf(m, n int, a []float32, lda int, tau []float32) error {
	return geqrf("SGEQRF", m, n, a, lda, tau)
}

func geqrf[T ref.Float](name string, m, n int, a []T, lda int, tau []T) error {
	if p := check.Geqrf(m, n, lda); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.GeqrfLens(m, n, lda, len(a), len(tau)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Geqrf(ref.RefKern[T](), ref.NBQR, ref.NXQR, m, n, a, lda, tau)
	return nil
}

// Dormqr is DORMQR: overwrites the m x n matrix c with Q*C, Q**T*C, C*Q or C*Q**T,
// Q being the orthogonal matrix Dgeqrf represents. See lapack.Dormqr.
func Dormqr(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau []float64, c []float64, ldc int) error {
	return ormqr("DORMQR", side, trans, m, n, k, a, lda, tau, c, ldc)
}

// Sormqr is SORMQR (sormqr.f), Dormqr in single precision.
func Sormqr(side blas.Side, trans blas.Transpose, m, n, k int, a []float32, lda int, tau []float32, c []float32, ldc int) error {
	return ormqr("SORMQR", side, trans, m, n, k, a, lda, tau, c, ldc)
}

func ormqr[T ref.Float](name string, side blas.Side, trans blas.Transpose, m, n, k int, a []T, lda int, tau []T, c []T, ldc int) error {
	side = blas.Side(check.Upper(byte(side)))                                 // LSAME ignores case
	trans = blas.Transpose(check.Upper(byte(trans)))                          // LSAME ignores case
	if p := check.Ormqr(byte(side), byte(trans), m, n, k, lda, ldc); p != 0 { // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 || k == 0 { // guard
		return nil
	}
	if p := check.OrmqrLens(byte(side), m, n, k, lda, ldc, len(a), len(tau), len(c)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Ormqr(ref.RefKern[T](), ref.NBORM, side == blas.Left, trans == blas.NoTrans, m, n, k, a, lda, tau, c, ldc)
	return nil
}

// Dorgqr is DORGQR: generates the m x n matrix Q with orthonormal columns, the
// first n columns of the product of k reflectors Dgeqrf returned. See
// lapack.Dorgqr.
func Dorgqr(m, n, k int, a []float64, lda int, tau []float64) error {
	return orgqr("DORGQR", m, n, k, a, lda, tau)
}

// Sorgqr is SORGQR (sorgqr.f), Dorgqr in single precision.
func Sorgqr(m, n, k int, a []float32, lda int, tau []float32) error {
	return orgqr("SORGQR", m, n, k, a, lda, tau)
}

func orgqr[T ref.Float](name string, m, n, k int, a []T, lda int, tau []T) error {
	if p := check.Orgqr(m, n, k, lda); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.OrgqrLens(m, n, k, lda, len(a), len(tau)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Orgqr(ref.RefKern[T](), ref.NBQR, ref.NXQR, m, n, k, a, lda, tau)
	return nil
}
