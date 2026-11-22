package lapack

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// qrNB and qrNX are the block size and the crossover of Geqrf and Orgqr, ormNB
// the block size of Ormqr: the values ILAENV answers in the reference (32, 128,
// 32). Variables only so that the tests can make small matrices span many panels.
var (
	qrNB  = ref.NBQR
	qrNX  = ref.NXQR
	ormNB = ref.NBORM
)

// tr, uplo and diag are the BLAS flags for the booleans of ref.Kern.
func tr(notA bool) blas.Transpose {
	if notA {
		return blas.NoTrans
	}
	return blas.Trans
}

func uplo(upper bool) blas.Uplo {
	if upper {
		return blas.Upper
	}
	return blas.Lower
}

func diag(nounit bool) blas.Diag {
	if nounit {
		return blas.NonUnit
	}
	return blas.Unit
}

// kern is the BLAS of package blas as the Kern the QR routines of package ref
// call: the blocked algorithms are the reference's, and their work is the packed
// Gemm and the Trmm; the rank-1 update of the reflectors is an Axpy per column
// (the reference's DGER, which skips the columns whose y is zero).
func (o *ops[T]) kern() *ref.Kern[T] {
	return &ref.Kern[T]{
		Gemm: func(notA, notB bool, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {
			must(o.gemm(tr(notA), tr(notB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
		},
		Trmm: func(upper, notA, nounit bool, m, n int, alpha T, a []T, lda int, b []T, ldb int) {
			must(o.trmm(blas.Right, uplo(upper), tr(notA), diag(nounit), m, n, alpha, a, lda, b, ldb))
		},
		Gemv: func(notA bool, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) {
			must(o.gemv(tr(notA), m, n, alpha, a, lda, x, incx, beta, y, incy))
		},
		Ger: func(m, n int, alpha T, x, y []T, a []T, lda int) {
			for j := 0; j < n; j++ {
				if y[j] != 0 {
					must(o.axpy(m, alpha*y[j], x, 1, a[j*lda:], 1))
				}
			}
		},
	}
}

var (
	dKern = dOps.kern()
	sKern = sOps.kern()
)

// Dgeqrf computes the QR factorization A = Q*R of the m x n column-major matrix
// a, as DGEQRF: R (upper trapezoidal, min(m, n) x n) overwrites the upper triangle
// of A, and Q = H(1) H(2) ... H(k), k = min(m, n), a product of elementary
// reflectors H(i) = I - tau(i)*v*v**T with v(1:i-1) = 0 and v(i) = 1, is stored
// below the diagonal: v(i+1:m) in A(i+1:m, i), the scalar in tau(i). Parameters,
// numbered as in the reference (which has a workspace WORK and its length LWORK
// as parameters 6 and 7: here it is allocated): m 1, n 2, a 3, lda 4, tau 5. a is
// checked against lda*(n-1)+m and tau against min(m, n), after the quick return
// min(m, n) = 0. See the package comment for the bound.
func Dgeqrf(m, n int, a []float64, lda int, tau []float64) error {
	return geqrf("DGEQRF", dKern, m, n, a, lda, tau)
}

// Sgeqrf is Dgeqrf in single precision (SGEQRF).
func Sgeqrf(m, n int, a []float32, lda int, tau []float32) error {
	return geqrf("SGEQRF", sKern, m, n, a, lda, tau)
}

func geqrf[T real](name string, kn *ref.Kern[T], m, n int, a []T, lda int, tau []T) error {
	if p := check.Geqrf(m, n, lda); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.GeqrfLens(m, n, lda, len(a), len(tau)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Geqrf(kn, qrNB, qrNX, m, n, a, lda, tau)
	return nil
}

// Dormqr overwrites the m x n matrix c with Q*C (side Left, trans NoTrans),
// Q**T*C (Left, Trans), C*Q (Right, NoTrans) or C*Q**T (Right, Trans), as DORMQR,
// where Q = H(1) H(2) ... H(k) is the orthogonal matrix of order nq (m for Left, n
// for Right) that Dgeqrf represents by its k reflectors in the columns of a (nq x
// k) and tau. Parameters: side 1, trans 2, m 3, n 4, k 5, a 6, lda 7, tau 8, c 9,
// ldc 10. trans is NoTrans or Trans only (a real Q has no conjugate transpose, and
// the reference refuses 'C' as parameter 2). k is 0 to nq. a is checked against
// lda*(k-1)+nq, tau against k and c against ldc*(n-1)+m, after the quick return
// m = 0, n = 0 or k = 0. a is used in place of a copy of its diagonal: the
// diagonal elements are set to 1 during the call and restored, so a must not be
// read by another goroutine meanwhile.
func Dormqr(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau []float64, c []float64, ldc int) error {
	return ormqr("DORMQR", dKern, side, trans, m, n, k, a, lda, tau, c, ldc)
}

// Sormqr is Dormqr in single precision (SORMQR).
func Sormqr(side blas.Side, trans blas.Transpose, m, n, k int, a []float32, lda int, tau []float32, c []float32, ldc int) error {
	return ormqr("SORMQR", sKern, side, trans, m, n, k, a, lda, tau, c, ldc)
}

func ormqr[T real](name string, kn *ref.Kern[T], sd blas.Side, trans blas.Transpose, m, n, k int, a []T, lda int, tau []T, c []T, ldc int) error {
	sd = blas.Side(check.Upper(byte(sd)))                                   // LSAME ignores case
	trans = blas.Transpose(check.Upper(byte(trans)))                        // LSAME ignores case
	if p := check.Ormqr(byte(sd), byte(trans), m, n, k, lda, ldc); p != 0 { // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 || k == 0 { // guard
		return nil
	}
	if p := check.OrmqrLens(byte(sd), m, n, k, lda, ldc, len(a), len(tau), len(c)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Ormqr(kn, ormNB, sd == blas.Left, trans == blas.NoTrans, m, n, k, a, lda, tau, c, ldc)
	return nil
}

// Dorgqr generates the m x n matrix Q with orthonormal columns, the first n
// columns of Q = H(1) H(2) ... H(k) of order m, as DORGQR, in place: on entry the
// first k columns of a hold the reflectors as Dgeqrf left them (columns k+1 to n
// are not referenced), on exit a is Q. Parameters: m 1, n 2, k 3, a 4, lda 5, tau
// 6; n is 0 to m, k 0 to n. a is checked against lda*(n-1)+m and tau against k,
// after the quick return n = 0. With k = 0, a becomes the first n columns of the
// identity. See the package comment for the bound.
func Dorgqr(m, n, k int, a []float64, lda int, tau []float64) error {
	return orgqr("DORGQR", dKern, m, n, k, a, lda, tau)
}

// Sorgqr is Dorgqr in single precision (SORGQR).
func Sorgqr(m, n, k int, a []float32, lda int, tau []float32) error {
	return orgqr("SORGQR", sKern, m, n, k, a, lda, tau)
}

func orgqr[T real](name string, kn *ref.Kern[T], m, n, k int, a []T, lda int, tau []T) error {
	if p := check.Orgqr(m, n, k, lda); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.OrgqrLens(m, n, k, lda, len(a), len(tau)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Orgqr(kn, qrNB, qrNX, m, n, k, a, lda, tau)
	return nil
}
