package lapack

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// Jobz says whether Syev computes the eigenvectors, as the JOBZ flag of DSYEV.
type Jobz byte

const (
	// EigenvaluesOnly is JOBZ = 'N'.
	EigenvaluesOnly Jobz = 'N'
	// EigenvaluesAndVectors is JOBZ = 'V'.
	EigenvaluesAndVectors Jobz = 'V'
)

// Compz is the COMPZ flag of DSTEQR: what Steqr does with Z.
type Compz byte

const (
	// CompzNone is COMPZ = 'N': only the eigenvalues are computed, Z is not
	// referenced.
	CompzNone Compz = 'N'
	// CompzOriginal is COMPZ = 'V': Z holds, on entry, the orthogonal matrix that
	// reduced a symmetric matrix to this tridiagonal one (Dorgtr's output), and
	// returns the eigenvectors of that matrix.
	CompzOriginal Compz = 'V'
	// CompzIdentity is COMPZ = 'I': Z is set to the identity first, and returns the
	// eigenvectors of the tridiagonal matrix.
	CompzIdentity Compz = 'I'
)

// Dsytrd reduces the n x n symmetric column-major matrix a, of which only the
// triangle uplo names is referenced, to symmetric tridiagonal form T = Q**T*A*Q,
// as DSYTRD: d (n) receives the diagonal of T and e (n-1) its off-diagonal; Q =
// H(n-1) ... H(2) H(1) (Upper) or H(1) H(2) ... H(n-1) (Lower), a product of
// elementary reflectors H(i) = I - tau(i)*v*v**T, is stored in the triangle
// opposite to the one e and d replace (Lower: v(i+2:n) in A(i+2:n, i), with
// v(i+1) = 1; Upper: v(1:i-1) in A(1:i-1, i+1), with v(i) = 1) and tau (n-1).
// Parameters, numbered as in the reference (which has WORK and LWORK as 8 and 9;
// here the workspace is allocated): uplo 1, n 2, a 3, lda 4, d 5, e 6, tau 7. a is
// checked against lda*(n-1)+n, d against n, e and tau against n-1, after the quick
// return n = 0. See the package comment for the bound.
func Dsytrd(uplo blas.Uplo, n int, a []float64, lda int, d, e, tau []float64) error {
	return sytrd("DSYTRD", dKern, uplo, n, a, lda, d, e, tau)
}

// Ssytrd is Dsytrd in single precision (SSYTRD).
func Ssytrd(uplo blas.Uplo, n int, a []float32, lda int, d, e, tau []float32) error {
	return sytrd("SSYTRD", sKern, uplo, n, a, lda, d, e, tau)
}

func sytrd[T real](name string, kn *ref.Kern[T], up blas.Uplo, n int, a []T, lda int, d, e, tau []T) error {
	up = blas.Uplo(check.Upper(byte(up)))           // LSAME ignores case
	if p := check.Sytrd(byte(up), n, lda); p != 0 { // guard
		return argErr(name, p)
	}
	if n == 0 { // guard
		return nil
	}
	if p := check.SytrdLens(n, lda, len(a), len(d), len(e), len(tau)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Sytrd(kn, ref.NBTRD, ref.NXTRD, up == blas.Upper, n, a, lda, d, e, tau)
	return nil
}

// Dorgtr generates the n x n orthogonal matrix Q that Dsytrd defined as a product
// of n-1 elementary reflectors, in place (DORGTR): uplo is the one Dsytrd was
// called with, a and tau its output. Parameters: uplo 1, n 2, a 3, lda 4, tau 5. a
// is checked against lda*(n-1)+n and tau against n-1, after the quick return n =
// 0. See the package comment for the bound.
func Dorgtr(uplo blas.Uplo, n int, a []float64, lda int, tau []float64) error {
	return orgtr("DORGTR", dKern, uplo, n, a, lda, tau)
}

// Sorgtr is Dorgtr in single precision (SORGTR).
func Sorgtr(uplo blas.Uplo, n int, a []float32, lda int, tau []float32) error {
	return orgtr("SORGTR", sKern, uplo, n, a, lda, tau)
}

func orgtr[T real](name string, kn *ref.Kern[T], up blas.Uplo, n int, a []T, lda int, tau []T) error {
	up = blas.Uplo(check.Upper(byte(up)))           // LSAME ignores case
	if p := check.Orgtr(byte(up), n, lda); p != 0 { // guard
		return argErr(name, p)
	}
	if n == 0 { // guard
		return nil
	}
	if p := check.OrgtrLens(n, lda, len(a), len(tau)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Orgtr(kn, ref.NBQL, ref.NXQL, qrNB, qrNX, up == blas.Upper, n, a, lda, tau)
	return nil
}

// Dsteqr computes all the eigenvalues, and optionally the eigenvectors, of the n x
// n symmetric tridiagonal matrix with diagonal d (n) and off-diagonal e (n-1) by
// the implicit QL or QR method, as DSTEQR. compz is CompzNone, CompzOriginal (z
// holds Dorgtr's Q on entry and returns the eigenvectors of the original matrix)
// or CompzIdentity (z returns those of the tridiagonal matrix). On exit d holds
// the eigenvalues in increasing order and e is destroyed. Parameters: compz 1, n
// 2, d 3, e 4, z 5, ldz 6 (the reference's WORK is 7, allocated). info is 0, i > 0
// when the algorithm did not converge in 30*n iterations (i is the number of
// off-diagonal elements not converged to zero: the eigenvalues in d are then those
// of a nearby problem and not sorted), or -p for an illegal parameter p, with err
// the *blas.ArgError. z is checked against ldz*(n-1)+n unless compz is CompzNone,
// after the quick return n = 0. See the package comment for the bound.
func Dsteqr(compz Compz, n int, d, e, z []float64, ldz int) (info int, err error) {
	return steqr("DSTEQR", compz, n, d, e, z, ldz)
}

// Ssteqr is Dsteqr in single precision (SSTEQR).
func Ssteqr(compz Compz, n int, d, e, z []float32, ldz int) (info int, err error) {
	return steqr("SSTEQR", compz, n, d, e, z, ldz)
}

func steqr[T real](name string, compz Compz, n int, d, e, z []T, ldz int) (int, error) {
	compz = Compz(check.Upper(byte(compz)))            // LSAME ignores case
	if p := check.Steqr(byte(compz), n, ldz); p != 0 { // guard
		return -p, argErr(name, p)
	}
	// n = 0 needs no quick return: the length checks ask for nothing and
	// ref.Steqr does no work (TestSteqrEmpty).
	if p := check.SteqrLens(byte(compz), n, ldz, len(d), len(e), len(z)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	return ref.Steqr(compzCode(compz), n, d, e, z, ldz), nil
}

// compzCode is the ICOMPZ of DSTEQR: 0 for 'N', 1 for 'V', 2 for 'I'.
func compzCode(c Compz) int {
	switch c {
	case CompzOriginal:
		return 1
	case CompzIdentity:
		return 2
	}
	return 0
}

// Dsterf computes all the eigenvalues of the n x n symmetric tridiagonal matrix
// with diagonal d (n) and off-diagonal e (n-1) by the Pal-Walker-Kahan variant of
// the QL or QR algorithm, as DSTERF. On exit d holds the eigenvalues in
// increasing order and e is destroyed. Parameters: n 1, d 2, e 3. info is 0, i > 0
// when the algorithm did not converge in 30*n iterations (i is the number of
// off-diagonal elements not converged to zero), or -p for an illegal parameter p,
// with err the *blas.ArgError. d is checked against n and e against n-1, after the
// quick return n <= 1. See the package comment for the bound.
func Dsterf(n int, d, e []float64) (info int, err error) {
	return sterf("DSTERF", n, d, e)
}

// Ssterf is Dsterf in single precision (SSTERF).
func Ssterf(n int, d, e []float32) (info int, err error) {
	return sterf("SSTERF", n, d, e)
}

func sterf[T real](name string, n int, d, e []T) (int, error) {
	if p := check.Sterf(n); p != 0 { // guard
		return -p, argErr(name, p)
	}
	if n <= 1 { // guard
		return 0, nil
	}
	if p := check.SterfLens(n, len(d), len(e)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	return ref.Sterf(n, d, e), nil
}

// Dsyev computes all the eigenvalues, and optionally the eigenvectors, of the n x
// n symmetric column-major matrix a, of which only the triangle uplo names is
// referenced, as DSYEV: the reduction to tridiagonal form (Dsytrd), then Dsterf
// for the eigenvalues alone or Dorgtr and Dsteqr for the eigenvectors. The matrix
// is scaled when its largest element is outside [sqrt(safmin/eps),
// sqrt(eps/safmin)], and the eigenvalues unscaled. On exit w (n) holds the
// eigenvalues in increasing order; with EigenvaluesAndVectors, a holds the
// orthonormal eigenvectors, column j for w(j); otherwise the triangle uplo names
// of a is destroyed. Parameters: jobz 1, uplo 2, n 3, a 4, lda 5, w 6 (the
// reference's WORK and LWORK, 7 and 8, are allocated). info is 0, i > 0 when the
// tridiagonal eigensolver did not converge (i off-diagonal elements not having
// converged to zero; the eigenvalues w(1:i) are then not reliable), or -p for an
// illegal parameter p, with err the *blas.ArgError. a is checked against
// lda*(n-1)+n and w against n, after the quick return n = 0. See the package
// comment for the bound.
func Dsyev(jobz Jobz, uplo blas.Uplo, n int, a []float64, lda int, w []float64) (info int, err error) {
	return syev("DSYEV", dKern, jobz, uplo, n, a, lda, w)
}

// Ssyev is Dsyev in single precision (SSYEV).
func Ssyev(jobz Jobz, uplo blas.Uplo, n int, a []float32, lda int, w []float32) (info int, err error) {
	return syev("SSYEV", sKern, jobz, uplo, n, a, lda, w)
}

func syev[T real](name string, kn *ref.Kern[T], jobz Jobz, up blas.Uplo, n int, a []T, lda int, w []T) (int, error) {
	jobz = Jobz(check.Upper(byte(jobz)))                       // LSAME ignores case
	up = blas.Uplo(check.Upper(byte(up)))                      // LSAME ignores case
	if p := check.Syev(byte(jobz), byte(up), n, lda); p != 0 { // guard
		return -p, argErr(name, p)
	}
	if n == 0 { // guard
		return 0, nil
	}
	if p := check.SyevLens(n, lda, len(a), len(w)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	return ref.Syev(kn, jobz == EigenvaluesAndVectors, up == blas.Upper, n, a, lda, w), nil
}
