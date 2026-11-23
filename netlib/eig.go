package netlib

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
	"github.com/go-blas/blas/lapack"
)

// The symmetric eigensolver, as plain-Go ports of Reference-LAPACK 3.12.0 (tag
// v3.12.0, SRC): dsyev.f, dsytrd.f, dsytd2.f, dlatrd.f, dorgtr.f, dorgql.f,
// dorg2l.f, dsteqr.f, dsterf.f and what they call (dlartg.f90, dlae2.f, dlaev2.f,
// dlasr.f, dlasrt.f, dlascl.f, dlanst.f, dlansy.f, and the level 1 and 2 BLAS of
// this package), and the single precision twins, with the signatures and the
// conventions of package lapack, which documents them. The block sizes are the
// reference's (ilaenv.f): 32 and 32 for Sytrd, 32 and 128 for Orgql, 32 and 128
// for Orgqr.

// Dsyev is DSYEV: the eigenvalues and optionally the eigenvectors of a symmetric
// matrix. See lapack.Dsyev.
func Dsyev(jobz lapack.Jobz, uplo blas.Uplo, n int, a []float64, lda int, w []float64) (info int, err error) {
	return syev("DSYEV", jobz, uplo, n, a, lda, w)
}

// Ssyev is SSYEV (ssyev.f), Dsyev in single precision.
func Ssyev(jobz lapack.Jobz, uplo blas.Uplo, n int, a []float32, lda int, w []float32) (info int, err error) {
	return syev("SSYEV", jobz, uplo, n, a, lda, w)
}

func syev[T ref.Float](name string, jobz lapack.Jobz, up blas.Uplo, n int, a []T, lda int, w []T) (int, error) {
	jobz = lapack.Jobz(check.Upper(byte(jobz)))                // LSAME ignores case
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
	return ref.Syev(ref.RefKern[T](), jobz == lapack.EigenvaluesAndVectors, up == blas.Upper, n, a, lda, w), nil
}

// Dsytrd is DSYTRD: the reduction of a symmetric matrix to tridiagonal form. See
// lapack.Dsytrd.
func Dsytrd(uplo blas.Uplo, n int, a []float64, lda int, d, e, tau []float64) error {
	return sytrd("DSYTRD", uplo, n, a, lda, d, e, tau)
}

// Ssytrd is SSYTRD (ssytrd.f), Dsytrd in single precision.
func Ssytrd(uplo blas.Uplo, n int, a []float32, lda int, d, e, tau []float32) error {
	return sytrd("SSYTRD", uplo, n, a, lda, d, e, tau)
}

func sytrd[T ref.Float](name string, up blas.Uplo, n int, a []T, lda int, d, e, tau []T) error {
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
	ref.Sytrd(ref.RefKern[T](), ref.NBTRD, ref.NXTRD, up == blas.Upper, n, a, lda, d, e, tau)
	return nil
}

// Dorgtr is DORGTR: the orthogonal matrix of the reduction Dsytrd made. See
// lapack.Dorgtr.
func Dorgtr(uplo blas.Uplo, n int, a []float64, lda int, tau []float64) error {
	return orgtr("DORGTR", uplo, n, a, lda, tau)
}

// Sorgtr is SORGTR (sorgtr.f), Dorgtr in single precision.
func Sorgtr(uplo blas.Uplo, n int, a []float32, lda int, tau []float32) error {
	return orgtr("SORGTR", uplo, n, a, lda, tau)
}

func orgtr[T ref.Float](name string, up blas.Uplo, n int, a []T, lda int, tau []T) error {
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
	ref.Orgtr(ref.RefKern[T](), ref.NBQL, ref.NXQL, ref.NBQR, ref.NXQR, up == blas.Upper, n, a, lda, tau)
	return nil
}

// Dsteqr is DSTEQR: the eigenvalues and optionally the eigenvectors of a symmetric
// tridiagonal matrix by the implicit QL or QR method. See lapack.Dsteqr.
func Dsteqr(compz lapack.Compz, n int, d, e, z []float64, ldz int) (info int, err error) {
	return steqr("DSTEQR", compz, n, d, e, z, ldz)
}

// Ssteqr is SSTEQR (ssteqr.f), Dsteqr in single precision.
func Ssteqr(compz lapack.Compz, n int, d, e, z []float32, ldz int) (info int, err error) {
	return steqr("SSTEQR", compz, n, d, e, z, ldz)
}

func steqr[T ref.Float](name string, compz lapack.Compz, n int, d, e, z []T, ldz int) (int, error) {
	compz = lapack.Compz(check.Upper(byte(compz)))     // LSAME ignores case
	if p := check.Steqr(byte(compz), n, ldz); p != 0 { // guard
		return -p, argErr(name, p)
	}
	// n = 0 needs no quick return (see lapack.steqr, TestSteqrEmpty).
	if p := check.SteqrLens(byte(compz), n, ldz, len(d), len(e), len(z)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	code := 0
	switch compz {
	case lapack.CompzOriginal:
		code = 1
	case lapack.CompzIdentity:
		code = 2
	}
	return ref.Steqr(code, n, d, e, z, ldz), nil
}

// Dsterf is DSTERF: the eigenvalues of a symmetric tridiagonal matrix by the
// Pal-Walker-Kahan variant of the QL or QR algorithm. See lapack.Dsterf.
func Dsterf(n int, d, e []float64) (info int, err error) {
	return sterf("DSTERF", n, d, e)
}

// Ssterf is SSTERF (ssterf.f), Dsterf in single precision.
func Ssterf(n int, d, e []float32) (info int, err error) {
	return sterf("SSTERF", n, d, e)
}

func sterf[T ref.Float](name string, n int, d, e []T) (int, error) {
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
