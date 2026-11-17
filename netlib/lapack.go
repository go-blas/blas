package netlib

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// The LAPACK routines of the repository's first LAPACK release, as plain-Go ports
// of Reference-LAPACK 3.12.0 (tag v3.12.0, SRC): dgetrf.f, dgetrf2.f, dgetrs.f,
// dlaswp.f, dpotrf.f, dpotrf2.f, dpotrs.f and the single precision twins. They
// have the signatures and the conventions of package lapack, which documents
// them: a factorization returns (info, err), INFO as LAPACK sets it (negative
// for an illegal argument, with a *blas.ArgError; positive for a singular U or a
// matrix that is not positive definite, which is a result and no error); a solve
// returns an error only. The block size is the reference's, 64 (ilaenv.f).

// Dgetrf is DGETRF (dgetrf.f, dgetrf2.f, dlaswp.f): the LU factorization with
// partial pivoting A = P*L*U of the m x n column-major matrix a. See
// lapack.Dgetrf.
func Dgetrf(m, n int, a []float64, lda int, ipiv []int) (info int, err error) {
	return getrf("DGETRF", ref.NB, m, n, a, lda, ipiv)
}

// Sgetrf is SGETRF (sgetrf.f), Dgetrf in single precision.
func Sgetrf(m, n int, a []float32, lda int, ipiv []int) (info int, err error) {
	return getrf("SGETRF", ref.NB, m, n, a, lda, ipiv)
}

func getrf[T ref.Float](name string, nb, m, n int, a []T, lda int, ipiv []int) (int, error) {
	if p := check.Getrf(m, n, lda); p != 0 { // guard
		return -p, argErr(name, p)
	}
	// Not a guard line: without this return, ref.Getrf2 on an empty matrix
	// recurses without end, and a mutant that dies of a stack overflow is found
	// by the runtime, not by an assertion, so guardmut cannot classify it.
	if m == 0 || n == 0 {
		return 0, nil
	}
	if p := check.GetrfLens(m, n, lda, len(a), len(ipiv)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	return ref.Getrf(nb, m, n, a, lda, ipiv), nil
}

// Dgetrs is DGETRS (dgetrs.f): solves A*X = B or A**T*X = B with the factors
// Dgetrf computed. See lapack.Dgetrs.
func Dgetrs(trans blas.Transpose, n, nrhs int, a []float64, lda int, ipiv []int, b []float64, ldb int) error {
	return getrs("DGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb)
}

// Sgetrs is SGETRS (sgetrs.f), Dgetrs in single precision.
func Sgetrs(trans blas.Transpose, n, nrhs int, a []float32, lda int, ipiv []int, b []float32, ldb int) error {
	return getrs("SGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb)
}

func getrs[T ref.Float](name string, trans blas.Transpose, n, nrhs int, a []T, lda int, ipiv []int, b []T, ldb int) error {
	trans = blas.Transpose(check.Upper(byte(trans)))              // LSAME ignores case
	if p := check.Getrs(byte(trans), n, nrhs, lda, ldb); p != 0 { // guard
		return argErr(name, p)
	}
	if n == 0 || nrhs == 0 { // guard
		return nil
	}
	if p := check.GetrsLens(n, nrhs, lda, ldb, len(a), len(ipiv), len(b)); p != 0 { // guard
		return argErr(name, p)
	}
	if p := check.Pivots(n, ipiv); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Getrs(trans == blas.NoTrans, n, nrhs, a, lda, ipiv, b, ldb)
	return nil
}

// Dpotrf is DPOTRF (dpotrf.f, dpotrf2.f): the Cholesky factorization A = U**T*U
// or A = L*L**T of the n x n symmetric positive definite column-major matrix a,
// of which only the triangle uplo names is referenced and overwritten. See
// lapack.Dpotrf.
func Dpotrf(uplo blas.Uplo, n int, a []float64, lda int) (info int, err error) {
	return potrf("DPOTRF", ref.NB, uplo, n, a, lda)
}

// Spotrf is SPOTRF (spotrf.f), Dpotrf in single precision.
func Spotrf(uplo blas.Uplo, n int, a []float32, lda int) (info int, err error) {
	return potrf("SPOTRF", ref.NB, uplo, n, a, lda)
}

func potrf[T ref.Float](name string, nb int, uplo blas.Uplo, n int, a []T, lda int) (int, error) {
	uplo = blas.Uplo(check.Upper(byte(uplo)))         // LSAME ignores case
	if p := check.Potrf(byte(uplo), n, lda); p != 0 { // guard
		return -p, argErr(name, p)
	}
	// Not a guard line, for the reason given at getrf: ref.Potrf2 recurses
	// without end on n = 0.
	if n == 0 {
		return 0, nil
	}
	if p := check.PotrfLens(n, lda, len(a)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	return ref.Potrf(nb, uplo == blas.Upper, n, a, lda), nil
}

// Dpotrs is DPOTRS (dpotrs.f): solves A*X = B with the factor Dpotrf computed.
// See lapack.Dpotrs.
func Dpotrs(uplo blas.Uplo, n, nrhs int, a []float64, lda int, b []float64, ldb int) error {
	return potrs("DPOTRS", uplo, n, nrhs, a, lda, b, ldb)
}

// Spotrs is SPOTRS (spotrs.f), Dpotrs in single precision.
func Spotrs(uplo blas.Uplo, n, nrhs int, a []float32, lda int, b []float32, ldb int) error {
	return potrs("SPOTRS", uplo, n, nrhs, a, lda, b, ldb)
}

func potrs[T ref.Float](name string, uplo blas.Uplo, n, nrhs int, a []T, lda int, b []T, ldb int) error {
	uplo = blas.Uplo(check.Upper(byte(uplo)))                    // LSAME ignores case
	if p := check.Potrs(byte(uplo), n, nrhs, lda, ldb); p != 0 { // guard
		return argErr(name, p)
	}
	if n == 0 || nrhs == 0 { // guard
		return nil
	}
	if p := check.PotrsLens(n, nrhs, lda, ldb, len(a), len(b)); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Potrs(uplo == blas.Upper, n, nrhs, a, lda, b, ldb)
	return nil
}
