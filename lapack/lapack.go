// Package lapack is the first LAPACK of the pure-Go BLAS of go-blas: the LU and
// Cholesky factorizations and their solves, in single and double precision, with
// the reference (netlib, LAPACK 3.12) signatures, column-major, so that a
// Fortran ABI shim (julia's libblastrampoline, for one) can call them exactly:
//
//	Sgetrf/Dgetrf  LU factorization with partial pivoting, A = P*L*U
//	Sgetrs/Dgetrs  solves A*X = B or A**T*X = B with the factors of getrf
//	Spotrf/Dpotrf  Cholesky factorization, A = U**T*U or A = L*L**T
//	Spotrs/Dpotrs  solves A*X = B with the factor of potrf
//
// # Conventions
//
// As in package blas: column-major storage with leading dimensions; flags
// (Transpose, Uplo) are case-insensitive; no routine panics on its arguments,
// each returns an *blas.ArgError numbered like xerbla (the parameter's position in
// the reference signature, LAPACK's INFO = -Param), a slice too short for what
// the call reads or writes being reported against its own parameter number, with
// overflow-safe arithmetic; the reference's quick returns (m = 0 or n = 0, ...)
// come before the slice-length checks.
//
// # The INFO of a factorization
//
// A positive INFO is a result, not an error: Getrf's U is exactly singular
// (U(i,i) = 0, the factorization having been completed), Potrf's matrix is not
// positive definite (the leading minor of order INFO is not, a NaN counting as
// not). The factorizations therefore return (info int, err error), info being
// exactly what LAPACK leaves in INFO: 0, i > 0, or -p < 0 for an illegal
// parameter p, in which case err is the *blas.ArgError whose Param is p and
// whose Info() is info. err is non-nil if and only if info < 0, so a caller that
// wants LAPACK's INFO reads info alone, and a Go caller tests err and then info.
// The solves (Getrs, Potrs) have no positive INFO: they return an error only,
// with Info() as the negative INFO. The pivots, ipiv, are []int and 1-based, as
// in LAPACK: row i was interchanged with row ipiv[i-1]. Getrs also refuses (as
// parameter 6) a pivot outside 1..n, which the reference would read out of
// bounds.
//
// # The contract
//
// An error bound per routine, asserted by the tests against a math/big oracle of
// 256 bits, and not equality with netlib (package netlib has the ports of the
// reference routines for that, in the reference's loop order, without FMA). With
// u the unit roundoff, gamma_k = k*u/(1-k*u), |.| elementwise, L^ and U^ the
// computed factors (L^ unit lower trapezoidal m x r, U^ upper trapezoidal r x n,
// r = min(m, n)), C^ the computed Cholesky factor and X^ the computed solution:
//
//	Getrf  |P A - L^ U^|     <= gamma_{3r+2} |L^| |U^|
//	Getrs  |P B - L^ U^ X^|  <= gamma_{2n+8} |L^| |U^| |X^|     (NoTrans; Trans: |B - U^T L^T P X^| <= gamma_{2n+8} |U^T| |L^T| |P X^|)
//	Potrf  |A - C^T C^|      <= gamma_{3n+3} |C^T| |C^|         (Upper; Lower: |A - C^ C^T|), on the triangle referenced
//	Potrs  |B - C^T C^ X^|   <= gamma_{2n+8} |C^T| |C^| |X^|     (Upper; Lower: |B - C^ C^T X^|)
//
// With INFO > 0 from Getrf the bound holds for the factorization as completed;
// with INFO > 0 from Potrf it holds on the leading block of order INFO - 1. A
// factorization followed by a solve satisfies |A X^ - B| <= (gamma_{3r+2} +
// gamma_{2n+8}) |P^T L^| |U^| |X^| by the triangle inequality. The bounds are
// first-order results widened to a non-asymptotic gamma form; docs/design.md,
// decision 10, gives the sources opened, what they state, and what the blocked
// algorithms rest on.
//
// # Algorithms
//
// Both factorizations are recursive (Toledo's recursive LU, as the reference's
// xGETRF2 and xPOTRF2) and blocked over the BLAS of package blas, so that the
// work is the packed GEMM: Getrf splits the columns in halves, the update being a
// Trsm and a Gemm, Potrf the order in halves, with a Trsm and a Syrk, down to a
// panel of a few columns factored by plain loops. No kernel was written. The
// solves are two Trsm calls and the row interchanges.
package lapack

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
)

type real interface{ float32 | float64 }

func argErr(routine string, p int) error {
	return &blas.ArgError{Routine: routine, Param: p}
}

// must panics on an error of a BLAS call. The calls this package makes are on
// views of matrices it has checked, with valid flags and dimensions: an error
// there is a bug of this package, not of the caller's arguments, and must not
// pass unnoticed.
func must(err error) {
	if err != nil {
		panic("lapack: internal BLAS call failed: " + err.Error())
	}
}

// ops are the three BLAS routines the algorithms call, for one element type.
type ops[T real] struct {
	gemm func(tA, tB blas.Transpose, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) error
	trsm func(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int) error
	syrk func(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha T, a []T, lda int, beta T, c []T, ldc int) error
}

var (
	dOps = &ops[float64]{blas.Dgemm, blas.Dtrsm, blas.Dsyrk}
	sOps = &ops[float32]{blas.Sgemm, blas.Strsm, blas.Ssyrk}
)

// Dgetrf computes the LU factorization with partial pivoting of the m x n
// column-major matrix a, A = P*L*U, as DGETRF: L is unit lower triangular (m x
// min(m, n)), U upper triangular (min(m, n) x n), both overwrite A, and ipiv
// (min(m, n) elements, 1-based) records the interchanges. Parameters, numbered
// as in the reference: m 1, n 2, a 3, lda 4, ipiv 5. info is 0, i > 0 when U(i,i)
// is exactly zero (the factorization is completed, but U is singular and a solve
// with it would divide by zero), or -p for an illegal parameter p, with err the
// *blas.ArgError. a is checked against lda*(n-1)+m and ipiv against min(m, n),
// after the quick return m = 0 or n = 0. See the package comment for the bound.
func Dgetrf(m, n int, a []float64, lda int, ipiv []int) (info int, err error) {
	return getrf("DGETRF", dOps, m, n, a, lda, ipiv)
}

// Sgetrf is Dgetrf in single precision (SGETRF).
func Sgetrf(m, n int, a []float32, lda int, ipiv []int) (info int, err error) {
	return getrf("SGETRF", sOps, m, n, a, lda, ipiv)
}

func getrf[T real](name string, o *ops[T], m, n int, a []T, lda int, ipiv []int) (int, error) {
	if p := check.Getrf(m, n, lda); p != 0 { // guard
		return -p, argErr(name, p)
	}
	// Not a guard line: with m = 0 or n = 0 the recursion below does nothing and
	// returns INFO 0, so disabling this return changes no behaviour, and a mutant
	// that cannot be told from the original cannot be killed.
	if m == 0 || n == 0 {
		return 0, nil
	}
	if p := check.GetrfLens(m, n, lda, len(a), len(ipiv)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	return getrfRec(o, m, n, a, lda, ipiv), nil
}

// Dgetrs solves A*X = B (trans NoTrans) or A**T*X = B (Trans, ConjTrans) for the
// n x nrhs matrix B, in place, with the LU factorization Dgetrf computed, as
// DGETRS. Parameters: trans 1, n 2, nrhs 3, a 4, lda 5, ipiv 6, b 7, ldb 8. a is
// checked against lda*(n-1)+n, ipiv against n and its values against 1..n (all
// parameter 6), b against ldb*(nrhs-1)+n, after the quick return n = 0 or nrhs
// = 0. A zero on the diagonal of U is not tested for, as in the reference.
func Dgetrs(trans blas.Transpose, n, nrhs int, a []float64, lda int, ipiv []int, b []float64, ldb int) error {
	return getrs("DGETRS", dOps, trans, n, nrhs, a, lda, ipiv, b, ldb)
}

// Sgetrs is Dgetrs in single precision (SGETRS).
func Sgetrs(trans blas.Transpose, n, nrhs int, a []float32, lda int, ipiv []int, b []float32, ldb int) error {
	return getrs("SGETRS", sOps, trans, n, nrhs, a, lda, ipiv, b, ldb)
}

func getrs[T real](name string, o *ops[T], trans blas.Transpose, n, nrhs int, a []T, lda int, ipiv []int, b []T, ldb int) error {
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
	getrsRec(o, trans == blas.NoTrans, n, nrhs, a, lda, ipiv, b, ldb)
	return nil
}

// Dpotrf computes the Cholesky factorization of the n x n symmetric positive
// definite column-major matrix a, A = U**T*U (uplo Upper) or A = L*L**T (Lower),
// as DPOTRF. Only the triangle uplo names is referenced, and overwritten by the
// factor; the other triangle is not touched. Parameters: uplo 1, n 2, a 3, lda 4.
// info is 0, i > 0 when the leading minor of order i is not positive definite
// (the factorization is not completed; a NaN on the diagonal counts as not
// positive), or -p for an illegal parameter p, with err the *blas.ArgError. a is
// checked against lda*(n-1)+n, after the quick return n = 0. See the package
// comment for the bound.
func Dpotrf(uplo blas.Uplo, n int, a []float64, lda int) (info int, err error) {
	return potrf("DPOTRF", dOps, uplo, n, a, lda)
}

// Spotrf is Dpotrf in single precision (SPOTRF).
func Spotrf(uplo blas.Uplo, n int, a []float32, lda int) (info int, err error) {
	return potrf("SPOTRF", sOps, uplo, n, a, lda)
}

func potrf[T real](name string, o *ops[T], uplo blas.Uplo, n int, a []T, lda int) (int, error) {
	uplo = blas.Uplo(check.Upper(byte(uplo)))         // LSAME ignores case
	if p := check.Potrf(byte(uplo), n, lda); p != 0 { // guard
		return -p, argErr(name, p)
	}
	// Not a guard line, for the reason given at getrf: n = 0 recurses on nothing.
	if n == 0 {
		return 0, nil
	}
	if p := check.PotrfLens(n, lda, len(a)); p != 0 { // guard
		return -p, argErr(name, p)
	}
	return potrfRec(o, uplo == blas.Upper, n, a, lda), nil
}

// Dpotrs solves A*X = B for the n x nrhs matrix B, in place, with the Cholesky
// factor Dpotrf computed, as DPOTRS. Parameters: uplo 1, n 2, nrhs 3, a 4, lda 5,
// b 6, ldb 7. a is checked against lda*(n-1)+n, b against ldb*(nrhs-1)+n, after
// the quick return n = 0 or nrhs = 0.
func Dpotrs(uplo blas.Uplo, n, nrhs int, a []float64, lda int, b []float64, ldb int) error {
	return potrs("DPOTRS", dOps, uplo, n, nrhs, a, lda, b, ldb)
}

// Spotrs is Dpotrs in single precision (SPOTRS).
func Spotrs(uplo blas.Uplo, n, nrhs int, a []float32, lda int, b []float32, ldb int) error {
	return potrs("SPOTRS", sOps, uplo, n, nrhs, a, lda, b, ldb)
}

func potrs[T real](name string, o *ops[T], uplo blas.Uplo, n, nrhs int, a []T, lda int, b []T, ldb int) error {
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
	potrsRec(o, uplo == blas.Upper, n, nrhs, a, lda, b, ldb)
	return nil
}
