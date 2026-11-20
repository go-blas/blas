// Package blas is a pure-Go BLAS with the reference (netlib) signatures and
// semantics: column-major matrices, leading dimensions, increments, and the
// scalar rules of the reference routines (beta = 0 never reads C, alpha = 0
// never reads A or B, the quick returns). Go's bounds checks cannot protect a
// caller that binds this package to Fortran, so no routine panics on its
// arguments: each returns an error, nil or a *ArgError.
//
// # Scope
//
// This release has the real routines in single and double precision: the level
// 3 routines Sgemm/Dgemm, Strsm/Dtrsm, Strmm/Dtrmm, Ssyrk/Dsyrk, Ssyr2k/Dsyr2k
// and Ssymm/Dsymm, Sgemv, Dgemv and the level 1 routines Sdot/Ddot, Saxpy/Daxpy,
// Sscal/Dscal, Snrm2/Dnrm2, Sasum/Dasum and Isamax/Idamax. Complex routines
// (Cgemm, Zgemm, Zdotc, ...) and the rest of LAPACK follow (the LU and Cholesky
// factorizations and solves are in package lapack, since v0.4.0), under the same naming
// convention, with complex64 and complex128 slices; nothing here will have to
// change for them. ConjTrans is in Transpose from the start: for real types it
// means Trans, as in netlib.
//
// # Contract
//
// The contract of each routine is a stated error bound, asserted against a
// math/big oracle of at least 256 bits, and not equality with another float64
// code. With u the unit roundoff (2^-53, or 2^-24 for float32) and
// gamma_k = k*u/(1-k*u):
//
//	Gemm   |C^ - C| <= gamma_{k+2} (|alpha| |op(A)| |op(B)| + |beta| |C0|)   elementwise
//	Trsm   |op(A) X^ - alpha B| <= gamma_{s+4} (|op(A)| |X^| + |alpha B|)   elementwise, s the order of A: a residual (backward error) bound
//	Trmm   |B^ - alpha op(A) B| <= gamma_{s+3} |alpha| |op(A)| |B|          elementwise
//	Syrk   |C^ - C| <= gamma_{k+2} (|alpha| |op(A)| |op(A)^T| + |beta| |C0|)   elementwise, on the triangle written
//	Syr2k  |C^ - C| <= gamma_{2k+2} (|alpha| (|op(A)||op(B)^T| + |op(B)||op(A)^T|) + |beta| |C0|)   likewise
//	Symm   |C^ - C| <= gamma_{s+2} (|alpha| |A| |B| + |beta| |C0|)           elementwise (|B| |A| for side Right)
//	Gemv   |y^ - y| <= gamma_{r+2} (|alpha| |op(A)| |x|      + |beta| |y0|)   elementwise, r the length of the sum
//	Dot    |d^ - d| <= gamma_n |x|.|y|
//	Axpy   |y^ - y| <= gamma_2 (|alpha x| + |y0|)             elementwise
//	Scal   |x^ - x| <= gamma_1 |alpha x|                     elementwise
//	Nrm2   |s^ - s| <= gamma_{n+2} s
//	Asum   |s^ - s| <= gamma_n sum|x|
//	Iamax  exact: the first index of the largest |x|
//
// FMA, blocking, packing and reordered sums are all allowed inside that bound,
// and are used: Dgemm and Sgemm run on the packed, cache-blocked, multicore
// SIMD kernel moved from go-ndarray/ndarray (AVX2/FMA on amd64, NEON on arm64,
// VSX on ppc64le, LASX on loong64, RVV on riscv64, the vector facility on
// s390x). Bit identity with netlib is not part of the contract; package netlib
// is the option for callers who want it: plain-Go ports of the reference
// routines, in the reference's loop order, without FMA.
package blas

import "strconv"

// Transpose says whether a matrix operand is used as is, transposed, or
// conjugate-transposed. The values are the characters the Fortran routines take.
type Transpose byte

const (
	NoTrans   Transpose = 'N'
	Trans     Transpose = 'T'
	ConjTrans Transpose = 'C' // for real types, the same as Trans
)

// Uplo says which triangle of a symmetric, Hermitian or triangular matrix is
// referenced. It is type-independent, like Diag and Side, and is taken by the
// level 3 routines Trsm, Trmm, Syrk, Syr2k and Symm.
type Uplo byte

const (
	Upper Uplo = 'U'
	Lower Uplo = 'L'
)

// Diag says whether a triangular matrix has a unit diagonal.
type Diag byte

const (
	NonUnit Diag = 'N'
	Unit    Diag = 'U'
)

// Side says on which side of a product a matrix operand stands.
type Side byte

const (
	Left  Side = 'L'
	Right Side = 'R'
)

// ArgError reports an invalid argument the way the reference's xerbla does:
// by the routine and the number of the parameter, counted from 1 in the
// reference signature (so Param is 8 for the LDA of DGEMM). LAPACK's info is
// -Param.
//
// Where the reference has no test, because Fortran cannot know the length of an
// array, a slice too short for what the call reads or writes is reported against
// the slice's own parameter number.
type ArgError struct {
	Routine string // upper case, as xerbla prints it: "DGEMM"
	Param   int
}

func (e *ArgError) Error() string {
	return "blas: " + e.Routine + ": parameter " + strconv.Itoa(e.Param) + " had an illegal value"
}

// Info is the LAPACK convention for the same fact: -Param.
func (e *ArgError) Info() int { return -e.Param }

// argErr wraps the nonzero parameter number p returned by the checks of package
// check as an *ArgError for routine.
func argErr(routine string, p int) error {
	return &ArgError{Routine: routine, Param: p}
}
