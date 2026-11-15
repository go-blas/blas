// Package check validates BLAS arguments the way the reference routines do,
// plus the one thing the reference cannot: that each slice is long enough for
// what the call reads or writes.
//
// Every function returns the number of the first offending parameter, in the
// position the Fortran routine gives it (xerbla's INFO), or 0 when the call is
// valid. Packages blas and netlib wrap the number in a *blas.ArgError. A slice
// that is too short is reported against the slice's own parameter number.
//
// All size arithmetic is overflow-safe on 32-bit targets: a length that does
// not fit an int is longer than any slice, so it is reported as too short.
package check

import "math/bits"

// Transpose operation characters, as the Fortran routines take them.
const (
	NoTrans   = 'N'
	Trans     = 'T'
	ConjTrans = 'C'
)

// Upper returns c in upper case if it is an ASCII letter: the reference's LSAME
// ignores case, so 'n' and 'N' are the same flag. The routines apply it once at
// entry, before the checks.
func Upper(c byte) byte {
	if c >= 'a' && c <= 'z' {
		return c - 'a' + 'A'
	}
	return c
}

// ValidTrans reports whether c is one of the three transpose operations.
func ValidTrans(c byte) bool {
	return c == NoTrans || c == Trans || c == ConjTrans
}

// Need returns the least length of a slice holding a column-major rows x cols
// matrix with leading dimension ld: ld*(cols-1)+rows, and 0 for an empty
// matrix. ok is false when that length does not fit an int.
func Need(rows, cols, ld int) (n int, ok bool) {
	if rows == 0 || cols == 0 {
		return 0, true
	}
	hi, lo := bits.Mul(uint(ld), uint(cols-1))
	sum, carry := bits.Add(lo, uint(rows), 0)
	if hi != 0 || carry != 0 || sum > uint(maxInt) {
		return 0, false
	}
	return int(sum), true
}

// NeedVec returns the least length of a slice holding n elements spaced |inc|
// apart: 1+(n-1)*|inc|, and 0 for n <= 0. ok is false when it does not fit.
func NeedVec(n, inc int) (int, bool) {
	if n <= 0 {
		return 0, true
	}
	step := uint(inc)
	if inc < 0 {
		step = -step // |inc| as unsigned, also for the most negative int
	}
	hi, lo := bits.Mul(step, uint(n-1))
	sum, carry := bits.Add(lo, 1, 0)
	if hi != 0 || carry != 0 || sum > uint(maxInt) {
		return 0, false
	}
	return int(sum), true
}

const maxInt = int(^uint(0) >> 1)

// short reports whether a slice of length have cannot hold need elements, need
// being a result of Need or NeedVec.
func short(have, need int, ok bool) bool { return !ok || have < need }

// Gemm checks the dimension parameters of xGEMM(TRANSA, TRANSB, M, N, K, ALPHA,
// A, LDA, B, LDB, BETA, C, LDC), in the reference's order.
func Gemm(tA, tB byte, m, n, k, lda, ldb, ldc int) int {
	nrowA, nrowB := m, k
	if tA != NoTrans {
		nrowA = k
	}
	if tB != NoTrans {
		nrowB = n
	}
	switch {
	case !ValidTrans(tA): // guard
		return 1
	case !ValidTrans(tB): // guard
		return 2
	case m < 0: // guard
		return 3
	case n < 0: // guard
		return 4
	case k < 0: // guard
		return 5
	case lda < max(1, nrowA): // guard
		return 8
	case ldb < max(1, nrowB): // guard
		return 10
	case ldc < max(1, m): // guard
		return 13
	}
	return 0
}

// GemmLens checks the slice lengths of a GEMM whose dimensions Gemm accepted
// and that is not a quick return. readsAB is false when alpha is zero, in which
// case A and B are never read and may be empty.
func GemmLens(tA, tB byte, m, n, k, lda, ldb, ldc, lenA, lenB, lenC int, readsAB bool) int {
	rowsA, colsA := m, k
	if tA != NoTrans {
		rowsA, colsA = k, m
	}
	rowsB, colsB := k, n
	if tB != NoTrans {
		rowsB, colsB = n, k
	}
	if readsAB {
		nA, okA := Need(rowsA, colsA, lda)
		if short(lenA, nA, okA) { // guard
			return 7
		}
		nB, okB := Need(rowsB, colsB, ldb)
		if short(lenB, nB, okB) { // guard
			return 9
		}
	}
	nC, okC := Need(m, n, ldc)
	if short(lenC, nC, okC) { // guard
		return 12
	}
	return 0
}

// NeedStrided returns the least length of a slice holding a rows x cols matrix
// whose element (i, j) is at i*rs+j*cs: 1+(rows-1)*rs+(cols-1)*cs, and 0 for an
// empty matrix. rs and cs are at least 1. ok is false when it does not fit.
func NeedStrided(rows, cols, rs, cs int) (n int, ok bool) {
	if rows == 0 || cols == 0 {
		return 0, true
	}
	h1, l1 := bits.Mul(uint(rs), uint(rows-1))
	h2, l2 := bits.Mul(uint(cs), uint(cols-1))
	sum, c1 := bits.Add(l1, l2, 0)
	sum, c2 := bits.Add(sum, 1, 0)
	if h1 != 0 || h2 != 0 || c1 != 0 || c2 != 0 || sum > uint(maxInt) {
		return 0, false
	}
	return int(sum), true
}

// GemmStrided checks the dimension and stride parameters of xGEMMSTRIDED(M, N,
// K, ALPHA, A, RSA, CSA, B, RSB, CSB, BETA, C, RSC, CSC), in the order of its
// parameters: 1 M, 2 N, 3 K, 6 RSA, 7 CSA, 9 RSB, 10 CSB, 13 RSC, 14 CSC. Every
// stride of A and B is at least 1. C must be row-major (CSC = 1, RSC >=
// max(1, N)) or column-major (RSC = 1, CSC >= max(1, M)). A CSC below 1, and a
// column-major C whose CSC is below M, are reported against CSC (14); any other
// C (RSC below N, RSC below 1, neither stride 1) against RSC (13).
func GemmStrided(m, n, k, rsa, csa, rsb, csb, rsc, csc int) int {
	switch {
	case m < 0: // guard
		return 1
	case n < 0: // guard
		return 2
	case k < 0: // guard
		return 3
	case rsa < 1: // guard
		return 6
	case csa < 1: // guard
		return 7
	case rsb < 1: // guard
		return 9
	case csb < 1: // guard
		return 10
	case csc < 1: // guard
		return 14
	case csc == 1 && rsc < max(1, n): // guard
		return 13
	case csc != 1 && rsc != 1: // guard
		return 13
	case csc != 1 && csc < max(1, m): // guard
		return 14
	}
	return 0
}

// GemmStridedLens checks the slice lengths of a GEMMSTRIDED whose parameters
// GemmStrided accepted and that is not a quick return. readsAB is false when
// alpha is zero, in which case A and B are never read and may be empty.
func GemmStridedLens(m, n, k, rsa, csa, rsb, csb, rsc, csc, lenA, lenB, lenC int, readsAB bool) int {
	if readsAB {
		nA, okA := NeedStrided(m, k, rsa, csa)
		if short(lenA, nA, okA) { // guard
			return 5
		}
		nB, okB := NeedStrided(k, n, rsb, csb)
		if short(lenB, nB, okB) { // guard
			return 8
		}
	}
	nC, okC := NeedStrided(m, n, rsc, csc)
	if short(lenC, nC, okC) { // guard
		return 12
	}
	return 0
}

// Gemv checks the parameters of xGEMV(TRANS, M, N, ALPHA, A, LDA, X, INCX,
// BETA, Y, INCY), in the reference's order.
func Gemv(t byte, m, n, lda, incx, incy int) int {
	switch {
	case !ValidTrans(t): // guard
		return 1
	case m < 0: // guard
		return 2
	case n < 0: // guard
		return 3
	case lda < max(1, m): // guard
		return 6
	case incx == 0: // guard
		return 8
	case incy == 0: // guard
		return 11
	}
	return 0
}

// GemvLens checks the slice lengths of a GEMV that is not a quick return.
// readsAX is false when alpha is zero: A and x are then never read.
func GemvLens(t byte, m, n, lda, incx, incy, lenA, lenX, lenY int, readsAX bool) int {
	lenx, leny := n, m
	if t != NoTrans {
		lenx, leny = m, n
	}
	if readsAX {
		nA, okA := Need(m, n, lda)
		if short(lenA, nA, okA) { // guard
			return 5
		}
		nX, okX := NeedVec(lenx, incx)
		if short(lenX, nX, okX) { // guard
			return 7
		}
	}
	nY, okY := NeedVec(leny, incy)
	if short(lenY, nY, okY) { // guard
		return 10
	}
	return 0
}

// Vec1 checks one vector operand of a level 1 routine: x holds n elements
// spaced inc apart, and param is its parameter number. A non-positive n reads
// nothing and is valid.
func Vec1(param, n, inc, lenX int) int {
	need, ok := NeedVec(n, inc)
	if short(lenX, need, ok) { // guard
		return param
	}
	return 0
}
