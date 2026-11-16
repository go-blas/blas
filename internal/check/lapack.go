package check

// The checks of the LAPACK routines. As for BLAS, each function returns the
// number of the first offending parameter, in the position of the reference
// signature (xerbla's INFO is minus that number), or 0. A slice too short is
// reported against its own parameter number, after the quick returns.

// Getrf checks the dimension parameters of xGETRF(M, N, A, LDA, IPIV, INFO), in
// the reference's order (dgetrf.f): M 1, N 2, LDA 4.
func Getrf(m, n, lda int) int {
	switch {
	case m < 0: // guard
		return 1
	case n < 0: // guard
		return 2
	case lda < max(1, m): // guard
		return 4
	}
	return 0
}

// GetrfLens checks the slice lengths of a GETRF whose dimensions Getrf accepted
// and that is not a quick return: A 3, IPIV 5 (min(m, n) pivots are written).
func GetrfLens(m, n, lda, lenA, lenIpiv int) int {
	nA, okA := Need(m, n, lda)
	if short(lenA, nA, okA) { // guard
		return 3
	}
	if lenIpiv < min(m, n) { // guard
		return 5
	}
	return 0
}

// Getrs checks the dimension parameters of xGETRS(TRANS, N, NRHS, A, LDA, IPIV,
// B, LDB, INFO), in the reference's order (dgetrs.f): TRANS 1, N 2, NRHS 3, LDA
// 5, LDB 8.
func Getrs(trans byte, n, nrhs, lda, ldb int) int {
	switch {
	case !ValidTrans(trans): // guard
		return 1
	case n < 0: // guard
		return 2
	case nrhs < 0: // guard
		return 3
	case lda < max(1, n): // guard
		return 5
	case ldb < max(1, n): // guard
		return 8
	}
	return 0
}

// GetrsLens checks the slice lengths of a GETRS whose dimensions Getrs accepted
// and that is not a quick return: A 4, IPIV 6, B 7.
func GetrsLens(n, nrhs, lda, ldb, lenA, lenIpiv, lenB int) int {
	nA, okA := Need(n, n, lda)
	if short(lenA, nA, okA) { // guard
		return 4
	}
	if lenIpiv < n { // guard
		return 6
	}
	nB, okB := Need(n, nrhs, ldb)
	if short(lenB, nB, okB) { // guard
		return 7
	}
	return 0
}

// Pivots checks the values of the n pivots a GETRS applies: each must be a row
// index of the n x n problem, 1 to n, as dgetrf writes them. The reference does
// not test it (Fortran would read outside the array); an index out of range is
// reported against IPIV, parameter 6, as a slice too short is.
func Pivots(n int, ipiv []int) int {
	for _, p := range ipiv[:n] {
		if p < 1 || p > n { // guard
			return 6
		}
	}
	return 0
}

// Potrf checks the dimension parameters of xPOTRF(UPLO, N, A, LDA, INFO), in the
// reference's order (dpotrf.f): UPLO 1, N 2, LDA 4.
func Potrf(uplo byte, n, lda int) int {
	switch {
	case !validUplo(uplo): // guard
		return 1
	case n < 0: // guard
		return 2
	case lda < max(1, n): // guard
		return 4
	}
	return 0
}

// PotrfLens checks the slice length of a POTRF whose dimensions Potrf accepted
// and that is not a quick return: A 3.
func PotrfLens(n, lda, lenA int) int {
	nA, okA := Need(n, n, lda)
	if short(lenA, nA, okA) { // guard
		return 3
	}
	return 0
}

// Potrs checks the dimension parameters of xPOTRS(UPLO, N, NRHS, A, LDA, B, LDB,
// INFO), in the reference's order (dpotrs.f): UPLO 1, N 2, NRHS 3, LDA 5, LDB 7.
func Potrs(uplo byte, n, nrhs, lda, ldb int) int {
	switch {
	case !validUplo(uplo): // guard
		return 1
	case n < 0: // guard
		return 2
	case nrhs < 0: // guard
		return 3
	case lda < max(1, n): // guard
		return 5
	case ldb < max(1, n): // guard
		return 7
	}
	return 0
}

// PotrsLens checks the slice lengths of a POTRS whose dimensions Potrs accepted
// and that is not a quick return: A 4, B 6.
func PotrsLens(n, nrhs, lda, ldb, lenA, lenB int) int {
	nA, okA := Need(n, n, lda)
	if short(lenA, nA, okA) { // guard
		return 4
	}
	nB, okB := Need(n, nrhs, ldb)
	if short(lenB, nB, okB) { // guard
		return 6
	}
	return 0
}
