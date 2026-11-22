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

// Geqrf checks the dimension parameters of xGEQRF(M, N, A, LDA, TAU, WORK, LWORK,
// INFO), in the reference's order (dgeqrf.f): M 1, N 2, LDA 4. The workspace
// (WORK, LWORK) has no counterpart here: it is allocated.
func Geqrf(m, n, lda int) int {
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

// GeqrfLens checks the slice lengths of a GEQRF whose dimensions Geqrf accepted
// (min(m, n) = 0 needs nothing of either slice): A 3, TAU 5 (min(m, n) scalars
// are written).
func GeqrfLens(m, n, lda, lenA, lenTau int) int {
	nA, okA := Need(m, n, lda)
	if short(lenA, nA, okA) { // guard
		return 3
	}
	if lenTau < min(m, n) { // guard
		return 5
	}
	return 0
}

// Ormqr checks the dimension parameters of xORMQR(SIDE, TRANS, M, N, K, A, LDA,
// TAU, C, LDC, WORK, LWORK, INFO), in the reference's order (dormqr.f): SIDE 1,
// TRANS 2 ('N' or 'T' only, a real Q has no conjugate transpose: 'C' is refused),
// M 3, N 4, K 5 (0 to the order nq of Q, M for Left, N for Right), LDA 7, LDC 10.
// Flags are upper case.
func Ormqr(side, trans byte, m, n, k, lda, ldc int) int {
	nq := n
	if side == 'L' {
		nq = m
	}
	switch {
	case !validSide(side): // guard
		return 1
	case trans != 'N' && trans != 'T': // guard
		return 2
	case m < 0: // guard
		return 3
	case n < 0: // guard
		return 4
	case k < 0 || k > nq: // guard
		return 5
	case lda < max(1, nq): // guard
		return 7
	case ldc < max(1, m): // guard
		return 10
	}
	return 0
}

// OrmqrLens checks the slice lengths of an ORMQR whose dimensions Ormqr accepted
// and that is not a quick return (m, n or k = 0): A 6 (nq x k), TAU 8 (k), C 9.
func OrmqrLens(side byte, m, n, k, lda, ldc, lenA, lenTau, lenC int) int {
	nq := n
	if side == 'L' {
		nq = m
	}
	nA, okA := Need(nq, k, lda)
	if short(lenA, nA, okA) { // guard
		return 6
	}
	if lenTau < k { // guard
		return 8
	}
	nC, okC := Need(m, n, ldc)
	if short(lenC, nC, okC) { // guard
		return 9
	}
	return 0
}

// Orgqr checks the dimension parameters of xORGQR(M, N, K, A, LDA, TAU, WORK,
// LWORK, INFO), in the reference's order (dorgqr.f): M 1, N 2 (0 to M), K 3 (0 to
// N), LDA 5.
func Orgqr(m, n, k, lda int) int {
	switch {
	case m < 0: // guard
		return 1
	case n < 0 || n > m: // guard
		return 2
	case k < 0 || k > n: // guard
		return 3
	case lda < max(1, m): // guard
		return 5
	}
	return 0
}

// OrgqrLens checks the slice lengths of an ORGQR whose dimensions Orgqr accepted
// accepted (n = 0 needs nothing of either slice, k <= n being checked): A 4 (m x n), TAU 6 (k).
func OrgqrLens(m, n, k, lda, lenA, lenTau int) int {
	nA, okA := Need(m, n, lda)
	if short(lenA, nA, okA) { // guard
		return 4
	}
	if lenTau < k { // guard
		return 6
	}
	return 0
}
