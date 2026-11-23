package check

// The checks of the symmetric level 2 routines Symv and Syr2, and of the symmetric
// eigensolver's LAPACK routines. Same conventions as check.go and lapack.go.

// Symv checks the dimension parameters of xSYMV(UPLO, N, ALPHA, A, LDA, X, INCX,
// BETA, Y, INCY), in the reference's order (dsymv.f): UPLO 1, N 2, LDA 5, INCX 7,
// INCY 10.
func Symv(uplo byte, n, lda, incx, incy int) int {
	switch {
	case !validUplo(uplo): // guard
		return 1
	case n < 0: // guard
		return 2
	case lda < max(1, n): // guard
		return 5
	case incx == 0: // guard
		return 7
	case incy == 0: // guard
		return 10
	}
	return 0
}

// SymvLens checks the slice lengths of a SYMV that is not a quick return: A 4, X 6,
// Y 9. readsAX is false when alpha is zero: A and x are then never read.
func SymvLens(n, lda, incx, incy, lenA, lenX, lenY int, readsAX bool) int {
	if readsAX {
		nA, okA := Need(n, n, lda)
		if short(lenA, nA, okA) { // guard
			return 4
		}
		nX, okX := NeedVec(n, incx)
		if short(lenX, nX, okX) { // guard
			return 6
		}
	}
	nY, okY := NeedVec(n, incy)
	if short(lenY, nY, okY) { // guard
		return 9
	}
	return 0
}

// Syr2 checks the dimension parameters of xSYR2(UPLO, N, ALPHA, X, INCX, Y, INCY,
// A, LDA), in the reference's order (dsyr2.f): UPLO 1, N 2, INCX 5, INCY 7, LDA 9.
func Syr2(uplo byte, n, incx, incy, lda int) int {
	switch {
	case !validUplo(uplo): // guard
		return 1
	case n < 0: // guard
		return 2
	case incx == 0: // guard
		return 5
	case incy == 0: // guard
		return 7
	case lda < max(1, n): // guard
		return 9
	}
	return 0
}

// Syr2Lens checks the slice lengths of a SYR2 that is not a quick return: X 4, Y 6,
// A 8.
func Syr2Lens(n, incx, incy, lda, lenX, lenY, lenA int) int {
	nX, okX := NeedVec(n, incx)
	if short(lenX, nX, okX) { // guard
		return 4
	}
	nY, okY := NeedVec(n, incy)
	if short(lenY, nY, okY) { // guard
		return 6
	}
	nA, okA := Need(n, n, lda)
	if short(lenA, nA, okA) { // guard
		return 8
	}
	return 0
}

// Sytrd checks the dimension parameters of xSYTRD(UPLO, N, A, LDA, D, E, TAU, WORK,
// LWORK, INFO), in the reference's order (dsytrd.f): UPLO 1, N 2, LDA 4. The
// workspace has no counterpart here: it is allocated.
func Sytrd(uplo byte, n, lda int) int {
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

// SytrdLens checks the slice lengths of a SYTRD that is not a quick return (n = 0):
// A 3, D 5 (n), E 6 (n-1), TAU 7 (n-1).
func SytrdLens(n, lda, lenA, lenD, lenE, lenTau int) int {
	nA, okA := Need(n, n, lda)
	if short(lenA, nA, okA) { // guard
		return 3
	}
	if lenD < n { // guard
		return 5
	}
	if lenE < n-1 { // guard
		return 6
	}
	if lenTau < n-1 { // guard
		return 7
	}
	return 0
}

// Orgtr checks the dimension parameters of xORGTR(UPLO, N, A, LDA, TAU, WORK,
// LWORK, INFO), in the reference's order (dorgtr.f): UPLO 1, N 2, LDA 4.
func Orgtr(uplo byte, n, lda int) int { return Sytrd(uplo, n, lda) }

// OrgtrLens checks the slice lengths of an ORGTR that is not a quick return (n =
// 0): A 3, TAU 5 (n-1).
func OrgtrLens(n, lda, lenA, lenTau int) int {
	nA, okA := Need(n, n, lda)
	if short(lenA, nA, okA) { // guard
		return 3
	}
	if lenTau < n-1 { // guard
		return 5
	}
	return 0
}

// Steqr checks the dimension parameters of xSTEQR(COMPZ, N, D, E, Z, LDZ, WORK,
// INFO), in the reference's order (dsteqr.f): COMPZ 1 ('N', 'V' or 'I'), N 2, LDZ 6
// (at least 1, and at least n when eigenvectors are wanted). The flag is upper
// case.
func Steqr(compz byte, n, ldz int) int {
	switch {
	case compz != 'N' && compz != 'V' && compz != 'I': // guard
		return 1
	case n < 0: // guard
		return 2
	case ldz < 1 || (compz != 'N' && ldz < max(1, n)): // guard
		return 6
	}
	return 0
}

// SteqrLens checks the slice lengths of a STEQR whose dimensions Steqr accepted and
// that is not a quick return (n = 0): D 3 (n), E 4 (n-1), Z 5 (n x n, unless compz
// is 'N', when it is not referenced).
func SteqrLens(compz byte, n, ldz, lenD, lenE, lenZ int) int {
	if lenD < n { // guard
		return 3
	}
	if lenE < n-1 { // guard
		return 4
	}
	if compz != 'N' {
		nZ, okZ := Need(n, n, ldz)
		if short(lenZ, nZ, okZ) { // guard
			return 5
		}
	}
	return 0
}

// Sterf checks the dimension parameter of xSTERF(N, D, E, INFO): N 1.
func Sterf(n int) int {
	if n < 0 { // guard
		return 1
	}
	return 0
}

// SterfLens checks the slice lengths of a STERF whose dimension Sterf accepted and
// that is not a quick return (n <= 1): D 2 (n), E 3 (n-1).
func SterfLens(n, lenD, lenE int) int {
	if lenD < n { // guard
		return 2
	}
	if lenE < n-1 { // guard
		return 3
	}
	return 0
}

// Syev checks the dimension parameters of xSYEV(JOBZ, UPLO, N, A, LDA, W, WORK,
// LWORK, INFO), in the reference's order (dsyev.f): JOBZ 1 ('N' or 'V'), UPLO 2, N
// 3, LDA 5. Flags are upper case.
func Syev(jobz, uplo byte, n, lda int) int {
	switch {
	case jobz != 'N' && jobz != 'V': // guard
		return 1
	case !validUplo(uplo): // guard
		return 2
	case n < 0: // guard
		return 3
	case lda < max(1, n): // guard
		return 5
	}
	return 0
}

// SyevLens checks the slice lengths of a SYEV that is not a quick return (n = 0):
// A 4, W 6 (n).
func SyevLens(n, lda, lenA, lenW int) int {
	nA, okA := Need(n, n, lda)
	if short(lenA, nA, okA) { // guard
		return 4
	}
	if lenW < n { // guard
		return 6
	}
	return 0
}
