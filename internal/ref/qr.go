package ref

import "math"

// Ports of the reference LAPACK 3.12.0 routines (Reference-LAPACK/lapack, tag
// v3.12.0, SRC) of the QR factorization: dgeqrf.f, dgeqr2.f, dlarfg.f, dlarf.f,
// dlarft.f, dlarfb.f, dormqr.f, dorm2r.f, dorgqr.f, dorg2r.f, dlapy2.f, iladlc.f,
// iladlr.f, and the level 2 BLAS dger.f and dtrmv.f they call, with their single
// precision twins, written once, generic over float32 and float64. Same rules as
// the rest of this package: the loops are the reference's, 0-based indices into
// column-major slices, no fused multiply-add, the arguments already checked, and
// the reference's quick returns the caller's. Only what QR calls is ported: DLARFT
// and DLARFB for DIRECT = 'F' and STOREV = 'C' (the reflectors of a QR
// factorization are stored by columns, forward), DTRMV for an upper, non-unit,
// non-transposed matrix, DGER and DGEMV with unit increments. The workspace is
// unbounded (LWORK large enough), as a caller that sized it by a query has it, and
// is allocated here; the block sizes are the parameters that ILAENV would answer.
//
// The calls the blocked algorithms make to the level 3 BLAS, and the rank-1
// update and the matrix-vector product of DLARF, go through a Kern, so that the
// reference's own BLAS ports (RefKern) or another implementation (package lapack's)
// can be plugged in: the factorization algorithms are one code.

const (
	// NBQR and NXQR are what ILAENV answers (ilaenv.f, tag v3.12.0) for DGEQRF and
	// DORGQR: NB = 32 (ISPEC 1), NX = 128 (ISPEC 3), single and double; NBMIN is 2.
	NBQR = 32
	NXQR = 128
	// NBORM is the block size DORMQR uses: min(NBMAX = 64, ILAENV answer 32).
	NBORM = 32
)

// Kern is the BLAS a QR routine calls. Each function is the reference routine
// with its flags as booleans (notA: A is not transposed; upper, nounit), unit
// increments for Gemv and Ger, and no quick return: the methods below carry the
// ones of the reference BLAS. Trmm is the right-sided one (B := alpha*B*op(A)),
// the only one DLARFB calls.
type Kern[T Float] struct {
	Gemm func(notA, notB bool, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int)
	Trmm func(upper, notA, nounit bool, m, n int, alpha T, a []T, lda int, b []T, ldb int)
	Gemv func(notA bool, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int)
	Ger  func(m, n int, alpha T, x, y []T, a []T, lda int)
	// Symv, Syr2 and Syr2k are the BLAS the symmetric eigensolver (eig.go) adds:
	// unit increments, Syr2k with the flags as booleans (upper, notA).
	Symv  func(upper bool, n int, alpha T, a []T, lda int, x []T, beta T, y []T)
	Syr2  func(upper bool, n int, alpha T, x, y []T, a []T, lda int)
	Syr2k func(upper, notA bool, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int)
}

// RefKern is the Kern of this package's ports of the reference BLAS.
func RefKern[T Float]() *Kern[T] {
	return &Kern[T]{
		Gemm: Gemm[T],
		Trmm: func(upper, notA, nounit bool, m, n int, alpha T, a []T, lda int, b []T, ldb int) {
			Trmm(false, upper, notA, nounit, m, n, alpha, a, lda, b, ldb)
		},
		Gemv: Gemv[T],
		Ger:  Ger[T],
		Symv: func(upper bool, n int, alpha T, a []T, lda int, x []T, beta T, y []T) {
			Symv(upper, n, alpha, a, lda, x, 1, beta, y, 1)
		},
		Syr2: func(upper bool, n int, alpha T, x, y []T, a []T, lda int) {
			Syr2(upper, n, alpha, x, 1, y, 1, a, lda)
		},
		Syr2k: Syr2k[T],
	}
}

// The calls below carry the quick returns of the reference BLAS, so that a Kern
// need not.
func (k *Kern[T]) gemm(notA, notB bool, m, n, kk int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {
	if m == 0 || n == 0 || ((alpha == 0 || kk == 0) && beta == 1) {
		return
	}
	k.Gemm(notA, notB, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (k *Kern[T]) trmm(upper, notA, nounit bool, m, n int, alpha T, a []T, lda int, b []T, ldb int) {
	if m == 0 || n == 0 {
		return
	}
	k.Trmm(upper, notA, nounit, m, n, alpha, a, lda, b, ldb)
}

func (k *Kern[T]) gemv(notA bool, m, n int, alpha T, a []T, lda int, x []T, beta T, y []T) {
	if m == 0 || n == 0 || (alpha == 0 && beta == 1) {
		return
	}
	k.Gemv(notA, m, n, alpha, a, lda, x, 1, beta, y, 1)
}

func (k *Kern[T]) ger(m, n int, alpha T, x, y []T, a []T, lda int) {
	if m == 0 || n == 0 || alpha == 0 {
		return
	}
	k.Ger(m, n, alpha, x, y, a, lda)
}

// Ger is xGER with unit increments: A := alpha*x*y**T + A, the m x n matrix a,
// the columns whose y is zero skipped (dger.f), alpha != 0 and m, n > 0.
func Ger[T Float](m, n int, alpha T, x, y []T, a []T, lda int) {
	for j := 0; j < n; j++ {
		if y[j] != 0 {
			temp := T(alpha * y[j])
			for i := 0; i < m; i++ {
				a[i+j*lda] = a[i+j*lda] + T(x[i]*temp)
			}
		}
	}
}

// trmvUN is xTRMV('Upper', 'No transpose', 'Non-unit', n, a, lda, x, 1), n > 0:
// x := A*x for the upper triangular n x n matrix a (dtrmv.f).
func trmvUN[T Float](n int, a []T, lda int, x []T) {
	for j := 0; j < n; j++ {
		if x[j] != 0 {
			temp := x[j]
			for i := 0; i < j; i++ {
				x[i] = x[i] + T(temp*a[i+j*lda])
			}
			x[j] = T(x[j] * a[j+j*lda])
		}
	}
}

// Iladlc is ILADLC (iladlc.f): the index (1-based) of the last non-zero column
// of the m x n matrix a, 0 if there is none; m > 0 unless n = 0.
func Iladlc[T Float](m, n int, a []T, lda int) int {
	// The reference's quick test of the two corners is left out: the scan below
	// gives the same index (0 for n = 0) and reads nothing when n = 0.
	for j := n; j >= 1; j-- {
		for i := 0; i < m; i++ {
			if a[i+(j-1)*lda] != 0 {
				return j
			}
		}
	}
	return 0
}

// Iladlr is ILADLR (iladlr.f): the index (1-based) of the last non-zero row of
// the m x n matrix a, 0 if there is none; n > 0 unless m = 0.
func Iladlr[T Float](m, n int, a []T, lda int) int {
	// The reference's quick test of the corners is left out: the scan below gives
	// the same index (0 for m = 0) and reads nothing when m = 0 or n = 0.
	// Scan up each column tracking the last zero row seen.
	last := 0
	for j := 0; j < n; j++ {
		i := m
		for i >= 1 && a[(i-1)+j*lda] == 0 {
			i--
		}
		last = max(last, i)
	}
	return last
}

// huge is DLAMCH('Overflow') / SLAMCH('Overflow').
func huge[T Float]() T {
	largest := math.MaxFloat32
	if bits64[T]() {
		largest = math.MaxFloat64
	}
	return T(largest)
}

// sign is the Fortran SIGN(a, b): |a| with the sign of b, a negative zero b
// counting as negative (as gfortran, and the IEEE signbit).
func sign[T Float](a, b T) T {
	a = abs(a)
	if math.Signbit(float64(b)) {
		return -a
	}
	return a
}

// Lapy2 is xLAPY2 (dlapy2.f, 3.12): sqrt(x**2 + y**2) avoiding unnecessary
// overflow, a NaN argument being returned (y's if both are).
func Lapy2[T Float](x, y T) T {
	xNaN, yNaN := x != x, y != y
	var r T
	if xNaN {
		r = x
	}
	if yNaN {
		r = y
	}
	if !(xNaN || yNaN) {
		xa, ya := abs(x), abs(y)
		w, z := max(xa, ya), min(xa, ya)
		if z == 0 || w > huge[T]() {
			r = w
		} else {
			q := T(z / w)
			r = T(w * sqrt(T(1+T(q*q))))
		}
	}
	return r
}

// larfgSafmin is DLAMCH('S') / DLAMCH('E') of dlarfg.f: the smallest normal
// number over the relative machine epsilon, 2^-1022 / 2^-53 for float64 and
// 2^-126 / 2^-24 for float32.
func larfgSafmin[T Float]() T {
	if bits64[T]() {
		return T(math.Ldexp(1, -1022+53))
	}
	return T(math.Ldexp(1, -126+24))
}

// Larfg is xLARFG (dlarfg.f): it generates the elementary reflector H = I -
// tau*v*v**T of order n such that H*[alpha; x] = [beta; 0], H**T*H = I, v(1) = 1.
// On exit *alpha is beta and x (n-1 elements, stride incx) holds v(2:n); tau is
// returned.
func Larfg[T Float](n int, alpha *T, x []T, incx int) (tau T) {
	// N <= 1 is TAU = 0 in the reference; here Nrm2 of no element is 0, which
	// is the H = I branch below.
	xnorm := Nrm2(n-1, x, incx)
	if xnorm == 0 {
		// H = I.
		return 0
	}
	al := *alpha
	beta := -sign(Lapy2(al, xnorm), al)
	safmin := larfgSafmin[T]()
	knt := 0
	if abs(beta) < safmin {
		// xnorm, beta may be inaccurate; scale x and recompute them.
		rsafmn := T(1 / safmin)
		for {
			knt++
			Scal(n-1, rsafmn, x, incx)
			beta = T(beta * rsafmn)
			al = T(al * rsafmn)
			if !(abs(beta) < safmin && knt < 20) {
				break
			}
		}
		// New beta is at most 1, at least safmin.
		xnorm = Nrm2(n-1, x, incx)
		beta = -sign(Lapy2(al, xnorm), al)
	}
	tau = T(T(beta-al) / beta)
	Scal(n-1, T(1/T(al-beta)), x, incx)
	// If alpha is subnormal, it may lose relative accuracy.
	for j := 0; j < knt; j++ {
		beta = T(beta * safmin)
	}
	*alpha = beta
	return tau
}

// Larf is xLARF with incv = 1 (dlarf.f): it applies the elementary reflector H =
// I - tau*v*v**T to the m x n matrix c, from the left (left) or the right, using
// work (n elements for left, m for right) as workspace. The scan for the last
// non-zero of v and the last non-zero column (row) of c is the reference's, which
// limits the work done.
func Larf[T Float](k *Kern[T], left bool, m, n int, v []T, tau T, c []T, ldc int, work []T) {
	lastv, lastc := 0, 0
	if tau != 0 {
		// Set up variables for scanning v: lastv begins pointing to the end of v.
		if left {
			lastv = m
		} else {
			lastv = n
		}
		// Look for the last non-zero row in v.
		for lastv > 0 && v[lastv-1] == 0 {
			lastv--
		}
		switch {
		case lastv == 0:
			// Nothing to scan: the operations below are null.
		case left:
			// Scan for the last non-zero column in c(1:lastv,:).
			lastc = Iladlc(lastv, n, c, ldc)
		default:
			// Scan for the last non-zero row in c(:,1:lastv).
			lastc = Iladlr(m, lastv, c, ldc)
		}
	}
	// Note that lastc = 0 renders the BLAS operations null.
	if lastv > 0 {
		if left {
			// w(1:lastc,1) := c(1:lastv,1:lastc)**T * v(1:lastv,1), then
			// c(1:lastv,1:lastc) := c - v * w**T.
			k.gemv(false, lastv, lastc, 1, c, ldc, v, 0, work)
			k.ger(lastv, lastc, -tau, v, work, c, ldc)
		} else {
			// w(1:lastc,1) := c(1:lastc,1:lastv) * v(1:lastv,1), then
			// c(1:lastc,1:lastv) := c - w * v**T.
			k.gemv(true, lastc, lastv, 1, c, ldc, v, 0, work)
			k.ger(lastc, lastv, -tau, work, v, c, ldc)
		}
	}
}

// Geqr2 is xGEQR2 (dgeqr2.f): the unblocked QR factorization of the m x n matrix
// a, A = Q*R, Q = H(1) H(2) ... H(min(m,n)) the product of elementary reflectors
// stored below the diagonal with the scalars in tau. work has n elements.
func Geqr2[T Float](k *Kern[T], m, n int, a []T, lda int, tau []T, work []T) {
	for i := 1; i <= min(m, n); i++ {
		// Generate elementary reflector H(i) to annihilate A(i+1:m,i).
		ii := (i - 1) + (i-1)*lda
		tau[i-1] = Larfg(m-i+1, &a[ii], a[(min(i+1, m)-1)+(i-1)*lda:], 1)
		if i < n {
			// Apply H(i) to A(i:m,i+1:n) from the left.
			aii := a[ii]
			a[ii] = 1
			Larf(k, true, m-i+1, n-i, a[ii:], tau[i-1], a[(i-1)+i*lda:], lda, work)
			a[ii] = aii
		}
	}
}

// Larft is xLARFT (dlarft.f) for DIRECT = 'F', STOREV = 'C': it forms the k x k
// upper triangular factor t of the block reflector H = H(1) H(2) ... H(k) =
// I - V*T*V**T, the n x k unit lower trapezoidal v holding the vectors by
// columns, n >= k, n > 0.
func Larft[T Float](kn *Kern[T], n, k int, v []T, ldv int, tau []T, t []T, ldt int) {
	prevlastv := n
	for i := 1; i <= k; i++ {
		prevlastv = max(i, prevlastv)
		if tau[i-1] == 0 {
			// H(i) = I.
			for j := 1; j <= i; j++ {
				t[(j-1)+(i-1)*ldt] = 0
			}
			continue
		}
		// General case.
		lastv := n
		for lastv > i && v[(lastv-1)+(i-1)*ldv] == 0 {
			lastv--
		}
		for j := 1; j <= i-1; j++ {
			t[(j-1)+(i-1)*ldt] = T(-tau[i-1] * v[(i-1)+(j-1)*ldv])
		}
		j := min(lastv, prevlastv)
		// T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)**T * V(i:j,i).
		kn.gemv(false, j-i, i-1, -tau[i-1], v[i:], ldv, v[i+(i-1)*ldv:], 1, t[(i-1)*ldt:])
		// T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i).
		if i > 1 {
			trmvUN(i-1, t, ldt, t[(i-1)*ldt:])
		}
		t[(i-1)+(i-1)*ldt] = tau[i-1]
		if i > 1 {
			prevlastv = max(prevlastv, lastv)
		} else {
			prevlastv = lastv
		}
	}
}

// Larfb is xLARFB (dlarfb.f) for DIRECT = 'F', STOREV = 'C': it applies the block
// reflector H = I - V*T*V**T (notrans) or its transpose H**T to the m x n matrix
// c from the left (left) or the right. v is nq x k unit lower trapezoidal (nq = m
// for left, n for right), t the k x k upper triangular factor of Larft, work an
// n x k (left) or m x k (right) matrix with leading dimension ldwork.
func Larfb[T Float](kn *Kern[T], left, notrans bool, m, n, k int, v []T, ldv int, t []T, ldt int, c []T, ldc int, work []T, ldwork int) {
	if m <= 0 || n <= 0 {
		return
	}
	if left {
		// Form H*C or H**T*C. W := C1**T (the first k rows of c).
		for j := 0; j < k; j++ {
			for i := 0; i < n; i++ {
				work[i+j*ldwork] = c[j+i*ldc]
			}
		}
		// W := W * V1.
		kn.trmm(false, true, false, n, k, 1, v, ldv, work, ldwork)
		if m > k {
			// W := W + C2**T * V2.
			kn.gemm(false, true, n, k, m-k, 1, c[k:], ldc, v[k:], ldv, 1, work, ldwork)
		}
		// W := W * T**T or W * T.
		kn.trmm(true, !notrans, true, n, k, 1, t, ldt, work, ldwork)
		if m > k {
			// C2 := C2 - V2 * W**T.
			kn.gemm(true, false, m-k, n, k, -1, v[k:], ldv, work, ldwork, 1, c[k:], ldc)
		}
		// W := W * V1**T.
		kn.trmm(false, false, false, n, k, 1, v, ldv, work, ldwork)
		// C1 := C1 - W**T.
		for j := 0; j < k; j++ {
			for i := 0; i < n; i++ {
				c[j+i*ldc] = c[j+i*ldc] - work[i+j*ldwork]
			}
		}
		return
	}
	// Form C*H or C*H**T. W := C1.
	for j := 0; j < k; j++ {
		for i := 0; i < m; i++ {
			work[i+j*ldwork] = c[i+j*ldc]
		}
	}
	// W := W * V1.
	kn.trmm(false, true, false, m, k, 1, v, ldv, work, ldwork)
	if n > k {
		// W := W + C2 * V2.
		kn.gemm(true, true, m, k, n-k, 1, c[k*ldc:], ldc, v[k:], ldv, 1, work, ldwork)
	}
	// W := W * T or W * T**T.
	kn.trmm(true, notrans, true, m, k, 1, t, ldt, work, ldwork)
	if n > k {
		// C2 := C2 - W * V2**T.
		kn.gemm(true, false, m, n-k, k, -1, work, ldwork, v[k:], ldv, 1, c[k*ldc:], ldc)
	}
	// W := W * V1**T.
	kn.trmm(false, false, false, m, k, 1, v, ldv, work, ldwork)
	// C1 := C1 - W.
	for j := 0; j < k; j++ {
		for i := 0; i < m; i++ {
			c[i+j*ldc] = c[i+j*ldc] - work[i+j*ldwork]
		}
	}
}

// Geqrf is xGEQRF (dgeqrf.f): the QR factorization of the m x n matrix a, m, n >=
// 1, blocked with panels of nb columns while more than nx remain (nb and nx as
// ILAENV answers: NBQR, NXQR), by Geqr2 and Larfb, the unblocked code for the
// rest. tau has min(m, n) elements.
func Geqrf[T Float](kn *Kern[T], nb, nx, m, n int, a []T, lda int, tau []T) {
	k := min(m, n)
	nbmin := 2
	if !(nb > 1 && nb < k) {
		nx = 0
	}
	blocked := nb >= nbmin && nb < k && nx < k
	ldwork := n
	wlen := n
	if blocked {
		wlen = ldwork * nb
	}
	work := make([]T, wlen)
	i := 1
	if blocked {
		// Use blocked code initially.
		for i = 1; i <= k-nx; i += nb {
			ib := min(k-i+1, nb)
			ii := (i - 1) + (i-1)*lda
			// Compute the QR factorization of the current block A(i:m,i:i+ib-1).
			Geqr2(kn, m-i+1, ib, a[ii:], lda, tau[i-1:], work)
			if i+ib <= n {
				// Form the triangular factor of the block reflector H = H(i) ... H(i+ib-1),
				// and apply H**T to A(i:m,i+ib:n) from the left.
				Larft(kn, m-i+1, ib, a[ii:], lda, tau[i-1:], work, ldwork)
				Larfb(kn, true, false, m-i+1, n-i-ib+1, ib, a[ii:], lda, work, ldwork, a[(i-1)+(i+ib-1)*lda:], lda, work[ib:], ldwork)
			}
		}
	}
	// Use unblocked code to factor the last or only block.
	if i <= k {
		Geqr2(kn, m-i+1, n-i+1, a[(i-1)+(i-1)*lda:], lda, tau[i-1:], work)
	}
}

// Org2r is xORG2R (dorg2r.f): it generates the m x n matrix Q with orthonormal
// columns, the first n columns of the product of k elementary reflectors H(1) ...
// H(k) of order m returned by Geqrf, in place, m >= n >= k >= 0, n > 0. work has n
// elements.
func Org2r[T Float](kn *Kern[T], m, n, k int, a []T, lda int, tau []T, work []T) {
	// Initialise columns k+1:n to columns of the unit matrix.
	for j := k + 1; j <= n; j++ {
		for l := 1; l <= m; l++ {
			a[(l-1)+(j-1)*lda] = 0
		}
		a[(j-1)+(j-1)*lda] = 1
	}
	for i := k; i >= 1; i-- {
		ii := (i - 1) + (i-1)*lda
		// Apply H(i) to A(i:m,i:n) from the left.
		if i < n {
			a[ii] = 1
			Larf(kn, true, m-i+1, n-i, a[ii:], tau[i-1], a[(i-1)+i*lda:], lda, work)
		}
		if i < m {
			Scal(m-i, T(-tau[i-1]), a[ii+1:], 1)
		}
		a[ii] = T(1 - tau[i-1])
		// Set A(1:i-1,i) to zero.
		for l := 1; l <= i-1; l++ {
			a[(l-1)+(i-1)*lda] = 0
		}
	}
}

// Orgqr is xORGQR (dorgqr.f): Org2r, blocked with panels of nb columns while more
// than nx remain (NBQR, NXQR), using Larft and Larfb; m >= n >= k >= 0, n > 0.
func Orgqr[T Float](kn *Kern[T], nb, nx, m, n, k int, a []T, lda int, tau []T) {
	nbmin := 2
	if !(nb > 1 && nb < k) {
		nx = 0
	}
	blocked := nb >= nbmin && nb < k && nx < k
	ldwork := n
	wlen := n
	if blocked {
		wlen = ldwork * nb
	}
	work := make([]T, wlen)
	var kk, ki int
	if blocked {
		// Use blocked code after the first block. The last kk columns are handled by
		// the block method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)
		// Set A(1:kk,kk+1:n) to zero.
		for j := kk + 1; j <= n; j++ {
			for i := 1; i <= kk; i++ {
				a[(i-1)+(j-1)*lda] = 0
			}
		}
	}
	// Use unblocked code for the last or only block.
	if kk < n {
		Org2r(kn, m-kk, n-kk, k-kk, a[kk+kk*lda:], lda, tau[kk:], work)
	}
	if kk > 0 {
		// Use blocked code.
		for i := ki + 1; i >= 1; i -= nb {
			ib := min(nb, k-i+1)
			ii := (i - 1) + (i-1)*lda
			if i+ib <= n {
				// Form the triangular factor of the block reflector, and apply H to
				// A(i:m,i+ib:n) from the left.
				Larft(kn, m-i+1, ib, a[ii:], lda, tau[i-1:], work, ldwork)
				Larfb(kn, true, true, m-i+1, n-i-ib+1, ib, a[ii:], lda, work, ldwork, a[(i-1)+(i+ib-1)*lda:], lda, work[ib:], ldwork)
			}
			// Apply H to rows i:m of current block.
			Org2r(kn, m-i+1, ib, ib, a[ii:], lda, tau[i-1:], work)
			// Set rows 1:i-1 of current block to zero.
			for j := i; j <= i+ib-1; j++ {
				for l := 1; l <= i-1; l++ {
					a[(l-1)+(j-1)*lda] = 0
				}
			}
		}
	}
}

// Orm2r is xORM2R (dorm2r.f): it overwrites the m x n matrix c with Q*C, Q**T*C,
// C*Q or C*Q**T, Q = H(1) H(2) ... H(k) of order nq (m for left, n for right) as
// Geqrf returned it in the nq x k matrix a, whose diagonal is restored on exit;
// m, n, k > 0. work has n (left) or m (right) elements.
func Orm2r[T Float](kn *Kern[T], left, notran bool, m, n, k int, a []T, lda int, tau []T, c []T, ldc int, work []T) {
	var i1, i2, i3 int
	if (left && !notran) || (!left && notran) {
		i1, i2, i3 = 1, k, 1
	} else {
		i1, i2, i3 = k, 1, -1
	}
	mi, ni, ic, jc := m, n, 1, 1
	for i := i1; (i3 > 0 && i <= i2) || (i3 < 0 && i >= i2); i += i3 {
		if left {
			// H(i) is applied to C(i:m,1:n).
			mi = m - i + 1
			ic = i
		} else {
			// H(i) is applied to C(1:m,i:n).
			ni = n - i + 1
			jc = i
		}
		// Apply H(i).
		ii := (i - 1) + (i-1)*lda
		aii := a[ii]
		a[ii] = 1
		Larf(kn, left, mi, ni, a[ii:], tau[i-1], c[(ic-1)+(jc-1)*ldc:], ldc, work)
		a[ii] = aii
	}
}

// Ormqr is xORMQR (dormqr.f): Orm2r, blocked with panels of nb columns (NBORM)
// using Larft and Larfb when 1 < nb < k; m, n, k > 0.
func Ormqr[T Float](kn *Kern[T], nb int, left, notran bool, m, n, k int, a []T, lda int, tau []T, c []T, ldc int) {
	nq, nw := n, m
	if left {
		nq, nw = m, n
	}
	const nbmin = 2
	if nb < nbmin || nb >= k {
		// Use unblocked code.
		Orm2r(kn, left, notran, m, n, k, a, lda, tau, c, ldc, make([]T, nw))
		return
	}
	// Use blocked code. T has leading dimension nb.
	ldwork := nw
	ldt := nb
	work := make([]T, ldwork*nb)
	t := make([]T, ldt*nb)
	var i1, i2, i3 int
	if (left && !notran) || (!left && notran) {
		i1, i2, i3 = 1, k, nb
	} else {
		i1, i2, i3 = ((k-1)/nb)*nb+1, 1, -nb
	}
	mi, ni, ic, jc := m, n, 1, 1
	for i := i1; (i3 > 0 && i <= i2) || (i3 < 0 && i >= i2); i += i3 {
		ib := min(nb, k-i+1)
		// Form the triangular factor of the block reflector H = H(i) ... H(i+ib-1).
		ii := (i - 1) + (i-1)*lda
		Larft(kn, nq-i+1, ib, a[ii:], lda, tau[i-1:], t, ldt)
		if left {
			// H or H**T is applied to C(i:m,1:n).
			mi = m - i + 1
			ic = i
		} else {
			// H or H**T is applied to C(1:m,i:n).
			ni = n - i + 1
			jc = i
		}
		// Apply H or H**T.
		Larfb(kn, left, notran, mi, ni, ib, a[ii:], lda, t, ldt, c[(ic-1)+(jc-1)*ldc:], ldc, work, ldwork)
	}
}
