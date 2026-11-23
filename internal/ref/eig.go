package ref

// Ports of the symmetric eigensolver of Reference-LAPACK 3.12.0 (tag v3.12.0, SRC):
// dsyev.f, dsytrd.f, dsytd2.f, dlatrd.f, dorgtr.f, dorgql.f, dorg2l.f, dsteqr.f,
// dsterf.f, and what they call, with their single precision twins, generic over
// float32 and float64. Same rules as qr.go: the loops are the reference's (the
// GO TO structure of dsteqr and dsterf is kept as loops with the same tests, in
// the same order), 0-based indices into column-major slices in the arguments but
// the reference's 1-based indices in the loop variables, written as i-1 where they
// index, no FMA, the arguments already checked, the reference's quick returns the
// caller's, and the workspace allocated.

// NBTRD and NXTRD are what ILAENV answers (ilaenv.f, v3.12.0) for DSYTRD: NB = 32
// (ISPEC 1), NX = 32 (ISPEC 3). NBQL and NXQL are what it answers for DORGQL: NB =
// 32, NX = 128. Variables only so that the tests can make small matrices span
// many panels; Syev reads them.
var (
	NBTRD = 32
	NXTRD = 32
	NBQL  = 32
	NXQL  = 128
)

// MaxIt is the MAXIT of DSTEQR and DSTERF, the number of QL or QR iterations per
// eigenvalue before they give up (NMAXIT = N*MAXIT in all). A variable only so
// that a test can force the non-convergence INFO > 0, which no matrix is known to
// provoke.
var MaxIt = 30

// The calls of the symmetric eigensolver to the BLAS, with the quick returns of the
// reference.
func (k *Kern[T]) gemvx(notA bool, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T) {
	// No quick return here: Gemv has the one of DGEMV (m = 0 or n = 0), and
	// alpha is +-1 in every call.
	k.Gemv(notA, m, n, alpha, a, lda, x, incx, beta, y, 1)
}

func (k *Kern[T]) symv(upper bool, n int, alpha T, a []T, lda int, x []T, beta T, y []T) {
	// No quick return: the reference's callers (Sytd2, Latrd) never pass n = 0, or
	// alpha = 0 with beta = 1.
	k.Symv(upper, n, alpha, a, lda, x, beta, y)
}

func (k *Kern[T]) syr2(upper bool, n int, alpha T, x, y []T, a []T, lda int) {
	// No quick return: Sytd2 calls it with alpha = -1 and n > 0.
	k.Syr2(upper, n, alpha, x, y, a, lda)
}

func (k *Kern[T]) syr2k(upper, notA bool, n, kk int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {
	// No quick return: Sytrd calls it with alpha = -1, beta = 1, n > 0, kk = nb > 0.
	k.Syr2k(upper, notA, n, kk, alpha, a, lda, b, ldb, beta, c, ldc)
}

// axpy is xAXPY with unit increments and the quick returns of the reference.
func axpy[T Float](n int, alpha T, x, y []T) {
	if n <= 0 || alpha == 0 {
		return
	}
	Axpy(n, alpha, x, 1, y, 1)
}

// Sytd2 is xSYTD2 (dsytd2.f): the unblocked reduction of the n x n symmetric
// matrix a, whose upper or lower triangle is stored, to the symmetric tridiagonal
// T = Q**T*A*Q, Q = H(n-1) ... H(1) (upper) or H(1) ... H(n-1) (lower): d (n) and e
// (n-1) hold T, the reflectors are in a, their scalars in tau (n-1), n > 0.
func Sytd2[T Float](kn *Kern[T], upper bool, n int, a []T, lda int, d, e, tau []T) {
	if upper {
		// Reduce the upper triangle of A.
		for i := n - 1; i >= 1; i-- {
			// Generate elementary reflector H(i) = I - tau*v*v**T to annihilate
			// A(1:i-1,i+1).
			taui := Larfg(i, &a[(i-1)+i*lda], a[i*lda:], 1)
			e[i-1] = a[(i-1)+i*lda]
			if taui != 0 {
				// Apply H(i) from both sides to A(1:i,1:i).
				a[(i-1)+i*lda] = 1
				// Compute x := tau * A * v, storing x in tau(1:i).
				kn.symv(upper, i, taui, a, lda, a[i*lda:], 0, tau)
				// Compute w := x - 1/2 tau * (x**T * v) * v.
				alpha := T(T(-0.5*taui) * Dot(i, tau, 1, a[i*lda:], 1))
				axpy(i, alpha, a[i*lda:], tau)
				// Apply the transformation as a rank-2 update A := A - v*w**T - w*v**T.
				kn.syr2(upper, i, -1, a[i*lda:], tau, a, lda)
				a[(i-1)+i*lda] = e[i-1]
			}
			d[i] = a[i+i*lda]
			tau[i-1] = taui
		}
		d[0] = a[0]
		return
	}
	// Reduce the lower triangle of A.
	for i := 1; i <= n-1; i++ {
		// Generate elementary reflector H(i) to annihilate A(i+2:n,i).
		taui := Larfg(n-i, &a[i+(i-1)*lda], a[min(i+2, n)-1+(i-1)*lda:], 1)
		e[i-1] = a[i+(i-1)*lda]
		if taui != 0 {
			// Apply H(i) from both sides to A(i+1:n,i+1:n).
			a[i+(i-1)*lda] = 1
			// Compute x := tau * A * v, storing x in tau(i:n-1).
			kn.symv(upper, n-i, taui, a[i+i*lda:], lda, a[i+(i-1)*lda:], 0, tau[i-1:])
			// Compute w := x - 1/2 tau * (x**T * v) * v.
			alpha := T(T(-0.5*taui) * Dot(n-i, tau[i-1:], 1, a[i+(i-1)*lda:], 1))
			axpy(n-i, alpha, a[i+(i-1)*lda:], tau[i-1:])
			// Apply the transformation as a rank-2 update.
			kn.syr2(upper, n-i, -1, a[i+(i-1)*lda:], tau[i-1:], a[i+i*lda:], lda)
			a[i+(i-1)*lda] = e[i-1]
		}
		d[i-1] = a[(i-1)+(i-1)*lda]
		tau[i-1] = taui
	}
	d[n-1] = a[(n-1)+(n-1)*lda]
}

// Latrd is xLATRD (dlatrd.f): it reduces nb rows and columns of the n x n
// symmetric matrix a to tridiagonal form by an orthogonal similarity, and returns
// in the n x nb matrix w (leading dimension ldw) the matrices the blocked update
// of the rest needs, n > 0.
func Latrd[T Float](kn *Kern[T], upper bool, n, nb int, a []T, lda int, e, tau []T, w []T, ldw int) {
	if upper {
		// Reduce the last nb columns of the upper triangle.
		for i := n; i >= n-nb+1; i-- {
			iw := i - n + nb
			// Index helpers: column i of a, column iw of w.
			ai := a[(i-1)*lda:]
			wi := w[(iw-1)*ldw:]
			if i < n {
				// Update A(1:i,i).
				kn.gemvx(true, i, n-i, -1, a[i*lda:], lda, w[(i-1)+iw*ldw:], ldw, 1, ai)
				kn.gemvx(true, i, n-i, -1, w[iw*ldw:], ldw, a[(i-1)+i*lda:], lda, 1, ai)
			}
			if i > 1 {
				// Generate elementary reflector H(i) to annihilate A(1:i-2,i).
				tau[i-2] = Larfg(i-1, &ai[i-2], ai, 1)
				e[i-2] = ai[i-2]
				ai[i-2] = 1
				// Compute W(1:i-1,i).
				kn.symv(true, i-1, 1, a, lda, ai, 0, wi)
				if i < n {
					kn.gemvx(false, i-1, n-i, 1, w[iw*ldw:], ldw, ai, 1, 0, wi[i:])
					kn.gemvx(true, i-1, n-i, -1, a[i*lda:], lda, wi[i:], 1, 1, wi)
					kn.gemvx(false, i-1, n-i, 1, a[i*lda:], lda, ai, 1, 0, wi[i:])
					kn.gemvx(true, i-1, n-i, -1, w[iw*ldw:], ldw, wi[i:], 1, 1, wi)
				}
				Scal(i-1, tau[i-2], wi, 1)
				alpha := T(T(-0.5*tau[i-2]) * Dot(i-1, wi, 1, ai, 1))
				axpy(i-1, alpha, ai, wi)
			}
		}
		return
	}
	// Reduce the first nb columns of the lower triangle.
	for i := 1; i <= nb; i++ {
		ai := a[(i-1)*lda:]
		wi := w[(i-1)*ldw:]
		// Update A(i:n,i).
		kn.gemvx(true, n-i+1, i-1, -1, a[i-1:], lda, w[i-1:], ldw, 1, ai[i-1:])
		kn.gemvx(true, n-i+1, i-1, -1, w[i-1:], ldw, a[i-1:], lda, 1, ai[i-1:])
		if i < n {
			// Generate elementary reflector H(i) to annihilate A(i+2:n,i).
			tau[i-1] = Larfg(n-i, &ai[i], ai[min(i+2, n)-1:], 1)
			e[i-1] = ai[i]
			ai[i] = 1
			// Compute W(i+1:n,i).
			kn.symv(false, n-i, 1, a[i+i*lda:], lda, ai[i:], 0, wi[i:])
			kn.gemvx(false, n-i, i-1, 1, w[i:], ldw, ai[i:], 1, 0, wi)
			kn.gemvx(true, n-i, i-1, -1, a[i:], lda, wi, 1, 1, wi[i:])
			kn.gemvx(false, n-i, i-1, 1, a[i:], lda, ai[i:], 1, 0, wi)
			kn.gemvx(true, n-i, i-1, -1, w[i:], ldw, wi, 1, 1, wi[i:])
			Scal(n-i, tau[i-1], wi[i:], 1)
			alpha := T(T(-0.5*tau[i-1]) * Dot(n-i, wi[i:], 1, ai[i:], 1))
			axpy(n-i, alpha, ai[i:], wi[i:])
		}
	}
}

// Sytrd is xSYTRD (dsytrd.f): the reduction of the n x n symmetric matrix a to
// tridiagonal form, as Sytd2, blocked with panels of nb columns (Latrd and a
// rank-2k update) while more than nx remain (NBTRD, NXTRD), n > 0.
func Sytrd[T Float](kn *Kern[T], nb, nx int, upper bool, n int, a []T, lda int, d, e, tau []T) {
	ldwork := n
	if nb > 1 && nb < n {
		// Determine when to cross over from blocked to unblocked code (the last
		// block is always handled by unblocked code).
		nx = max(nb, nx)
		if nx >= n {
			nx = n
		}
	} else {
		nb = 1
		nx = n
	}
	var work []T
	if nx < n {
		work = make([]T, ldwork*nb)
	}
	if upper {
		// Reduce the upper triangle of A. Columns 1:kk are handled by the unblocked
		// method.
		kk := n - ((n-nx+nb-1)/nb)*nb
		for i := n - nb + 1; i >= kk+1; i -= nb {
			// Reduce columns i:i+nb-1 to tridiagonal form and form the matrix W
			// which is needed to update the unreduced part of the matrix.
			Latrd(kn, upper, i+nb-1, nb, a, lda, e, tau, work, ldwork)
			// Update the unreduced submatrix A(1:i-1,1:i-1), using an update of the
			// form A := A - V*W**T - W*V**T.
			kn.syr2k(upper, true, i-1, nb, -1, a[(i-1)*lda:], lda, work, ldwork, 1, a, lda)
			// Copy superdiagonal elements back into A, and diagonal elements into d.
			for j := i; j <= i+nb-1; j++ {
				a[(j-2)+(j-1)*lda] = e[j-2]
				d[j-1] = a[(j-1)+(j-1)*lda]
			}
		}
		// Use unblocked code to reduce the last or only block.
		Sytd2(kn, upper, kk, a, lda, d, e, tau)
		return
	}
	// Reduce the lower triangle of A.
	i := 1
	for ; i <= n-nx; i += nb {
		// Reduce columns i:i+nb-1 to tridiagonal form and form the matrix W.
		ii := (i - 1) + (i-1)*lda
		Latrd(kn, upper, n-i+1, nb, a[ii:], lda, e[i-1:], tau[i-1:], work, ldwork)
		// Update the unreduced submatrix A(i+nb:n,i+nb:n).
		kn.syr2k(upper, true, n-i-nb+1, nb, -1, a[(i+nb-1)+(i-1)*lda:], lda, work[nb:], ldwork, 1, a[(i+nb-1)+(i+nb-1)*lda:], lda)
		// Copy subdiagonal elements back into A, and diagonal elements into d.
		for j := i; j <= i+nb-1; j++ {
			a[j+(j-1)*lda] = e[j-1]
			d[j-1] = a[(j-1)+(j-1)*lda]
		}
	}
	// Use unblocked code to reduce the last or only block.
	Sytd2(kn, upper, n-i+1, a[(i-1)+(i-1)*lda:], lda, d[i-1:], e[i-1:], tau[i-1:])
}

// LarftBC is xLARFT (dlarft.f) for DIRECT = 'B', STOREV = 'C': it forms the k x k
// lower triangular factor t of the block reflector H = H(k) ... H(2) H(1) = I -
// V*T*V**T, the n x k vectors v (unit upper trapezoidal at its last k rows) held
// by columns, n >= k, n > 0.
func LarftBC[T Float](kn *Kern[T], n, k int, v []T, ldv int, tau []T, t []T, ldt int) {
	prevlastv := 1
	for i := k; i >= 1; i-- {
		if tau[i-1] == 0 {
			// H(i) = I.
			for j := i; j <= k; j++ {
				t[(j-1)+(i-1)*ldt] = 0
			}
			continue
		}
		// General case.
		if i < k {
			// Skip any leading zeros.
			lastv := 1
			for ; lastv <= i-1; lastv++ {
				if v[(lastv-1)+(i-1)*ldv] != 0 {
					break
				}
			}
			for j := i + 1; j <= k; j++ {
				t[(j-1)+(i-1)*ldt] = T(-tau[i-1] * v[(n-k+i-1)+(j-1)*ldv])
			}
			j := max(lastv, prevlastv)
			// T(i+1:k,i) := - tau(i) * V(j:n-k+i,i+1:k)**T * V(j:n-k+i,i).
			kn.gemv(false, n-k+i-j, k-i, -tau[i-1], v[(j-1)+i*ldv:], ldv, v[(j-1)+(i-1)*ldv:], 1, t[i+(i-1)*ldt:])
			// T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i).
			if k-i > 0 {
				trmvLN(k-i, t[i+i*ldt:], ldt, t[i+(i-1)*ldt:])
			}
			if i > 1 {
				prevlastv = min(prevlastv, lastv)
			} else {
				prevlastv = lastv
			}
		}
		t[(i-1)+(i-1)*ldt] = tau[i-1]
	}
}

// LarfbBCL is xLARFB (dlarfb.f) for SIDE = 'L', DIRECT = 'B', STOREV = 'C': it
// applies the block reflector H = I - V*T*V**T (notrans) or H**T to the m x n
// matrix c from the left. v is m x k, unit upper triangular at its last k rows, t
// the k x k lower triangular factor of LarftBC, work an n x k matrix with leading
// dimension ldwork.
func LarfbBCL[T Float](kn *Kern[T], notrans bool, m, n, k int, v []T, ldv int, t []T, ldt int, c []T, ldc int, work []T, ldwork int) {
	// (Orgql calls it with m > 0 and n > 0 only: no quick return.)
	// W := C2**T (the last k rows of c).
	for j := 0; j < k; j++ {
		for i := 0; i < n; i++ {
			work[i+j*ldwork] = c[(m-k+j)+i*ldc]
		}
	}
	// W := W * V2.
	kn.trmm(true, true, false, n, k, 1, v[m-k:], ldv, work, ldwork)
	if m > k {
		// W := W + C1**T * V1.
		kn.gemm(false, true, n, k, m-k, 1, c, ldc, v, ldv, 1, work, ldwork)
	}
	// W := W * T**T or W * T.
	kn.trmm(false, !notrans, true, n, k, 1, t, ldt, work, ldwork)
	if m > k {
		// C1 := C1 - V1 * W**T.
		kn.gemm(true, false, m-k, n, k, -1, v, ldv, work, ldwork, 1, c, ldc)
	}
	// W := W * V2**T.
	kn.trmm(true, false, false, n, k, 1, v[m-k:], ldv, work, ldwork)
	// C2 := C2 - W**T.
	for j := 0; j < k; j++ {
		for i := 0; i < n; i++ {
			c[(m-k+j)+i*ldc] = c[(m-k+j)+i*ldc] - work[i+j*ldwork]
		}
	}
}

// Org2l is xORG2L (dorg2l.f): it generates the m x n matrix Q with orthonormal
// columns, the last n columns of the product of k elementary reflectors H(k) ...
// H(2) H(1) of order m as Sytrd (upper) returned them, in place, m >= n >= k >= 0,
// n > 0. work has n elements.
func Org2l[T Float](kn *Kern[T], m, n, k int, a []T, lda int, tau []T, work []T) {
	// The reference first sets columns 1:n-k to columns of the unit matrix; Orgql
	// is only ever called here with n = k (Orgtr), where there are none.
	for i := 1; i <= k; i++ {
		ii := n - k + i
		// Apply H(i) to A(1:m-k+i,1:n-k+i) from the left.
		a[(m-n+ii-1)+(ii-1)*lda] = 1
		Larf(kn, true, m-n+ii, ii-1, a[(ii-1)*lda:], tau[i-1], a, lda, work)
		Scal(m-n+ii-1, T(-tau[i-1]), a[(ii-1)*lda:], 1)
		a[(m-n+ii-1)+(ii-1)*lda] = T(1 - tau[i-1])
		// Set A(m-k+i+1:m,n-k+i) to zero.
		for l := m - n + ii + 1; l <= m; l++ {
			a[(l-1)+(ii-1)*lda] = 0
		}
	}
}

// Orgql is xORGQL (dorgql.f): Org2l, blocked with panels of nb columns while more
// than nx remain (NBQL, NXQL), using LarftBC and LarfbBCL; m >= n >= k >= 0, n > 0.
func Orgql[T Float](kn *Kern[T], nb, nx, m, n, k int, a []T, lda int, tau []T) {
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
	kk := 0
	if blocked {
		// Use blocked code after the first block. The last kk columns are handled
		// by the block method.
		kk = min(k, ((k-nx+nb-1)/nb)*nb)
		// Set A(m-kk+1:m,1:n-kk) to zero.
		for j := 1; j <= n-kk; j++ {
			for i := m - kk + 1; i <= m; i++ {
				a[(i-1)+(j-1)*lda] = 0
			}
		}
	}
	// Use unblocked code for the first or only block.
	Org2l(kn, m-kk, n-kk, k-kk, a, lda, tau, work)
	if kk > 0 {
		// Use blocked code.
		for i := k - kk + 1; i <= k; i += nb {
			ib := min(nb, k-i+1)
			if n-k+i > 1 {
				// Form the triangular factor of the block reflector H = H(i+ib-1) ...
				// H(i+1) H(i).
				LarftBC(kn, m-k+i+ib-1, ib, a[(n-k+i-1)*lda:], lda, tau[i-1:], work, ldwork)
				// Apply H to A(1:m-k+i+ib-1,1:n-k+i-1) from the left.
				LarfbBCL(kn, true, m-k+i+ib-1, n-k+i-1, ib, a[(n-k+i-1)*lda:], lda, work, ldwork, a, lda, work[ib:], ldwork)
			}
			// Apply H to rows 1:m-k+i+ib-1 of the current block.
			Org2l(kn, m-k+i+ib-1, ib, ib, a[(n-k+i-1)*lda:], lda, tau[i-1:], work)
			// Set rows m-k+i+ib:m of the current block to zero.
			for j := n - k + i; j <= n-k+i+ib-1; j++ {
				for l := m - k + i + ib; l <= m; l++ {
					a[(l-1)+(j-1)*lda] = 0
				}
			}
		}
	}
}

// Orgtr is xORGTR (dorgtr.f): it generates the n x n orthogonal matrix Q of the
// reduction Sytrd made, in place, n > 0.
func Orgtr[T Float](kn *Kern[T], nbql, nxql, nbqr, nxqr int, upper bool, n int, a []T, lda int, tau []T) {
	if upper {
		// Q was determined by Sytrd with UPLO = 'U'. Shift the vectors which define
		// the elementary reflectors one column to the left, and set the last row and
		// column of Q to those of the unit matrix.
		for j := 1; j <= n-1; j++ {
			for i := 1; i <= j-1; i++ {
				a[(i-1)+(j-1)*lda] = a[(i-1)+j*lda]
			}
			a[(n-1)+(j-1)*lda] = 0
		}
		for i := 1; i <= n-1; i++ {
			a[(i-1)+(n-1)*lda] = 0
		}
		a[(n-1)+(n-1)*lda] = 1
		// Generate Q(1:n-1,1:n-1).
		if n > 1 {
			Orgql(kn, nbql, nxql, n-1, n-1, n-1, a, lda, tau)
		}
		return
	}
	// Q was determined by Sytrd with UPLO = 'L'. Shift the vectors one column to
	// the right, and set the first row and column of Q to those of the unit matrix.
	for j := n; j >= 2; j-- {
		a[(j-1)*lda] = 0
		for i := j + 1; i <= n; i++ {
			a[(i-1)+(j-1)*lda] = a[(i-1)+(j-2)*lda]
		}
	}
	a[0] = 1
	for i := 2; i <= n; i++ {
		a[i-1] = 0
	}
	if n > 1 {
		// Generate Q(2:n,2:n).
		Orgqr(kn, nbqr, nxqr, n-1, n-1, n-1, a[1+lda:], lda, tau)
	}
}

// Sterf is xSTERF (dsterf.f): all the eigenvalues of the n x n symmetric
// tridiagonal matrix with diagonal d and off-diagonal e (both destroyed, the
// eigenvalues in increasing order in d), by the Pal-Walker-Kahan variant of the QL
// or QR algorithm. info > 0: the algorithm failed to find all the eigenvalues in
// n*MaxIt iterations, info being the number of off-diagonal elements not
// converged to zero.
func Sterf[T Float](n int, d, e []T) (info int) {
	// n >= 2: the callers return for n <= 1.
	eps := epsRound[T]()
	eps2 := T(eps * eps)
	safmin := SafeMin[T]()
	safmax := T(1 / safmin)
	ssfmax := T(sqrt(safmax) / 3)
	ssfmin := T(sqrt(safmin) / eps2)
	nmaxit := n * MaxIt
	var sigma T
	jtot := 0
	// Determine where the matrix splits and choose QL or QR iteration for each
	// block, according to whether top or bottom diagonal element is smaller.
	l1 := 1
	for l1 <= n {
		if l1 > 1 {
			e[l1-2] = 0
		}
		m := n
		for mm := l1; mm <= n-1; mm++ {
			if abs(e[mm-1]) <= T(T(sqrt(abs(d[mm-1]))*sqrt(abs(d[mm])))*eps) {
				e[mm-1] = 0
				m = mm
				break
			}
		}
		l := l1
		lsv := l
		lend := m
		lendsv := lend
		l1 = m + 1
		if lend == l {
			continue
		}
		// Scale submatrix in rows and columns l to lend.
		anorm := Lanst(lend-l+1, d[l-1:], e[l-1:])
		iscale := 0
		// (The reference returns here when anorm == 0; a block split at every
		// zero off-diagonal element has a nonzero one, so it cannot.)
		if anorm > ssfmax {
			iscale = 1
			Lascl('G', anorm, ssfmax, lend-l+1, 1, d[l-1:], n)
			Lascl('G', anorm, ssfmax, lend-l, 1, e[l-1:], n)
		} else if anorm < ssfmin {
			iscale = 2
			Lascl('G', anorm, ssfmin, lend-l+1, 1, d[l-1:], n)
			Lascl('G', anorm, ssfmin, lend-l, 1, e[l-1:], n)
		}
		for i := l; i <= lend-1; i++ {
			e[i-1] = T(e[i-1] * e[i-1])
		}
		// Choose between QL and QR iteration.
		if abs(d[lend-1]) < abs(d[l-1]) {
			lend = lsv
			l = lendsv
		}
		if lend >= l {
			// QL iteration. Look for a small subdiagonal element.
			for {
				m := lend
				if l != lend {
					for mm := l; mm <= lend-1; mm++ {
						if abs(e[mm-1]) <= T(eps2*abs(T(d[mm-1]*d[mm]))) {
							m = mm
							break
						}
					}
				}
				if m < lend {
					e[m-1] = 0
				}
				p := d[l-1]
				if m == l {
					// Eigenvalue found.
					d[l-1] = p
					l++
					if l <= lend {
						continue
					}
					break
				}
				// If the remaining matrix is 2 by 2, use Lae2 to compute its
				// eigenvalues.
				if m == l+1 {
					rte := sqrt(e[l-1])
					rt1, rt2 := Lae2(d[l-1], rte, d[l])
					d[l-1] = rt1
					d[l] = rt2
					e[l-1] = 0
					l += 2
					if l <= lend {
						continue
					}
					break
				}
				if jtot == nmaxit {
					break
				}
				jtot++
				// Form the shift.
				rte := sqrt(e[l-1])
				sigma = T(T(d[l]-p) / T(2*rte))
				r := Lapy2(sigma, 1)
				sigma = T(p - T(rte/T(sigma+sign(r, sigma))))
				c := T(1)
				var s T
				gamma := T(d[m-1] - sigma)
				p = T(gamma * gamma)
				// Inner loop.
				for i := m - 1; i >= l; i-- {
					bb := e[i-1]
					r = T(p + bb)
					if i != m-1 {
						e[i] = T(s * r)
					}
					oldc := c
					c = T(p / r)
					s = T(bb / r)
					oldgam := gamma
					alpha := d[i-1]
					gamma = T(T(c*T(alpha-sigma)) - T(s*oldgam))
					d[i] = T(oldgam + T(alpha-gamma))
					if c != 0 {
						p = T(T(gamma*gamma) / c)
					} else {
						p = T(oldc * bb)
					}
				}
				e[l-1] = T(s * p)
				d[l-1] = T(sigma + gamma)
			}
		} else {
			// QR iteration. Look for a small superdiagonal element.
			for {
				m := lend
				for mm := l; mm >= lend+1; mm-- {
					if abs(e[mm-2]) <= T(eps2*abs(T(d[mm-1]*d[mm-2]))) {
						m = mm
						break
					}
				}
				if m > lend {
					e[m-2] = 0
				}
				p := d[l-1]
				if m == l {
					// Eigenvalue found.
					d[l-1] = p
					l--
					if l >= lend {
						continue
					}
					break
				}
				// If the remaining matrix is 2 by 2, use Lae2.
				if m == l-1 {
					rte := sqrt(e[l-2])
					rt1, rt2 := Lae2(d[l-1], rte, d[l-2])
					d[l-1] = rt1
					d[l-2] = rt2
					e[l-2] = 0
					l -= 2
					if l >= lend {
						continue
					}
					break
				}
				if jtot == nmaxit {
					break
				}
				jtot++
				// Form the shift.
				rte := sqrt(e[l-2])
				sigma = T(T(d[l-2]-p) / T(2*rte))
				r := Lapy2(sigma, 1)
				sigma = T(p - T(rte/T(sigma+sign(r, sigma))))
				c := T(1)
				var s T
				gamma := T(d[m-1] - sigma)
				p = T(gamma * gamma)
				// Inner loop.
				for i := m; i <= l-1; i++ {
					bb := e[i-1]
					r = T(p + bb)
					if i != m {
						e[i-2] = T(s * r)
					}
					oldc := c
					c = T(p / r)
					s = T(bb / r)
					oldgam := gamma
					alpha := d[i]
					gamma = T(T(c*T(alpha-sigma)) - T(s*oldgam))
					d[i-1] = T(oldgam + T(alpha-gamma))
					if c != 0 {
						p = T(T(gamma*gamma) / c)
					} else {
						p = T(oldc * bb)
					}
				}
				e[l-2] = T(s * p)
				d[l-1] = T(sigma + gamma)
			}
		}
		// Undo scaling if necessary.
		if iscale == 1 {
			Lascl('G', ssfmax, anorm, lendsv-lsv+1, 1, d[lsv-1:], n)
		}
		if iscale == 2 {
			Lascl('G', ssfmin, anorm, lendsv-lsv+1, 1, d[lsv-1:], n)
		}
		// Check for no convergence to an eigenvalue after a total of n*MaxIt
		// iterations.
		if jtot >= nmaxit {
			for i := 1; i <= n-1; i++ {
				if e[i-1] != 0 {
					info++
				}
			}
			return info
		}
	}
	// Sort the eigenvalues in increasing order.
	Lasrt(n, d)
	return 0
}

// Steqr is xSTEQR (dsteqr.f): all the eigenvalues, and optionally the
// eigenvectors, of the n x n symmetric tridiagonal matrix with diagonal d and
// off-diagonal e (both destroyed, the eigenvalues in increasing order in d), by
// the implicit QL or QR method. icompz: 0 eigenvalues only, 1 the eigenvectors of
// the original matrix, whose reduction's orthogonal matrix is in z on entry, 2 the
// eigenvectors of the tridiagonal matrix (z is set to the identity first). info >
// 0: n*MaxIt iterations were not enough, info being the number of off-diagonal
// elements not converged to zero.
func Steqr[T Float](icompz, n int, d, e []T, z []T, ldz int) (info int) {
	// The reference's n = 1 quick return (z = 1 for icompz = 2) is not needed: the
	// general path sets z to the identity, finds no block and leaves d alone
	// (TestOneByOne). n = 0 likewise does nothing (TestSteqrEmpty).
	// Determine the unit roundoff and over/underflow thresholds.
	eps := epsRound[T]()
	eps2 := T(eps * eps)
	safmin := SafeMin[T]()
	safmax := T(1 / safmin)
	ssfmax := T(sqrt(safmax) / 3)
	ssfmin := T(sqrt(safmin) / eps2)
	if icompz == 2 {
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				z[i+j*ldz] = 0
			}
			z[j+j*ldz] = 1
		}
	}
	work := make([]T, 2*n)
	nmaxit := n * MaxIt
	jtot := 0
	// Determine where the matrix splits and choose QL or QR iteration for each
	// block, according to whether top or bottom diagonal element is smaller.
	l1 := 1
	nm1 := n - 1
	for l1 <= n {
		if l1 > 1 {
			e[l1-2] = 0
		}
		m := n
		for mm := l1; mm <= nm1; mm++ {
			tst := abs(e[mm-1])
			if tst == 0 {
				m = mm
				break
			}
			if tst <= T(T(sqrt(abs(d[mm-1]))*sqrt(abs(d[mm])))*eps) {
				e[mm-1] = 0
				m = mm
				break
			}
		}
		l := l1
		lsv := l
		lend := m
		lendsv := lend
		l1 = m + 1
		if lend == l {
			continue
		}
		// Scale submatrix in rows and columns l to lend.
		anorm := Lanst(lend-l+1, d[l-1:], e[l-1:])
		iscale := 0
		// (The reference returns here when anorm == 0; a block split at every
		// zero off-diagonal element has a nonzero one, so it cannot.)
		if anorm > ssfmax {
			iscale = 1
			Lascl('G', anorm, ssfmax, lend-l+1, 1, d[l-1:], n)
			Lascl('G', anorm, ssfmax, lend-l, 1, e[l-1:], n)
		} else if anorm < ssfmin {
			iscale = 2
			Lascl('G', anorm, ssfmin, lend-l+1, 1, d[l-1:], n)
			Lascl('G', anorm, ssfmin, lend-l, 1, e[l-1:], n)
		}
		// Choose between QL and QR iteration.
		if abs(d[lend-1]) < abs(d[l-1]) {
			lend = lsv
			l = lendsv
		}
		if lend > l {
			// QL iteration. Look for a small subdiagonal element.
			for {
				m := lend
				if l != lend {
					for mm := l; mm <= lend-1; mm++ {
						tst := T(abs(e[mm-1]) * abs(e[mm-1]))
						if tst <= T(T(T(eps2*abs(d[mm-1]))*abs(d[mm]))+safmin) {
							m = mm
							break
						}
					}
				}
				if m < lend {
					e[m-1] = 0
				}
				p := d[l-1]
				if m == l {
					// Eigenvalue found.
					d[l-1] = p
					l++
					if l <= lend {
						continue
					}
					break
				}
				// If the remaining matrix is 2 by 2, use Lae2 or Laev2 to compute its
				// eigensystem.
				if m == l+1 {
					var rt1, rt2 T
					if icompz > 0 {
						var c, s T
						rt1, rt2, c, s = Laev2(d[l-1], e[l-1], d[l])
						work[l-1] = c
						work[n-1+l-1] = s
						LasrRV(false, n, 2, work[l-1:], work[n-1+l-1:], z[(l-1)*ldz:], ldz)
					} else {
						rt1, rt2 = Lae2(d[l-1], e[l-1], d[l])
					}
					d[l-1] = rt1
					d[l] = rt2
					e[l-1] = 0
					l += 2
					if l <= lend {
						continue
					}
					break
				}
				if jtot == nmaxit {
					break
				}
				jtot++
				// Form the shift.
				g := T(T(d[l]-p) / T(2*e[l-1]))
				r := Lapy2(g, 1)
				g = T(T(d[m-1]-p) + T(e[l-1]/T(g+sign(r, g))))
				s := T(1)
				c := T(1)
				p = 0
				// Inner loop.
				for i := m - 1; i >= l; i-- {
					f := T(s * e[i-1])
					b := T(c * e[i-1])
					c, s, r = Lartg(g, f)
					if i != m-1 {
						e[i] = r
					}
					g = T(d[i] - p)
					r = T(T(T(d[i-1]-g)*s) + T(T(2*c)*b))
					p = T(s * r)
					d[i] = T(g + p)
					g = T(T(c*r) - b)
					// If eigenvectors are desired, then save the rotations.
					if icompz > 0 {
						work[i-1] = c
						work[n-1+i-1] = -s
					}
				}
				// If eigenvectors are desired, then apply the saved rotations.
				if icompz > 0 {
					mm := m - l + 1
					LasrRV(false, n, mm, work[l-1:], work[n-1+l-1:], z[(l-1)*ldz:], ldz)
				}
				d[l-1] = T(d[l-1] - p)
				e[l-1] = g
			}
		} else {
			// QR iteration. Look for a small superdiagonal element.
			for {
				m := lend
				if l != lend {
					for mm := l; mm >= lend+1; mm-- {
						tst := T(abs(e[mm-2]) * abs(e[mm-2]))
						if tst <= T(T(T(eps2*abs(d[mm-1]))*abs(d[mm-2]))+safmin) {
							m = mm
							break
						}
					}
				}
				if m > lend {
					e[m-2] = 0
				}
				p := d[l-1]
				if m == l {
					// Eigenvalue found.
					d[l-1] = p
					l--
					if l >= lend {
						continue
					}
					break
				}
				// If the remaining matrix is 2 by 2, use Lae2 or Laev2.
				if m == l-1 {
					var rt1, rt2 T
					if icompz > 0 {
						var c, s T
						rt1, rt2, c, s = Laev2(d[l-2], e[l-2], d[l-1])
						work[m-1] = c
						work[n-1+m-1] = s
						LasrRV(true, n, 2, work[m-1:], work[n-1+m-1:], z[(l-2)*ldz:], ldz)
					} else {
						rt1, rt2 = Lae2(d[l-2], e[l-2], d[l-1])
					}
					d[l-2] = rt1
					d[l-1] = rt2
					e[l-2] = 0
					l -= 2
					if l >= lend {
						continue
					}
					break
				}
				if jtot == nmaxit {
					break
				}
				jtot++
				// Form the shift.
				g := T(T(d[l-2]-p) / T(2*e[l-2]))
				r := Lapy2(g, 1)
				g = T(T(d[m-1]-p) + T(e[l-2]/T(g+sign(r, g))))
				s := T(1)
				c := T(1)
				p = 0
				// Inner loop.
				for i := m; i <= l-1; i++ {
					f := T(s * e[i-1])
					b := T(c * e[i-1])
					c, s, r = Lartg(g, f)
					if i != m {
						e[i-2] = r
					}
					g = T(d[i-1] - p)
					r = T(T(T(d[i]-g)*s) + T(T(2*c)*b))
					p = T(s * r)
					d[i-1] = T(g + p)
					g = T(T(c*r) - b)
					// If eigenvectors are desired, then save the rotations.
					if icompz > 0 {
						work[i-1] = c
						work[n-1+i-1] = s
					}
				}
				// If eigenvectors are desired, then apply the saved rotations.
				if icompz > 0 {
					mm := l - m + 1
					LasrRV(true, n, mm, work[m-1:], work[n-1+m-1:], z[(m-1)*ldz:], ldz)
				}
				d[l-1] = T(d[l-1] - p)
				e[l-2] = g
			}
		}
		// Undo scaling if necessary.
		if iscale == 1 {
			Lascl('G', ssfmax, anorm, lendsv-lsv+1, 1, d[lsv-1:], n)
			Lascl('G', ssfmax, anorm, lendsv-lsv, 1, e[lsv-1:], n)
		} else if iscale == 2 {
			Lascl('G', ssfmin, anorm, lendsv-lsv+1, 1, d[lsv-1:], n)
			Lascl('G', ssfmin, anorm, lendsv-lsv, 1, e[lsv-1:], n)
		}
		// Check for no convergence to an eigenvalue after a total of n*MaxIt
		// iterations.
		if jtot >= nmaxit {
			for i := 1; i <= n-1; i++ {
				if e[i-1] != 0 {
					info++
				}
			}
			return info
		}
	}
	// Order the eigenvalues and eigenvectors.
	if icompz == 0 {
		// Use quick sort.
		Lasrt(n, d)
		return 0
	}
	// Use selection sort to minimize swaps of eigenvectors.
	for ii := 2; ii <= n; ii++ {
		i := ii - 1
		k := i
		p := d[i-1]
		for j := ii; j <= n; j++ {
			if d[j-1] < p {
				k = j
				p = d[j-1]
			}
		}
		if k != i {
			d[k-1] = d[i-1]
			d[i-1] = p
			Swap(n, z[(i-1)*ldz:], 1, z[(k-1)*ldz:], 1)
		}
	}
	return 0
}

// Syev is xSYEV (dsyev.f): all the eigenvalues, and optionally the eigenvectors,
// of the n x n symmetric matrix a, whose upper or lower triangle is stored, the
// matrix being scaled when its largest element is outside the range [sqrt(safmin
// /eps), sqrt(eps/safmin)]. The eigenvalues are in w in increasing order; if wantz,
// a is overwritten by the orthonormal eigenvectors. info > 0: Steqr or Sterf
// failed to converge, info off-diagonal elements of an intermediate tridiagonal
// form not having converged to zero. n > 0.
func Syev[T Float](kn *Kern[T], wantz, upper bool, n int, a []T, lda int, w []T) (info int) {
	if n == 1 {
		w[0] = a[0]
		if wantz {
			a[0] = 1
		}
		return 0
	}
	// Get the machine constants.
	safmin := SafeMin[T]()
	eps := epsPrec[T]()
	smlnum := T(safmin / eps)
	bignum := T(1 / smlnum)
	rmin := sqrt(smlnum)
	rmax := sqrt(bignum)
	// Scale the matrix to the allowable range, if necessary.
	anrm := Lansy(upper, n, a, lda)
	iscale := false
	var sigma T
	if anrm > 0 && anrm < rmin {
		iscale = true
		sigma = T(rmin / anrm)
	} else if anrm > rmax {
		iscale = true
		sigma = T(rmax / anrm)
	}
	if iscale {
		typ := byte('L')
		if upper {
			typ = 'U'
		}
		Lascl(typ, 1, sigma, n, n, a, lda)
	}
	// Reduce the symmetric matrix to tridiagonal form.
	e := make([]T, n)
	tau := make([]T, n)
	Sytrd(kn, NBTRD, NXTRD, upper, n, a, lda, w, e, tau)
	// For eigenvalues only, call Sterf. For eigenvectors, first call Orgtr to
	// generate the orthogonal matrix, then Steqr.
	if !wantz {
		info = Sterf(n, w, e)
	} else {
		Orgtr(kn, NBQL, NXQL, NBQR, NXQR, upper, n, a, lda, tau)
		info = Steqr(1, n, w, e, a, lda)
	}
	// If the matrix was scaled, then rescale the eigenvalues appropriately.
	if iscale {
		imax := n
		if info != 0 {
			imax = info - 1
		}
		if imax > 0 {
			Scal(imax, T(1/sigma), w, 1)
		}
	}
	return info
}
