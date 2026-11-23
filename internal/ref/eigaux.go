package ref

import "math"

// Ports of the auxiliary routines of the symmetric eigensolver, Reference-LAPACK
// 3.12.0 (tag v3.12.0, SRC): dlartg.f90, dlascl.f, dlasrt.f, dlanst.f, dlansy.f,
// dlae2.f, dlaev2.f, dlasr.f, dlamch.f and the level 2 BLAS dtrmv.f for a lower
// non-unit matrix, with their single precision twins. Same rules as qr.go: 0-based
// indices, no FMA, the arguments already checked. Only the variants the
// eigensolver calls are ported: DLASCL for TYPE = 'G', 'L', 'U'; DLASRT for ID =
// 'I'; DLANST and DLANSY for NORM = 'M'; DLASR for SIDE = 'R', PIVOT = 'V'.

// epsRound is DLAMCH('E'): the relative machine epsilon, 2^-53 for float64 and
// 2^-24 for float32 (rounding arithmetic).
func epsRound[T Float]() T {
	if bits64[T]() {
		return T(math.Ldexp(1, -53))
	}
	return T(math.Ldexp(1, -24))
}

// epsPrec is DLAMCH('P'): eps*base, 2^-52 for float64 and 2^-23 for float32.
func epsPrec[T Float]() T { return 2 * epsRound[T]() }

// Lartg is xLARTG (dlartg.f90, 3.12): the plane rotation [c s; -s c] taking [f;
// g] to [r; 0], by the safe-scaling algorithm of Anderson (2017).
func Lartg[T Float](f, g T) (c, s, r T) {
	safmin := SafeMin[T]()
	safmax := T(1 / safmin)
	rtmin := sqrt(safmin)
	rtmax := sqrt(T(safmax / 2))
	f1, g1 := abs(f), abs(g)
	switch {
	case g == 0:
		return 1, 0, f
	case f == 0:
		return 0, sign(T(1), g), g1
	case f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax:
		d := sqrt(T(T(f*f) + T(g*g)))
		c = T(f1 / d)
		r = sign(d, f)
		s = T(g / r)
		return c, s, r
	}
	u := min(safmax, max(safmin, f1, g1))
	fs := T(f / u)
	gs := T(g / u)
	d := sqrt(T(T(fs*fs) + T(gs*gs)))
	c = T(abs(fs) / d)
	r = sign(d, f)
	s = T(gs / r)
	r = T(r * u)
	return c, s, r
}

// Lascl is xLASCL (dlascl.f) for TYPE = 'G' (general), 'L' (lower triangular) and
// 'U' (upper triangular), the m x n matrix a: it multiplies a by cto/cfrom without
// over- or underflow. cfrom must be neither zero nor NaN, nor cto NaN.
func Lascl[T Float](typ byte, cfrom, cto T, m, n int, a []T, lda int) {
	// m = 0 or n = 0 needs no quick return: the loops below do not run (the
	// reference returns first, and reads nothing either way).
	smlnum := SafeMin[T]()
	bignum := T(1 / smlnum)
	cfromc, ctoc := cfrom, cto
	for done := false; !done; {
		var mul T
		cfrom1 := T(cfromc * smlnum)
		if cfrom1 == cfromc {
			// cfromc is an inf: multiply by a correctly signed zero for finite ctoc,
			// or a NaN if ctoc is infinite.
			mul = T(ctoc / cfromc)
			done = true
		} else {
			cto1 := T(ctoc / bignum)
			switch {
			case cto1 == ctoc:
				// ctoc is 0 or an inf: it serves as the multiplication factor.
				mul = ctoc
				done = true
				cfromc = 1
			case abs(cfrom1) > abs(ctoc) && ctoc != 0:
				mul = smlnum
				cfromc = cfrom1
			case abs(cto1) > abs(cfromc):
				mul = bignum
				ctoc = cto1
			default:
				mul = T(ctoc / cfromc)
				done = true
			}
		}
		for j := 0; j < n; j++ {
			lo, hi := 0, m-1
			switch typ {
			case 'L':
				lo = j
			case 'U':
				hi = min(j, m-1)
			}
			for i := lo; i <= hi; i++ {
				a[i+j*lda] = T(a[i+j*lda] * mul)
			}
		}
	}
}

// Lasrt is xLASRT for ID = 'I' (dlasrt.f): it sorts d into increasing order by
// quicksort with insertion sort for short partitions, as the reference does.
func Lasrt[T Float](n int, d []T) {
	const sel = 20
	// n <= 1 needs no quick return: the partition loop leaves such a range
	// unchanged (TestLasrtEdges).
	var stack [32][2]int
	stkpnt := 0
	stack[0] = [2]int{1, n}
	for stkpnt >= 0 {
		start, endd := stack[stkpnt][0], stack[stkpnt][1]
		stkpnt--
		if endd-start <= sel && endd-start > 0 {
			// Insertion sort on d(start:endd).
			for i := start + 1; i <= endd; i++ {
				for j := i; j >= start+1; j-- {
					if d[j-1] < d[j-2] {
						d[j-1], d[j-2] = d[j-2], d[j-1]
					} else {
						break
					}
				}
			}
		} else if endd-start > sel {
			// Partition d(start:endd) and stack the parts, the largest first, the
			// partition entry being the median of three.
			d1, d2, d3 := d[start-1], d[endd-1], d[(start+endd)/2-1]
			var dmnmx T
			if d1 < d2 {
				switch {
				case d3 < d1:
					dmnmx = d1
				case d3 < d2:
					dmnmx = d3
				default:
					dmnmx = d2
				}
			} else {
				switch {
				case d3 < d2:
					dmnmx = d2
				case d3 < d1:
					dmnmx = d3
				default:
					dmnmx = d1
				}
			}
			i, j := start-1, endd+1
			for {
				for {
					j--
					if !(d[j-1] > dmnmx) {
						break
					}
				}
				for {
					i++
					if !(d[i-1] < dmnmx) {
						break
					}
				}
				if i < j {
					d[i-1], d[j-1] = d[j-1], d[i-1]
					continue
				}
				break
			}
			if j-start > endd-j-1 {
				stkpnt++
				stack[stkpnt] = [2]int{start, j}
				stkpnt++
				stack[stkpnt] = [2]int{j + 1, endd}
			} else {
				stkpnt++
				stack[stkpnt] = [2]int{j + 1, endd}
				stkpnt++
				stack[stkpnt] = [2]int{start, j}
			}
		}
	}
}

// Lanst is xLANST('M') (dlanst.f): the largest absolute value of the symmetric
// tridiagonal matrix with diagonal d and off-diagonal e, a NaN being returned if
// there is one, n > 0.
func Lanst[T Float](n int, d, e []T) T {
	anorm := abs(d[n-1])
	for i := 0; i < n-1; i++ {
		if sum := abs(d[i]); anorm < sum || sum != sum {
			anorm = sum
		}
		if sum := abs(e[i]); anorm < sum || sum != sum {
			anorm = sum
		}
	}
	return anorm
}

// Lansy is xLANSY('M') (dlansy.f): the largest absolute value of the triangle of
// the symmetric matrix a, a NaN being returned if there is one.
func Lansy[T Float](upper bool, n int, a []T, lda int) T {
	var value T
	for j := 0; j < n; j++ {
		lo, hi := j, n-1
		if upper {
			lo, hi = 0, j
		}
		for i := lo; i <= hi; i++ {
			if sum := abs(a[i+j*lda]); value < sum || sum != sum {
				value = sum
			}
		}
	}
	return value
}

// Lae2 is xLAE2 (dlae2.f): the eigenvalues rt1 (larger in absolute value) and rt2
// of the symmetric 2 x 2 matrix [a b; b c].
func Lae2[T Float](a, b, c T) (rt1, rt2 T) {
	sm := T(a + c)
	df := T(a - c)
	adf := abs(df)
	tb := T(b + b)
	ab := abs(tb)
	acmx, acmn := c, a
	if abs(a) > abs(c) {
		acmx, acmn = a, c
	}
	var rt T
	switch {
	case adf > ab:
		q := T(ab / adf)
		rt = T(adf * sqrt(T(1+T(q*q))))
	case adf < ab:
		q := T(adf / ab)
		rt = T(ab * sqrt(T(1+T(q*q))))
	default:
		// Includes the case ab = adf = 0.
		rt = T(ab * sqrt(T(2)))
	}
	switch {
	case sm < 0:
		rt1 = T(0.5 * T(sm-rt))
		rt2 = T(T(T(acmx/rt1)*acmn) - T(T(b/rt1)*b))
	case sm > 0:
		rt1 = T(0.5 * T(sm+rt))
		rt2 = T(T(T(acmx/rt1)*acmn) - T(T(b/rt1)*b))
	default:
		// Includes the case rt1 = rt2 = 0.
		rt1 = T(0.5 * rt)
		rt2 = T(-0.5 * rt)
	}
	return rt1, rt2
}

// Laev2 is xLAEV2 (dlaev2.f): Lae2 and the unit right eigenvector (cs1, sn1) for
// rt1, so that [cs1 sn1; -sn1 cs1] diagonalizes [a b; b c].
func Laev2[T Float](a, b, c T) (rt1, rt2, cs1, sn1 T) {
	sm := T(a + c)
	df := T(a - c)
	adf := abs(df)
	tb := T(b + b)
	ab := abs(tb)
	acmx, acmn := c, a
	if abs(a) > abs(c) {
		acmx, acmn = a, c
	}
	var rt T
	switch {
	case adf > ab:
		q := T(ab / adf)
		rt = T(adf * sqrt(T(1+T(q*q))))
	case adf < ab:
		q := T(adf / ab)
		rt = T(ab * sqrt(T(1+T(q*q))))
	default:
		rt = T(ab * sqrt(T(2)))
	}
	var sgn1 int
	switch {
	case sm < 0:
		rt1 = T(0.5 * T(sm-rt))
		sgn1 = -1
		rt2 = T(T(T(acmx/rt1)*acmn) - T(T(b/rt1)*b))
	case sm > 0:
		rt1 = T(0.5 * T(sm+rt))
		sgn1 = 1
		rt2 = T(T(T(acmx/rt1)*acmn) - T(T(b/rt1)*b))
	default:
		rt1 = T(0.5 * rt)
		rt2 = T(-0.5 * rt)
		sgn1 = 1
	}
	// Compute the eigenvector.
	var cs T
	var sgn2 int
	if df >= 0 {
		cs = T(df + rt)
		sgn2 = 1
	} else {
		cs = T(df - rt)
		sgn2 = -1
	}
	acs := abs(cs)
	if acs > ab {
		ct := T(-tb / cs)
		sn1 = T(1 / sqrt(T(1+T(ct*ct))))
		cs1 = T(ct * sn1)
	} else if ab == 0 {
		cs1 = 1
		sn1 = 0
	} else {
		tn := T(-cs / tb)
		cs1 = T(1 / sqrt(T(1+T(tn*tn))))
		sn1 = T(tn * cs1)
	}
	if sgn1 == sgn2 {
		tn := cs1
		cs1 = -sn1
		sn1 = tn
	}
	return rt1, rt2, cs1, sn1
}

// LasrRV is xLASR with SIDE = 'R', PIVOT = 'V' (dlasr.f): it applies the sequence
// of n-1 plane rotations (c(j), s(j)), in the plane (j, j+1), to the m x n matrix a
// from the right, A := A*P**T, forward (j = 1 .. n-1) or backward.
func LasrRV[T Float](forward bool, m, n int, c, s []T, a []T, lda int) {
	rot := func(j int) {
		ctemp, stemp := c[j], s[j]
		if ctemp != 1 || stemp != 0 {
			for i := 0; i < m; i++ {
				temp := a[i+(j+1)*lda]
				a[i+(j+1)*lda] = T(ctemp*temp) - T(stemp*a[i+j*lda])
				a[i+j*lda] = T(stemp*temp) + T(ctemp*a[i+j*lda])
			}
		}
	}
	if forward {
		for j := 0; j < n-1; j++ {
			rot(j)
		}
		return
	}
	for j := n - 2; j >= 0; j-- {
		rot(j)
	}
}

// trmvLN is xTRMV('Lower', 'No transpose', 'Non-unit', n, a, lda, x, 1), n > 0:
// x := A*x for the lower triangular n x n matrix a (dtrmv.f).
func trmvLN[T Float](n int, a []T, lda int, x []T) {
	for j := n - 1; j >= 0; j-- {
		if x[j] != 0 {
			temp := x[j]
			for i := n - 1; i > j; i-- {
				x[i] = x[i] + T(temp*a[i+j*lda])
			}
			x[j] = T(x[j] * a[j+j*lda])
		}
	}
}
