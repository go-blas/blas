// Package ref holds plain-Go ports of the reference BLAS 3.12 routines this
// module ships (Reference-LAPACK/lapack, tag v3.12.0, BLAS/SRC): dgemm.f,
// sgemm.f, dgemv.f, sgemv.f, ddot.f, sdot.f, daxpy.f, saxpy.f, dscal.f,
// sscal.f, dnrm2.f90, snrm2.f90, dasum.f, sasum.f, idamax.f and isamax.f. The
// double and single precision routines differ only in the element type and, for
// nrm2, in four constants, so each is written once, generic over float32 and
// float64.
//
// The loops are the reference's, in its order, with one rule of this port: no
// fused multiply-add. Go is allowed to fuse x*y+z on arm64, ppc64le, riscv64
// and s390x; every product here is wrapped in an explicit conversion, which the
// Go specification says rounds and so forbids fusing. The results are therefore
// the same on every architecture.
//
// The routines assume valid arguments, which package check establishes and the
// caller has already verified: indices are 0-based into column-major slices.
// They are not the public API; packages blas and netlib wrap them.
package ref

import "math"

// Float is the element types of the real BLAS.
type Float interface{ float32 | float64 }

// Gemm is xGEMM: C := alpha*op(A)*op(B) + beta*C, with op(X) = X or X**T, A
// m x k, B k x n and C m x n, all column-major. notA and notB say that A and B
// are not transposed. It is dgemm.f/sgemm.f without the argument tests.
func Gemm[T Float](notA, notB bool, m, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {
	// The quick return of the reference (m = 0, n = 0, or alpha = 0 or k = 0
	// with beta = 1) is the caller's, which has also checked the arguments.
	// And if alpha.eq.zero.
	if alpha == 0 {
		if beta == 0 {
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					c[i+j*ldc] = 0
				}
			}
		} else {
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					c[i+j*ldc] = T(beta * c[i+j*ldc])
				}
			}
		}
		return
	}
	switch {
	case notB && notA:
		// Form C := alpha*A*B + beta*C.
		for j := 0; j < n; j++ {
			scaleCol(c[j*ldc:j*ldc+m], beta)
			for l := 0; l < k; l++ {
				temp := T(alpha * b[l+j*ldb])
				for i := 0; i < m; i++ {
					c[i+j*ldc] = c[i+j*ldc] + T(temp*a[i+l*lda])
				}
			}
		}
	case notB:
		// Form C := alpha*A**T*B + beta*C.
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				var temp T
				for l := 0; l < k; l++ {
					temp = temp + T(a[l+i*lda]*b[l+j*ldb])
				}
				if beta == 0 {
					c[i+j*ldc] = T(alpha * temp)
				} else {
					c[i+j*ldc] = T(alpha*temp) + T(beta*c[i+j*ldc])
				}
			}
		}
	case notA:
		// Form C := alpha*A*B**T + beta*C.
		for j := 0; j < n; j++ {
			scaleCol(c[j*ldc:j*ldc+m], beta)
			for l := 0; l < k; l++ {
				temp := T(alpha * b[j+l*ldb])
				for i := 0; i < m; i++ {
					c[i+j*ldc] = c[i+j*ldc] + T(temp*a[i+l*lda])
				}
			}
		}
	default:
		// Form C := alpha*A**T*B**T + beta*C.
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				var temp T
				for l := 0; l < k; l++ {
					temp = temp + T(a[l+i*lda]*b[j+l*ldb])
				}
				if beta == 0 {
					c[i+j*ldc] = T(alpha * temp)
				} else {
					c[i+j*ldc] = T(alpha*temp) + T(beta*c[i+j*ldc])
				}
			}
		}
	}
}

// scaleCol is the beta step of one column of C in the A*B forms: zero when
// beta is zero (C is not read), unchanged when it is one, else beta*C.
func scaleCol[T Float](col []T, beta T) {
	switch beta {
	case 0:
		clear(col)
	case 1:
	default:
		for i := range col {
			col[i] = T(beta * col[i])
		}
	}
}

// vecStart is the 0-based index of the first element of an n-element vector
// with increment inc: the reference's KX = 1-(LENX-1)*INCX for a negative one.
func vecStart(n, inc int) int {
	if inc > 0 {
		return 0
	}
	return -(n - 1) * inc
}

// Gemv is xGEMV: y := alpha*op(A)*x + beta*y, A m x n column-major, with
// op(A) = A when notA and A**T otherwise. It is dgemv.f/sgemv.f without the
// argument tests.
func Gemv[T Float](notA bool, m, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) {
	// The quick return of the reference is the caller's.
	lenx, leny := n, m
	if !notA {
		lenx, leny = m, n
	}
	kx, ky := vecStart(lenx, incx), vecStart(leny, incy)
	// First form y := beta*y.
	if beta != 1 {
		iy := ky
		for i := 0; i < leny; i++ {
			if beta == 0 {
				y[iy] = 0
			} else {
				y[iy] = T(beta * y[iy])
			}
			iy += incy
		}
	}
	if alpha == 0 {
		return
	}
	if notA {
		// Form y := alpha*A*x + y.
		jx := kx
		for j := 0; j < n; j++ {
			temp := T(alpha * x[jx])
			iy := ky
			for i := 0; i < m; i++ {
				y[iy] = y[iy] + T(temp*a[i+j*lda])
				iy += incy
			}
			jx += incx
		}
		return
	}
	// Form y := alpha*A**T*x + y.
	jy := ky
	for j := 0; j < n; j++ {
		var temp T
		ix := kx
		for i := 0; i < m; i++ {
			temp = temp + T(a[i+j*lda]*x[ix])
			ix += incx
		}
		y[jy] = y[jy] + T(alpha*temp)
		jy += incy
	}
}

// Dot is xDOT: the sum of x[i]*y[i], accumulated in order from i = 0.
func Dot[T Float](n int, x []T, incx int, y []T, incy int) T {
	var temp T
	if n <= 0 {
		return 0
	}
	ix, iy := vecStart(n, incx), vecStart(n, incy)
	for i := 0; i < n; i++ {
		temp = temp + T(x[ix]*y[iy])
		ix += incx
		iy += incy
	}
	return temp
}

// Axpy is xAXPY: y := alpha*x + y, for n > 0 and alpha != 0 (the reference's
// quick returns are the caller's).
func Axpy[T Float](n int, alpha T, x []T, incx int, y []T, incy int) {
	ix, iy := vecStart(n, incx), vecStart(n, incy)
	for i := 0; i < n; i++ {
		y[iy] = y[iy] + T(alpha*x[ix])
		ix += incx
		iy += incy
	}
}

// Scal is xSCAL: x := alpha*x, for n > 0, incx > 0 (the reference's quick
// returns, which include alpha = 1, are the caller's).
func Scal[T Float](n int, alpha T, x []T, incx int) {
	for i, ix := 0, 0; i < n; i, ix = i+1, ix+incx {
		x[ix] = T(alpha * x[ix])
	}
}

// Asum is xASUM: the sum of |x[i]|, for n > 0 and incx > 0 (otherwise the
// caller returns 0).
func Asum[T Float](n int, x []T, incx int) T {
	var temp T
	for i, ix := 0, 0; i < n; i, ix = i+1, ix+incx {
		temp = temp + abs(x[ix])
	}
	return temp
}

// Iamax is IxAMAX: the 1-based index of the first element of largest absolute
// value, for n >= 1 and incx > 0 (otherwise the caller returns 0).
func Iamax[T Float](n int, x []T, incx int) int {
	best, big := 1, abs(x[0])
	for i, ix := 1, incx; i < n; i, ix = i+1, ix+incx {
		if v := abs(x[ix]); v > big {
			best, big = i+1, v
		}
	}
	return best
}

func abs[T Float](v T) T {
	if v < 0 {
		return -v
	}
	if v == 0 {
		return 0 // -0 -> +0
	}
	return v
}

// Nrm2 is xNRM2, the 3.10+ algorithm of Anderson (LAWN 250, after Blue):
// the sum of squares is accumulated in three bins, small, medium and big values,
// so that it neither underflows nor overflows when the norm itself is
// representable. A negative increment walks the vector from its far end, as in
// the reference.
func Nrm2[T Float](n int, x []T, incx int) T {
	if n <= 0 {
		return 0
	}
	tsml, tbig, ssml, sbig := nrm2Consts[T]()
	largest := math.MaxFloat32
	if bits64[T]() {
		largest = math.MaxFloat64
	}
	maxN := T(largest)
	var abig, amed, asml T
	notbig := true
	ix := vecStart(n, incx)
	for i := 0; i < n; i++ {
		ax := abs(x[ix])
		switch {
		case ax > tbig:
			t := T(ax * sbig)
			abig = abig + T(t*t)
			notbig = false
		case ax < tsml:
			if notbig {
				t := T(ax * ssml)
				asml = asml + T(t*t)
			}
		default:
			amed = amed + T(ax*ax)
		}
		ix += incx
	}
	var scl, sumsq T
	switch {
	case abig > 0:
		// Put the existing sum of squares into one of the other bins.
		if amed > 0 || amed > maxN || amed != amed {
			abig = abig + T(T(amed*sbig)*sbig)
		}
		scl = 1 / sbig
		sumsq = abig
	case asml > 0:
		// Combine abig and amed or amed and asml if more than one accumulator
		// was used.
		if amed > 0 || amed > maxN || amed != amed {
			amed = sqrt(amed)
			asml = sqrt(asml) / ssml
			ymin, ymax := asml, amed
			if asml > amed {
				ymin, ymax = amed, asml
			}
			scl = 1
			r := T(ymin / ymax)
			sumsq = T(ymax*ymax) * (1 + T(r*r))
		} else {
			scl = 1 / ssml
			sumsq = asml
		}
	default:
		// Only one accumulator is used.
		scl = 1
		sumsq = amed
	}
	return T(scl * sqrt(sumsq))
}

// bits64 reports whether T is float64.
func bits64[T Float]() bool {
	var z T
	_, ok := any(z).(float64)
	return ok
}

func sqrt[T Float](v T) T { return T(math.Sqrt(float64(v))) }

// nrm2Consts returns tsml, tbig, ssml and sbig of dnrm2.f90 and snrm2.f90,
// powers of two computed from the exponent range and digits of the type:
//
//	tsml = 2**ceil((minexponent-1)/2)       tbig = 2**floor((maxexponent-digits+1)/2)
//	ssml = 2**-floor((minexponent-digits)/2)  sbig = 2**-ceil((maxexponent+digits-1)/2)
//
// with minexponent -1021, maxexponent 1024, digits 53 for float64 and -125, 128,
// 24 for float32.
func nrm2Consts[T Float]() (tsml, tbig, ssml, sbig T) {
	if bits64[T]() {
		return T(math.Ldexp(1, -511)), T(math.Ldexp(1, 486)), T(math.Ldexp(1, 537)), T(math.Ldexp(1, -538))
	}
	return T(math.Ldexp(1, -63)), T(math.Ldexp(1, 52)), T(math.Ldexp(1, 75)), T(math.Ldexp(1, -76))
}
