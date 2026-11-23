package ref

// Ports of the reference BLAS 3.12.0 routines dsymv.f, dsyr2.f, drot.f, dswap.f
// and dcopy.f (and their single precision twins), generic over float32 and
// float64. Same rules as ref.go: the loops are the reference's, 0-based indices,
// every product wrapped to forbid FMA, the arguments already checked, and the
// quick returns of the reference the caller's.

// Symv is xSYMV: y := alpha*A*x + beta*y, A the n x n symmetric matrix whose
// upper (upper) or lower triangle is stored in a, n > 0.
func Symv[T Float](upper bool, n int, alpha T, a []T, lda int, x []T, incx int, beta T, y []T, incy int) {
	kx, ky := vecStart(n, incx), vecStart(n, incy)
	// First form y := beta*y.
	if beta != 1 {
		iy := ky
		for i := 0; i < n; i++ {
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
	jx, jy := kx, ky
	for j := 0; j < n; j++ {
		temp1 := T(alpha * x[jx])
		var temp2 T
		ix, iy := kx, ky
		if upper {
			for i := 0; i < j; i++ {
				y[iy] = y[iy] + T(temp1*a[i+j*lda])
				temp2 = temp2 + T(a[i+j*lda]*x[ix])
				ix += incx
				iy += incy
			}
			y[jy] = T(y[jy]+T(temp1*a[j+j*lda])) + T(alpha*temp2)
		} else {
			y[jy] = y[jy] + T(temp1*a[j+j*lda])
			ix, iy = jx, jy
			for i := j + 1; i < n; i++ {
				ix += incx
				iy += incy
				y[iy] = y[iy] + T(temp1*a[i+j*lda])
				temp2 = temp2 + T(a[i+j*lda]*x[ix])
			}
			y[jy] = y[jy] + T(alpha*temp2)
		}
		jx += incx
		jy += incy
	}
}

// Syr2 is xSYR2: A := alpha*x*y**T + alpha*y*x**T + A on the upper or lower
// triangle of the n x n symmetric matrix a, n > 0, alpha != 0.
func Syr2[T Float](upper bool, n int, alpha T, x []T, incx int, y []T, incy int, a []T, lda int) {
	kx, ky := vecStart(n, incx), vecStart(n, incy)
	jx, jy := kx, ky
	for j := 0; j < n; j++ {
		if x[jx] != 0 || y[jy] != 0 {
			temp1 := T(alpha * y[jy])
			temp2 := T(alpha * x[jx])
			lo, hi := j, n-1
			ix, iy := jx, jy
			if upper {
				lo, hi = 0, j
				ix, iy = kx, ky
			}
			for i := lo; i <= hi; i++ {
				a[i+j*lda] = T(a[i+j*lda]+T(x[ix]*temp1)) + T(y[iy]*temp2)
				ix += incx
				iy += incy
			}
		}
		jx += incx
		jy += incy
	}
}

// Rot is xROT: the plane rotation of the vectors x and y, n > 0.
func Rot[T Float](n int, x []T, incx int, y []T, incy int, c, s T) {
	ix, iy := vecStart(n, incx), vecStart(n, incy)
	for i := 0; i < n; i++ {
		temp := T(c*x[ix]) + T(s*y[iy])
		y[iy] = T(c*y[iy]) - T(s*x[ix])
		x[ix] = temp
		ix += incx
		iy += incy
	}
}

// Swap is xSWAP: it exchanges x and y, n > 0.
func Swap[T Float](n int, x []T, incx int, y []T, incy int) {
	ix, iy := vecStart(n, incx), vecStart(n, incy)
	for i := 0; i < n; i++ {
		x[ix], y[iy] = y[iy], x[ix]
		ix += incx
		iy += incy
	}
}

// Copy is xCOPY: y := x, n > 0.
func Copy[T Float](n int, x []T, incx int, y []T, incy int) {
	ix, iy := vecStart(n, incx), vecStart(n, incy)
	for i := 0; i < n; i++ {
		y[iy] = x[ix]
		ix += incx
		iy += incy
	}
}
