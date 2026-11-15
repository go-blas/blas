package blas_test

import (
	"errors"
	"math"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

// The fuzz targets drive every routine, of blas and of netlib, with arbitrary
// dimensions, leading dimensions, increments and slice lengths, valid or not.
// Each asserts, per input:
//
//   - nothing panics;
//   - blas and netlib agree on whether the call is an error, and on which
//     parameter (the checks are the same code, so a divergence is a bug in one
//     wrapper);
//   - an error leaves every output exactly as it was;
//   - a call without error, on finite data of moderate magnitude, is within the
//     stated bound of the math/big oracle, for both implementations.
//
// The seeds in testdata/fuzz are replayed by go test.

// modest reports whether every value is zero, or finite and of a magnitude
// whose products and sums neither overflow nor go subnormal: the domain on
// which the bounds are stated.
func modest[T float](vs ...[]T) bool {
	var hi, lo float64 = 1e15, 1e-30
	var z T
	if _, ok := any(z).(float32); ok {
		hi, lo = 1e5, 1e-10
	}
	for _, s := range vs {
		for _, v := range s {
			f := math.Abs(float64(v))
			if f != 0 && (!(f <= hi) || f < lo) { // !(f <= hi) is true for NaN
				return false
			}
		}
	}
	return true
}

func scalarsModest[T float](vs ...T) bool { return modest(vs) }

// fuzzData returns n values; flavor 0 is ordinary data, the others plant a NaN,
// an infinity, a huge or a tiny value in it.
func fuzzData[T float](seed uint64, n int, flavor uint8) []T {
	if n <= 0 {
		return nil
	}
	v := oracle.Fill[T](newRNG(seed), n)
	i := int(seed>>8) % n
	switch flavor % 5 {
	case 1:
		v[i] = T(math.NaN())
	case 2:
		v[i] = T(math.Inf(1))
	case 3:
		v[i] = T(math.MaxFloat32)
	case 4:
		v[i] = T(math.SmallestNonzeroFloat32)
	}
	return v
}

// sameErr reports whether two errors are both nil, or both *ArgError with the
// same routine and parameter.
func sameErr(a, b error) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	var ea, eb *blas.ArgError
	return errors.As(a, &ea) && errors.As(b, &eb) && *ea == *eb
}

// matLen is the length a rows x cols matrix with leading dimension ld needs,
// plus a delta that may make it too short.
func matLen(rows, cols, ld int, delta int8) int {
	n := 0
	if rows > 0 && cols > 0 {
		n = ld*(cols-1) + rows
	}
	return max(0, n+int(delta)%3)
}

func vecLen(n, inc int, delta int8) int {
	l := 0
	if n > 0 {
		l = 1 + (n-1)*max(inc, -inc)
	}
	return max(0, l+int(delta)%3)
}

func fuzzGemm[T float](t *testing.T, tA, tB byte, m8, n8, k8, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta T, flavor uint8, seed uint64, impls []impl[T]) {
	m, n, k := int(m8)%40, int(n8)%40, int(k8)%40
	trA, trB := blas.Transpose(tA), blas.Transpose(tB)
	rowsA, colsA, rowsB, colsB := m, k, k, n
	if trA != blas.NoTrans {
		rowsA, colsA = k, m
	}
	if trB != blas.NoTrans {
		rowsB, colsB = n, k
	}
	lda, ldb, ldc := max(1, rowsA)+int(ldaD)%4, max(1, rowsB)+int(ldbD)%4, max(1, m)+int(ldcD)%4
	a := fuzzData[T](seed, matLen(rowsA, colsA, lda, dA), flavor)
	b := fuzzData[T](seed+1, matLen(rowsB, colsB, ldb, dB), flavor)
	c0 := fuzzData[T](seed+2, matLen(m, n, ldc, dC), 0)
	var errs [2]error
	var outs [2][]T
	for i, im := range impls {
		outs[i] = clone(c0)
		errs[i] = im.gemm(trA, trB, m, n, k, alpha, a, lda, b, ldb, beta, outs[i], ldc)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil {
		if !same(outs[0], c0) || !same(outs[1], c0) {
			t.Fatal("an error modified C")
		}
		return
	}
	// A quick return writes nothing, and checks no slice it would not touch.
	if m == 0 || n == 0 || ((alpha == 0 || k == 0) && beta == 1) {
		if !same(outs[0], c0) || !same(outs[1], c0) {
			t.Fatal("a quick return changed C")
		}
		return
	}
	if flavor%5 != 0 || !modest(a, b, c0) || !scalarsModest(alpha, beta) {
		return
	}
	// The oracle reads only what the reference reads: A and B are never read for
	// alpha = 0, C not for beta = 0.
	r := oracle.Gemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c0, ldc)
	g := oracle.Gamma[T](k + 2)
	for i := range outs {
		for j := 0; j < n; j++ {
			for x := 0; x < m; x++ {
				within(t, impls[i].name, outs[i][x+j*ldc], r, x+j*m, g, g)
			}
		}
	}
}

func fuzzGemv[T float](t *testing.T, tr byte, m8, n8, ldaD int8, incx8, incy8 int8, dA, dX, dY int8, alpha, beta T, flavor uint8, seed uint64, impls []impl[T]) {
	m, n := int(m8)%40, int(n8)%40
	incx, incy := int(incx8)%5, int(incy8)%5
	lenx, leny := n, m
	if blas.Transpose(tr) != blas.NoTrans {
		lenx, leny = m, n
	}
	lda := max(1, m) + int(ldaD)%4
	a := fuzzData[T](seed, matLen(m, n, lda, dA), flavor)
	x := fuzzData[T](seed+1, vecLen(lenx, incx, dX), flavor)
	y0 := fuzzData[T](seed+2, vecLen(leny, incy, dY), 0)
	var errs [2]error
	var outs [2][]T
	for i, im := range impls {
		outs[i] = clone(y0)
		errs[i] = im.gemv(blas.Transpose(tr), m, n, alpha, a, lda, x, incx, beta, outs[i], incy)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil {
		if !same(outs[0], y0) || !same(outs[1], y0) {
			t.Fatal("an error modified y")
		}
		return
	}
	if flavor%5 != 0 || !modest(a, x, y0) || !scalarsModest(alpha, beta) || m == 0 || n == 0 || (alpha == 0 && beta == 1) {
		return
	}
	r := oracle.Gemv(tr, m, n, alpha, a, lda, x, incx, beta, y0, incy)
	g := oracle.Gamma[T](lenx + 2)
	for i := range outs {
		for e := 0; e < leny; e++ {
			within(t, impls[i].name, outs[i][oracle.Idx(leny, incy, e)], r, e, g, g)
		}
	}
}

func fuzzDot[T float](t *testing.T, n8, incx8, incy8, dX, dY int8, flavor uint8, seed uint64, impls []impl[T]) {
	n, incx, incy := int(n8)%40, int(incx8)%5, int(incy8)%5
	x := fuzzData[T](seed, vecLen(n, incx, dX), flavor)
	y := fuzzData[T](seed+1, vecLen(n, incy, dY), flavor)
	var errs [2]error
	var got [2]T
	for i, im := range impls {
		got[i], errs[i] = im.dot(n, x, incx, y, incy)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || flavor%5 != 0 || !modest(x, y) || n <= 0 {
		return
	}
	r := oracle.Dot(n, x, incx, y, incy)
	g := oracle.Gamma[T](n)
	for i := range got {
		within(t, impls[i].name, got[i], r, 0, g, g)
	}
}

func fuzzAxpy[T float](t *testing.T, n8, incx8, incy8, dX, dY int8, alpha T, flavor uint8, seed uint64, impls []impl[T]) {
	n, incx, incy := int(n8)%40, int(incx8)%5, int(incy8)%5
	x := fuzzData[T](seed, vecLen(n, incx, dX), flavor)
	y0 := fuzzData[T](seed+1, vecLen(n, incy, dY), 0)
	var errs [2]error
	var outs [2][]T
	for i, im := range impls {
		outs[i] = clone(y0)
		errs[i] = im.axpy(n, alpha, x, incx, outs[i], incy)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil {
		if !same(outs[0], y0) || !same(outs[1], y0) {
			t.Fatal("an error modified y")
		}
		return
	}
	// incy = 0 accumulates into one cell; the oracle's elementwise form does not apply.
	if flavor%5 != 0 || !modest(x, y0) || !scalarsModest(alpha) || n <= 0 || alpha == 0 || incy == 0 {
		return
	}
	r := oracle.Axpy(n, alpha, x, incx, y0, incy)
	g := oracle.Gamma[T](2)
	for i := range outs {
		for e := 0; e < n; e++ {
			within(t, impls[i].name, outs[i][oracle.Idx(n, incy, e)], r, e, g, g)
		}
	}
}

func fuzzScal[T float](t *testing.T, n8, incx8, dX int8, alpha T, flavor uint8, seed uint64, impls []impl[T]) {
	n, incx := int(n8)%40, int(incx8)%5
	x0 := fuzzData[T](seed, vecLen(n, incx, dX), flavor)
	var errs [2]error
	var outs [2][]T
	for i, im := range impls {
		outs[i] = clone(x0)
		errs[i] = im.scal(n, alpha, outs[i], incx)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil {
		if !same(outs[0], x0) || !same(outs[1], x0) {
			t.Fatal("an error modified x")
		}
		return
	}
	if flavor%5 != 0 || !modest(x0) || !scalarsModest(alpha) || n <= 0 || incx <= 0 || alpha == 1 {
		return
	}
	r := oracle.Scal(n, alpha, x0, incx)
	g := oracle.Gamma[T](1)
	for i := range outs {
		for e := 0; e < n; e++ {
			within(t, impls[i].name, outs[i][e*incx], r, e, g, g)
		}
	}
}

func fuzzNrm2[T float](t *testing.T, n8, incx8, dX int8, flavor uint8, seed uint64, impls []impl[T]) {
	n, incx := int(n8)%40, int(incx8)%5
	x := fuzzData[T](seed, vecLen(n, incx, dX), flavor)
	var errs [2]error
	var got [2]T
	for i, im := range impls {
		got[i], errs[i] = im.nrm2(n, x, incx)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	// nrm2 has no magnitude restriction: it is built not to overflow, but the
	// oracle's bound is only stated for the representable, normal results.
	if errs[0] != nil || flavor%5 != 0 || !modest(x) || n <= 0 {
		return
	}
	r := oracle.Nrm2(n, x, incx)
	g := oracle.Gamma[T](n + 2)
	for i := range got {
		within(t, impls[i].name, got[i], r, 0, g, g)
	}
}

func fuzzAsum[T float](t *testing.T, n8, incx8, dX int8, flavor uint8, seed uint64, impls []impl[T]) {
	n, incx := int(n8)%40, int(incx8)%5
	x := fuzzData[T](seed, vecLen(n, incx, dX), flavor)
	var errs [2]error
	var got [2]T
	for i, im := range impls {
		got[i], errs[i] = im.asum(n, x, incx)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || flavor%5 != 0 || !modest(x) || n <= 0 || incx <= 0 {
		return
	}
	r := oracle.Asum(n, x, incx)
	g := oracle.Gamma[T](n)
	for i := range got {
		within(t, impls[i].name, got[i], r, 0, g, g)
	}
}

func fuzzIamax[T float](t *testing.T, n8, incx8, dX int8, flavor uint8, seed uint64, impls []impl[T]) {
	n, incx := int(n8)%40, int(incx8)%5
	x := fuzzData[T](seed, vecLen(n, incx, dX), flavor)
	var errs [2]error
	var got [2]int
	for i, im := range impls {
		got[i], errs[i] = im.iamax(n, x, incx)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil {
		return
	}
	if got[0] != got[1] {
		t.Fatalf("blas = %d, netlib = %d", got[0], got[1])
	}
	if n >= 1 && incx > 0 && (got[0] < 1 || got[0] > n) {
		t.Fatalf("index %d outside 1..%d", got[0], n)
	}
	if (n < 1 || incx <= 0) && got[0] != 0 {
		t.Fatalf("index %d for an empty call", got[0])
	}
}

func FuzzDgemm(f *testing.F) {
	f.Add(byte('N'), byte('N'), int8(5), int8(6), int8(7), int8(0), int8(1), int8(2), int8(0), int8(0), int8(0), 1.5, -0.5, uint8(0), uint64(1))
	f.Add(byte('T'), byte('C'), int8(9), int8(3), int8(30), int8(2), int8(0), int8(0), int8(0), int8(-1), int8(0), 1.0, 0.0, uint8(0), uint64(2))
	f.Add(byte('N'), byte('T'), int8(0), int8(3), int8(4), int8(0), int8(0), int8(0), int8(0), int8(0), int8(0), 2.0, 1.0, uint8(1), uint64(3))
	f.Fuzz(func(t *testing.T, tA, tB byte, m, n, k, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float64, flavor uint8, seed uint64) {
		fuzzGemm(t, tA, tB, m, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, impls64)
	})
}

func FuzzSgemm(f *testing.F) {
	f.Add(byte('N'), byte('N'), int8(5), int8(6), int8(7), int8(0), int8(1), int8(2), int8(0), int8(0), int8(0), float32(1.5), float32(-0.5), uint8(0), uint64(1))
	f.Add(byte('T'), byte('T'), int8(10), int8(7), int8(33), int8(0), int8(3), int8(0), int8(0), int8(0), int8(-2), float32(0.5), float32(0), uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, tA, tB byte, m, n, k, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float32, flavor uint8, seed uint64) {
		fuzzGemm(t, tA, tB, m, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, impls32)
	})
}

func FuzzDgemv(f *testing.F) {
	f.Add(byte('N'), int8(6), int8(7), int8(0), int8(1), int8(1), int8(0), int8(0), int8(0), 1.5, 0.5, uint8(0), uint64(1))
	f.Add(byte('T'), int8(9), int8(4), int8(1), int8(-2), int8(3), int8(0), int8(-1), int8(0), -1.0, 0.0, uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, tr byte, m, n, ldaD, incx, incy, dA, dX, dY int8, alpha, beta float64, flavor uint8, seed uint64) {
		fuzzGemv(t, tr, m, n, ldaD, incx, incy, dA, dX, dY, alpha, beta, flavor, seed, impls64)
	})
}

func FuzzSgemv(f *testing.F) {
	f.Add(byte('N'), int8(6), int8(7), int8(0), int8(1), int8(1), int8(0), int8(0), int8(0), float32(1.5), float32(0.5), uint8(0), uint64(1))
	f.Add(byte('C'), int8(9), int8(4), int8(1), int8(1), int8(-3), int8(0), int8(0), int8(-1), float32(-1), float32(2), uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, tr byte, m, n, ldaD, incx, incy, dA, dX, dY int8, alpha, beta float32, flavor uint8, seed uint64) {
		fuzzGemv(t, tr, m, n, ldaD, incx, incy, dA, dX, dY, alpha, beta, flavor, seed, impls32)
	})
}

func FuzzDdot(f *testing.F) {
	f.Add(int8(10), int8(1), int8(1), int8(0), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(-2), int8(3), int8(0), int8(-1), uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, incy, dX, dY int8, flavor uint8, seed uint64) {
		fuzzDot(t, n, incx, incy, dX, dY, flavor, seed, impls64)
	})
}

func FuzzSdot(f *testing.F) {
	f.Add(int8(10), int8(1), int8(1), int8(0), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(2), int8(-3), int8(0), int8(0), uint8(1), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, incy, dX, dY int8, flavor uint8, seed uint64) {
		fuzzDot(t, n, incx, incy, dX, dY, flavor, seed, impls32)
	})
}

func FuzzDaxpy(f *testing.F) {
	f.Add(int8(10), int8(1), int8(1), int8(0), int8(0), 2.5, uint8(0), uint64(1))
	f.Add(int8(7), int8(-2), int8(3), int8(0), int8(-1), -1.0, uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, incy, dX, dY int8, alpha float64, flavor uint8, seed uint64) {
		fuzzAxpy(t, n, incx, incy, dX, dY, alpha, flavor, seed, impls64)
	})
}

func FuzzSaxpy(f *testing.F) {
	f.Add(int8(10), int8(1), int8(1), int8(0), int8(0), float32(2.5), uint8(0), uint64(1))
	f.Add(int8(7), int8(2), int8(-3), int8(-1), int8(0), float32(0), uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, incy, dX, dY int8, alpha float32, flavor uint8, seed uint64) {
		fuzzAxpy(t, n, incx, incy, dX, dY, alpha, flavor, seed, impls32)
	})
}

func FuzzDscal(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), 2.5, uint8(0), uint64(1))
	f.Add(int8(7), int8(3), int8(-1), 0.0, uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, alpha float64, flavor uint8, seed uint64) {
		fuzzScal(t, n, incx, dX, alpha, flavor, seed, impls64)
	})
}

func FuzzSscal(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), float32(2.5), uint8(0), uint64(1))
	f.Add(int8(7), int8(-3), int8(0), float32(0), uint8(0), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, alpha float32, flavor uint8, seed uint64) {
		fuzzScal(t, n, incx, dX, alpha, flavor, seed, impls32)
	})
}

func FuzzDnrm2(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(-3), int8(-1), uint8(2), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, flavor uint8, seed uint64) {
		fuzzNrm2(t, n, incx, dX, flavor, seed, impls64)
	})
}

func FuzzSnrm2(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(2), int8(0), uint8(3), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, flavor uint8, seed uint64) {
		fuzzNrm2(t, n, incx, dX, flavor, seed, impls32)
	})
}

func FuzzDasum(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(3), int8(-1), uint8(1), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, flavor uint8, seed uint64) {
		fuzzAsum(t, n, incx, dX, flavor, seed, impls64)
	})
}

func FuzzSasum(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(3), int8(0), uint8(4), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, flavor uint8, seed uint64) {
		fuzzAsum(t, n, incx, dX, flavor, seed, impls32)
	})
}

func FuzzIdamax(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(2), int8(0), uint8(1), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, flavor uint8, seed uint64) {
		fuzzIamax(t, n, incx, dX, flavor, seed, impls64)
	})
}

func FuzzIsamax(f *testing.F) {
	f.Add(int8(10), int8(1), int8(0), uint8(0), uint64(1))
	f.Add(int8(7), int8(-2), int8(0), uint8(2), uint64(2))
	f.Fuzz(func(t *testing.T, n, incx, dX int8, flavor uint8, seed uint64) {
		fuzzIamax(t, n, incx, dX, flavor, seed, impls32)
	})
}
