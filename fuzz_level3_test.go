package blas_test

import (
	"math"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

// Fuzz targets of the level 3 routines, with the properties of the others (see
// fuzz_test.go): no panic; blas and netlib agree on the error; an error or a
// quick return leaves the outputs as they were; with a planted NaN or infinity
// the two are of the same class; on finite data of moderate magnitude each is
// within its stated bound of the oracle. The diagonal block order of package blas
// is chosen by the input (blk), so that matrices of up to 39 rows span several
// blocks.

// fuzzBlock sets the diagonal block order from blk: the default for 0, else 1..24.
func fuzzBlock(t *testing.T, blk uint8) {
	t.Helper()
	if bs := int(blk) % 25; bs != 0 {
		blas.SetBlockSize(t, bs)
	}
}

type triFn[T float] func(blas.Side, blas.Uplo, blas.Transpose, blas.Diag, int, int, T, []T, int, []T, int) error

func fuzzTri[T float](t *testing.T, trsm bool, side, uplo, tA, diag byte, m8, n8, ldaD, ldbD, dA, dB int8, alpha T, flavor uint8, seed uint64, blk uint8, impls []impl[T]) {
	fuzzBlock(t, blk)
	side, uplo, tA, diag = upper(side), upper(uplo), upper(tA), upper(diag)
	m, n := int(m8)%40, int(n8)%40
	s := n
	if side == 'L' {
		s = m
	}
	lda, ldb := max(1, s)+int(ldaD)%4, max(1, m)+int(ldbD)%4
	a := fuzzData[T](seed, matLen(s, s, lda, dA), flavor)
	b0 := fuzzData[T](seed+1, matLen(m, n, ldb, dB), flavor)
	if trsm {
		// Random triangles are exponentially ill-conditioned: a solve overflows in
		// one order of summation and not in another, which the class check must not
		// see. Make the finite entries those of a well-conditioned triangle (a
		// diagonal in [1, 2), an off-diagonal below 1/(2s)), as triMatrix does; a
		// planted NaN or infinity stays.
		for p, v := range a {
			if f := float64(v); !math.IsNaN(f) && !math.IsInf(f, 0) && s > 0 && lda > 0 {
				if p%lda == p/lda {
					a[p] = T(math.Copysign(1+math.Abs(f)/256, f))
				} else {
					a[p] = T(f / float64(256*s))
				}
			}
		}
	}
	var errs [2]error
	var outs [2][]T
	for i, im := range impls {
		outs[i] = clone(b0)
		f := triFn[T](im.trmm)
		if trsm {
			f = im.trsm
		}
		errs[i] = f(blas.Side(side), blas.Uplo(uplo), blas.Transpose(tA), blas.Diag(diag), m, n, alpha, a, lda, outs[i], ldb)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || m == 0 || n == 0 {
		if !same(outs[0], b0) || !same(outs[1], b0) {
			t.Fatal("an error or a quick return modified B")
		}
		return
	}
	if scalarsModest(alpha) {
		classes(t, flavor, outs[0], outs[1])
	}
	if alpha == 0 { // B is zero, whatever it held; A is not read, and may be short
		if !zeroBlock(outs[0], m, n, ldb) || !zeroBlock(outs[1], m, n, ldb) {
			t.Fatal("alpha = 0 did not zero B")
		}
		return
	}
	if flavor%5 != 0 || !modest(a, b0) || !scalarsModest(alpha) {
		return
	}
	for i := range outs {
		if trsm {
			if !modest(outs[i]) && !allZero(outs[i]) {
				continue // the solution overflows or underflows: the bound is not stated there
			}
			trsmWithin(t, impls[i].name, side, uplo, tA, diag, m, n, alpha, a, lda, b0, outs[i], ldb)
			continue
		}
		td := triDense(a, lda, s, uplo == 'U', diag == 'U')
		var r oracle.Result
		if side == 'L' {
			r = oracle.Gemm(tA, 'N', m, n, m, alpha, td, s, b0, ldb, T(0), nil, 0)
		} else {
			r = oracle.Gemm('N', tA, m, n, n, alpha, b0, ldb, td, s, T(0), nil, 0)
		}
		g := oracle.Gamma[T](s + 3)
		for j := 0; j < n; j++ {
			for x := 0; x < m; x++ {
				within(t, impls[i].name, outs[i][x+j*ldb], r, x+j*m, g, g)
			}
		}
	}
}

// zeroBlock reports whether the m x n matrix in c, leading dimension ld, is zero.
func zeroBlock[T float](c []T, m, n, ld int) bool {
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			if c[i+j*ld] != 0 {
				return false
			}
		}
	}
	return true
}

func allZero[T float](v []T) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

// fuzzRank covers Syrk (b == false) and Syr2k.
func fuzzRank[T float](t *testing.T, two bool, uplo, tr byte, n8, k8, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta T, flavor uint8, seed uint64, blk uint8, impls []impl[T]) {
	fuzzBlock(t, blk)
	uplo, tr = upper(uplo), upper(tr)
	n, k := int(n8)%40, int(k8)%40
	rows, cols := n, k
	if tr != 'N' {
		rows, cols = k, n
	}
	lda, ldb, ldc := max(1, rows)+int(ldaD)%4, max(1, rows)+int(ldbD)%4, max(1, n)+int(ldcD)%4
	a := fuzzData[T](seed, matLen(rows, cols, lda, dA), flavor)
	b := fuzzData[T](seed+1, matLen(rows, cols, ldb, dB), flavor)
	c0 := fuzzData[T](seed+2, matLen(n, n, ldc, dC), 0)
	var errs [2]error
	var outs [2][]T
	for i, im := range impls {
		outs[i] = clone(c0)
		if two {
			errs[i] = im.syr2k(blas.Uplo(uplo), blas.Transpose(tr), n, k, alpha, a, lda, b, ldb, beta, outs[i], ldc)
		} else {
			errs[i] = im.syrk(blas.Uplo(uplo), blas.Transpose(tr), n, k, alpha, a, lda, beta, outs[i], ldc)
		}
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || n == 0 || ((alpha == 0 || k == 0) && beta == 1) {
		if !same(outs[0], c0) || !same(outs[1], c0) {
			t.Fatal("an error or a quick return modified C")
		}
		return
	}
	// The other triangle is never written.
	for i := range outs {
		otherTriangleIntact(t, impls[i].name, outs[i], c0, n, ldc, uplo == 'U')
	}
	if scalarsModest(alpha, beta) {
		classes(t, flavor, tri(outs[0], n, ldc, uplo == 'U'), tri(outs[1], n, ldc, uplo == 'U'))
	}
	if flavor%5 != 0 || !modest(a, b, c0) || !scalarsModest(alpha, beta) {
		return
	}
	// The oracle reads only what the reference reads: A and B are not read for
	// alpha = 0, C not for beta = 0.
	var r oracle.Result
	kk := k + 2
	if two {
		r, kk = syr2kExact(tr, n, k, alpha, a, lda, b, ldb, beta, c0, ldc), 2*k+2
	} else {
		r = syrkExact(tr, n, k, alpha, a, lda, beta, c0, ldc)
	}
	for i := range outs {
		triWithin(t, impls[i].name, outs[i], ldc, n, uplo == 'U', r, kk)
	}
}

// tri returns the elements of the triangle of an n x n matrix, in order.
func tri[T float](c []T, n, ld int, upper bool) []T {
	var v []T
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			if i == j || (i < j) == upper {
				v = append(v, c[i+j*ld])
			}
		}
	}
	return v
}

func fuzzSymm[T float](t *testing.T, side, uplo byte, m8, n8, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta T, flavor uint8, seed uint64, blk uint8, impls []impl[T]) {
	fuzzBlock(t, blk)
	side, uplo = upper(side), upper(uplo)
	m, n := int(m8)%40, int(n8)%40
	s := n
	if side == 'L' {
		s = m
	}
	lda, ldb, ldc := max(1, s)+int(ldaD)%4, max(1, m)+int(ldbD)%4, max(1, m)+int(ldcD)%4
	a := fuzzData[T](seed, matLen(s, s, lda, dA), flavor)
	b := fuzzData[T](seed+1, matLen(m, n, ldb, dB), flavor)
	c0 := fuzzData[T](seed+2, matLen(m, n, ldc, dC), 0)
	var errs [2]error
	var outs [2][]T
	for i, im := range impls {
		outs[i] = clone(c0)
		errs[i] = im.symm(blas.Side(side), blas.Uplo(uplo), m, n, alpha, a, lda, b, ldb, beta, outs[i], ldc)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("blas and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || m == 0 || n == 0 || (alpha == 0 && beta == 1) {
		if !same(outs[0], c0) || !same(outs[1], c0) {
			t.Fatal("an error or a quick return modified C")
		}
		return
	}
	if scalarsModest(alpha, beta) {
		classes(t, flavor, outs[0], outs[1])
	}
	if flavor%5 != 0 || !modest(a, b, c0) || !scalarsModest(alpha, beta) {
		return
	}
	if alpha == 0 { // A is not read and may be short: the oracle's dense copy of it must not be
		a = make([]T, matLen(s, s, lda, 0))
	}
	r := symmExact(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c0, ldc)
	g := oracle.Gamma[T](s + 2)
	for i := range outs {
		for j := 0; j < n; j++ {
			for x := 0; x < m; x++ {
				within(t, impls[i].name, outs[i][x+j*ldc], r, x+j*m, g, g)
			}
		}
	}
}

func FuzzDtrsm(f *testing.F) {
	f.Add(byte('L'), byte('U'), byte('N'), byte('N'), int8(6), int8(5), int8(0), int8(1), int8(0), int8(0), 1.5, uint8(0), uint64(1), uint8(4))
	f.Add(byte('R'), byte('L'), byte('T'), byte('U'), int8(9), int8(30), int8(2), int8(0), int8(0), int8(-1), -0.5, uint8(0), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, side, uplo, tA, diag byte, m, n, ldaD, ldbD, dA, dB int8, alpha float64, flavor uint8, seed uint64, blk uint8) {
		fuzzTri(t, true, side, uplo, tA, diag, m, n, ldaD, ldbD, dA, dB, alpha, flavor, seed, blk, impls64)
	})
}

func FuzzStrsm(f *testing.F) {
	f.Add(byte('L'), byte('L'), byte('T'), byte('N'), int8(7), int8(4), int8(0), int8(0), int8(0), int8(0), float32(1.5), uint8(0), uint64(1), uint8(3))
	f.Add(byte('R'), byte('U'), byte('C'), byte('U'), int8(5), int8(22), int8(1), int8(2), int8(0), int8(0), float32(2), uint8(1), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, side, uplo, tA, diag byte, m, n, ldaD, ldbD, dA, dB int8, alpha float32, flavor uint8, seed uint64, blk uint8) {
		fuzzTri(t, true, side, uplo, tA, diag, m, n, ldaD, ldbD, dA, dB, alpha, flavor, seed, blk, impls32)
	})
}

func FuzzDtrmm(f *testing.F) {
	f.Add(byte('L'), byte('U'), byte('N'), byte('N'), int8(6), int8(5), int8(0), int8(1), int8(0), int8(0), 1.5, uint8(0), uint64(1), uint8(4))
	f.Add(byte('R'), byte('L'), byte('T'), byte('U'), int8(9), int8(30), int8(2), int8(0), int8(0), int8(-1), -0.5, uint8(0), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, side, uplo, tA, diag byte, m, n, ldaD, ldbD, dA, dB int8, alpha float64, flavor uint8, seed uint64, blk uint8) {
		fuzzTri(t, false, side, uplo, tA, diag, m, n, ldaD, ldbD, dA, dB, alpha, flavor, seed, blk, impls64)
	})
}

func FuzzStrmm(f *testing.F) {
	f.Add(byte('L'), byte('L'), byte('T'), byte('N'), int8(7), int8(4), int8(0), int8(0), int8(0), int8(0), float32(1.5), uint8(0), uint64(1), uint8(3))
	f.Add(byte('R'), byte('U'), byte('C'), byte('U'), int8(5), int8(22), int8(1), int8(2), int8(0), int8(0), float32(2), uint8(1), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, side, uplo, tA, diag byte, m, n, ldaD, ldbD, dA, dB int8, alpha float32, flavor uint8, seed uint64, blk uint8) {
		fuzzTri(t, false, side, uplo, tA, diag, m, n, ldaD, ldbD, dA, dB, alpha, flavor, seed, blk, impls32)
	})
}

func FuzzDsyrk(f *testing.F) {
	f.Add(byte('U'), byte('N'), int8(6), int8(5), int8(0), int8(0), int8(1), int8(0), int8(0), int8(0), 1.5, -0.5, uint8(0), uint64(1), uint8(4))
	f.Add(byte('L'), byte('T'), int8(9), int8(30), int8(2), int8(0), int8(0), int8(-1), int8(0), int8(0), 1.0, 0.0, uint8(0), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, uplo, tr byte, n, k, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float64, flavor uint8, seed uint64, blk uint8) {
		fuzzRank(t, false, uplo, tr, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk, impls64)
	})
}

func FuzzSsyrk(f *testing.F) {
	f.Add(byte('U'), byte('T'), int8(7), int8(4), int8(0), int8(0), int8(0), int8(0), int8(0), int8(0), float32(1.5), float32(0.5), uint8(0), uint64(1), uint8(3))
	f.Add(byte('L'), byte('C'), int8(5), int8(22), int8(1), int8(2), int8(0), int8(0), int8(0), int8(-2), float32(2), float32(0), uint8(1), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, uplo, tr byte, n, k, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float32, flavor uint8, seed uint64, blk uint8) {
		fuzzRank(t, false, uplo, tr, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk, impls32)
	})
}

func FuzzDsyr2k(f *testing.F) {
	f.Add(byte('U'), byte('N'), int8(6), int8(5), int8(0), int8(0), int8(1), int8(0), int8(0), int8(0), 1.5, -0.5, uint8(0), uint64(1), uint8(4))
	f.Add(byte('L'), byte('T'), int8(9), int8(30), int8(2), int8(0), int8(0), int8(-1), int8(0), int8(0), 1.0, 0.0, uint8(0), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, uplo, tr byte, n, k, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float64, flavor uint8, seed uint64, blk uint8) {
		fuzzRank(t, true, uplo, tr, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk, impls64)
	})
}

func FuzzSsyr2k(f *testing.F) {
	f.Add(byte('U'), byte('T'), int8(7), int8(4), int8(0), int8(0), int8(0), int8(0), int8(0), int8(0), float32(1.5), float32(0.5), uint8(0), uint64(1), uint8(3))
	f.Add(byte('L'), byte('C'), int8(5), int8(22), int8(1), int8(2), int8(0), int8(0), int8(0), int8(-2), float32(2), float32(0), uint8(1), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, uplo, tr byte, n, k, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float32, flavor uint8, seed uint64, blk uint8) {
		fuzzRank(t, true, uplo, tr, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk, impls32)
	})
}

func FuzzDsymm(f *testing.F) {
	f.Add(byte('L'), byte('U'), int8(6), int8(5), int8(0), int8(1), int8(0), int8(0), int8(0), int8(0), 1.5, -0.5, uint8(0), uint64(1), uint8(4))
	f.Add(byte('R'), byte('L'), int8(9), int8(30), int8(2), int8(0), int8(0), int8(-1), int8(0), int8(0), 1.0, 0.0, uint8(0), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, side, uplo byte, m, n, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float64, flavor uint8, seed uint64, blk uint8) {
		fuzzSymm(t, side, uplo, m, n, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk, impls64)
	})
}

func FuzzSsymm(f *testing.F) {
	f.Add(byte('L'), byte('L'), int8(7), int8(4), int8(0), int8(0), int8(0), int8(0), int8(0), int8(0), float32(1.5), float32(0.5), uint8(0), uint64(1), uint8(3))
	f.Add(byte('R'), byte('U'), int8(5), int8(22), int8(1), int8(2), int8(0), int8(0), int8(0), int8(-2), float32(2), float32(0), uint8(1), uint64(2), uint8(0))
	f.Fuzz(func(t *testing.T, side, uplo byte, m, n, ldaD, ldbD, ldcD, dA, dB, dC int8, alpha, beta float32, flavor uint8, seed uint64, blk uint8) {
		fuzzSymm(t, side, uplo, m, n, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk, impls32)
	})
}
