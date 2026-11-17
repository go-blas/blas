package lapack_test

import (
	"errors"
	"math"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
	"github.com/go-blas/blas/lapack"
)

// The fuzz targets drive lapack and netlib with arbitrary dimensions, leading
// dimensions, slice lengths, panel widths and data, valid or not. Each asserts:
// nothing panics; the two agree on the *blas.ArgError (same routine and
// parameter) and on INFO; an error leaves every output as it was; on finite data
// of moderate magnitude both are within the stated bound. The seeds are in
// testdata/fuzz and replayed by go test.

func sameErr(a, b error) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	var ea, eb *blas.ArgError
	return errors.As(a, &ea) && errors.As(b, &eb) && *ea == *eb
}

// modest: every value zero or finite of a magnitude whose products neither
// overflow nor go subnormal.
func modest[T float](vs ...[]T) bool {
	hi, lo := 1e15, 1e-30
	var z T
	if _, ok := any(z).(float32); ok {
		hi, lo = 1e5, 1e-10
	}
	for _, s := range vs {
		for _, v := range s {
			f := math.Abs(float64(v))
			if f != 0 && (!(f <= hi) || f < lo) {
				return false
			}
		}
	}
	return true
}

// plant puts a NaN, an infinity, a huge or a tiny value into v, by flavor.
func plant[T float](v []T, seed uint64, flavor uint8) {
	if len(v) == 0 {
		return
	}
	i := int(seed>>8) % len(v)
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
}

// resize makes v hold n+d%3 elements (never negative), new ones the sentinel.
func resize[T float](v []T, n int, d int8) []T {
	l := max(0, n+int(d)%3)
	out := make([]T, l)
	for i := range out {
		out[i] = sentinel
	}
	copy(out, v)
	return out
}

func setFuzzLeaf(t *testing.T, lf uint8) {
	t.Helper()
	if l := int(lf) % 20; l > 0 {
		lapack.SetLeaf(t, l)
	}
}

func fuzzGetrf[T float](t *testing.T, m8, n8, ldaD, dA, dI, zc int8, flavor uint8, seed uint64, lf uint8, impls []impl[T]) {
	setFuzzLeaf(t, lf)
	m, n := int(m8)%40, int(n8)%40
	lda := max(1, m) + int(ldaD)%4
	rng := newRNG(seed)
	full := general[T](rng, max(m, 0), max(n, 0), max(lda, 1))
	zeroCol := -1                              // the index of a column made exactly zero, if it holds a pivot
	if n > 0 && m > 0 && zc >= 0 && lda >= m { // a zero column: a singular matrix
		c := int(zc) % n
		for i := 0; i < m; i++ {
			full[i+c*lda] = 0
		}
		if c < min(m, n) {
			zeroCol = c
		}
	}
	plant(full, seed, flavor)
	a0 := resize(full, matLen(max(m, 0), max(n, 0), max(lda, 1)), dA)
	nip := min(max(m, 0), max(n, 0))
	var errs [2]error
	var infos [2]int
	var as [2][]T
	var ps [2][]int
	for i, im := range impls {
		as[i] = clone(a0)
		ps[i] = make([]int, max(0, nip+int(dI)%3))
		for k := range ps[i] {
			ps[i][k] = -7
		}
		infos[i], errs[i] = im.getrf(m, n, as[i], lda, ps[i])
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("lapack and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || m == 0 || n == 0 {
		var ae *blas.ArgError
		if errs[0] != nil && (!errors.As(errs[0], &ae) || infos[0] != ae.Info() || infos[0] != infos[1]) {
			t.Fatalf("info %d, %d for %v", infos[0], infos[1], errs[0])
		}
		for i := range as {
			if !same(as[i], a0) {
				t.Fatal("an error or a quick return modified A")
			}
			for _, p := range ps[i] {
				if p != -7 {
					t.Fatal("an error or a quick return modified ipiv")
				}
			}
		}
		return
	}
	if flavor%5 != 0 || !modest(a0) {
		return // the bounds are stated on finite data of moderate magnitude
	}
	// INFO is not compared for equality on finite data: a pivot that cancels to
	// exactly zero in one order of evaluation (the recursion of lapack, the blocks
	// of netlib, FMA) and not in the other is legitimate, and the first zero
	// pivot is then not the same. What is asserted is what rounding cannot
	// change: a column made exactly zero (before its pivot, so that every update
	// of it is exact) is a zero pivot at that column or before, in both. The bound
	// below holds for the factorization as completed, whatever INFO says.
	if zeroCol >= 0 {
		for i := range infos {
			if infos[i] < 1 || infos[i] > zeroCol+1 {
				t.Fatalf("%s: info %d with column %d exactly zero, want 1..%d", impls[i].name, infos[i], zeroCol, zeroCol+1)
			}
		}
	}
	for i := range as {
		if !modest(as[i]) {
			continue // growth overflowed: the bound is not stated there
		}
		getrfBound(t, impls[i].name, a0, as[i], ps[i], m, n, lda)
	}
}

func fuzzPotrf[T float](t *testing.T, up bool, n8, ldaD, dA, bad int8, flavor uint8, seed uint64, lf uint8, impls []impl[T]) {
	setFuzzLeaf(t, lf)
	n := int(n8) % 40
	lda := max(1, n) + int(ldaD)%4
	uplo := blas.Lower
	if up {
		uplo = blas.Upper
	}
	full := spd[T](newRNG(seed), max(n, 0), max(lda, 1), up)
	if n > 0 && bad < 0 && lda >= n { // a diagonal element that is not positive
		k := int(-bad) % n
		full[k+k*lda] = T(-1)
	}
	plant(full, seed, flavor)
	a0 := resize(full, matLen(max(n, 0), max(n, 0), max(lda, 1)), dA)
	var errs [2]error
	var infos [2]int
	var as [2][]T
	for i, im := range impls {
		as[i] = clone(a0)
		infos[i], errs[i] = im.potrf(uplo, n, as[i], lda)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("lapack and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || n == 0 {
		var ae *blas.ArgError
		if errs[0] != nil && (!errors.As(errs[0], &ae) || infos[0] != ae.Info() || infos[0] != infos[1]) {
			t.Fatalf("info %d, %d for %v", infos[0], infos[1], errs[0])
		}
		for i := range as {
			if !same(as[i], a0) {
				t.Fatal("an error or a quick return modified A")
			}
		}
		return
	}
	if flavor%5 != 0 {
		return
	}
	// Finite data: the same INFO, and the leading block before it is a factor. The
	// equality is safe here, unlike getrf's: the matrix is strictly diagonally
	// dominant with a diagonal of at least n+1 against off-diagonals below 1, so no
	// leading minor is near zero, and the only failure there can be is the
	// diagonal element made -1, whose minor is negative by a margin rounding
	// cannot close.
	if infos[0] != infos[1] {
		t.Fatalf("info %d (lapack), %d (netlib)", infos[0], infos[1])
	}
	k := n
	if infos[0] > 0 {
		k = infos[0] - 1
	}
	for i := range as {
		potrfBound(t, impls[i].name, a0, as[i], k, lda, up)
	}
}

func fuzzGetrs[T float](t *testing.T, tr byte, n8, nrhs8, ldaD, ldbD, dA, dB, dI int8, pv, flavor uint8, seed uint64, lf uint8, impls []impl[T]) {
	setFuzzLeaf(t, lf)
	n, nrhs := int(n8)%40, int(nrhs8)%40
	lda, ldb := max(1, n)+int(ldaD)%4, max(1, n)+int(ldbD)%4
	rng := newRNG(seed)
	// Factors of a random matrix, so that the solve has something to solve with.
	f := general[T](rng, max(n, 0), max(n, 0), max(lda, 1))
	fp := make([]int, max(n, 0))
	if n > 0 && lda >= n {
		impls[1].getrf(n, n, f, lda, fp)
	}
	plant(f, seed, flavor)
	b := general[T](rng, max(n, 0), max(nrhs, 0), max(ldb, 1))
	plant(b, seed+1, flavor)
	if n > 0 {
		switch pv % 8 { // a pivot outside 1..n
		case 1:
			fp[seed%uint64(n)] = 0
		case 2:
			fp[seed%uint64(n)] = n + 1
		}
	}
	a0 := resize(f, matLen(max(n, 0), max(n, 0), max(lda, 1)), dA)
	b0 := resize(b, matLen(max(n, 0), max(nrhs, 0), max(ldb, 1)), dB)
	p0 := append([]int(nil), fp...)
	if l := max(0, n+int(dI)%3); l <= len(p0) {
		p0 = p0[:l]
	} else {
		base := len(p0)
		p0 = append(p0, make([]int, l-base)...)
		for k := base; k < l; k++ {
			p0[k] = 1
		}
	}
	var errs [2]error
	var xs [2][]T
	for i, im := range impls {
		xs[i] = clone(b0)
		errs[i] = im.getrs(blas.Transpose(tr), n, nrhs, a0, lda, p0, xs[i], ldb)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("lapack and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || n == 0 || nrhs == 0 {
		for i := range xs {
			if !same(xs[i], b0) {
				t.Fatal("an error or a quick return modified B")
			}
		}
		return
	}
	if flavor%5 != 0 || !modest(a0, b0) {
		return
	}
	for i := range xs {
		if modest(xs[i]) && !math.IsNaN(float64(xs[i][0])) {
			getrsBound(t, impls[i].name, upper(tr), a0, p0, n, nrhs, lda, b0, xs[i], ldb)
		}
	}
}

func fuzzPotrs[T float](t *testing.T, up bool, n8, nrhs8, ldaD, ldbD, dA, dB int8, flavor uint8, seed uint64, lf uint8, impls []impl[T]) {
	setFuzzLeaf(t, lf)
	n, nrhs := int(n8)%40, int(nrhs8)%40
	lda, ldb := max(1, n)+int(ldaD)%4, max(1, n)+int(ldbD)%4
	uplo := blas.Lower
	if up {
		uplo = blas.Upper
	}
	rng := newRNG(seed)
	f := spd[T](rng, max(n, 0), max(lda, 1), up)
	if n > 0 && lda >= n {
		impls[1].potrf(uplo, n, f, lda)
	}
	plant(f, seed, flavor)
	b := general[T](rng, max(n, 0), max(nrhs, 0), max(ldb, 1))
	plant(b, seed+1, flavor)
	a0 := resize(f, matLen(max(n, 0), max(n, 0), max(lda, 1)), dA)
	b0 := resize(b, matLen(max(n, 0), max(nrhs, 0), max(ldb, 1)), dB)
	var errs [2]error
	var xs [2][]T
	for i, im := range impls {
		xs[i] = clone(b0)
		errs[i] = im.potrs(uplo, n, nrhs, a0, lda, xs[i], ldb)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("lapack and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || n == 0 || nrhs == 0 {
		for i := range xs {
			if !same(xs[i], b0) {
				t.Fatal("an error or a quick return modified B")
			}
		}
		return
	}
	if flavor%5 != 0 || !modest(b0) { // the other triangle of A holds NaN, never read
		return
	}
	for i := range xs {
		if modest(xs[i]) {
			potrsBound(t, impls[i].name, a0, n, nrhs, lda, b0, xs[i], ldb, up)
		}
	}
}

var _ = oracle.Prec

func FuzzDgetrf(f *testing.F) {
	f.Add(int8(7), int8(5), int8(0), int8(0), int8(0), int8(-1), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, m, n, ldaD, dA, dI, zc int8, flavor uint8, seed uint64, lf uint8) {
		fuzzGetrf(t, m, n, ldaD, dA, dI, zc, flavor, seed, lf, impls64)
	})
}

func FuzzSgetrf(f *testing.F) {
	f.Add(int8(7), int8(5), int8(0), int8(0), int8(0), int8(-1), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, m, n, ldaD, dA, dI, zc int8, flavor uint8, seed uint64, lf uint8) {
		fuzzGetrf(t, m, n, ldaD, dA, dI, zc, flavor, seed, lf, impls32)
	})
}

func FuzzDpotrf(f *testing.F) {
	f.Add(true, int8(7), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, up bool, n, ldaD, dA, bad int8, flavor uint8, seed uint64, lf uint8) {
		fuzzPotrf(t, up, n, ldaD, dA, bad, flavor, seed, lf, impls64)
	})
}

func FuzzSpotrf(f *testing.F) {
	f.Add(true, int8(7), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, up bool, n, ldaD, dA, bad int8, flavor uint8, seed uint64, lf uint8) {
		fuzzPotrf(t, up, n, ldaD, dA, bad, flavor, seed, lf, impls32)
	})
}

func FuzzDgetrs(f *testing.F) {
	f.Add(byte('N'), int8(7), int8(3), int8(0), int8(0), int8(0), int8(0), int8(0), uint8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, tr byte, n, nrhs, ldaD, ldbD, dA, dB, dI int8, pv, flavor uint8, seed uint64, lf uint8) {
		fuzzGetrs(t, tr, n, nrhs, ldaD, ldbD, dA, dB, dI, pv, flavor, seed, lf, impls64)
	})
}

func FuzzSgetrs(f *testing.F) {
	f.Add(byte('N'), int8(7), int8(3), int8(0), int8(0), int8(0), int8(0), int8(0), uint8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, tr byte, n, nrhs, ldaD, ldbD, dA, dB, dI int8, pv, flavor uint8, seed uint64, lf uint8) {
		fuzzGetrs(t, tr, n, nrhs, ldaD, ldbD, dA, dB, dI, pv, flavor, seed, lf, impls32)
	})
}

func FuzzDpotrs(f *testing.F) {
	f.Add(true, int8(7), int8(3), int8(0), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, up bool, n, nrhs, ldaD, ldbD, dA, dB int8, flavor uint8, seed uint64, lf uint8) {
		fuzzPotrs(t, up, n, nrhs, ldaD, ldbD, dA, dB, flavor, seed, lf, impls64)
	})
}

func FuzzSpotrs(f *testing.F) {
	f.Add(true, int8(7), int8(3), int8(0), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, up bool, n, nrhs, ldaD, ldbD, dA, dB int8, flavor uint8, seed uint64, lf uint8) {
		fuzzPotrs(t, up, n, nrhs, ldaD, ldbD, dA, dB, flavor, seed, lf, impls32)
	})
}
