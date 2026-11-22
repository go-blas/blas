package lapack_test

import (
	"testing"

	"github.com/go-blas/blas"
)

// The fuzz targets of the QR routines drive lapack and netlib as the others do
// (fuzz_test.go): arbitrary dimensions, leading dimensions, slice lengths, block
// sizes and data. Each asserts that nothing panics; that the two agree on the
// *blas.ArgError; that an error or a quick return leaves every output as it was;
// that on finite data of moderate magnitude both are within the bound against
// the exact product of the stored reflectors; and, for the default block sizes
// (netlib has no others), that NaN and Inf reach the same elements in both (lapack against the netlib port: both
// run internal/ref, so this compares the kernels, not the Fortran reference).

// fuzzQRConfig applies the block configuration sel selects and says whether it
// is the default one.
func fuzzQRConfig(t *testing.T, sel uint8) bool {
	t.Helper()
	cs := qrConfigs("lapack")
	c := cs[int(sel)%len(cs)]
	setQR(t, c)
	return c == (qrConfig{})
}

func dimOf(v int8) int { return int(v) % 40 }

func fuzzGeqrf[T float](t *testing.T, m8, n8, ldaD, dA, dT int8, flavor uint8, seed uint64, cfg uint8, ims []qimpl[T]) {
	dflt := fuzzQRConfig(t, cfg)
	m, n := dimOf(m8), dimOf(n8)
	lda := max(1, m) + int(ldaD)%4
	rng := newRNG(seed)
	full := general[T](rng, max(m, 0), max(n, 0), max(lda, 1))
	plant(full, seed, flavor)
	a0 := resize(full, matLen(max(m, 0), max(n, 0), max(lda, 1)), dA)
	t0 := resize[T](nil, min(max(m, 0), max(n, 0)), dT)
	var errs [2]error
	var as, ts [2][]T
	for i, im := range ims {
		as[i], ts[i] = clone(a0), clone(t0)
		errs[i] = im.geqrf(m, n, as[i], lda, ts[i])
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("lapack and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || m <= 0 || n <= 0 {
		for i := range as {
			if !same(as[i], a0) || !same(ts[i], t0) {
				t.Fatal("an error or a quick return modified its arguments")
			}
		}
		return
	}
	if dflt {
		classSame(t, "a", as[0], as[1])
		classSame(t, "tau", ts[0], ts[1])
	}
	if flavor%5 != 0 || !modest(a0) {
		return
	}
	for i := range as {
		if !modest(as[i], ts[i]) {
			continue
		}
		q := exactQ(as[i], m, min(m, n), lda, ts[i])
		b0 := toBM(a0, m, n, lda)
		g := qrGamma[T](m, n)
		colBound(t, ims[i].name+": A - Q R", sub(b0, mul(q, rFactor(as[i], m, n, lda))), b0, g)
		orthBound(t, ims[i].name+": Q", q, g)
	}
}

// reflectors returns the nq x k reflectors of a factorization of random data, in
// a matrix of leading dimension lda (the padding the sentinel) and its tau.
func reflectors[T float](geqrf func(m, n int, a []T, lda int, tau []T) error, seed uint64, nq, k, lda int) ([]T, []T) {
	f := general[T](newRNG(seed), nq, k, max(1, nq))
	tau := make([]T, k)
	if nq > 0 && k > 0 {
		if err := geqrf(nq, k, f, max(1, nq), tau); err != nil {
			panic(err)
		}
	}
	a := general[T](newRNG(seed+1), nq, k, max(1, lda))
	for j := 0; j < k; j++ {
		copy(a[j*lda:j*lda+nq], f[j*nq:(j+1)*nq])
	}
	return a, tau
}

func fuzzOrmqr[T float](t *testing.T, sdSel, trSel uint8, m8, n8, k8, ldaD, ldcD, dA, dT, dC int8, flavor uint8, seed uint64, cfg uint8, ims []qimpl[T]) {
	dflt := fuzzQRConfig(t, cfg)
	sides := []blas.Side{'L', 'R', 'l', 'r', 'X'}
	trans := []blas.Transpose{'N', 'T', 'n', 't', 'C', 'X'}
	sd, tr := sides[int(sdSel)%len(sides)], trans[int(trSel)%len(trans)]
	m, n, k := dimOf(m8), dimOf(n8), dimOf(k8)
	nq := n
	if sd == 'L' || sd == 'l' {
		nq = m
	}
	lda, ldc := max(1, nq)+int(ldaD)%4, max(1, m)+int(ldcD)%4
	rng := newRNG(seed)
	var a, tau []T
	if nq > 0 && k > 0 && k <= nq && lda >= nq {
		a, tau = reflectors(ims[1].geqrf, seed, nq, k, lda)
	} else {
		a = general[T](rng, max(nq, 0), max(k, 0), max(lda, 1))
		tau = general[T](rng, max(k, 0), 1, 1)
	}
	plant(a, seed, flavor)
	a0 := resize(a, matLen(max(nq, 0), max(k, 0), max(lda, 1)), dA)
	t0 := resize(tau, max(k, 0), dT)
	c0 := resize(general[T](rng, max(m, 0), max(n, 0), max(ldc, 1)), matLen(max(m, 0), max(n, 0), max(ldc, 1)), dC)
	var errs [2]error
	var as, cs [2][]T
	for i, im := range ims {
		as[i], cs[i] = clone(a0), clone(c0)
		errs[i] = im.ormqr(sd, tr, m, n, k, as[i], lda, clone(t0), cs[i], ldc)
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("lapack and netlib disagree: %v, %v", errs[0], errs[1])
	}
	for i := range as {
		if !same(as[i], a0) {
			t.Fatal("ormqr did not leave A as it was")
		}
	}
	if errs[0] != nil || m <= 0 || n <= 0 || k <= 0 {
		if !same(cs[0], c0) || !same(cs[1], c0) {
			t.Fatal("an error or a quick return modified C")
		}
		return
	}
	if dflt {
		classSame(t, "c", cs[0], cs[1])
	}
	if flavor%5 != 0 || !modest(a0, t0, c0) {
		return
	}
	q := exactQ(a0, nq, k, lda, t0)
	if upper(byte(tr)) == 'T' {
		q = q.T()
	}
	cb := toBM(c0, m, n, ldc)
	var want bm
	if upper(byte(sd)) == 'L' {
		want = mul(q, cb)
	} else {
		want = mul(cb, q)
	}
	for i := range cs {
		if modest(cs[i]) {
			froBound(t, ims[i].name+": C op(Q)", sub(toBM(cs[i], m, n, ldc), want), fro(cb), qrGamma[T](nq, k))
		}
	}
}

func fuzzOrgqr[T float](t *testing.T, m8, n8, k8, ldaD, dA, dT int8, flavor uint8, seed uint64, cfg uint8, ims []qimpl[T]) {
	dflt := fuzzQRConfig(t, cfg)
	m, n, k := dimOf(m8), dimOf(n8), dimOf(k8)
	lda := max(1, m) + int(ldaD)%4
	rng := newRNG(seed)
	var a, tau []T
	if m > 0 && n > 0 && n <= m && k >= 0 && k <= n && lda >= m {
		a, tau = reflectors(ims[1].geqrf, seed, m, k, lda)
		full := general[T](rng, m, n, lda) // columns k..n-1: anything, not referenced
		copy(full, a)
		a = full
	} else {
		a = general[T](rng, max(m, 0), max(n, 0), max(lda, 1))
		tau = general[T](rng, max(k, 0), 1, 1)
	}
	plant(a, seed, flavor)
	a0 := resize(a, matLen(max(m, 0), max(n, 0), max(lda, 1)), dA)
	t0 := resize(tau, max(k, 0), dT)
	var errs [2]error
	var as [2][]T
	for i, im := range ims {
		as[i] = clone(a0)
		errs[i] = im.orgqr(m, n, k, as[i], lda, clone(t0))
	}
	if !sameErr(errs[0], errs[1]) {
		t.Fatalf("lapack and netlib disagree: %v, %v", errs[0], errs[1])
	}
	if errs[0] != nil || n <= 0 {
		if !same(as[0], a0) || !same(as[1], a0) {
			t.Fatal("an error or a quick return modified A")
		}
		return
	}
	if dflt {
		classSame(t, "a", as[0], as[1])
	}
	if flavor%5 != 0 || !modest(a0, t0) {
		return
	}
	ex := exactQ(a0, m, k, lda, t0)
	want := newBM(m, n)
	copy(want.v, ex.v[:m*n])
	for i := range as {
		if !modest(as[i]) {
			continue
		}
		got := toBM(as[i], m, n, lda)
		g := qrGamma[T](m, n)
		colBound(t, ims[i].name+": Q - Q exact", sub(got, want), want, g)
		orthBound(t, ims[i].name+": Q", got, g)
	}
}

func FuzzDgeqrf(f *testing.F) {
	f.Add(int8(7), int8(5), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, m, n, ldaD, dA, dT int8, flavor uint8, seed uint64, cfg uint8) {
		fuzzGeqrf(t, m, n, ldaD, dA, dT, flavor, seed, cfg, qimpls64)
	})
}

func FuzzSgeqrf(f *testing.F) {
	f.Add(int8(7), int8(5), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, m, n, ldaD, dA, dT int8, flavor uint8, seed uint64, cfg uint8) {
		fuzzGeqrf(t, m, n, ldaD, dA, dT, flavor, seed, cfg, qimpls32)
	})
}

func FuzzDormqr(f *testing.F) {
	f.Add(uint8(0), uint8(0), int8(7), int8(5), int8(3), int8(0), int8(0), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, sd, tr uint8, m, n, k, ldaD, ldcD, dA, dT, dC int8, flavor uint8, seed uint64, cfg uint8) {
		fuzzOrmqr(t, sd, tr, m, n, k, ldaD, ldcD, dA, dT, dC, flavor, seed, cfg, qimpls64)
	})
}

func FuzzSormqr(f *testing.F) {
	f.Add(uint8(0), uint8(0), int8(7), int8(5), int8(3), int8(0), int8(0), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, sd, tr uint8, m, n, k, ldaD, ldcD, dA, dT, dC int8, flavor uint8, seed uint64, cfg uint8) {
		fuzzOrmqr(t, sd, tr, m, n, k, ldaD, ldcD, dA, dT, dC, flavor, seed, cfg, qimpls32)
	})
}

func FuzzDorgqr(f *testing.F) {
	f.Add(int8(7), int8(5), int8(3), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, m, n, k, ldaD, dA, dT int8, flavor uint8, seed uint64, cfg uint8) {
		fuzzOrgqr(t, m, n, k, ldaD, dA, dT, flavor, seed, cfg, qimpls64)
	})
}

func FuzzSorgqr(f *testing.F) {
	f.Add(int8(7), int8(5), int8(3), int8(0), int8(0), int8(0), uint8(0), uint64(1), uint8(0))
	f.Fuzz(func(t *testing.T, m, n, k, ldaD, dA, dT int8, flavor uint8, seed uint64, cfg uint8) {
		fuzzOrgqr(t, m, n, k, ldaD, dA, dT, flavor, seed, cfg, qimpls32)
	})
}
