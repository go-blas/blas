package lapack_test

import (
	"math"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/lapack"
	"github.com/go-blas/blas/netlib"
)

func TestEigArgErrors(t *testing.T) {
	eachE(t, eigArgT[float64], eigArgT[float32])
}

func eigArgT[T float](t *testing.T, im eimpl[T]) {
	a9, a8 := make([]T, 9), make([]T, 8)
	d3, d2 := make([]T, 3), make([]T, 2)
	e2, e1 := make([]T, 2), make([]T, 1)
	N, V := lapack.EigenvaluesOnly, lapack.EigenvaluesAndVectors

	sy := func(what string, p int, jobz lapack.Jobz, uplo blas.Uplo, n int, a []T, lda int, w []T) {
		t.Helper()
		info, err := im.syev(jobz, uplo, n, a, lda, w)
		wantArg(t, "syev "+what, err, p, &info)
	}
	sy("jobz", 1, 'X', 'L', 3, a9, 3, d3)
	sy("uplo", 2, V, 'X', 3, a9, 3, d3)
	sy("n", 3, V, 'L', -1, a9, 3, d3)
	sy("lda", 5, V, 'L', 3, a9, 2, d3)
	sy("lda zero", 5, V, 'U', 0, a9, 0, d3)
	sy("a short", 4, V, 'L', 3, a8, 3, d3)
	sy("w short", 6, N, 'U', 3, a9, 3, d2)
	// Lower case flags are accepted (LSAME).
	if info, err := im.syev('v', 'l', 0, nil, 1, nil); err != nil || info != 0 {
		t.Fatalf("syev lower case, n=0: info %d, err %v", info, err)
	}
	// Quick return comes after the scalar checks and before the lengths.
	if info, err := im.syev(V, 'L', 0, nil, 1, nil); err != nil || info != 0 {
		t.Fatalf("syev n=0: info %d, err %v", info, err)
	}
	// n = 1: w(1) = a(1,1), a(1,1) = 1.
	a1, w1 := []T{-2.5}, []T{0}
	if info, err := im.syev(V, 'U', 1, a1, 1, w1); err != nil || info != 0 || w1[0] != -2.5 || a1[0] != 1 {
		t.Fatalf("syev n=1: info %d, err %v, w %v, a %v", info, err, w1, a1)
	}

	st := func(what string, p int, uplo blas.Uplo, n int, a []T, lda int, d, e, tau []T) {
		t.Helper()
		wantArg(t, "sytrd "+what, im.sytrd(uplo, n, a, lda, d, e, tau), p, nil)
	}
	st("uplo", 1, 'X', 3, a9, 3, d3, e2, e2)
	st("n", 2, 'L', -1, a9, 3, d3, e2, e2)
	st("lda", 4, 'L', 3, a9, 2, d3, e2, e2)
	st("a short", 3, 'U', 3, a8, 3, d3, e2, e2)
	st("d short", 5, 'U', 3, a9, 3, d2, e2, e2)
	st("e short", 6, 'L', 3, a9, 3, d3, e1, e2)
	st("tau short", 7, 'L', 3, a9, 3, d3, e2, e1)
	if err := im.sytrd('L', 0, nil, 1, nil, nil, nil); err != nil {
		t.Fatalf("sytrd n=0: %v", err)
	}

	og := func(what string, p int, uplo blas.Uplo, n int, a []T, lda int, tau []T) {
		t.Helper()
		wantArg(t, "orgtr "+what, im.orgtr(uplo, n, a, lda, tau), p, nil)
	}
	og("uplo", 1, 'X', 3, a9, 3, e2)
	og("n", 2, 'L', -1, a9, 3, e2)
	og("lda", 4, 'L', 3, a9, 2, e2)
	og("a short", 3, 'U', 3, a8, 3, e2)
	og("tau short", 5, 'U', 3, a9, 3, e1)
	if err := im.orgtr('U', 0, nil, 1, nil); err != nil {
		t.Fatalf("orgtr n=0: %v", err)
	}

	sq := func(what string, p int, compz lapack.Compz, n int, d, e, z []T, ldz int) {
		t.Helper()
		info, err := im.steqr(compz, n, d, e, z, ldz)
		wantArg(t, "steqr "+what, err, p, &info)
	}
	sq("compz", 1, 'X', 3, d3, e2, a9, 3)
	sq("n", 2, lapack.CompzIdentity, -1, d3, e2, a9, 3)
	sq("ldz", 6, lapack.CompzIdentity, 3, d3, e2, a9, 2)
	sq("ldz zero", 6, lapack.CompzOriginal, 3, d3, e2, a9, 0)
	sq("d short", 3, lapack.CompzNone, 3, d2, e2, nil, 1)
	sq("e short", 4, lapack.CompzNone, 3, d3, e1, nil, 1)
	sq("z short", 5, lapack.CompzIdentity, 3, d3, e2, a8, 3)
	if info, err := im.steqr(lapack.CompzNone, 0, nil, nil, nil, 1); err != nil || info != 0 {
		t.Fatalf("steqr n=0: info %d, err %v", info, err)
	}

	sf := func(what string, p int, n int, d, e []T) {
		t.Helper()
		info, err := im.sterf(n, d, e)
		wantArg(t, "sterf "+what, err, p, &info)
	}
	sf("n", 1, -1, d3, e2)
	sf("d short", 2, 3, d2, e2)
	sf("e short", 3, 3, d3, e1)
	for _, n := range []int{0, 1} {
		if info, err := im.sterf(n, nil, nil); err != nil || info != 0 {
			t.Fatalf("sterf n=%d: info %d, err %v", n, info, err)
		}
	}
}

// TestEigNonConvergence makes the QL/QR iteration run out of its budget (one
// iteration per eigenvalue instead of 30): INFO > 0, the number of off-diagonal
// elements that did not reach zero, and the routines return normally.
func TestEigNonConvergence(t *testing.T) {
	eachE(t, nonConvT[float64], nonConvT[float32])
}

func nonConvT[T float](t *testing.T, im eimpl[T]) {
	lapack.SetMaxIt(t, 1)
	rng := newRNG(14)
	n := 24
	d, e := make([]T, n), make([]T, n-1)
	for i := range d {
		d[i] = T(rng.NormFloat64())
	}
	for i := range e {
		e[i] = T(rng.NormFloat64())
	}
	for name, run := range map[string]func() (int, error){
		"sterf":   func() (int, error) { return im.sterf(n, clone(d), clone(e)) },
		"steqr N": func() (int, error) { return im.steqr(lapack.CompzNone, n, clone(d), clone(e), nil, 1) },
		"steqr I": func() (int, error) {
			return im.steqr(lapack.CompzIdentity, n, clone(d), clone(e), make([]T, n*n), n)
		},
		"syev": func() (int, error) {
			a := make([]T, n*n)
			for i := 0; i < n; i++ {
				a[i+i*n] = d[i]
				if i < n-1 {
					a[i+1+i*n] = e[i]
				}
			}
			return im.syev(lapack.EigenvaluesAndVectors, 'L', n, a, n, make([]T, n))
		},
	} {
		info, err := run()
		if err != nil || info <= 0 || info > n-1 {
			t.Fatalf("%s: info %d, err %v, want 1..%d", name, info, err, n-1)
		}
	}
}

// TestEigNaNInf: non-finite data terminates, without panic, with any result.
func TestEigNaNInf(t *testing.T) {
	eachE(t, nanInfE[float64], nanInfE[float32])
}

func nanInfE[T float](t *testing.T, im eimpl[T]) {
	rng := newRNG(15)
	bad := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
	for _, n := range []int{2, 3, 5, 17, 40} {
		for _, b := range bad {
			for _, pos := range []int{0, n / 2, n*n - 1} {
				for _, up := range []bool{true, false} {
					a := denseSym[T]("dense/random", n, n, up, 1, rng)
					a[pos] = T(b)
					w := make([]T, n)
					for _, jz := range []lapack.Jobz{lapack.EigenvaluesOnly, lapack.EigenvaluesAndVectors} {
						im.syev(jz, uploOf(up), n, clone(a), n, w)
					}
				}
			}
			d, e := make([]T, n), make([]T, n-1)
			for i := range d {
				d[i] = T(rng.NormFloat64())
			}
			for i := range e {
				e[i] = T(rng.NormFloat64())
			}
			d[n/2] = T(b)
			im.sterf(n, clone(d), clone(e))
			im.steqr(lapack.CompzIdentity, n, clone(d), clone(e), make([]T, n*n), n)
			e[n/2-1] = T(b)
			d[n/2] = 1
			im.sterf(n, clone(d), clone(e))
			im.steqr(lapack.CompzNone, n, clone(d), clone(e), nil, 1)
		}
	}
}

// FuzzSyev drives Dsyev and netlib.Dsyev with arbitrary orders, leading
// dimensions, slice lengths and data. Nothing panics; the two agree on the
// ArgError and on INFO; an error leaves the output as it was; on finite data of
// moderate magnitude the eigenvalues ascend and the residual is small.
func FuzzSyev(f *testing.F) {
	f.Add(uint8(5), uint64(1), true, true, uint8(0), int8(0), int8(0))
	f.Add(uint8(33), uint64(2), false, true, uint8(0), int8(1), int8(0))
	f.Add(uint8(7), uint64(3), true, false, uint8(1), int8(0), int8(-1))
	f.Add(uint8(40), uint64(4), false, false, uint8(3), int8(0), int8(0))
	f.Add(uint8(2), uint64(5), true, true, uint8(2), int8(-1), int8(0))
	f.Fuzz(func(t *testing.T, n8 uint8, seed uint64, vec, up bool, flavor uint8, dl, dlen int8) {
		n := int(n8) % 48
		rng := newRNG(seed)
		lda := max(0, n+int(dl)%3)
		a := make([]float64, max(0, matLen(n, n, lda)+int(dlen)%3))
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		plant(a, seed, flavor)
		w := make([]float64, max(0, n+int(dlen)%2))
		jz := lapack.EigenvaluesOnly
		if vec {
			jz = lapack.EigenvaluesAndVectors
		}
		a1, a2 := clone(a), clone(a)
		w1, w2 := clone(w), clone(w)
		i1, e1 := lapack.Dsyev(jz, uploOf(up), n, a1, lda, w1)
		i2, e2 := netlib.Dsyev(jz, uploOf(up), n, a2, lda, w2)
		if !sameErr(e1, e2) || i1 != i2 {
			t.Fatalf("lapack (info %d, %v) and netlib (info %d, %v) disagree", i1, e1, i2, e2)
		}
		if e1 != nil && (!same(a1, a) || !same(w1, w)) {
			t.Fatalf("an error modified the output")
		}
		if e1 != nil || i1 != 0 || n == 0 || !modest(a) {
			return
		}
		for i := 1; i < n; i++ {
			if w1[i] < w1[i-1] {
				t.Fatalf("not ascending")
			}
		}
		if !vec {
			return
		}
		at := func(i, j int) float64 { // the matrix, from the triangle named
			if (up && i > j) || (!up && i < j) {
				i, j = j, i
			}
			return a[i+j*lda]
		}
		var anorm float64
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				anorm += at(i, j) * at(i, j)
			}
		}
		anorm = math.Sqrt(anorm)
		for k := 0; k < n; k++ {
			var r2 float64
			for i := 0; i < n; i++ {
				s := -w1[k] * a1[i+k*lda]
				for j := 0; j < n; j++ {
					s += at(i, j) * a1[j+k*lda]
				}
				r2 += s * s
			}
			if math.Sqrt(r2) > 1e-11*float64(n)*(anorm+1e-300) {
				t.Fatalf("residual of eigenpair %d: %g, ||A|| %g", k, math.Sqrt(r2), anorm)
			}
		}
	})
}
