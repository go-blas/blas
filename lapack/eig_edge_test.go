package lapack_test

import (
	"testing"

	"github.com/go-blas/blas/lapack"
	"github.com/go-blas/blas/netlib"
)

// TestOneByOne: n = 1 returns the element itself, exactly, even where the general
// path would scale (the reference returns before the scaling of Syev and Steqr),
// and the eigenvector is 1.
func TestOneByOne(t *testing.T) {
	for _, v := range []float64{3, -5, 0, 1e300, -1e300, 1e-300, 5e-324, 1.7e308} {
		for _, jz := range []lapack.Jobz{lapack.EigenvaluesOnly, lapack.EigenvaluesAndVectors} {
			a, w := []float64{v}, []float64{-1}
			if info, err := lapack.Dsyev(jz, 'L', 1, a, 1, w); info != 0 || err != nil || w[0] != v {
				t.Fatalf("Dsyev %c %g: %d %v w=%g", jz, v, info, err, w[0])
			}
			if jz == lapack.EigenvaluesAndVectors && a[0] != 1 {
				t.Fatalf("Dsyev %g: vector %g", v, a[0])
			}
			a32, w32 := []float32{float32(v)}, []float32{-1}
			if info, err := lapack.Ssyev(jz, 'U', 1, a32, 1, w32); info != 0 || err != nil || w32[0] != float32(v) {
				t.Fatalf("Ssyev %c %g: %d %v w=%g", jz, v, info, err, w32[0])
			}
		}
		for _, c := range []lapack.Compz{lapack.CompzNone, lapack.CompzIdentity} {
			d, z := []float64{v}, []float64{7}
			if info, err := lapack.Dsteqr(c, 1, d, nil, z, 1); info != 0 || err != nil || d[0] != v {
				t.Fatalf("Dsteqr %c %g: %d %v d=%g", c, v, info, err, d[0])
			}
			if c == lapack.CompzIdentity && z[0] != 1 {
				t.Fatalf("Dsteqr I %g: z=%g", v, z[0])
			}
			if c == lapack.CompzNone && z[0] != 7 {
				t.Fatalf("Dsteqr N %g: z touched: %g", v, z[0])
			}
		}
	}
}

// TestSteqrEmpty: n = 0 returns before any slice is read, with nil slices, for
// every COMPZ (the reference quick-returns after its argument checks).
func TestSteqrEmpty(t *testing.T) {
	for _, c := range []lapack.Compz{lapack.CompzNone, lapack.CompzOriginal, lapack.CompzIdentity} {
		if info, err := lapack.Dsteqr(c, 0, nil, nil, nil, 1); info != 0 || err != nil {
			t.Fatalf("Dsteqr %c: %d %v", c, info, err)
		}
		if info, err := lapack.Ssteqr(c, 0, nil, nil, nil, 1); info != 0 || err != nil {
			t.Fatalf("Ssteqr %c: %d %v", c, info, err)
		}
		if info, err := netlib.Dsteqr(c, 0, nil, nil, nil, 1); info != 0 || err != nil {
			t.Fatalf("netlib Dsteqr %c: %d %v", c, info, err)
		}
		if info, err := netlib.Ssteqr(c, 0, nil, nil, nil, 1); info != 0 || err != nil {
			t.Fatalf("netlib Ssteqr %c: %d %v", c, info, err)
		}
	}
}
