package lapack_test

import (
	"strconv"
	"testing"

	"github.com/go-blas/blas/lapack"
)

// BenchmarkSyev: Syev against netlib's port, eigenvalues only ("N") and with
// eigenvectors ("V"), lower triangle, n = 64 to 512. The metric is the time of
// one call (the matrix is refreshed by a copy, n^2 against n^3 work).
func BenchmarkSyev(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512} {
		for _, jz := range []lapack.Jobz{lapack.EigenvaluesOnly, lapack.EigenvaluesAndVectors} {
			name := "/n=" + strconv.Itoa(n) + "/" + string(rune(jz))
			b.Run("float64"+name, func(b *testing.B) { benchSyev(b, eimpls64, n, jz) })
			b.Run("float32"+name, func(b *testing.B) { benchSyev(b, eimpls32, n, jz) })
		}
	}
}

func benchSyev[T float](b *testing.B, impls []eimpl[T], n int, jz lapack.Jobz) {
	a0 := denseSym[T]("dense/random", n, n, false, 1, newRNG(3))
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			a, w := make([]T, len(a0)), make([]T, n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, a0)
				if info, err := im.syev(jz, 'L', n, a, n, w); err != nil || info != 0 {
					b.Fatal(info, err)
				}
			}
		})
	}
}
