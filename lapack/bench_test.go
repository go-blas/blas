package lapack_test

import (
	"strconv"
	"testing"

	"github.com/go-blas/blas"
)

// Benchmarks of lapack against netlib's ports of the reference, n = 64 to 1024.
// Each iteration refreshes the matrix with a copy (n^2 against n^3 work). The
// rate counts the flops of the operation: 2n^3/3 for getrf, n^3/3 for potrf,
// 2 n^2 nrhs for the solves (nrhs = 64).

var benchSizes = []int{64, 128, 256, 512, 1024}

const benchRHS = 64

func report(b *testing.B, flops float64) {
	b.ReportMetric(flops*float64(b.N)/float64(b.Elapsed().Nanoseconds()), "Gflop/s")
}

func BenchmarkGetrf(b *testing.B) {
	for _, n := range benchSizes {
		b.Run("float64/n="+strconv.Itoa(n), func(b *testing.B) { benchGetrf(b, impls64, n) })
		b.Run("float32/n="+strconv.Itoa(n), func(b *testing.B) { benchGetrf(b, impls32, n) })
	}
}

func benchGetrf[T float](b *testing.B, impls []impl[T], n int) {
	a0 := general[T](newRNG(1), n, n, n)
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			a, ipiv := make([]T, len(a0)), make([]int, n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, a0)
				if _, err := im.getrf(n, n, a, n, ipiv); err != nil {
					b.Fatal(err)
				}
			}
			report(b, 2*float64(n)*float64(n)*float64(n)/3)
		})
	}
}

func BenchmarkPotrf(b *testing.B) {
	for _, n := range benchSizes {
		b.Run("float64/n="+strconv.Itoa(n), func(b *testing.B) { benchPotrf(b, impls64, n) })
		b.Run("float32/n="+strconv.Itoa(n), func(b *testing.B) { benchPotrf(b, impls32, n) })
	}
}

func benchPotrf[T float](b *testing.B, impls []impl[T], n int) {
	a0 := spd[T](newRNG(2), n, n, false)
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			a := make([]T, len(a0))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, a0)
				if _, err := im.potrf(blas.Lower, n, a, n); err != nil {
					b.Fatal(err)
				}
			}
			report(b, float64(n)*float64(n)*float64(n)/3)
		})
	}
}

func BenchmarkGetrs(b *testing.B) {
	for _, n := range benchSizes {
		b.Run("float64/n="+strconv.Itoa(n), func(b *testing.B) { benchGetrs(b, impls64, n) })
		b.Run("float32/n="+strconv.Itoa(n), func(b *testing.B) { benchGetrs(b, impls32, n) })
	}
}

func benchGetrs[T float](b *testing.B, impls []impl[T], n int) {
	f := general[T](newRNG(3), n, n, n)
	ipiv := make([]int, n)
	if _, err := impls[1].getrf(n, n, f, n, ipiv); err != nil {
		b.Fatal(err)
	}
	b0 := general[T](newRNG(4), n, benchRHS, n)
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			x := make([]T, len(b0))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(x, b0)
				if err := im.getrs(blas.NoTrans, n, benchRHS, f, n, ipiv, x, n); err != nil {
					b.Fatal(err)
				}
			}
			report(b, 2*float64(n)*float64(n)*benchRHS)
		})
	}
}

func BenchmarkPotrs(b *testing.B) {
	for _, n := range benchSizes {
		b.Run("float64/n="+strconv.Itoa(n), func(b *testing.B) { benchPotrs(b, impls64, n) })
		b.Run("float32/n="+strconv.Itoa(n), func(b *testing.B) { benchPotrs(b, impls32, n) })
	}
}

func benchPotrs[T float](b *testing.B, impls []impl[T], n int) {
	f := spd[T](newRNG(5), n, n, false)
	if _, err := impls[1].potrf(blas.Lower, n, f, n); err != nil {
		b.Fatal(err)
	}
	b0 := general[T](newRNG(6), n, benchRHS, n)
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			x := make([]T, len(b0))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(x, b0)
				if err := im.potrs(blas.Lower, n, benchRHS, f, n, x, n); err != nil {
					b.Fatal(err)
				}
			}
			report(b, 2*float64(n)*float64(n)*benchRHS)
		})
	}
}

// QR: n x n matrices; the flops counted are the operation's own: 4n^3/3 for
// geqrf, 4n^3/3 for orgqr (k = n), 2n^3 for ormqr (Left, NoTrans, C n x n, k = n:
// 4mnk - 2nk^2).

func BenchmarkGeqrf(b *testing.B) {
	for _, n := range benchSizes {
		b.Run("float64/n="+strconv.Itoa(n), func(b *testing.B) { benchGeqrf(b, qimpls64, n) })
		b.Run("float32/n="+strconv.Itoa(n), func(b *testing.B) { benchGeqrf(b, qimpls32, n) })
	}
}

func benchGeqrf[T float](b *testing.B, impls []qimpl[T], n int) {
	a0 := general[T](newRNG(1), n, n, n)
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			a, tau := make([]T, len(a0)), make([]T, n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, a0)
				if err := im.geqrf(n, n, a, n, tau); err != nil {
					b.Fatal(err)
				}
			}
			report(b, 4*float64(n)*float64(n)*float64(n)/3)
		})
	}
}

func BenchmarkOrgqr(b *testing.B) {
	for _, n := range benchSizes {
		b.Run("float64/n="+strconv.Itoa(n), func(b *testing.B) { benchOrgqr(b, qimpls64, n) })
		b.Run("float32/n="+strconv.Itoa(n), func(b *testing.B) { benchOrgqr(b, qimpls32, n) })
	}
}

func benchOrgqr[T float](b *testing.B, impls []qimpl[T], n int) {
	f, tau := general[T](newRNG(1), n, n, n), make([]T, n)
	if err := impls[1].geqrf(n, n, f, n, tau); err != nil {
		b.Fatal(err)
	}
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			a := make([]T, len(f))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, f)
				if err := im.orgqr(n, n, n, a, n, tau); err != nil {
					b.Fatal(err)
				}
			}
			report(b, 4*float64(n)*float64(n)*float64(n)/3)
		})
	}
}

func BenchmarkOrmqr(b *testing.B) {
	for _, n := range benchSizes {
		b.Run("float64/n="+strconv.Itoa(n), func(b *testing.B) { benchOrmqr(b, qimpls64, n) })
		b.Run("float32/n="+strconv.Itoa(n), func(b *testing.B) { benchOrmqr(b, qimpls32, n) })
	}
}

func benchOrmqr[T float](b *testing.B, impls []qimpl[T], n int) {
	f, tau := general[T](newRNG(1), n, n, n), make([]T, n)
	if err := impls[1].geqrf(n, n, f, n, tau); err != nil {
		b.Fatal(err)
	}
	c0 := general[T](newRNG(2), n, n, n)
	for _, im := range impls {
		b.Run(im.name, func(b *testing.B) {
			c := make([]T, len(c0))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(c, c0)
				if err := im.ormqr(blas.Left, blas.NoTrans, n, n, n, f, n, tau, c, n); err != nil {
					b.Fatal(err)
				}
			}
			report(b, 2*float64(n)*float64(n)*float64(n))
		})
	}
}
