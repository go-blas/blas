package blas_test

import (
	"testing"

	"github.com/go-blas/blas"
)

// Benchmarks of the level 3 routines against netlib's ports of the reference,
// the same call on the same data. Run them on one core, on a quiet machine:
//
//	GOMAXPROCS=1 go test -run '^$' -bench 'Level3' -benchtime 20x -cpu 1 .
//
// and compare the "blas" and "netlib" lines of each size.

var level3Sizes = []int{128, 512}

func benchTri[T float](b *testing.B, im impl[T], trsm bool) {
	for _, n := range level3Sizes {
		a := triMatrix[T](newRNG(1), n, n, false, false, true)
		b0 := matrix[T](newRNG(2), n, n, n)
		x := make([]T, len(b0))
		b.Run(itoa(n), func(b *testing.B) {
			f := im.trmm
			if trsm {
				f = im.trsm
			}
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				copy(x, b0)
				if err := f(blas.Left, blas.Lower, blas.NoTrans, blas.NonUnit, n, n, 1, a, n, x, n); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(n)*float64(n)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "Gflop/s")
		})
	}
}

func benchRank[T float](b *testing.B, im impl[T], two bool) {
	for _, n := range level3Sizes {
		a, bb := matrix[T](newRNG(1), n, n, n), matrix[T](newRNG(2), n, n, n)
		c := cMatrix[T](newRNG(3), n, n, true)
		b.Run(itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			flops := float64(n) * float64(n) * float64(n)
			for i := 0; i < b.N; i++ {
				var err error
				if two {
					err = im.syr2k(blas.Upper, blas.NoTrans, n, n, 1, a, n, bb, n, 1, c, n)
				} else {
					err = im.syrk(blas.Upper, blas.NoTrans, n, n, 1, a, n, 1, c, n)
				}
				if err != nil {
					b.Fatal(err)
				}
			}
			if two {
				flops *= 2
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "Gflop/s")
		})
	}
}

func benchSymm[T float](b *testing.B, im impl[T]) {
	for _, n := range level3Sizes {
		a := symMatrix[T](newRNG(1), n, n, true)
		bb, c := matrix[T](newRNG(2), n, n, n), matrix[T](newRNG(3), n, n, n)
		b.Run(itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := im.symm(blas.Left, blas.Upper, n, n, 1, a, n, bb, n, 1, c, n); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(2*float64(n)*float64(n)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "Gflop/s")
		})
	}
}

func eachBench(b *testing.B, f64 func(*testing.B, impl[float64]), f32 func(*testing.B, impl[float32])) {
	for _, im := range impls64 {
		b.Run("float64/"+im.name, func(b *testing.B) { f64(b, im) })
	}
	for _, im := range impls32 {
		b.Run("float32/"+im.name, func(b *testing.B) { f32(b, im) })
	}
}

func BenchmarkLevel3Trsm(b *testing.B) {
	eachBench(b, func(b *testing.B, im impl[float64]) { benchTri(b, im, true) }, func(b *testing.B, im impl[float32]) { benchTri(b, im, true) })
}

func BenchmarkLevel3Trmm(b *testing.B) {
	eachBench(b, func(b *testing.B, im impl[float64]) { benchTri(b, im, false) }, func(b *testing.B, im impl[float32]) { benchTri(b, im, false) })
}

func BenchmarkLevel3Syrk(b *testing.B) {
	eachBench(b, func(b *testing.B, im impl[float64]) { benchRank(b, im, false) }, func(b *testing.B, im impl[float32]) { benchRank(b, im, false) })
}

func BenchmarkLevel3Syr2k(b *testing.B) {
	eachBench(b, func(b *testing.B, im impl[float64]) { benchRank(b, im, true) }, func(b *testing.B, im impl[float32]) { benchRank(b, im, true) })
}

func BenchmarkLevel3Symm(b *testing.B) {
	eachBench(b, benchSymm[float64], benchSymm[float32])
}
