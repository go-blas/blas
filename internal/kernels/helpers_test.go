package kernels

import (
	"math/rand"
	"runtime"
)

// withThresholds runs f with GemmThreshold set to gemm and restores it after. A
// low value flips every product onto the parallel path, a high one onto the
// serial path.
func withThresholds(_, gemm int, f func()) {
	og := GemmThreshold
	GemmThreshold = gemm
	defer func() { GemmThreshold = og }()
	f()
}

// withMaxProcs runs f with GOMAXPROCS forced to n and restores it after. The
// worker count of the parallel GEMM is derived from GOMAXPROCS, so a test that
// must hit the `workers > rows` clamp pins it high here; otherwise the clamp is
// unreachable on a low-core runner and the coverage gate fails there only.
func withMaxProcs(n int, f func()) {
	old := runtime.GOMAXPROCS(n)
	defer runtime.GOMAXPROCS(old)
	f()
}

func randVec(n int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	a := make([]float64, n)
	for i := range a {
		a[i] = r.NormFloat64() * 100
	}
	return a
}
