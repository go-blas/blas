package kernels

import "runtime"

// spinUntil waits for done to become true, which other running goroutines
// will make so: a spin, then a spin that yields (runtime.Gosched) so an
// oversubscribed machine still makes progress. It never parks, because parking
// is exactly the cost the GEMM workers avoid: waking threads one after another.
//
// Moved from go-ndarray/ndarray internal/kernels/parallel.go; the rest of that
// file (the elementwise and reduction fan-out) is not GEMM and stays there.
func spinUntil(done func() bool) {
	for i := 0; !done(); i++ {
		if i >= spinIters {
			runtime.Gosched()
		}
	}
}

// spinIters is how many times spinUntil polls before it starts yielding. A var
// so tests can make every wait yield.
var spinIters = 1 << 10
