package blas

import (
	"slices"
	"testing"

	"github.com/go-blas/blas/internal/ref"
)

// Without a float32 kernel the level 3 routines run the reference loops: the
// result is bit for bit that of package ref, which this drives on every machine.
func TestLevel3WithoutKernel(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panicked: %v", r)
		}
	}()
	saved := sHW.mul
	sHW.mul = nil
	defer func() { sHW.mul = saved }()
	const n = 9
	mk := func(seed float32) []float32 {
		v := make([]float32, n*n)
		for i := range v {
			v[i] = seed + float32(i%7)*0.25 + float32(i%3)
		}
		for i := 0; i < n; i++ {
			v[i+i*n] += 20
		}
		return v
	}
	a, b, c := mk(1), mk(2), mk(3)
	run := func(name string, got func(x []float32) error, want func(x []float32)) {
		x, y := slices.Clone(c), slices.Clone(c)
		if err := got(x); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want(y)
		if !slices.Equal(x, y) {
			t.Fatalf("%s without a kernel differs from the reference loops", name)
		}
	}
	run("Strsm", func(x []float32) error { return Strsm(Left, Upper, NoTrans, NonUnit, n, n, 2, a, n, x, n) },
		func(y []float32) { ref.Trsm(true, true, true, true, n, n, 2, a, n, y, n) })
	run("Strmm", func(x []float32) error { return Strmm(Right, Lower, Trans, Unit, n, n, 2, a, n, x, n) },
		func(y []float32) { ref.Trmm(false, false, false, false, n, n, 2, a, n, y, n) })
	run("Ssyrk", func(x []float32) error { return Ssyrk(Upper, Trans, n, n, 2, a, n, 3, x, n) },
		func(y []float32) { ref.Syrk(true, false, n, n, 2, a, n, 3, y, n) })
	run("Ssyr2k", func(x []float32) error { return Ssyr2k(Lower, NoTrans, n, n, 2, a, n, b, n, 3, x, n) },
		func(y []float32) { ref.Syr2k(false, true, n, n, 2, a, n, b, n, 3, y, n) })
	run("Ssymm", func(x []float32) error { return Ssymm(Left, Lower, n, n, 2, a, n, b, n, 3, x, n) },
		func(y []float32) { ref.Symm(true, false, n, n, 2, a, n, b, n, 3, y, n) })
}

func TestBlockSizeDefault(t *testing.T) {
	if blockSize < 16 {
		t.Fatalf("blockSize = %d", blockSize)
	}
}
