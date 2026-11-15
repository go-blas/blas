package blas

import (
	"testing"

	"github.com/go-blas/blas/internal/kernels"
)

func TestKernelIf(t *testing.T) {
	f := func() int { return 1 }
	if kernelIf(true, f) == nil {
		t.Fatal("kernelIf(true) = nil")
	}
	if kernelIf(false, f) != nil {
		t.Fatal("kernelIf(false) != nil")
	}
}

func TestHasSgemmKernel(t *testing.T) {
	if got, want := HasSgemmKernel(), kernels.HasGemm32(); got != want {
		t.Fatalf("HasSgemmKernel() = %v, kernels.HasGemm32() = %v", got, want)
	}
	saved := sHW.mul
	sHW.mul = nil
	defer func() { sHW.mul = saved }()
	if HasSgemmKernel() {
		t.Fatal("HasSgemmKernel() = true without a kernel")
	}
}

// Without a float32 kernel (a CPU without one, or a target that has none) Sgemm
// runs the reference loops; this drives that path on every machine.
func TestSgemmWithoutKernel(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panicked: %v", r)
		}
	}()
	saved := sHW.mul
	sHW.mul = nil
	defer func() { sHW.mul = saved }()
	a, b := []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}
	c := make([]float32, 4)
	if err := Sgemm(NoTrans, NoTrans, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2); err != nil {
		t.Fatal(err)
	}
	// column-major: A = [1 3; 2 4], B = [5 7; 6 8], AB = [23 31; 34 46]
	want := []float32{23, 34, 31, 46}
	for i := range want {
		if c[i] != want[i] {
			t.Fatalf("c = %v, want %v", c, want)
		}
	}
}

// Without a float32 kernel SgemmStrided runs plain loops over the strides, for
// every beta; this drives that path on every machine.
func TestSgemmStridedWithoutKernel(t *testing.T) {
	saved := sHW.mul
	sHW.mul = nil
	defer func() { sHW.mul = saved }()
	a, b := []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8} // row-major A = [1 2; 3 4], B = [5 6; 7 8]
	for _, tc := range []struct {
		alpha, beta float32
		want        []float32 // row-major, from C = [1 1; 1 1]
	}{
		{1, 0, []float32{19, 22, 43, 50}},
		{1, 1, []float32{20, 23, 44, 51}},
		{2, 2, []float32{40, 46, 88, 102}},
	} {
		c := []float32{1, 1, 1, 1}
		if err := SgemmStrided(2, 2, 2, tc.alpha, a, 2, 1, b, 2, 1, tc.beta, c, 2, 1); err != nil {
			t.Fatal(err)
		}
		for i := range c {
			if c[i] != tc.want[i] {
				t.Fatalf("alpha, beta = %v, %v: c = %v, want %v", tc.alpha, tc.beta, c, tc.want)
			}
		}
	}
}
