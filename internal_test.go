package blas

import "testing"

func TestKernelIf(t *testing.T) {
	f := func() int { return 1 }
	if kernelIf(true, f) == nil {
		t.Fatal("kernelIf(true) = nil")
	}
	if kernelIf(false, f) != nil {
		t.Fatal("kernelIf(false) != nil")
	}
}

// Without a float32 kernel (a CPU without one, or a target that has none) Sgemm
// runs the reference loops; this drives that path on every machine.
func TestSgemmWithoutKernel(t *testing.T) {
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
