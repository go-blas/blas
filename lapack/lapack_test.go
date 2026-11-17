package lapack_test

import (
	"errors"
	"math"
	"strconv"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/lapack"
)

// sizes cross the recursion (leaf 1, 2, 3, 16) and, for netlib, the block size
// 64 of ilaenv: 63, 64, 65 and a size with a ragged last block.
var shapes = [][2]int{
	{1, 1}, {1, 5}, {5, 1}, {2, 2}, {3, 3}, {5, 5}, {7, 4}, {4, 7}, {16, 16},
	{17, 17}, {33, 20}, {20, 33}, {40, 40}, {63, 63}, {64, 64}, {65, 65}, {70, 130}, {130, 70}, {130, 130},
}

func nameOf(parts ...any) string {
	s := ""
	for _, p := range parts {
		switch v := p.(type) {
		case int:
			s += "/" + strconv.Itoa(v)
		case string:
			s += "/" + v
		case byte:
			s += "/" + string(rune(v))
		}
	}
	return s
}

func TestGetrfBound(t *testing.T) {
	each(t, getrfBoundT[float64], getrfBoundT[float32])
}

func getrfBoundT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		for _, sh := range shapes {
			m, n := sh[0], sh[1]
			if l == 1 && m*n > 40*40 {
				continue // the exact oracle is cubic; netlib and the default leaf cover the large ones
			}
			for _, pad := range []int{0, 3} {
				t.Run(nameOf(l, m, n, pad), func(t *testing.T) {
					setLeaf(t, l)
					rng := newRNG(uint64(m*1000 + n*10 + pad))
					lda := m + pad
					a0 := general[T](rng, m, n, lda)
					a := clone(a0)
					ipiv := make([]int, min(m, n))
					info, err := im.getrf(m, n, a, lda, ipiv)
					if err != nil || info != 0 {
						t.Fatalf("info %d, err %v on a random matrix", info, err)
					}
					padUntouched(t, "a", a, m, max(lda, 1))
					getrfBound(t, im.name, a0, a, ipiv, m, n, lda)
				})
			}
		}
	}
}

func TestGetrfSingular(t *testing.T) {
	each(t, getrfSingularT[float64], getrfSingularT[float32])
}

// A zero column makes U(j,j) exactly zero: INFO is j+1 as netlib returns it, the
// factorization is completed, and its bound still holds.
func getrfSingularT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		for _, n := range []int{1, 2, 5, 17, 40, 70} {
			for _, j := range []int{0, n / 2, n - 1} {
				setLeaf(t, l)
				rng := newRNG(uint64(n*100 + j))
				a0 := general[T](rng, n, n, n)
				for i := 0; i < n; i++ {
					a0[i+j*n] = 0
				}
				a := clone(a0)
				ipiv := make([]int, n)
				info, err := im.getrf(n, n, a, n, ipiv)
				if err != nil || info != j+1 {
					t.Fatalf("leaf %d n %d zero column %d: info %d, err %v, want %d", l, n, j, info, err, j+1)
				}
				getrfBound(t, im.name, a0, a, ipiv, n, n, n)
			}
		}
	}
	// The zero matrix and a matrix whose second pivot vanishes.
	for _, l := range leaves(im.name) {
		setLeaf(t, l)
		z := make([]T, 9)
		ipiv := make([]int, 3)
		if info, err := im.getrf(3, 3, z, 3, ipiv); err != nil || info != 1 {
			t.Fatalf("zero matrix: info %d, err %v", info, err)
		}
		a := []T{1, 1, 1, 1, 1, 1, 3, 5, 4} // the second column equals the first: U(2,2) = 0
		if info, err := im.getrf(3, 3, a, 3, ipiv); err != nil || info != 2 {
			t.Fatalf("dependent columns: info %d, err %v, want 2", info, err)
		}
	}
}

// A pivot column of one element, a subnormal: the column is divided, not
// multiplied by the reciprocal (which would overflow), as the reference does.
func TestGetrfTinyPivot(t *testing.T) {
	each(t, tinyT[float64], tinyT[float32])
}

func tinyT[T float](t *testing.T, im impl[T]) {
	var z T
	_, is32 := any(z).(float32)
	tiny := T(math.Ldexp(1, -1070))
	if is32 {
		tiny = T(math.Ldexp(1, -145))
	}
	for _, l := range leaves(im.name) {
		setLeaf(t, l)
		col := []T{tiny, 3 * tiny, 2 * tiny, tiny}
		a := clone(col)
		ipiv := make([]int, 1)
		if info, err := im.getrf(4, 1, a, 4, ipiv); err != nil || info != 0 {
			t.Fatalf("info %d, err %v", info, err)
		}
		if ipiv[0] != 2 || a[0] != 3*tiny {
			t.Fatalf("pivot %d, a[0] = %v", ipiv[0], a[0])
		}
		// After the interchange of rows 1 and 2 the column is (3t, t, 2t, t), and
		// each element below the pivot is divided by it.
		if a[1] != col[0]/col[1] || a[2] != col[2]/col[1] || a[3] != col[3]/col[1] {
			t.Fatalf("scaled column %v", a)
		}
	}
}

func TestGetrfRowVector(t *testing.T) {
	each(t, rowT[float64], rowT[float32])
}

// One row: just the pivot and INFO.
func rowT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		setLeaf(t, l)
		ipiv := make([]int, 1)
		if info, err := im.getrf(1, 3, []T{2, 4, 6}, 1, ipiv); err != nil || info != 0 || ipiv[0] != 1 {
			t.Fatalf("info %d, err %v, ipiv %v", info, err, ipiv)
		}
		if info, err := im.getrf(1, 3, []T{0, 4, 6}, 1, ipiv); err != nil || info != 1 {
			t.Fatalf("zero row: info %d, err %v", info, err)
		}
	}
}

func TestGetrfNaNInf(t *testing.T) {
	each(t, nanInfT[float64], nanInfT[float32])
}

// refOf is the netlib implementation of the same element type: the reference
// the non-finite values must travel like.
func refOf[T float]() impl[T] {
	var z T
	if _, ok := any(z).(float32); ok {
		return any(impls32[1]).(impl[T])
	}
	return any(impls64[1]).(impl[T])
}

// classEq reports whether two values are of the same class: NaN with NaN, an
// infinity with the infinity of the same sign, a finite value with a finite one.
func classEq[T float](a, b T) bool {
	x, y := float64(a), float64(b)
	switch {
	case math.IsNaN(x) || math.IsNaN(y):
		return math.IsNaN(x) && math.IsNaN(y)
	case math.IsInf(x, 0) || math.IsInf(y, 0):
		return x == y
	}
	return true
}

// NaN and Inf in the data travel as in the reference: the factors of lapack
// and netlib are of the same class at every element and the pivots are the
// same, whatever the panel width; and no panic, no error. The pivot search keeps
// a leading NaN, as IDAMAX.
func nanInfT[T float](t *testing.T, im impl[T]) {
	inf := T(math.Inf(1))
	nl := refOf[T]()
	for _, l := range leaves(im.name) {
		setLeaf(t, l)
		for _, bad := range []T{nan[T](), inf, -inf} {
			for _, n := range []int{3, 20} {
				// Planted at the middle of the diagonal, in the first row, in the
				// last column, and in a matrix whose first column is zero (the
				// infinity then meets the zero multipliers of the update).
				for variant := 0; variant < 4; variant++ {
					a := general[T](newRNG(uint64(n)), n, n, n)
					switch variant {
					case 0:
						a[n/2+(n/2)*n] = bad
					case 1:
						a[(n-1)*n] = bad
					case 2:
						a[n-1+(n-1)*n] = bad
					case 3:
						for i := 0; i < n; i++ {
							a[i] = 0
						}
						a[(n-1)*n] = bad
					}
					want, got := clone(a), clone(a)
					wp, gp := make([]int, n), make([]int, n)
					if _, err := nl.getrf(n, n, want, n, wp); err != nil {
						t.Fatal(err)
					}
					if _, err := im.getrf(n, n, got, n, gp); err != nil {
						t.Fatal(err)
					}
					for i := range want {
						if !classEq(got[i], want[i]) {
							t.Fatalf("leaf %d n %d variant %d bad %v: element %d is %v, netlib %v", l, n, variant, bad, i, got[i], want[i])
						}
					}
					for i := range wp {
						if gp[i] != wp[i] {
							t.Fatalf("leaf %d n %d variant %d bad %v: pivot %d is %d, netlib %d", l, n, variant, bad, i, gp[i], wp[i])
						}
					}
				}
			}
		}
		// A zero pivot column over an infinity: DGETF2 and DGETRF2 still run the
		// update, and 0*Inf is NaN at U(2,2), in every implementation and panel width.
		z := []T{0, 0, inf, 1}
		zp := make([]int, 2)
		info, err := im.getrf(2, 2, z, 2, zp)
		if err != nil || info != 1 || !math.IsNaN(float64(z[3])) || !math.IsInf(float64(z[2]), 1) {
			t.Fatalf("leaf %d: info %d, err %v, factors %v: want info 1 and U(2,2) NaN", l, info, err, z)
		}
		// A NaN first in the column is the pivot: the interchange is skipped.
		a := []T{nan[T](), 5, 7, 1}
		ipiv := make([]int, 1)
		if _, err := im.getrf(4, 1, a, 4, ipiv); err != nil || ipiv[0] != 1 {
			t.Fatalf("NaN first: ipiv %v, err %v", ipiv, err)
		}
	}
}

func TestPotrfBound(t *testing.T) {
	each(t, potrfBoundT[float64], potrfBoundT[float32])
}

func potrfBoundT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		for _, n := range []int{1, 2, 3, 5, 16, 17, 33, 40, 63, 64, 65, 130} {
			if l == 1 && n > 40 {
				continue
			}
			for _, up := range []bool{true, false} {
				for _, pad := range []int{0, 3} {
					setLeaf(t, l)
					rng := newRNG(uint64(n*10 + pad))
					lda := n + pad
					a0 := spd[T](rng, n, lda, up)
					a := clone(a0)
					uplo := blas.Lower
					if up {
						uplo = blas.Upper
					}
					info, err := im.potrf(uplo, n, a, lda)
					if err != nil || info != 0 {
						t.Fatalf("leaf %d n %d up %v: info %d, err %v", l, n, up, info, err)
					}
					padUntouched(t, "a", a, n, lda)
					// The other triangle is never read or written: it holds NaN still.
					for j := 0; j < n; j++ {
						for i := 0; i < n; i++ {
							if i != j && (i < j) != up && !math.IsNaN(float64(a[i+j*lda])) {
								t.Fatalf("other triangle written at (%d,%d)", i, j)
							}
						}
					}
					potrfBound(t, im.name, a0, a, n, lda, up)
				}
			}
		}
	}
}

func TestPotrfNotPositiveDefinite(t *testing.T) {
	each(t, potrfNPDT[float64], potrfNPDT[float32])
}

// A diagonal element made negative (or NaN, or zero) fails exactly at its own
// leading minor: INFO is its index, 1-based, in every block position.
func potrfNPDT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		for _, n := range []int{1, 2, 5, 17, 40, 70, 130} {
			if l == 1 && n > 40 {
				continue
			}
			for _, k := range []int{0, n / 3, n - 1} {
				for _, bad := range []T{-1, 0, nan[T]()} {
					for _, up := range []bool{true, false} {
						setLeaf(t, l)
						a0 := spd[T](newRNG(uint64(n+k)), n, n, up)
						a0[k+k*n] = bad
						a := clone(a0)
						uplo := blas.Lower
						if up {
							uplo = blas.Upper
						}
						info, err := im.potrf(uplo, n, a, n)
						if err != nil || info != k+1 {
							t.Fatalf("leaf %d n %d k %d bad %v up %v: info %d, err %v, want %d", l, n, k, bad, up, info, err, k+1)
						}
						// The leading block before the failure is a Cholesky factor.
						potrfBound(t, im.name, a0, a, k, n, up)
					}
				}
			}
		}
	}
}

func TestPotrfInf(t *testing.T) {
	each(t, potrfInfT[float64], potrfInfT[float32])
}

func potrfInfT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		setLeaf(t, l)
		// An infinite diagonal element is positive: the factorization goes on
		// (the row below it divides to zero), INFO 0, and the factors are of the
		// same class as netlib's.
		for _, up := range []bool{true, false} {
			uplo := blas.Lower
			if up {
				uplo = blas.Upper
			}
			a := spd[T](newRNG(1), 20, 20, up)
			a[0] = T(math.Inf(1))
			want := clone(a)
			winfo, werr := refOf[T]().potrf(uplo, 20, want, 20)
			info, err := im.potrf(uplo, 20, a, 20)
			if err != nil || werr != nil || info != 0 || winfo != 0 {
				t.Fatalf("leaf %d up %v: info %d, err %v (netlib %d, %v), want 0", l, up, info, err, winfo, werr)
			}
			for i := range a {
				if !classEq(a[i], want[i]) {
					t.Fatalf("leaf %d up %v: element %d is %v, netlib %v", l, up, i, a[i], want[i])
				}
			}
		}
	}
}

func TestGetrsBound(t *testing.T) {
	each(t, getrsBoundT[float64], getrsBoundT[float32])
}

func getrsBoundT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		for _, n := range []int{1, 2, 5, 17, 40, 65} {
			for _, nrhs := range []int{1, 3, 33} {
				if (l == 1 && n > 17) || (n == 65 && nrhs > 3) {
					continue
				}
				for _, tr := range []byte{'N', 'T', 'C', 'n', 't', 'c'} {
					setLeaf(t, l)
					rng := newRNG(uint64(n*100 + nrhs))
					lda, ldb := n+2, n+1
					f := general[T](rng, n, n, lda)
					ipiv := make([]int, n)
					if info, err := im.getrf(n, n, f, lda, ipiv); err != nil || info != 0 {
						t.Fatalf("getrf: %d %v", info, err)
					}
					b0 := general[T](rng, n, nrhs, ldb)
					x := clone(b0)
					if err := im.getrs(blas.Transpose(tr), n, nrhs, f, lda, ipiv, x, ldb); err != nil {
						t.Fatal(err)
					}
					padUntouched(t, "b", x, n, ldb)
					getrsBound(t, im.name, upper(tr), f, ipiv, n, nrhs, lda, b0, x, ldb)
				}
			}
		}
	}
}

func TestPotrsBound(t *testing.T) {
	each(t, potrsBoundT[float64], potrsBoundT[float32])
}

func potrsBoundT[T float](t *testing.T, im impl[T]) {
	for _, l := range leaves(im.name) {
		for _, n := range []int{1, 2, 5, 17, 40, 65} {
			for _, nrhs := range []int{1, 3, 33} {
				if (l == 1 && n > 17) || (n == 65 && nrhs > 3) {
					continue
				}
				for _, up := range []bool{true, false} {
					setLeaf(t, l)
					rng := newRNG(uint64(n*100 + nrhs))
					lda, ldb := n+2, n+1
					f := spd[T](rng, n, lda, up)
					uplo := blas.Lower
					if up {
						uplo = blas.Upper
					}
					if info, err := im.potrf(uplo, n, f, lda); err != nil || info != 0 {
						t.Fatalf("potrf: %d %v", info, err)
					}
					b0 := general[T](rng, n, nrhs, ldb)
					x := clone(b0)
					if err := im.potrs(uplo, n, nrhs, f, lda, x, ldb); err != nil {
						t.Fatal(err)
					}
					padUntouched(t, "b", x, n, ldb)
					potrsBound(t, im.name, f, n, nrhs, lda, b0, x, ldb, up)
				}
			}
		}
	}
}

// The solve from a factorization recovers a planted solution to the accuracy the
// conditioning allows: a sanity check of the whole chain on a system with a known
// answer, independent of the residual machinery above.
func TestSolveKnown(t *testing.T) {
	a := []float64{4, 1, 0, 1, 3, 1, 0, 1, 2} // symmetric positive definite
	x := []float64{1, 2, 3}
	b := []float64{4*1 + 1*2, 1*1 + 3*2 + 1*3, 1*2 + 2*3}
	f := append([]float64(nil), a...)
	ipiv := make([]int, 3)
	if info, err := lapack.Dgetrf(3, 3, f, 3, ipiv); err != nil || info != 0 {
		t.Fatal(info, err)
	}
	got := append([]float64(nil), b...)
	if err := lapack.Dgetrs(blas.NoTrans, 3, 1, f, 3, ipiv, got, 3); err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if math.Abs(got[i]-x[i]) > 1e-12 {
			t.Fatalf("getrs: %v", got)
		}
	}
	c := append([]float64(nil), a...)
	if info, err := lapack.Dpotrf(blas.Lower, 3, c, 3); err != nil || info != 0 {
		t.Fatal(info, err)
	}
	got = append([]float64(nil), b...)
	if err := lapack.Dpotrs(blas.Lower, 3, 1, c, 3, got, 3); err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if math.Abs(got[i]-x[i]) > 1e-12 {
			t.Fatalf("potrs: %v", got)
		}
	}
}

func TestMust(t *testing.T) {
	lapack.Must(nil)
	defer func() {
		if recover() == nil {
			t.Fatal("must did not panic on an error")
		}
	}()
	lapack.Must(errors.New("boom"))
}
