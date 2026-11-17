package ref

import (
	"math"
	"math/rand/v2"
	"testing"
)

// naiveLaswp is dlaswp.f read plainly: row i interchanged with row
// ipiv[k1+(i-k1)*|incx|], one column at a time.
func naiveLaswp(n int, a []float64, lda, k1, k2 int, ipiv []int, incx int) {
	if incx == 0 {
		return
	}
	idx := func(i int) int { return k1 + (i-k1)*max(incx, -incx) }
	rows := make([]int, 0)
	for i := k1; i <= k2; i++ {
		rows = append(rows, i)
	}
	if incx < 0 {
		for l, r := 0, len(rows)-1; l < r; l, r = l+1, r-1 {
			rows[l], rows[r] = rows[r], rows[l]
		}
	}
	for _, i := range rows {
		ip := ipiv[idx(i)-1]
		for c := 0; c < n; c++ {
			a[i-1+c*lda], a[ip-1+c*lda] = a[ip-1+c*lda], a[i-1+c*lda]
		}
	}
}

func TestLaswp(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 31, 32, 33, 64, 70} {
		for _, incx := range []int{1, -1, 0} {
			const m = 9
			ipiv := make([]int, m)
			for i := range ipiv {
				ipiv[i] = i + 1 + rng.IntN(m-i) // 1-based, at or below the diagonal
			}
			a := make([]float64, m*max(n, 1))
			for i := range a {
				a[i] = rng.Float64()
			}
			want := append([]float64(nil), a...)
			naiveLaswp(n, want, m, 2, 8, ipiv, incx)
			Laswp(n, a, m, 2, 8, ipiv, incx)
			for i := range a {
				if a[i] != want[i] {
					t.Fatalf("n %d incx %d: element %d is %v, want %v", n, incx, i, a[i], want[i])
				}
			}
		}
	}
}

// matmulLU forms P^T L U from the factors of Getrf (m x n, pivots 1-based) and
// compares it with the original: a float64 sanity check of the ports at block
// sizes the public routines never use (the reference's is 64).
func TestGetrfBlockSizes(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, nb := range []int{-1, 0, 1, 2, 3, 7, 64, 1000} {
		for _, sh := range [][2]int{{1, 1}, {5, 5}, {9, 4}, {4, 9}, {20, 20}, {31, 17}} {
			m, n := sh[0], sh[1]
			a0 := make([]float64, m*n)
			for i := range a0 {
				a0[i] = rng.Float64() - 0.5
			}
			a := append([]float64(nil), a0...)
			ipiv := make([]int, min(m, n))
			if info := Getrf(nb, m, n, a, m, ipiv); info != 0 {
				t.Fatalf("nb %d %dx%d: info %d", nb, m, n, info)
			}
			// P A = L U, with P applied by the interchanges in order.
			r := min(m, n)
			pa := append([]float64(nil), a0...)
			Laswp(n, pa, m, 1, r, ipiv, 1)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var s float64
					for k := 0; k <= min(i, j) && k < r; k++ {
						l := a[i+k*m]
						if i == k {
							l = 1
						}
						s += l * a[k+j*m]
					}
					if math.Abs(s-pa[i+j*m]) > 1e-12 {
						t.Fatalf("nb %d %dx%d: (P A - L U)(%d,%d) = %g", nb, m, n, i, j, s-pa[i+j*m])
					}
				}
			}
		}
	}
}

func TestPotrfBlockSizes(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, nb := range []int{-1, 0, 1, 2, 3, 7, 64, 1000} {
		for _, n := range []int{1, 2, 5, 17, 30} {
			for _, up := range []bool{true, false} {
				a0 := make([]float64, n*n)
				for j := 0; j < n; j++ {
					for i := 0; i < n; i++ {
						switch {
						case i == j:
							a0[i+j*n] = float64(n) + 1
						default:
							a0[i+j*n] = 0.5 * (rng.Float64() - 0.5)
							a0[j+i*n] = a0[i+j*n]
						}
					}
				}
				a := append([]float64(nil), a0...)
				if info := Potrf(nb, up, n, a, n); info != 0 {
					t.Fatalf("nb %d n %d: info %d", nb, n, info)
				}
				for i := 0; i < n; i++ {
					for j := 0; j < n; j++ {
						if (i <= j) != up && i != j {
							continue
						}
						var s float64
						for k := 0; k < n; k++ {
							if up {
								if k <= i && k <= j {
									s += a[k+i*n] * a[k+j*n]
								}
							} else if k <= i && k <= j {
								s += a[i+k*n] * a[j+k*n]
							}
						}
						if math.Abs(s-a0[i+j*n]) > 1e-12 {
							t.Fatalf("nb %d n %d up %v: (A - C'C)(%d,%d) = %g", nb, n, up, i, j, s-a0[i+j*n])
						}
					}
				}
			}
		}
	}
}

func TestSafeMin(t *testing.T) {
	if SafeMin[float64]() != math.Ldexp(1, -1022) || SafeMin[float32]() != float32(math.Ldexp(1, -126)) {
		t.Fatal("SafeMin")
	}
}
