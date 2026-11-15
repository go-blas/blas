package blas_test

import (
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
)

// triShapes are the (m, n) of the triangular routines: the empty ones, the unit
// one, odd sizes, and sizes that span several blocks of 4 and 7 (the default of
// 64 is covered by triBig).
var triShapes = [][2]int{{0, 5}, {5, 0}, {1, 1}, {6, 5}, {13, 9}, {37, 11}, {9, 37}}

// triBig are shapes that span several blocks of the default size.
var triBig = [][2]int{{150, 5}, {5, 150}}

func transFlags() []blas.Transpose {
	return []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans, 'n', 't', 'c'}
}

// forEachTri calls f for every combination of side, uplo, trans and diag (and
// the lower case spellings of some), on the given shapes, with padded leading
// dimensions on every other shape.
func forEachTri(shapes [][2]int, f func(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n, pad int)) {
	for si, sh := range shapes {
		for _, side := range sides {
			for _, uplo := range uplos {
				for _, tA := range trans2 {
					for _, diag := range diags {
						f(side, uplo, tA, diag, sh[0], sh[1], 3*(si%2))
					}
				}
			}
		}
	}
}

func orderOf(side blas.Side, m, n int) int {
	if side == blas.Left {
		return m
	}
	return n
}

func testTrsmBound[T float](t *testing.T, im impl[T]) {
	for _, bs := range blockSizes(im.name) {
		blas.SetBlockSize(t, bs)
		shapes := triShapes
		if bs == 64 {
			shapes = append(append([][2]int{}, triShapes...), triBig...)
		}
		rng := newRNG(11)
		forEachTri(shapes, func(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n, pad int) {
			s := orderOf(side, m, n)
			for _, alpha := range []T{1, -0.75, 2.5} {
				lda, ldb := max(1, s)+pad, max(1, m)+pad
				a := triMatrix[T](rng, s, lda, uplo == blas.Upper, diag == blas.Unit, true)
				b := matrix[T](rng, m, n, ldb)
				a0, b0 := clone(a), clone(b)
				if err := im.trsm(side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb); err != nil {
					t.Fatal(err)
				}
				what := flagName(byte(side), byte(uplo), byte(tA), byte(diag)) + " " + sprint(shape{m, n, s}, [2]float64{float64(alpha), 0}, pad) + " nb=" + itoa(bs)
				trsmWithin(t, what, byte(side), byte(uplo), byte(tA), byte(diag), m, n, alpha, a, lda, b0, b, ldb)
				padUntouched(t, what+" B", b, m, ldb)
				if !same(a, a0) {
					t.Fatalf("%s: A was modified", what)
				}
			}
		})
	}
}

func TestTrsmBound(t *testing.T) { each(t, testTrsmBound[float64], testTrsmBound[float32]) }

func testTrmmBound[T float](t *testing.T, im impl[T]) {
	for _, bs := range blockSizes(im.name) {
		blas.SetBlockSize(t, bs)
		shapes := triShapes
		if bs == 64 {
			shapes = append(append([][2]int{}, triShapes...), triBig...)
		}
		rng := newRNG(12)
		forEachTri(shapes, func(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n, pad int) {
			s := orderOf(side, m, n)
			for _, alpha := range []T{1, -0.75, 2.5} {
				lda, ldb := max(1, s)+pad, max(1, m)+pad
				a := triMatrix[T](rng, s, lda, uplo == blas.Upper, diag == blas.Unit, false)
				b := matrix[T](rng, m, n, ldb)
				a0, b0 := clone(a), clone(b)
				// op(A) * B (Left) or B * op(A) (Right), exactly, from the dense triangle.
				td := triDense(a, lda, s, uplo == blas.Upper, diag == blas.Unit)
				var r oracle.Result
				if side == blas.Left {
					r = oracle.Gemm(byte(tA), 'N', m, n, m, alpha, td, s, b0, ldb, T(0), nil, 0)
				} else {
					r = oracle.Gemm('N', byte(tA), m, n, n, alpha, b0, ldb, td, s, T(0), nil, 0)
				}
				if err := im.trmm(side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb); err != nil {
					t.Fatal(err)
				}
				what := flagName(byte(side), byte(uplo), byte(tA), byte(diag)) + " " + sprint(shape{m, n, s}, [2]float64{float64(alpha), 0}, pad) + " nb=" + itoa(bs)
				g := oracle.Gamma[T](s + 3)
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						within(t, what, b[i+j*ldb], r, i+j*m, g, g)
					}
				}
				padUntouched(t, what+" B", b, m, ldb)
				if !same(a, a0) {
					t.Fatalf("%s: A was modified", what)
				}
			}
		})
	}
}

func TestTrmmBound(t *testing.T) { each(t, testTrmmBound[float64], testTrmmBound[float32]) }

// alpha = 0 sets B to zero, whatever B holds, and does not read A.
func testTriAlphaZero[T float](t *testing.T, im impl[T]) {
	rng := newRNG(13)
	for name, f := range map[string]func(blas.Side, blas.Uplo, blas.Transpose, blas.Diag, int, int, T, []T, int, []T, int) error{"trsm": im.trsm, "trmm": im.trmm} {
		for _, side := range sides {
			m, n := 7, 5
			b := matrix[T](rng, m, n, m+2)
			b[0], b[3] = nan[T](), T(1)
			if err := f(side, blas.Lower, blas.Trans, blas.NonUnit, m, n, 0, nil, 7, b, m+2); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					if b[i+j*(m+2)] != 0 {
						t.Fatalf("%s %c: alpha = 0 left B[%d,%d] = %v", name, side, i, j, b[i+j*(m+2)])
					}
				}
			}
			padUntouched(t, name+" B", b, m, m+2)
		}
	}
}

func TestTriAlphaZero(t *testing.T) { each(t, testTriAlphaZero[float64], testTriAlphaZero[float32]) }

// Quick returns touch nothing and check no slice: an empty B, or a short A, with
// m = 0 or n = 0.
func testTriQuickReturn[T float](t *testing.T, im impl[T]) {
	for name, f := range map[string]func(blas.Side, blas.Uplo, blas.Transpose, blas.Diag, int, int, T, []T, int, []T, int) error{"trsm": im.trsm, "trmm": im.trmm} {
		for _, call := range []func() error{
			func() error { return f(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, 5, 0, 1, nil, 5, nil, 5) },
			func() error { return f(blas.Right, blas.Upper, blas.NoTrans, blas.NonUnit, 0, 5, 1, nil, 5, nil, 1) },
		} {
			if err := call(); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
		}
	}
}

func TestTriQuickReturn(t *testing.T) {
	each(t, testTriQuickReturn[float64], testTriQuickReturn[float32])
}

// A zero diagonal of a non-unit A is not tested for, as in the reference: the
// solve yields infinities, with no panic and no error.
func testTrsmSingular[T float](t *testing.T, im impl[T]) {
	a := []T{0, 0, 0, 1} // A = [0 0; 0 1], upper
	b := []T{1, 1}
	if err := im.trsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, 2, 1, 1, a, 2, b, 2); err != nil {
		t.Fatal(err)
	}
	if b[0] < 1e30 { // 1/0
		t.Fatalf("a singular solve gave %v, want an infinity", b)
	}
}

func TestTrsmSingular(t *testing.T) { each(t, testTrsmSingular[float64], testTrsmSingular[float32]) }

// ConjTrans is Trans for real types, and the lower case flags are the upper case
// ones, bit for bit.
func testTriFlagSpellings[T float](t *testing.T, im impl[T]) {
	rng := newRNG(14)
	for _, bs := range blockSizes(im.name) {
		blas.SetBlockSize(t, bs)
		m, n := 11, 9
		for _, side := range sides {
			s := orderOf(side, m, n)
			a := triMatrix[T](rng, s, s, true, false, true)
			b0 := matrix[T](rng, m, n, m)
			for _, f := range []func(blas.Side, blas.Uplo, blas.Transpose, blas.Diag, int, int, T, []T, int, []T, int) error{im.trsm, im.trmm} {
				want := clone(b0)
				if err := f(side, blas.Upper, blas.Trans, blas.NonUnit, m, n, 1.5, a, s, want, m); err != nil {
					t.Fatal(err)
				}
				for _, sp := range [][3]byte{{'U', 'C', 'N'}, {'u', 't', 'n'}, {'u', 'c', 'N'}} {
					got := clone(b0)
					sd := blas.Side(side | 0x20)
					if err := f(sd, blas.Uplo(sp[0]), blas.Transpose(sp[1]), blas.Diag(sp[2]), m, n, 1.5, a, s, got, m); err != nil {
						t.Fatalf("%c %c %c: %v", sd, sp[0], sp[1], err)
					}
					if !same(got, want) {
						t.Fatalf("spelling %c%c%c%c differs from the upper case one", sd, sp[0], sp[1], sp[2])
					}
				}
			}
		}
	}
}

func TestTriFlagSpellings(t *testing.T) {
	each(t, testTriFlagSpellings[float64], testTriFlagSpellings[float32])
}

// netlib's level 3 routines, like Dgemm, are the reference loops without fused
// multiply-add: B(2) + temp*A(2,1) with the pair of TestNetlibIsNotFused is 0
// unfused and 2^-54 fused (arm64's compiler fuses x*y+z by default).
func TestNetlibLevel3IsNotFused(t *testing.T) {
	t.Run("float64", func(t *testing.T) {
		x := 1 + 0x1p-27
		a := []float64{1, x, 0, 1} // lower unit, A(2,1) = x
		b := []float64{x, -(1 + 0x1p-26)}
		if err := impls64[1].trmm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit, 2, 1, 1, a, 2, b, 2); err != nil {
			t.Fatal(err)
		}
		if b[1] != 0 {
			t.Fatalf("netlib.Dtrmm = %v, want the unfused 0", b[1])
		}
	})
	t.Run("float32", func(t *testing.T) {
		x := float32(1 + 0x1p-13)
		a := []float32{1, x, 0, 1}
		b := []float32{x, -float32(1 + 0x1p-12)}
		if err := impls32[1].trmm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit, 2, 1, 1, a, 2, b, 2); err != nil {
			t.Fatal(err)
		}
		if b[1] != 0 {
			t.Fatalf("netlib.Strmm = %v, want the unfused 0", b[1])
		}
	})
}

// syrkShapes are the (n, k) of Syrk and Syr2k: the empty ones, odd sizes, sizes
// spanning several blocks of 4 and 7, and a contraction longer than the kernel's.
var syrkShapes = [][2]int{{0, 3}, {5, 0}, {1, 1}, {6, 5}, {13, 7}, {37, 3}, {9, 19}, {17, 300}}

var syrkBig = [][2]int{{150, 5}}

// syrkAB are the scalar pairs of Syrk and Syr2k: the reference branches on 0 and
// 1 for both.
var syrkAB = [][2]float64{{1, 0}, {1, 1}, {2.5, 0}, {2.5, 1}, {-1, 1.5}, {0.75, -2}, {0, 0}, {0, 3}}

func testSyrkBound[T float](t *testing.T, im impl[T]) {
	for _, bs := range blockSizes(im.name) {
		blas.SetBlockSize(t, bs)
		shapes := syrkShapes
		if bs == 64 {
			shapes = append(append([][2]int{}, syrkShapes...), syrkBig...)
		}
		rng := newRNG(15)
		for si, sh := range shapes {
			n, k := sh[0], sh[1]
			for _, uplo := range uplos {
				for _, tr := range trans2 {
					for ai, ab := range syrkAB {
						if n*n*k > 100000 && ai != 3 { // the oracle's cost
							continue
						}
						pad := 3 * ((si + ai) % 2)
						alpha, beta := T(ab[0]), T(ab[1])
						rows, cols := n, k
						if tr == blas.Trans {
							rows, cols = k, n
						}
						lda, ldc := max(1, rows)+pad, max(1, n)+pad
						a := matrix[T](rng, rows, cols, lda)
						c := cMatrix[T](rng, n, ldc, uplo == blas.Upper)
						a0, c0 := clone(a), clone(c)
						r := syrkExact(byte(tr), n, k, alpha, a, lda, beta, c0, ldc)
						if beta == 0 && n > 0 {
							for j := 0; j < n; j++ { // C must not be read: NaN in the triangle
								lo, hi := 0, j+1
								if uplo == blas.Lower {
									lo, hi = j, n
								}
								for i := lo; i < hi; i++ {
									c[i+j*ldc] = nan[T]()
								}
							}
						}
						if err := im.syrk(uplo, tr, n, k, alpha, a, lda, beta, c, ldc); err != nil {
							t.Fatal(err)
						}
						what := flagName(byte(uplo), byte(tr)) + " " + sprint(shape{n, n, k}, ab, pad) + " nb=" + itoa(bs)
						triWithin(t, what, c, ldc, n, uplo == blas.Upper, r, k+2)
						otherTriangleIntact(t, what, c, c0, n, ldc, uplo == blas.Upper)
						padUntouched(t, what+" C", c, n, ldc)
						if !same(a, a0) {
							t.Fatalf("%s: A was modified", what)
						}
					}
				}
			}
		}
	}
}

func TestSyrkBound(t *testing.T) { each(t, testSyrkBound[float64], testSyrkBound[float32]) }

func testSyr2kBound[T float](t *testing.T, im impl[T]) {
	for _, bs := range blockSizes(im.name) {
		blas.SetBlockSize(t, bs)
		shapes := syrkShapes
		if bs == 64 {
			shapes = append(append([][2]int{}, syrkShapes...), syrkBig...)
		}
		rng := newRNG(16)
		for si, sh := range shapes {
			n, k := sh[0], sh[1]
			for _, uplo := range uplos {
				for _, tr := range trans2 {
					for ai, ab := range syrkAB {
						if n*n*k > 50000 && ai != 3 {
							continue
						}
						pad := 3 * ((si + ai) % 2)
						alpha, beta := T(ab[0]), T(ab[1])
						rows, cols := n, k
						if tr == blas.Trans {
							rows, cols = k, n
						}
						lda, ldb, ldc := max(1, rows)+pad, max(1, rows)+pad+1, max(1, n)+pad
						a, b := matrix[T](rng, rows, cols, lda), matrix[T](rng, rows, cols, ldb)
						c := cMatrix[T](rng, n, ldc, uplo == blas.Upper)
						a0, b0, c0 := clone(a), clone(b), clone(c)
						r := syr2kExact(byte(tr), n, k, alpha, a, lda, b, ldb, beta, c0, ldc)
						if beta == 0 && n > 0 {
							for j := 0; j < n; j++ {
								lo, hi := 0, j+1
								if uplo == blas.Lower {
									lo, hi = j, n
								}
								for i := lo; i < hi; i++ {
									c[i+j*ldc] = nan[T]()
								}
							}
						}
						if err := im.syr2k(uplo, tr, n, k, alpha, a, lda, b, ldb, beta, c, ldc); err != nil {
							t.Fatal(err)
						}
						what := flagName(byte(uplo), byte(tr)) + " " + sprint(shape{n, n, k}, ab, pad) + " nb=" + itoa(bs)
						triWithin(t, what, c, ldc, n, uplo == blas.Upper, r, 2*k+2)
						otherTriangleIntact(t, what, c, c0, n, ldc, uplo == blas.Upper)
						padUntouched(t, what+" C", c, n, ldc)
						if !same(a, a0) || !same(b, b0) {
							t.Fatalf("%s: an input was modified", what)
						}
					}
				}
			}
		}
	}
}

func TestSyr2kBound(t *testing.T) { each(t, testSyr2kBound[float64], testSyr2kBound[float32]) }

// Quick returns and the scalar rules of Syrk and Syr2k: alpha = 0 reads neither A
// nor B, and scales the triangle of C by beta (to zero, without reading it, for
// beta = 0); the quick returns write nothing and check no slice.
func testSyrkScalarRules[T float](t *testing.T, im impl[T]) {
	rng := newRNG(17)
	n, k := 6, 4
	for _, uplo := range uplos {
		for _, beta := range []T{0, 3} {
			c := cMatrix[T](rng, n, n, uplo == blas.Upper)
			want := clone(c)
			for j := 0; j < n; j++ {
				for i := 0; i < n; i++ {
					if i == j || (i < j) == (uplo == blas.Upper) {
						want[i+j*n] = beta * c[i+j*n]
						if beta == 0 {
							c[i+j*n] = nan[T]()
						}
					}
				}
			}
			c2 := clone(c)
			if err := im.syrk(uplo, blas.NoTrans, n, k, 0, nil, n, beta, c, n); err != nil {
				t.Fatal(err)
			}
			if !same(c, want) {
				t.Fatalf("syrk alpha=0 beta=%v: C is not beta*C on the triangle", beta)
			}
			if err := im.syr2k(uplo, blas.Trans, n, k, 0, nil, k, nil, k, beta, c2, n); err != nil {
				t.Fatal(err)
			}
			if !same(c2, want) {
				t.Fatalf("syr2k alpha=0 beta=%v: C is not beta*C on the triangle", beta)
			}
		}
	}
	// k = 0 with beta != 1: C := beta*C, whatever the transpose.
	for _, tr := range trans2 {
		c := cMatrix[T](rng, n, n, true)
		want := clone(c)
		for j := 0; j < n; j++ {
			for i := 0; i <= j; i++ {
				want[i+j*n] = 2 * c[i+j*n]
			}
		}
		c2 := clone(c)
		if err := im.syrk(blas.Upper, tr, n, 0, 5, nil, n, 2, c, n); err != nil {
			t.Fatal(err)
		}
		if err := im.syr2k(blas.Upper, tr, n, 0, 5, nil, n, nil, n, 2, c2, n); err != nil {
			t.Fatal(err)
		}
		if !same(c, want) || !same(c2, want) {
			t.Fatalf("k=0 %c: C != beta*C", tr)
		}
	}
	// Quick returns: n = 0, and alpha = 0 or k = 0 with beta = 1, leave C alone and
	// check no slice.
	c := []T{nan[T]()}
	for _, call := range []func() error{
		func() error { return im.syrk(blas.Upper, blas.NoTrans, 0, 3, 1, nil, 1, 0, c, 1) },
		func() error { return im.syrk(blas.Upper, blas.NoTrans, 5, 3, 0, nil, 5, 1, c, 5) },
		func() error { return im.syrk(blas.Lower, blas.Trans, 5, 0, 1, nil, 1, 1, c, 5) },
		func() error { return im.syr2k(blas.Upper, blas.NoTrans, 0, 3, 1, nil, 1, nil, 1, 0, c, 1) },
		func() error { return im.syr2k(blas.Upper, blas.NoTrans, 5, 3, 0, nil, 5, nil, 5, 1, c, 5) },
		func() error { return im.syr2k(blas.Lower, blas.Trans, 5, 0, 1, nil, 1, nil, 1, 1, c, 5) },
	} {
		if err := call(); err != nil {
			t.Fatal(err)
		}
		if c[0] == c[0] {
			t.Fatal("a quick return wrote C")
		}
	}
}

func TestSyrkScalarRules(t *testing.T) {
	each(t, testSyrkScalarRules[float64], testSyrkScalarRules[float32])
}

// symmShapes are the (m, n) of Symm.
var symmShapes = [][2]int{{0, 5}, {5, 0}, {1, 1}, {6, 5}, {13, 9}, {37, 11}, {9, 37}, {70, 6}, {6, 70}}

func testSymmBound[T float](t *testing.T, im impl[T]) {
	rng := newRNG(18)
	for si, sh := range symmShapes {
		m, n := sh[0], sh[1]
		for _, side := range sides {
			for _, uplo := range uplos {
				for ai, ab := range syrkAB {
					pad := 3 * ((si + ai) % 2)
					alpha, beta := T(ab[0]), T(ab[1])
					s := orderOf(side, m, n)
					lda, ldb, ldc := max(1, s)+pad, max(1, m)+pad+1, max(1, m)+pad
					a := symMatrix[T](rng, s, lda, uplo == blas.Upper)
					b := matrix[T](rng, m, n, ldb)
					c := matrix[T](rng, m, n, ldc)
					a0, b0, c0 := clone(a), clone(b), clone(c)
					r := symmExact(byte(side), byte(uplo), m, n, alpha, a, lda, b, ldb, beta, c0, ldc)
					if beta == 0 && len(c) > 0 {
						c[0] = nan[T]()
					}
					if err := im.symm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc); err != nil {
						t.Fatal(err)
					}
					what := flagName(byte(side), byte(uplo)) + " " + sprint(shape{m, n, s}, ab, pad)
					g := oracle.Gamma[T](s + 2)
					for j := 0; j < n; j++ {
						for i := 0; i < m; i++ {
							within(t, what, c[i+j*ldc], r, i+j*m, g, g)
						}
					}
					padUntouched(t, what+" C", c, m, ldc)
					if !same(a, a0) || !same(b, b0) {
						t.Fatalf("%s: an input was modified", what)
					}
				}
			}
		}
	}
}

func TestSymmBound(t *testing.T) { each(t, testSymmBound[float64], testSymmBound[float32]) }

// Quick returns of Symm, and k = 0 like cases: alpha = 0 reads neither A nor B.
func testSymmScalarRules[T float](t *testing.T, im impl[T]) {
	rng := newRNG(19)
	m, n := 5, 4
	for _, side := range sides {
		for _, beta := range []T{0, 2} {
			c := matrix[T](rng, m, n, m)
			want := make([]T, len(c))
			for i := range c {
				want[i] = beta * c[i]
			}
			if beta == 0 {
				c[0] = nan[T]()
			}
			if err := im.symm(side, blas.Upper, m, n, 0, nil, 5, nil, 5, beta, c, m); err != nil {
				t.Fatal(err)
			}
			if !same(c, want) {
				t.Fatalf("symm %c alpha=0 beta=%v: C != beta*C", side, beta)
			}
		}
	}
	c := []T{nan[T]()}
	for _, call := range []func() error{
		func() error { return im.symm(blas.Left, blas.Upper, 0, 3, 1, nil, 1, nil, 1, 0, c, 1) },
		func() error { return im.symm(blas.Right, blas.Upper, 4, 0, 1, nil, 1, nil, 4, 0, c, 4) },
		func() error { return im.symm(blas.Left, blas.Lower, 5, 3, 0, nil, 5, nil, 5, 1, c, 5) },
	} {
		if err := call(); err != nil {
			t.Fatal(err)
		}
		if c[0] == c[0] {
			t.Fatal("a quick return wrote C")
		}
	}
}

func TestSymmScalarRules(t *testing.T) {
	each(t, testSymmScalarRules[float64], testSymmScalarRules[float32])
}

// The lower case spellings, and ConjTrans, of Syrk, Syr2k and Symm, bit for bit.
func testRank2FlagSpellings[T float](t *testing.T, im impl[T]) {
	rng := newRNG(20)
	n, k := 9, 5
	a, b := matrix[T](rng, k, n, k), matrix[T](rng, k, n, k)
	c0 := cMatrix[T](rng, n, n, true)
	c1, c2, c3, c4 := clone(c0), clone(c0), clone(c0), clone(c0)
	if err := im.syrk(blas.Upper, blas.Trans, n, k, 1.5, a, k, 0.5, c1, n); err != nil {
		t.Fatal(err)
	}
	if err := im.syrk('u', 'c', n, k, 1.5, a, k, 0.5, c2, n); err != nil || !same(c1, c2) {
		t.Fatalf("syrk 'u','c' differs from 'U','T' (%v)", err)
	}
	if err := im.syr2k(blas.Upper, blas.Trans, n, k, 1.5, a, k, b, k, 0.5, c3, n); err != nil {
		t.Fatal(err)
	}
	if err := im.syr2k('u', 't', n, k, 1.5, a, k, b, k, 0.5, c4, n); err != nil || !same(c3, c4) {
		t.Fatalf("syr2k 'u','t' differs from 'U','T' (%v)", err)
	}
	m, w := 7, 6
	sa := symMatrix[T](rng, m, m, false)
	sb, sc := matrix[T](rng, m, w, m), matrix[T](rng, m, w, m)
	sc2 := clone(sc)
	if err := im.symm(blas.Left, blas.Lower, m, w, 1.5, sa, m, sb, m, 0.5, sc, m); err != nil {
		t.Fatal(err)
	}
	if err := im.symm('l', 'l', m, w, 1.5, sa, m, sb, m, 0.5, sc2, m); err != nil || !same(sc, sc2) {
		t.Fatalf("symm 'l','l' differs from 'L','L' (%v)", err)
	}
}

func TestRank2FlagSpellings(t *testing.T) {
	each(t, testRank2FlagSpellings[float64], testRank2FlagSpellings[float32])
}

// The blocked routines are not just a function of the block size: for the same
// data and any block size the result is inside the bound, which the tests above
// check, and for integer data below 2^11, where every product and sum is exact,
// it is the same bit for bit as netlib's.
func testLevel3ExactIntegers[T float](t *testing.T, im impl[T]) {
	rng := newRNG(21)
	ints := func(n int) []T {
		v := make([]T, n)
		for i := range v {
			v[i] = T(rng.IntN(5) - 2)
		}
		return v
	}
	for _, bs := range blockSizes(im.name) {
		blas.SetBlockSize(t, bs)
		m, n := 17, 14
		a, b := ints(m*m), ints(m*n)
		nl := netlibImpl[T]()
		// Unit lower triangular with entries in -1..1 (the solution grows at most
		// like 2^m, exact in float32): the solve is exact in any order.
		unit := clone(a)
		for i := range unit {
			unit[i] = T(int(unit[i]) % 2)
		}
		x, want := clone(b), clone(b)
		if err := nl.trsm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit, m, n, 1, unit, m, want, m); err != nil {
			t.Fatal(err)
		}
		if err := im.trsm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit, m, n, 1, unit, m, x, m); err != nil {
			t.Fatal(err)
		}
		if !same(x, want) {
			t.Fatalf("nb=%d: unit trsm differs from netlib's", bs)
		}
		// Products with integer matrices: every sum is exact.
		for _, side := range sides {
			rows, cols := m, n
			if side == blas.Right {
				rows, cols = n, m
			}
			x, want = clone(b[:rows*cols]), clone(b[:rows*cols])
			if err := nl.trmm(side, blas.Upper, blas.Trans, blas.NonUnit, rows, cols, 2, a, m, want, rows); err != nil {
				t.Fatal(err)
			}
			if err := im.trmm(side, blas.Upper, blas.Trans, blas.NonUnit, rows, cols, 2, a, m, x, rows); err != nil {
				t.Fatal(err)
			}
			if !same(x, want) {
				t.Fatalf("nb=%d: integer trmm %c differs from netlib's", bs, side)
			}
		}
		c, c2 := ints(m*m), ints(m*m)
		cc := clone(c)
		if err := nl.syrk(blas.Lower, blas.NoTrans, m, n, 2, b, m, 3, c, m); err != nil {
			t.Fatal(err)
		}
		if err := im.syrk(blas.Lower, blas.NoTrans, m, n, 2, b, m, 3, cc, m); err != nil {
			t.Fatal(err)
		}
		if !same(c, cc) {
			t.Fatalf("nb=%d: integer syrk differs from netlib's", bs)
		}
		c, cc = clone(c2), clone(c2)
		if err := nl.syr2k(blas.Upper, blas.Trans, n, m, 2, b, m, b, m, 3, c, m); err != nil {
			t.Fatal(err)
		}
		if err := im.syr2k(blas.Upper, blas.Trans, n, m, 2, b, m, b, m, 3, cc, m); err != nil {
			t.Fatal(err)
		}
		if !same(c, cc) {
			t.Fatalf("nb=%d: integer syr2k differs from netlib's", bs)
		}
		c, cc = clone(c2), clone(c2)
		if err := nl.symm(blas.Right, blas.Upper, m, m, 2, a, m, c2, m, 3, c, m); err != nil {
			t.Fatal(err)
		}
		if err := im.symm(blas.Right, blas.Upper, m, m, 2, a, m, c2, m, 3, cc, m); err != nil {
			t.Fatal(err)
		}
		if !same(c, cc) {
			t.Fatalf("nb=%d: integer symm differs from netlib's", bs)
		}
	}
}

func TestLevel3ExactIntegers(t *testing.T) {
	each(t, testLevel3ExactIntegers[float64], testLevel3ExactIntegers[float32])
}

// netlibImpl is the netlib implementation of the element type T.
func netlibImpl[T float]() impl[T] {
	var z T
	if _, ok := any(z).(float32); ok {
		return any(impls32[1]).(impl[T])
	}
	return any(impls64[1]).(impl[T])
}
