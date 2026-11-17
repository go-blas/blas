package lapack_test

import (
	"errors"
	"testing"

	"github.com/go-blas/blas"
)

// wantArg asserts err is a *blas.ArgError for param p and that its Info is -p,
// and, for the factorizations, that info is -p as well.
func wantArg(t *testing.T, what string, err error, p int, info *int) {
	t.Helper()
	var ae *blas.ArgError
	if !errors.As(err, &ae) || ae.Param != p || ae.Info() != -p {
		t.Fatalf("%s: err %v, want parameter %d", what, err, p)
	}
	if info != nil && *info != -p {
		t.Fatalf("%s: info %d, want %d", what, *info, -p)
	}
}

func wantNone(t *testing.T, what string, err error, info int) {
	t.Helper()
	if err != nil || info != 0 {
		t.Fatalf("%s: info %d, err %v, want none", what, info, err)
	}
}

func TestArgErrors(t *testing.T) {
	each(t, argErrT[float64], argErrT[float32])
}

func argErrT[T float](t *testing.T, im impl[T]) {
	a9, a8 := make([]T, 9), make([]T, 8)
	b9, b8 := make([]T, 9), make([]T, 8)
	p3, p2 := []int{1, 2, 3}, []int{1, 2}

	gf := func(what string, p int, m, n int, a []T, lda int, ipiv []int) {
		t.Helper()
		info, err := im.getrf(m, n, a, lda, ipiv)
		wantArg(t, "getrf "+what, err, p, &info)
	}
	gf("m", 1, -1, 3, a9, 3, p3)
	gf("n", 2, 3, -1, a9, 3, p3)
	gf("lda", 4, 3, 3, a9, 2, p3)
	gf("lda zero", 4, 0, 3, a9, 0, p3)
	gf("a short", 3, 3, 3, a8, 3, p3)
	gf("ipiv short", 5, 3, 3, a9, 3, p2)
	// Quick returns come first: nothing else is looked at, not even a nil slice.
	for _, c := range [][2]int{{0, 3}, {3, 0}, {0, 0}} {
		info, err := im.getrf(c[0], c[1], nil, max(1, c[0]), nil)
		wantNone(t, "getrf quick", err, info)
	}

	gs := func(what string, p int, tr blas.Transpose, n, nrhs int, a []T, lda int, ipiv []int, b []T, ldb int) {
		t.Helper()
		wantArg(t, "getrs "+what, im.getrs(tr, n, nrhs, a, lda, ipiv, b, ldb), p, nil)
	}
	gs("trans", 1, 'X', 3, 3, a9, 3, p3, b9, 3)
	gs("n", 2, 'N', -1, 3, a9, 3, p3, b9, 3)
	gs("nrhs", 3, 'N', 3, -1, a9, 3, p3, b9, 3)
	gs("a short", 4, 'N', 3, 3, a8, 3, p3, b9, 3)
	gs("lda", 5, 'N', 3, 3, a9, 2, p3, b9, 3)
	gs("ipiv short", 6, 'N', 3, 3, a9, 3, p2, b9, 3)
	gs("pivot zero", 6, 'N', 3, 3, a9, 3, []int{1, 0, 3}, b9, 3)
	gs("pivot above n", 6, 'T', 3, 3, a9, 3, []int{1, 2, 4}, b9, 3)
	gs("pivot negative", 6, 'T', 3, 3, a9, 3, []int{-1, 2, 3}, b9, 3)
	gs("b short", 7, 'N', 3, 3, a9, 3, p3, b8, 3)
	gs("ldb", 8, 'N', 3, 3, a9, 3, p3, b9, 2)
	// Quick returns before the slice checks; transposes in lower case and C are valid.
	for _, tr := range []blas.Transpose{'n', 't', 'c', 'C'} {
		if err := im.getrs(tr, 0, 3, nil, 1, nil, nil, 1); err != nil {
			t.Fatal(err)
		}
		if err := im.getrs(tr, 3, 0, nil, 3, nil, nil, 3); err != nil {
			t.Fatal(err)
		}
	}

	pf := func(what string, p int, uplo blas.Uplo, n int, a []T, lda int) {
		t.Helper()
		info, err := im.potrf(uplo, n, a, lda)
		wantArg(t, "potrf "+what, err, p, &info)
	}
	pf("uplo", 1, 'X', 3, a9, 3)
	pf("n", 2, 'U', -1, a9, 3)
	pf("a short", 3, 'L', 3, a8, 3)
	pf("lda", 4, 'U', 3, a9, 2)
	info, err := im.potrf('U', 0, nil, 1)
	wantNone(t, "potrf quick", err, info)

	ps := func(what string, p int, uplo blas.Uplo, n, nrhs int, a []T, lda int, b []T, ldb int) {
		t.Helper()
		wantArg(t, "potrs "+what, im.potrs(uplo, n, nrhs, a, lda, b, ldb), p, nil)
	}
	ps("uplo", 1, 'X', 3, 3, a9, 3, b9, 3)
	ps("n", 2, 'U', -1, 3, a9, 3, b9, 3)
	ps("nrhs", 3, 'U', 3, -1, a9, 3, b9, 3)
	ps("a short", 4, 'L', 3, 3, a8, 3, b9, 3)
	ps("lda", 5, 'U', 3, 3, a9, 2, b9, 3)
	ps("b short", 6, 'U', 3, 3, a9, 3, b8, 3)
	ps("ldb", 7, 'L', 3, 3, a9, 3, b9, 2)
	for _, up := range []blas.Uplo{'u', 'l'} {
		if err := im.potrs(up, 0, 3, nil, 1, nil, 1); err != nil {
			t.Fatal(err)
		}
		if err := im.potrs(up, 3, 0, nil, 3, nil, 3); err != nil {
			t.Fatal(err)
		}
	}
}

// Lower case flags are the same as upper case, and the routine name in the error
// is the reference's.
func TestFlagsAndNames(t *testing.T) {
	each(t, flagsT[float64], flagsT[float32])
}

func flagsT[T float](t *testing.T, im impl[T]) {
	for _, tr := range []byte{'n', 't', 'c'} {
		n := 5
		f := general[T](newRNG(3), n, n, n)
		ipiv := make([]int, n)
		if _, err := im.getrf(n, n, f, n, ipiv); err != nil {
			t.Fatal(err)
		}
		b := general[T](newRNG(4), n, 2, n)
		b2 := clone(b)
		if err := im.getrs(blas.Transpose(tr), n, 2, f, n, ipiv, b, n); err != nil {
			t.Fatal(err)
		}
		if err := im.getrs(blas.Transpose(upper(tr)), n, 2, f, n, ipiv, b2, n); err != nil {
			t.Fatal(err)
		}
		if !same(b, b2) {
			t.Fatalf("flag %c differs from its upper case", tr)
		}
	}
	_, err := im.getrf(-1, 1, nil, 1, nil)
	var ae *blas.ArgError
	if !errors.As(err, &ae) || (ae.Routine != "DGETRF" && ae.Routine != "SGETRF") {
		t.Fatalf("routine name: %v", err)
	}
	for _, up := range []blas.Uplo{'u', 'l'} {
		a := spd[T](newRNG(5), 6, 6, up == 'u')
		a2 := clone(a)
		if _, err := im.potrf(up, 6, a, 6); err != nil {
			t.Fatal(err)
		}
		if _, err := im.potrf(blas.Uplo(upper(byte(up))), 6, a2, 6); err != nil {
			t.Fatal(err)
		}
		if !same(a, a2) {
			t.Fatalf("uplo %c differs from its upper case", up)
		}
	}
}

// lapack and netlib agree on every argument error and on INFO, over a grid that
// includes invalid dimensions and short slices. Both call the same internal/check,
// so this compares two callers of one checker and cannot see a wrong number in it:
// TestArgErrors, with its table of numbers from the reference signatures, is the
// independent check, and this test the one that the wrappers order their checks
// and quick returns alike.
func TestAgreeOnArguments(t *testing.T) {
	defer noPanic(t)
	for _, m := range []int{-1, 0, 1, 4} {
		for _, n := range []int{-1, 0, 1, 4} {
			for _, lda := range []int{0, 1, 4} {
				for _, la := range []int{0, 5, 16} {
					for _, lp := range []int{0, 2, 4} {
						a1, a2 := make([]float64, la), make([]float64, la)
						i1, e1 := impls64[0].getrf(m, n, a1, lda, make([]int, lp))
						i2, e2 := impls64[1].getrf(m, n, a2, lda, make([]int, lp))
						if i1 != i2 || (e1 == nil) != (e2 == nil) || (e1 != nil && e1.Error() != e2.Error()) {
							t.Fatalf("getrf %d %d %d %d %d: (%d, %v) vs (%d, %v)", m, n, lda, la, lp, i1, e1, i2, e2)
						}
						e3 := impls64[0].potrs('L', m, n, a1, lda, a1, lda)
						e4 := impls64[1].potrs('L', m, n, a2, lda, a2, lda)
						if (e3 == nil) != (e4 == nil) || (e3 != nil && e3.Error() != e4.Error()) {
							t.Fatalf("potrs %d %d %d %d: %v vs %v", m, n, lda, la, e3, e4)
						}
					}
				}
			}
		}
	}
}
