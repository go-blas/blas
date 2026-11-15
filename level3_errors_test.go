package blas_test

import (
	"testing"

	"github.com/go-blas/blas"
)

// The argument errors of the level 3 routines: every parameter of the reference's
// tests (numbered as xerbla numbers them), every slice that is too short, sizes
// that overflow an int, the order in which the first bad parameter wins, and the
// calls that look wrong and are not.

// testTriErrors runs the cases of Trsm and Trmm, which have one signature and one
// set of tests.
func testTriErrors[T float](t *testing.T, routine string, f func(blas.Side, blas.Uplo, blas.Transpose, blas.Diag, int, int, T, []T, int, []T, int) error) {
	v := make([]T, 100)
	L, R, U, Lo, N, T_, NU, UN := blas.Left, blas.Right, blas.Upper, blas.Lower, blas.NoTrans, blas.Trans, blas.NonUnit, blas.Unit
	g := func(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, a []T, lda int, b []T, ldb int) func() error {
		return func() error { return f(side, uplo, tA, diag, m, n, 1, a, lda, b, ldb) }
	}
	runErrCases(t, routine, []errCase{
		{"side", 1, g('X', U, N, NU, 3, 3, v, 3, v, 3)},
		{"uplo", 2, g(L, 'x', N, NU, 3, 3, v, 3, v, 3)},
		{"trans", 3, g(L, U, 'Q', NU, 3, 3, v, 3, v, 3)},
		{"diag", 4, g(L, U, N, 'Z', 3, 3, v, 3, v, 3)},
		{"m", 5, g(L, U, N, NU, -1, 3, v, 3, v, 3)},
		{"n", 6, g(L, U, N, NU, 3, -1, v, 3, v, 3)},
		{"lda left", 9, g(L, U, N, NU, 3, 5, v, 2, v, 3)},
		{"lda right", 9, g(R, U, N, NU, 3, 5, v, 4, v, 3)},
		{"lda trans", 9, g(L, Lo, T_, UN, 4, 2, v, 3, v, 4)},
		{"lda zero", 9, g(L, U, N, NU, 0, 3, v, 0, v, 1)},
		{"ldb", 11, g(L, U, N, NU, 3, 3, v, 3, v, 2)},
		{"ldb zero", 11, g(R, U, N, NU, 0, 3, v, 3, v, 0)},
		{"a short left", 8, g(L, U, N, NU, 3, 3, v[:8], 3, v, 3)},
		{"a short right", 8, g(R, Lo, T_, UN, 2, 4, v[:15], 4, v, 2)},
		{"a short padded", 8, g(L, U, N, NU, 3, 3, v[:12], 5, v, 3)},
		{"b short", 10, g(L, U, N, NU, 3, 3, v, 3, v[:8], 3)},
		{"b short padded", 10, g(R, U, N, NU, 3, 3, v, 3, v[:12], 5)},
		{"a size overflows int", 8, g(L, U, N, NU, 3, 3, v, maxInt, v, 3)},
		{"b size overflows int", 10, g(L, U, N, NU, 3, 3, v, 3, v, maxInt)},
		{"a size overflows by one", 8, g(L, U, N, NU, 2, 1, v, maxInt, v, 2)},
	})
	runErrCases(t, routine, []errCase{
		{"side before uplo", 1, g('X', 'X', 'X', 'X', -1, -1, nil, 0, nil, 0)},
		{"uplo before trans", 2, g(L, 'X', 'X', 'X', -1, -1, nil, 0, nil, 0)},
		{"trans before diag", 3, g(L, U, 'X', 'X', -1, -1, nil, 0, nil, 0)},
		{"diag before m", 4, g(L, U, N, 'X', -1, -1, nil, 0, nil, 0)},
		{"m before n", 5, g(L, U, N, NU, -1, -1, nil, 0, nil, 0)},
		{"n before lda", 6, g(L, U, N, NU, 3, -1, nil, 0, nil, 0)},
		{"lda before ldb", 9, g(L, U, N, NU, 3, 3, nil, 1, nil, 1)},
		{"ldb before slices", 11, g(L, U, N, NU, 3, 3, nil, 3, nil, 1)},
		{"a before b", 8, g(L, U, N, NU, 3, 3, nil, 3, nil, 3)},
	})
	// Not errors: A is not read with alpha = 0, and an empty product reads nothing.
	for i, call := range []func() error{
		func() error { return f(L, U, N, NU, 3, 3, 0, nil, 3, v, 3) },
		func() error { return f(R, U, N, NU, 3, 3, 1, v, 3, v[:9], 3) },
		func() error { return f(L, U, N, NU, 0, 3, 1, nil, 1, nil, 1) },
		func() error { return f(R, U, N, NU, 3, 0, 1, nil, 1, nil, 3) },
	} {
		if err := call(); err != nil {
			t.Errorf("%s valid call %d: %v", routine, i, err)
		}
	}
}

func TestTrsmErrors(t *testing.T) {
	each(t, func(t *testing.T, im impl[float64]) { testTriErrors(t, "DTRSM", im.trsm) },
		func(t *testing.T, im impl[float32]) { testTriErrors(t, "STRSM", im.trsm) })
}

func TestTrmmErrors(t *testing.T) {
	each(t, func(t *testing.T, im impl[float64]) { testTriErrors(t, "DTRMM", im.trmm) },
		func(t *testing.T, im impl[float32]) { testTriErrors(t, "STRMM", im.trmm) })
}

func testSyrkErrors[T float](t *testing.T, im impl[T], routine string) {
	v := make([]T, 100)
	U, Lo, N, T_ := blas.Upper, blas.Lower, blas.NoTrans, blas.Trans
	g := func(uplo blas.Uplo, tr blas.Transpose, n, k int, a []T, lda int, c []T, ldc int) func() error {
		return func() error { return im.syrk(uplo, tr, n, k, 1, a, lda, 1, c, ldc) }
	}
	runErrCases(t, routine, []errCase{
		{"uplo", 1, g('X', N, 3, 3, v, 3, v, 3)},
		{"trans", 2, g(U, 'x', 3, 3, v, 3, v, 3)},
		{"n", 3, g(U, N, -1, 3, v, 3, v, 3)},
		{"k", 4, g(U, N, 3, -1, v, 3, v, 3)},
		{"lda notrans", 7, g(U, N, 5, 3, v, 4, v, 5)},
		{"lda trans", 7, g(U, T_, 3, 5, v, 4, v, 3)},
		{"lda zero", 7, g(Lo, N, 0, 3, v, 0, v, 1)},
		{"ldc", 10, g(U, N, 3, 3, v, 3, v, 2)},
		{"a short", 6, g(U, N, 3, 3, v[:8], 3, v, 3)},
		{"a short trans", 6, g(U, T_, 3, 4, v[:11], 4, v, 3)},
		{"a short padded", 6, g(Lo, N, 3, 3, v[:12], 5, v, 3)},
		{"c short", 9, g(U, N, 3, 3, v, 3, v[:8], 3)},
		{"c short padded", 9, g(Lo, T_, 3, 3, v, 3, v[:12], 5)},
		{"a size overflows int", 6, g(U, N, 3, 3, v, maxInt, v, 3)},
		{"c size overflows int", 9, g(U, N, 3, 3, v, 3, v, maxInt)},
		{"a size overflows by one", 6, g(U, N, 1, 2, v, maxInt, v, 1)},
	})
	runErrCases(t, routine, []errCase{
		{"uplo before trans", 1, g('X', 'X', -1, -1, nil, 0, nil, 0)},
		{"trans before n", 2, g(U, 'X', -1, -1, nil, 0, nil, 0)},
		{"n before k", 3, g(U, N, -1, -1, nil, 0, nil, 0)},
		{"k before lda", 4, g(U, N, 3, -1, nil, 0, nil, 0)},
		{"lda before ldc", 7, g(U, N, 3, 3, nil, 1, nil, 1)},
		{"ldc before slices", 10, g(U, N, 3, 3, nil, 3, nil, 1)},
		{"a before c", 6, g(U, N, 3, 3, nil, 3, nil, 3)},
	})
	for i, call := range []func() error{
		func() error { return im.syrk(U, N, 3, 3, 0, nil, 3, 1.5, v, 3) }, // alpha = 0 reads no A
		func() error { return im.syrk(U, N, 3, 3, 1, v, 3, 1, v[:9], 3) },
		func() error { return im.syrk(U, N, 0, 3, 1, nil, 1, 1, nil, 1) },
	} {
		if err := call(); err != nil {
			t.Errorf("%s valid call %d: %v", routine, i, err)
		}
	}
}

func TestSyrkErrors(t *testing.T) {
	each(t, func(t *testing.T, im impl[float64]) { testSyrkErrors(t, im, "DSYRK") },
		func(t *testing.T, im impl[float32]) { testSyrkErrors(t, im, "SSYRK") })
}

func testSyr2kErrors[T float](t *testing.T, im impl[T], routine string) {
	v := make([]T, 100)
	U, Lo, N, T_ := blas.Upper, blas.Lower, blas.NoTrans, blas.Trans
	g := func(uplo blas.Uplo, tr blas.Transpose, n, k int, a []T, lda int, b []T, ldb int, c []T, ldc int) func() error {
		return func() error { return im.syr2k(uplo, tr, n, k, 1, a, lda, b, ldb, 1, c, ldc) }
	}
	runErrCases(t, routine, []errCase{
		{"uplo", 1, g('X', N, 3, 3, v, 3, v, 3, v, 3)},
		{"trans", 2, g(U, 'x', 3, 3, v, 3, v, 3, v, 3)},
		{"n", 3, g(U, N, -1, 3, v, 3, v, 3, v, 3)},
		{"k", 4, g(U, N, 3, -1, v, 3, v, 3, v, 3)},
		{"lda notrans", 7, g(U, N, 5, 3, v, 4, v, 5, v, 5)},
		{"lda trans", 7, g(U, T_, 3, 5, v, 4, v, 5, v, 3)},
		{"lda zero", 7, g(Lo, N, 0, 3, v, 0, v, 1, v, 1)},
		{"ldb notrans", 9, g(U, N, 5, 3, v, 5, v, 4, v, 5)},
		{"ldb trans", 9, g(U, T_, 3, 5, v, 5, v, 4, v, 3)},
		{"ldc", 12, g(U, N, 3, 3, v, 3, v, 3, v, 2)},
		{"a short", 6, g(U, N, 3, 3, v[:8], 3, v, 3, v, 3)},
		{"a short trans", 6, g(U, T_, 3, 4, v[:11], 4, v, 4, v, 3)},
		{"b short", 8, g(U, N, 3, 3, v, 3, v[:8], 3, v, 3)},
		{"b short padded", 8, g(Lo, T_, 3, 3, v, 3, v[:12], 5, v, 3)},
		{"c short", 11, g(U, N, 3, 3, v, 3, v, 3, v[:8], 3)},
		{"c short padded", 11, g(Lo, N, 3, 3, v, 3, v, 3, v[:12], 5)},
		{"a size overflows int", 6, g(U, N, 3, 3, v, maxInt, v, 3, v, 3)},
		{"b size overflows int", 8, g(U, N, 3, 3, v, 3, v, maxInt, v, 3)},
		{"c size overflows int", 11, g(U, N, 3, 3, v, 3, v, 3, v, maxInt)},
		{"a size overflows by one", 6, g(U, N, 1, 2, v, maxInt, v, 1, v, 1)},
	})
	runErrCases(t, routine, []errCase{
		{"uplo before trans", 1, g('X', 'X', -1, -1, nil, 0, nil, 0, nil, 0)},
		{"trans before n", 2, g(U, 'X', -1, -1, nil, 0, nil, 0, nil, 0)},
		{"n before k", 3, g(U, N, -1, -1, nil, 0, nil, 0, nil, 0)},
		{"k before lda", 4, g(U, N, 3, -1, nil, 0, nil, 0, nil, 0)},
		{"lda before ldb", 7, g(U, N, 3, 3, nil, 1, nil, 1, nil, 1)},
		{"ldb before ldc", 9, g(U, N, 3, 3, nil, 3, nil, 1, nil, 1)},
		{"ldc before slices", 12, g(U, N, 3, 3, nil, 3, nil, 3, nil, 1)},
		{"a before b before c", 6, g(U, N, 3, 3, nil, 3, nil, 3, nil, 3)},
	})
	for i, call := range []func() error{
		func() error { return im.syr2k(U, N, 3, 3, 0, nil, 3, nil, 3, 1.5, v, 3) }, // alpha = 0 reads neither A nor B
		func() error { return im.syr2k(U, N, 3, 3, 1, v, 3, v, 3, 1, v[:9], 3) },
		func() error { return im.syr2k(U, N, 0, 3, 1, nil, 1, nil, 1, 1, nil, 1) },
	} {
		if err := call(); err != nil {
			t.Errorf("%s valid call %d: %v", routine, i, err)
		}
	}
}

func TestSyr2kErrors(t *testing.T) {
	each(t, func(t *testing.T, im impl[float64]) { testSyr2kErrors(t, im, "DSYR2K") },
		func(t *testing.T, im impl[float32]) { testSyr2kErrors(t, im, "SSYR2K") })
}

func testSymmErrors[T float](t *testing.T, im impl[T], routine string) {
	v := make([]T, 100)
	L, R, U := blas.Left, blas.Right, blas.Upper
	g := func(side blas.Side, uplo blas.Uplo, m, n int, a []T, lda int, b []T, ldb int, c []T, ldc int) func() error {
		return func() error { return im.symm(side, uplo, m, n, 1, a, lda, b, ldb, 1, c, ldc) }
	}
	runErrCases(t, routine, []errCase{
		{"side", 1, g('X', U, 3, 3, v, 3, v, 3, v, 3)},
		{"uplo", 2, g(L, 'x', 3, 3, v, 3, v, 3, v, 3)},
		{"m", 3, g(L, U, -1, 3, v, 3, v, 3, v, 3)},
		{"n", 4, g(L, U, 3, -1, v, 3, v, 3, v, 3)},
		{"lda left", 7, g(L, U, 3, 5, v, 2, v, 3, v, 3)},
		{"lda right", 7, g(R, U, 3, 5, v, 4, v, 3, v, 3)},
		{"lda zero", 7, g(L, U, 0, 3, v, 0, v, 1, v, 1)},
		{"ldb", 9, g(L, U, 3, 3, v, 3, v, 2, v, 3)},
		{"ldc", 12, g(L, U, 3, 3, v, 3, v, 3, v, 2)},
		{"a short left", 6, g(L, U, 3, 3, v[:8], 3, v, 3, v, 3)},
		{"a short right", 6, g(R, U, 2, 4, v[:15], 4, v, 2, v, 2)},
		{"a short padded", 6, g(L, U, 3, 3, v[:12], 5, v, 3, v, 3)},
		{"b short", 8, g(L, U, 3, 3, v, 3, v[:8], 3, v, 3)},
		{"c short", 11, g(L, U, 3, 3, v, 3, v, 3, v[:8], 3)},
		{"c short padded", 11, g(R, U, 3, 3, v, 3, v, 3, v[:12], 5)},
		{"a size overflows int", 6, g(L, U, 3, 3, v, maxInt, v, 3, v, 3)},
		{"b size overflows int", 8, g(L, U, 3, 3, v, 3, v, maxInt, v, 3)},
		{"c size overflows int", 11, g(L, U, 3, 3, v, 3, v, 3, v, maxInt)},
		{"a size overflows by one", 6, g(L, U, 2, 1, v, maxInt, v, 2, v, 2)},
	})
	runErrCases(t, routine, []errCase{
		{"side before uplo", 1, g('X', 'X', -1, -1, nil, 0, nil, 0, nil, 0)},
		{"uplo before m", 2, g(L, 'X', -1, -1, nil, 0, nil, 0, nil, 0)},
		{"m before n", 3, g(L, U, -1, -1, nil, 0, nil, 0, nil, 0)},
		{"n before lda", 4, g(L, U, 3, -1, nil, 0, nil, 0, nil, 0)},
		{"lda before ldb", 7, g(L, U, 3, 3, nil, 1, nil, 1, nil, 1)},
		{"ldb before ldc", 9, g(L, U, 3, 3, nil, 3, nil, 1, nil, 1)},
		{"ldc before slices", 12, g(L, U, 3, 3, nil, 3, nil, 3, nil, 1)},
		{"a before b before c", 6, g(L, U, 3, 3, nil, 3, nil, 3, nil, 3)},
	})
	for i, call := range []func() error{
		func() error { return im.symm(L, U, 3, 3, 0, nil, 3, nil, 3, 1.5, v, 3) }, // alpha = 0 reads neither A nor B
		func() error { return im.symm(R, U, 3, 3, 1, v, 3, v, 3, 1, v[:9], 3) },
		func() error { return im.symm(L, U, 0, 3, 1, nil, 1, nil, 1, 1, nil, 1) },
	} {
		if err := call(); err != nil {
			t.Errorf("%s valid call %d: %v", routine, i, err)
		}
	}
}

func TestSymmErrors(t *testing.T) {
	each(t, func(t *testing.T, im impl[float64]) { testSymmErrors(t, im, "DSYMM") },
		func(t *testing.T, im impl[float32]) { testSymmErrors(t, im, "SSYMM") })
}
