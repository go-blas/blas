package netlib

import (
	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/check"
	"github.com/go-blas/blas/internal/ref"
)

// Dtrsm is DTRSM (dtrsm.f): B := alpha*inv(op(A))*B or B := alpha*B*inv(op(A)),
// A triangular, column-major. See blas.Dtrsm for the arguments and the errors.
func Dtrsm(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) error {
	return trsm("DTRSM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb)
}

// Strsm is STRSM (strsm.f).
func Strsm(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) error {
	return trsm("STRSM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb)
}

func trsm[T ref.Float](name string, side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int) error {
	side, uplo, tA, diag = blas.Side(check.Upper(byte(side))), blas.Uplo(check.Upper(byte(uplo))), blas.Transpose(check.Upper(byte(tA))), blas.Diag(check.Upper(byte(diag))) // LSAME ignores case
	if p := check.Tr(byte(side), byte(uplo), byte(tA), byte(diag), m, n, lda, ldb); p != 0 {                                                                                 // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 { // guard
		return nil
	}
	if p := check.TrLens(byte(side), m, n, lda, ldb, len(a), len(b), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Trsm(side == blas.Left, uplo == blas.Upper, tA == blas.NoTrans, diag == blas.NonUnit, m, n, alpha, a, lda, b, ldb)
	return nil
}

// Dtrmm is DTRMM (dtrmm.f): B := alpha*op(A)*B or B := alpha*B*op(A), A
// triangular, column-major. See blas.Dtrmm.
func Dtrmm(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) error {
	return trmm("DTRMM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb)
}

// Strmm is STRMM (strmm.f).
func Strmm(side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) error {
	return trmm("STRMM", side, uplo, tA, diag, m, n, alpha, a, lda, b, ldb)
}

func trmm[T ref.Float](name string, side blas.Side, uplo blas.Uplo, tA blas.Transpose, diag blas.Diag, m, n int, alpha T, a []T, lda int, b []T, ldb int) error {
	side, uplo, tA, diag = blas.Side(check.Upper(byte(side))), blas.Uplo(check.Upper(byte(uplo))), blas.Transpose(check.Upper(byte(tA))), blas.Diag(check.Upper(byte(diag))) // LSAME ignores case
	if p := check.Tr(byte(side), byte(uplo), byte(tA), byte(diag), m, n, lda, ldb); p != 0 {                                                                                 // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 { // guard
		return nil
	}
	if p := check.TrLens(byte(side), m, n, lda, ldb, len(a), len(b), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Trmm(side == blas.Left, uplo == blas.Upper, tA == blas.NoTrans, diag == blas.NonUnit, m, n, alpha, a, lda, b, ldb)
	return nil
}

// Dsyrk is DSYRK (dsyrk.f): C := alpha*A*A**T + beta*C or C := alpha*A**T*A +
// beta*C on one triangle of C. See blas.Dsyrk.
func Dsyrk(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) error {
	return syrk("DSYRK", uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

// Ssyrk is SSYRK (ssyrk.f).
func Ssyrk(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) error {
	return syrk("SSYRK", uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func syrk[T ref.Float](name string, uplo blas.Uplo, trans blas.Transpose, n, k int, alpha T, a []T, lda int, beta T, c []T, ldc int) error {
	uplo, trans = blas.Uplo(check.Upper(byte(uplo))), blas.Transpose(check.Upper(byte(trans))) // LSAME ignores case
	if p := check.Syrk(byte(uplo), byte(trans), n, k, lda, ldc); p != 0 {                      // guard
		return argErr(name, p)
	}
	if n == 0 || ((alpha == 0 || k == 0) && beta == 1) { // guard
		return nil
	}
	if p := check.SyrkLens(byte(trans), n, k, lda, ldc, len(a), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Syrk(uplo == blas.Upper, trans == blas.NoTrans, n, k, alpha, a, lda, beta, c, ldc)
	return nil
}

// Dsyr2k is DSYR2K (dsyr2k.f): C := alpha*A*B**T + alpha*B*A**T + beta*C or C :=
// alpha*A**T*B + alpha*B**T*A + beta*C on one triangle of C. See blas.Dsyr2k.
func Dsyr2k(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) error {
	return syr2k("DSYR2K", uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Ssyr2k is SSYR2K (ssyr2k.f).
func Ssyr2k(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	return syr2k("SSYR2K", uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func syr2k[T ref.Float](name string, uplo blas.Uplo, trans blas.Transpose, n, k int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) error {
	uplo, trans = blas.Uplo(check.Upper(byte(uplo))), blas.Transpose(check.Upper(byte(trans))) // LSAME ignores case
	if p := check.Syr2k(byte(uplo), byte(trans), n, k, lda, ldb, ldc); p != 0 {                // guard
		return argErr(name, p)
	}
	if n == 0 || ((alpha == 0 || k == 0) && beta == 1) { // guard
		return nil
	}
	if p := check.Syr2kLens(byte(trans), n, k, lda, ldb, ldc, len(a), len(b), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Syr2k(uplo == blas.Upper, trans == blas.NoTrans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	return nil
}

// Dsymm is DSYMM (dsymm.f): C := alpha*A*B + beta*C or C := alpha*B*A + beta*C, A
// symmetric, column-major. See blas.Dsymm.
func Dsymm(side blas.Side, uplo blas.Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) error {
	return symm("DSYMM", side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Ssymm is SSYMM (ssymm.f).
func Ssymm(side blas.Side, uplo blas.Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	return symm("SSYMM", side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

func symm[T ref.Float](name string, side blas.Side, uplo blas.Uplo, m, n int, alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) error {
	side, uplo = blas.Side(check.Upper(byte(side))), blas.Uplo(check.Upper(byte(uplo))) // LSAME ignores case
	if p := check.Symm(byte(side), byte(uplo), m, n, lda, ldb, ldc); p != 0 {           // guard
		return argErr(name, p)
	}
	if m == 0 || n == 0 || (alpha == 0 && beta == 1) { // guard
		return nil
	}
	if p := check.SymmLens(byte(side), m, n, lda, ldb, ldc, len(a), len(b), len(c), alpha != 0); p != 0 { // guard
		return argErr(name, p)
	}
	ref.Symm(side == blas.Left, uplo == blas.Upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
	return nil
}
