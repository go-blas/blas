package lapack

import (
	"testing"

	"github.com/go-blas/blas/internal/ref"
)

// SetLeaf sets the panel width of the recursion for the rest of the test.
func SetLeaf(t testing.TB, n int) {
	t.Helper()
	old := leaf
	leaf = n
	t.Cleanup(func() { leaf = old })
}

// SetQR sets the block size and the crossover of Geqrf and Orgqr and the block
// size of Ormqr for the rest of the test.
func SetQR(t testing.TB, nb, nx, onb int) {
	t.Helper()
	ob, ox, oo := qrNB, qrNX, ormNB
	qrNB, qrNX, ormNB = nb, nx, onb
	t.Cleanup(func() { qrNB, qrNX, ormNB = ob, ox, oo })
}

// QR is the current block size, crossover and Ormqr block size.
func QR() (nb, nx, onb int) { return qrNB, qrNX, ormNB }

// Leaf is the current panel width.
func Leaf() int { return leaf }

// Must is must, for the test of its panic.
func Must(err error) { must(err) }

// SetEig sets the block size and crossover of Sytrd and of the QL generation of
// Orgtr, and the block size and crossover of the QR generation of Orgtr (qrNB,
// qrNX, shared with Orgqr), for the rest of the test.
func SetEig(t testing.TB, trdNB, trdNX, qlNB, qlNX, qrnb, qrnx int) {
	t.Helper()
	o := [6]int{ref.NBTRD, ref.NXTRD, ref.NBQL, ref.NXQL, qrNB, qrNX}
	ref.NBTRD, ref.NXTRD, ref.NBQL, ref.NXQL, qrNB, qrNX = trdNB, trdNX, qlNB, qlNX, qrnb, qrnx
	t.Cleanup(func() { ref.NBTRD, ref.NXTRD, ref.NBQL, ref.NXQL, qrNB, qrNX = o[0], o[1], o[2], o[3], o[4], o[5] })
}

// SetMaxIt sets MAXIT, the QL or QR iterations per eigenvalue Steqr and Sterf
// allow (30), so that a test can make them fail to converge.
func SetMaxIt(t testing.TB, n int) {
	t.Helper()
	old := ref.MaxIt
	ref.MaxIt = n
	t.Cleanup(func() { ref.MaxIt = old })
}
