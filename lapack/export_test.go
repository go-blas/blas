package lapack

import "testing"

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
