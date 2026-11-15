package blas

import "testing"

// SetBlockSize sets the order of the diagonal blocks of the level 3 routines for
// the rest of the test, so that small matrices span several blocks.
func SetBlockSize(t testing.TB, n int) {
	t.Helper()
	old := blockSize
	blockSize = n
	t.Cleanup(func() { blockSize = old })
}

// BlockSize is the order of the diagonal blocks.
func BlockSize() int { return blockSize }
