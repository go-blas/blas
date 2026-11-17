package lapack

import "testing"

// SetLeaf sets the panel width of the recursion for the rest of the test.
func SetLeaf(t testing.TB, n int) {
	t.Helper()
	old := leaf
	leaf = n
	t.Cleanup(func() { leaf = old })
}

// Leaf is the current panel width.
func Leaf() int { return leaf }

// Must is must, for the test of its panic.
func Must(err error) { must(err) }
