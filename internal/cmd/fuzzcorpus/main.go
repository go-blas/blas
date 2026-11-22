// Command fuzzcorpus writes the committed fuzz corpus, testdata/fuzz/<Target>/,
// in the format go test replays: for every target, one entry per interesting
// class of argument (each parameter of the reference's argument tests, a slice
// one element short, a zero dimension, a quick return, a negative increment, a
// planted NaN or infinity). Run from internal/cmd:
//
//	go run ./fuzzcorpus -root ../..
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// v is one typed argument of a corpus entry.
type v struct {
	kind string
	val  string
}

func i8(n int) v                 { return v{"int8", fmt.Sprint(n)} }
func b(c byte) v                 { return v{"byte", fmt.Sprintf("'%c'", c)} }
func u8(n int) v                 { return v{"uint8", fmt.Sprint(n)} }
func u64(n int) v                { return v{"uint64", fmt.Sprint(max(n, -n))} }
func f(kind string, x float64) v { return v{kind, fmt.Sprint(x)} }

func entry(args ...v) string {
	var sb strings.Builder
	sb.WriteString("go test fuzz v1\n")
	for _, a := range args {
		fmt.Fprintf(&sb, "%s(%s)\n", a.kind, a.val)
	}
	return sb.String()
}

func main() {
	root := flag.String("root", ".", "the repository root")
	flag.Parse()
	out := map[string][]string{}
	add := func(target string, args ...v) { out[target] = append(out[target], entry(args...)) }

	for _, p := range []struct{ target, kind string }{{"FuzzDgemm", "float64"}, {"FuzzSgemm", "float32"}} {
		// tA, tB, m, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed
		g := func(tA, tB byte, m, n, k, la, lb, lc, dA, dB, dC int, al, be float64, flavor int) {
			add(p.target, b(tA), b(tB), i8(m), i8(n), i8(k), i8(la), i8(lb), i8(lc), i8(dA), i8(dB), i8(dC), f(p.kind, al), f(p.kind, be), u8(flavor), u64(m*100+n*10+k))
		}
		g('N', 'N', 7, 5, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0)
		g('N', 'T', 7, 5, 3, 1, 2, 3, 0, 0, 0, 2.5, 1, 0)
		g('T', 'N', 7, 5, 3, 0, 0, 0, 0, 0, 0, -1, 0.5, 0)
		g('T', 'T', 13, 17, 19, 3, 3, 3, 0, 0, 0, 0.75, 2, 0)
		g('C', 'C', 4, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 0)
		g('n', 't', 5, 4, 3, 0, 0, 0, 0, 0, 0, 1.5, 0.5, 0) // lower case
		g('t', 'c', 5, 4, 3, 0, 0, 0, 0, 0, 0, 1.5, 0.5, 1)
		g('X', 'N', 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0)    // param 1
		g('N', 'Y', 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0)    // param 2
		g('N', 'N', -1, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0)   // param 3
		g('N', 'N', 3, -1, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0)   // param 4
		g('N', 'N', 3, 3, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0)   // param 5
		g('N', 'N', 3, 3, 3, -1, 0, 0, 0, 0, 0, 1, 1, 0)   // param 8
		g('N', 'N', 3, 3, 3, 0, -1, 0, 0, 0, 0, 1, 1, 0)   // param 10
		g('N', 'N', 3, 3, 3, 0, 0, -1, 0, 0, 0, 1, 1, 0)   // param 13
		g('N', 'N', 6, 6, 6, 0, 0, 0, -1, 0, 0, 1, 1, 0)   // A one short
		g('N', 'T', 6, 6, 6, 0, 0, 0, 0, -1, 0, 1, 1, 0)   // B one short
		g('N', 'N', 6, 6, 6, 0, 0, 0, 0, 0, -1, 1, 1, 0)   // C one short
		g('N', 'N', 0, 5, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0)    // m = 0
		g('N', 'N', 5, 5, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0)    // k = 0, quick return
		g('N', 'N', 5, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)    // k = 0, C := 0
		g('N', 'N', 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 3, 0)    // alpha = 0
		g('N', 'N', 5, 5, 5, 0, 0, 0, -2, -2, 0, 0, 3, 0)  // alpha = 0, A and B short: not an error
		g('N', 'N', 5, 5, 5, 0, 0, 0, 0, 0, 0, 1, 0, 1)    // NaN planted
		g('T', 'N', 5, 5, 5, 0, 0, 0, 0, 0, 0, 1, 1, 2)    // infinity planted
		g('N', 'N', 39, 39, 39, 0, 0, 0, 0, 0, 0, 1, 1, 0) // parallel kernel path
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDgemmStrided", "float64"}, {"FuzzSgemmStrided", "float32"}} {
		// kinds (A | B<<2 | C<<4 | invalid-C<<6), m, n, k, padA, padB, padC, dA, dB, dC, alpha, beta, flavor, seed
		g := func(kinds uint8, m, n, k, pa, pb, pc, dA, dB, dC int, al, be float64, flavor int) {
			add(p.target, u8(int(kinds)), i8(m), i8(n), i8(k), i8(pa), i8(pb), i8(pc), i8(dA), i8(dB), i8(dC), f(p.kind, al), f(p.kind, be), u8(flavor), u64(m*100+n*10+k+int(kinds)))
		}
		g(0x00, 7, 5, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0)   // row-major everywhere
		g(0x15, 7, 5, 3, 0, 0, 0, 0, 0, 0, 2.5, 1, 0) // column-major everywhere
		g(0x1e, 13, 17, 19, 3, 2, 1, 0, 0, 0, 0.75, 2, 0)
		g(0x2d, 9, 4, 6, 1, 0, 3, 0, 0, 0, -1, 0.5, 0)
		g(0x0b, 6, 8, 37, 0, 0, 0, 0, 0, 0, 1, 1, 0)
		g(0x00, 39, 39, 39, 0, 0, 0, 0, 0, 0, 1, 1, 0) // parallel kernel path
		g(0x20, 5, 5, 5, 0, 0, 0, 0, 0, 0, 1, 1, 0)    // C with both strides above 1
		g(0x40, 3, 3, 3, 0, 0, -1, 0, 0, 0, 1, 1, 0)   // invalid C stride
		g(0x40, 3, 3, 3, 0, 0, 3, 0, 0, 0, 1, 1, 0)
		g(0x00, 6, 6, 6, 0, 0, 0, -1, 0, 0, 1, 1, 0)  // A one short
		g(0x00, 6, 6, 6, 0, 0, 0, 0, -1, 0, 1, 1, 0)  // B one short
		g(0x00, 6, 6, 6, 0, 0, 0, 0, 0, -1, 1, 1, 0)  // C one short
		g(0x00, 0, 5, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0)   // m = 0
		g(0x00, 5, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)   // k = 0, C := 0
		g(0x00, 5, 5, 5, 0, 0, 0, -2, -2, 0, 0, 3, 0) // alpha = 0, A and B short
		g(0x00, 5, 5, 5, 0, 0, 0, 0, 0, 0, 1, 0, 1)   // NaN planted
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDgemv", "float64"}, {"FuzzSgemv", "float32"}} {
		// tr, m, n, ldaD, incx, incy, dA, dX, dY, alpha, beta, flavor, seed
		g := func(tr byte, m, n, la, ix, iy, dA, dX, dY int, al, be float64, flavor int) {
			add(p.target, b(tr), i8(m), i8(n), i8(la), i8(ix), i8(iy), i8(dA), i8(dX), i8(dY), f(p.kind, al), f(p.kind, be), u8(flavor), u64(m*100+n))
		}
		g('N', 7, 5, 0, 1, 1, 0, 0, 0, 1, 0, 0)
		g('T', 7, 5, 2, 1, 1, 0, 0, 0, 2.5, 1, 0)
		g('C', 7, 5, 0, 2, -1, 0, 0, 0, -1, 0.5, 0)
		g('t', 7, 5, 0, 1, 1, 0, 0, 0, 2, 1, 0) // lower case
		g('N', 7, 5, 0, -3, 2, 0, 0, 0, 0.75, 2, 0)
		g('Q', 3, 3, 0, 1, 1, 0, 0, 0, 1, 1, 0)   // param 1
		g('N', -1, 3, 0, 1, 1, 0, 0, 0, 1, 1, 0)  // param 2
		g('N', 3, -1, 0, 1, 1, 0, 0, 0, 1, 1, 0)  // param 3
		g('N', 3, 3, -1, 1, 1, 0, 0, 0, 1, 1, 0)  // param 6
		g('N', 3, 3, 0, 0, 1, 0, 0, 0, 1, 1, 0)   // param 8
		g('N', 3, 3, 0, 1, 0, 0, 0, 0, 1, 1, 0)   // param 11
		g('N', 6, 6, 0, 1, 1, -1, 0, 0, 1, 1, 0)  // A short
		g('N', 6, 6, 0, 1, 1, 0, -1, 0, 1, 1, 0)  // x short
		g('N', 6, 6, 0, 1, 1, 0, 0, -1, 1, 1, 0)  // y short
		g('N', 0, 5, 0, 1, 1, 0, 0, 0, 1, 1, 0)   // m = 0
		g('N', 5, 5, 0, 1, 1, 0, 0, 0, 0, 1, 0)   // quick return
		g('T', 5, 5, 0, 1, 1, -2, -2, 0, 0, 0, 0) // alpha = 0: A and x not read
		g('N', 5, 5, 0, 1, 1, 0, 0, 0, 1, 0, 1)   // NaN planted
		g('T', 39, 3, 0, 1, 1, 0, 0, 0, 1, 1, 2)  // infinity planted
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDdot", ""}, {"FuzzSdot", ""}} {
		// n, incx, incy, dX, dY, flavor, seed
		g := func(n, ix, iy, dX, dY, flavor int) {
			add(p.target, i8(n), i8(ix), i8(iy), i8(dX), i8(dY), u8(flavor), u64(n*10+flavor))
		}
		g(10, 1, 1, 0, 0, 0)
		g(7, -2, 3, 0, 0, 0)
		g(0, 1, 1, 0, 0, 0)
		g(5, 0, 1, 0, 0, 0)
		g(5, 1, 0, 0, 0, 0)
		g(5, 1, 1, -1, 0, 0) // x short
		g(5, 1, 1, 0, -1, 0) // y short
		g(5, 1, 1, 0, 0, 1)  // NaN
		g(5, 2, -2, 0, 0, 2) // infinity
		g(39, 1, 1, 0, 0, 0)
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDaxpy", "float64"}, {"FuzzSaxpy", "float32"}} {
		// n, incx, incy, dX, dY, alpha, flavor, seed
		g := func(n, ix, iy, dX, dY int, al float64, flavor int) {
			add(p.target, i8(n), i8(ix), i8(iy), i8(dX), i8(dY), f(p.kind, al), u8(flavor), u64(n*10+flavor))
		}
		g(10, 1, 1, 0, 0, 2.5, 0)
		g(7, -2, 3, 0, 0, -1, 0)
		g(0, 1, 1, 0, 0, 1, 0)
		g(5, 1, 1, 0, 0, 0, 0) // alpha = 0
		g(5, 1, 1, -2, -2, 0, 0)
		g(5, 1, 1, -1, 0, 1, 0)
		g(5, 1, 1, 0, -1, 1, 0)
		g(5, 1, 0, 0, 0, 1, 0)
		g(5, 1, 1, 0, 0, 1, 1)
		g(39, 1, 1, 0, 0, 3, 0)
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDscal", "float64"}, {"FuzzSscal", "float32"}} {
		// n, incx, dX, alpha, flavor, seed
		g := func(n, ix, dX int, al float64, flavor int) {
			add(p.target, i8(n), i8(ix), i8(dX), f(p.kind, al), u8(flavor), u64(n*10+flavor))
		}
		g(10, 1, 0, 2.5, 0)
		g(7, 3, 0, -1, 0)
		g(0, 1, 0, 2, 0)
		g(5, 0, 0, 2, 0)
		g(5, -1, 0, 2, 0)
		g(5, 1, 0, 1, 0)
		g(5, 1, 0, 0, 1) // 0*NaN
		g(5, 1, -1, 2, 0)
		g(39, 2, 0, 0.5, 0)
	}
	for _, t := range []string{"FuzzDnrm2", "FuzzSnrm2", "FuzzDasum", "FuzzSasum", "FuzzIdamax", "FuzzIsamax"} {
		// n, incx, dX, flavor, seed
		g := func(n, ix, dX, flavor int) { add(t, i8(n), i8(ix), i8(dX), u8(flavor), u64(n*10+flavor)) }
		g(10, 1, 0, 0)
		g(7, 3, 0, 0)
		g(7, -3, 0, 0)
		g(0, 1, 0, 0)
		g(5, 0, 0, 0)
		g(5, 1, -1, 0) // x short
		g(5, 1, 0, 1)  // NaN
		g(5, 1, 0, 2)  // infinity
		g(5, 1, 0, 3)  // huge
		g(5, 1, 0, 4)  // tiny
		g(39, 1, 0, 0)
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDtrsm", "float64"}, {"FuzzStrsm", "float32"}, {"FuzzDtrmm", "float64"}, {"FuzzStrmm", "float32"}} {
		// side, uplo, tA, diag, m, n, ldaD, ldbD, dA, dB, alpha, flavor, seed, blk
		g := func(side, uplo, tA, diag byte, m, n, la, lb, dA, dB int, al float64, flavor, blk int) {
			add(p.target, b(side), b(uplo), b(tA), b(diag), i8(m), i8(n), i8(la), i8(lb), i8(dA), i8(dB), f(p.kind, al), u8(flavor), u64(m*100+n*10+int(side)), u8(blk))
		}
		for _, side := range []byte{'L', 'R'} {
			g(side, 'U', 'N', 'N', 7, 5, 0, 0, 0, 0, 1, 0, 0)
			g(side, 'L', 'T', 'N', 7, 5, 2, 1, 0, 0, 2.5, 0, 0)
			g(side, 'U', 'T', 'U', 13, 9, 0, 3, 0, 0, -1, 0, 4)
			g(side, 'L', 'C', 'U', 9, 13, 3, 0, 0, 0, 0.75, 0, 7)
			g(side, 'u', 'n', 'u', 11, 6, 0, 0, 0, 0, 1.5, 0, 3) // lower case
			g(side, 'L', 'N', 'N', 39, 39, 0, 0, 0, 0, 1, 0, 5)  // several blocks
			g(side, 'U', 'N', 'N', 5, 5, 0, 0, 0, 0, 0, 0, 0)    // alpha = 0
			g(side, 'U', 'N', 'N', 5, 5, 0, 0, -2, 0, 0, 0, 0)   // alpha = 0, A short: not an error
			g(side, 'U', 'N', 'N', 5, 5, 0, 0, 0, 0, 1, 1, 2)    // NaN planted
			g(side, 'L', 'T', 'N', 5, 5, 0, 0, 0, 0, 1, 2, 2)    // infinity planted
			g(side, 'U', 'N', 'N', 0, 5, 0, 0, 0, 0, 1, 0, 0)    // m = 0
			g(side, 'U', 'N', 'N', 5, 0, 0, 0, 0, 0, 1, 0, 0)    // n = 0
		}
		g('X', 'U', 'N', 'N', 3, 3, 0, 0, 0, 0, 1, 0, 0)  // param 1
		g('L', 'X', 'N', 'N', 3, 3, 0, 0, 0, 0, 1, 0, 0)  // param 2
		g('L', 'U', 'X', 'N', 3, 3, 0, 0, 0, 0, 1, 0, 0)  // param 3
		g('L', 'U', 'N', 'X', 3, 3, 0, 0, 0, 0, 1, 0, 0)  // param 4
		g('L', 'U', 'N', 'N', -1, 3, 0, 0, 0, 0, 1, 0, 0) // param 5
		g('L', 'U', 'N', 'N', 3, -1, 0, 0, 0, 0, 1, 0, 0) // param 6
		g('L', 'U', 'N', 'N', 3, 3, -1, 0, 0, 0, 1, 0, 0) // param 9
		g('L', 'U', 'N', 'N', 3, 3, 0, -1, 0, 0, 1, 0, 0) // param 11
		g('L', 'U', 'N', 'N', 6, 6, 0, 0, -1, 0, 1, 0, 0) // A one short
		g('R', 'L', 'T', 'N', 6, 6, 0, 0, 0, -1, 1, 0, 0) // B one short
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDsyrk", "float64"}, {"FuzzSsyrk", "float32"}, {"FuzzDsyr2k", "float64"}, {"FuzzSsyr2k", "float32"}} {
		// uplo, trans, n, k, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk
		g := func(uplo, tr byte, n, k, la, lb, lc, dA, dB, dC int, al, be float64, flavor, blk int) {
			add(p.target, b(uplo), b(tr), i8(n), i8(k), i8(la), i8(lb), i8(lc), i8(dA), i8(dB), i8(dC), f(p.kind, al), f(p.kind, be), u8(flavor), u64(n*100+k*10+int(uplo)), u8(blk))
		}
		for _, uplo := range []byte{'U', 'L'} {
			g(uplo, 'N', 7, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
			g(uplo, 'T', 7, 5, 2, 1, 3, 0, 0, 0, 2.5, 1, 0, 4)
			g(uplo, 'C', 13, 9, 0, 3, 0, 0, 0, 0, -1, 0.5, 0, 7)
			g(uplo+32, 'n', 9, 13, 3, 0, 0, 0, 0, 0, 0.75, 2, 0, 3) // lower case
			g(uplo, 'N', 39, 39, 0, 0, 0, 0, 0, 0, 1, 1, 0, 5)      // several blocks
			g(uplo, 'T', 5, 5, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0)        // alpha = 0
			g(uplo, 'N', 5, 5, 0, 0, 0, -2, -2, 0, 0, 3, 0, 0)      // alpha = 0, A and B short: not an error
			g(uplo, 'N', 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)        // k = 0, C := 0
			g(uplo, 'N', 5, 5, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2)        // NaN planted
			g(uplo, 'T', 5, 5, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2)        // infinity planted
			g(uplo, 'N', 0, 5, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)        // n = 0
		}
		g('X', 'N', 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)  // param 1
		g('U', 'X', 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)  // param 2
		g('U', 'N', -1, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0) // param 3
		g('U', 'N', 3, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0) // param 4
		g('U', 'N', 3, 3, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0) // param 7
		g('U', 'N', 3, 3, 0, -1, 0, 0, 0, 0, 1, 1, 0, 0) // param 9 (syr2k)
		g('U', 'N', 3, 3, 0, 0, -1, 0, 0, 0, 1, 1, 0, 0) // ldc
		g('U', 'N', 6, 6, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0) // A one short
		g('U', 'T', 6, 6, 0, 0, 0, 0, -1, 0, 1, 1, 0, 0) // B one short
		g('L', 'N', 6, 6, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0) // C one short
	}
	for _, p := range []struct{ target, kind string }{{"FuzzDsymm", "float64"}, {"FuzzSsymm", "float32"}} {
		// side, uplo, m, n, ldaD, ldbD, ldcD, dA, dB, dC, alpha, beta, flavor, seed, blk
		g := func(side, uplo byte, m, n, la, lb, lc, dA, dB, dC int, al, be float64, flavor, blk int) {
			add(p.target, b(side), b(uplo), i8(m), i8(n), i8(la), i8(lb), i8(lc), i8(dA), i8(dB), i8(dC), f(p.kind, al), f(p.kind, be), u8(flavor), u64(m*100+n*10+int(side)), u8(blk))
		}
		for _, side := range []byte{'L', 'R'} {
			for _, uplo := range []byte{'U', 'L'} {
				g(side, uplo, 7, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
				g(side, uplo, 7, 5, 2, 1, 3, 0, 0, 0, 2.5, 1, 0, 0)
				g(side, uplo, 13, 9, 0, 3, 0, 0, 0, 0, -1, 0.5, 0, 0)
				g(side+32, uplo+32, 9, 13, 3, 0, 0, 0, 0, 0, 0.75, 2, 0, 0) // lower case
				g(side, uplo, 5, 5, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0)           // alpha = 0
				g(side, uplo, 5, 5, 0, 0, 0, -2, -2, 0, 0, 3, 0, 0)         // alpha = 0, A and B short: not an error
				g(side, uplo, 5, 5, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0)           // NaN planted
				g(side, uplo, 5, 5, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0)           // infinity planted
				g(side, uplo, 39, 39, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)         // parallel kernel path
				g(side, uplo, 0, 5, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)           // m = 0
			}
		}
		g('X', 'U', 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)  // param 1
		g('L', 'X', 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)  // param 2
		g('L', 'U', -1, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0) // param 3
		g('L', 'U', 3, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0) // param 4
		g('L', 'U', 3, 3, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0) // param 7
		g('L', 'U', 3, 3, 0, -1, 0, 0, 0, 0, 1, 1, 0, 0) // param 9
		g('L', 'U', 3, 3, 0, 0, -1, 0, 0, 0, 1, 1, 0, 0) // param 12
		g('L', 'U', 6, 6, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0) // A one short
		g('R', 'L', 6, 6, 0, 0, 0, 0, -1, 0, 1, 1, 0, 0) // B one short
		g('L', 'L', 6, 6, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0) // C one short
	}
	// LAPACK: package lapack has its own testdata, and its targets compare
	// lapack with netlib.
	addL := func(target string, args ...v) { out["lapack:"+target] = append(out["lapack:"+target], entry(args...)) }
	boolV := func(x bool) v { return v{"bool", fmt.Sprint(x)} }
	for _, p := range []struct{ prefix, kind string }{{"D", "float64"}, {"S", "float32"}} {
		// m, n, ldaD, dA, dI, zc, flavor, seed, leaf
		g := func(m, n, la, dA, dI, zc, flavor, lf int) {
			addL("Fuzz"+p.prefix+"getrf", i8(m), i8(n), i8(la), i8(dA), i8(dI), i8(zc), u8(flavor), u64(m*100+n*10+lf+flavor), u8(lf))
		}
		g(7, 5, 0, 0, 0, -1, 0, 0)
		g(5, 7, 2, 0, 0, -1, 0, 1)
		g(9, 9, 0, 0, 0, -1, 0, 2)
		g(11, 11, 3, 0, 0, -1, 0, 3)
		g(39, 39, 0, 0, 0, -1, 0, 0) // past the default panel
		g(38, 20, 1, 0, 0, -1, 0, 5)
		g(20, 38, 0, 0, 0, -1, 0, 7)
		g(12, 12, 0, 0, 0, 2, 0, 1) // a zero column: INFO 3
		g(30, 30, 0, 0, 0, 29, 0, 2)
		g(1, 1, 0, 0, 0, 0, 0, 0)
		g(1, 5, 0, 0, 0, -1, 0, 0)
		g(5, 1, 0, 0, 0, -1, 0, 0)
		g(-1, 3, 0, 0, 0, -1, 0, 0) // param 1
		g(3, -1, 0, 0, 0, -1, 0, 0) // param 2
		g(3, 3, -1, 0, 0, -1, 0, 0) // param 4
		g(6, 6, 0, -1, 0, -1, 0, 0) // A one short
		g(6, 6, 0, 0, -1, -1, 0, 0) // ipiv one short
		g(0, 5, 0, 0, 0, -1, 0, 0)  // m = 0
		g(5, 0, 0, 0, 0, -1, 0, 0)  // n = 0
		g(8, 8, 0, 0, 0, -1, 1, 1)  // NaN planted
		g(8, 8, 0, 0, 0, -1, 2, 1)  // infinity planted
		g(8, 8, 0, 0, 0, -1, 3, 1)  // huge
		g(8, 8, 0, 0, 0, -1, 4, 1)  // tiny
	}
	for _, p := range []struct{ prefix, kind string }{{"D", "float64"}, {"S", "float32"}} {
		// up, n, ldaD, dA, bad, flavor, seed, leaf
		g := func(up bool, n, la, dA, bad, flavor, lf int) {
			addL("Fuzz"+p.prefix+"potrf", boolV(up), i8(n), i8(la), i8(dA), i8(bad), u8(flavor), u64(n*10+lf+flavor), u8(lf))
		}
		for _, up := range []bool{true, false} {
			g(up, 7, 0, 0, 0, 0, 0)
			g(up, 9, 2, 0, 0, 0, 1)
			g(up, 11, 0, 0, 0, 0, 2)
			g(up, 13, 3, 0, 0, 0, 3)
			g(up, 39, 0, 0, 0, 0, 0)
			g(up, 1, 0, 0, 0, 0, 0)
			g(up, 12, 0, 0, -5, 0, 1) // diagonal 5 not positive: INFO 6
			g(up, 30, 0, 0, -1, 0, 2)
			g(up, 8, 0, 0, 0, 1, 1) // NaN planted
			g(up, 8, 0, 0, 0, 2, 1) // infinity planted
			g(up, 8, 0, 0, 0, 3, 1)
			g(up, 8, 0, 0, 0, 4, 1)
			g(up, 0, 0, 0, 0, 0, 0)  // n = 0
			g(up, -1, 0, 0, 0, 0, 0) // param 2
			g(up, 3, -1, 0, 0, 0, 0) // param 4
			g(up, 6, 0, -1, 0, 0, 0) // A one short
		}
	}
	for _, p := range []struct{ prefix, kind string }{{"D", "float64"}, {"S", "float32"}} {
		// tr, n, nrhs, ldaD, ldbD, dA, dB, dI, pv, flavor, seed, leaf
		g := func(tr byte, n, nrhs, la, lb, dA, dB, dI, pv, flavor, lf int) {
			addL("Fuzz"+p.prefix+"getrs", b(tr), i8(n), i8(nrhs), i8(la), i8(lb), i8(dA), i8(dB), i8(dI), u8(pv), u8(flavor), u64(n*10+nrhs+lf), u8(lf))
		}
		for _, tr := range []byte{'N', 'T', 'C', 'n', 't'} {
			g(tr, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0)
			g(tr, 9, 4, 2, 1, 0, 0, 0, 0, 0, 2)
			g(tr, 39, 3, 0, 0, 0, 0, 0, 0, 0, 0)
			g(tr, 5, 39, 0, 0, 0, 0, 0, 0, 0, 1)
			g(tr, 8, 2, 0, 0, 0, 0, 0, 0, 1, 1) // NaN planted
			g(tr, 8, 2, 0, 0, 0, 0, 0, 0, 2, 1) // infinity planted
			g(tr, 6, 2, 0, 0, 0, 0, 0, 1, 0, 1) // a pivot 0
			g(tr, 6, 2, 0, 0, 0, 0, 0, 2, 0, 1) // a pivot n+1
		}
		g('X', 3, 3, 0, 0, 0, 0, 0, 0, 0, 0)  // param 1
		g('N', -1, 3, 0, 0, 0, 0, 0, 0, 0, 0) // param 2
		g('N', 3, -1, 0, 0, 0, 0, 0, 0, 0, 0) // param 3
		g('N', 3, 3, -1, 0, 0, 0, 0, 0, 0, 0) // param 5
		g('N', 3, 3, 0, -1, 0, 0, 0, 0, 0, 0) // param 8
		g('N', 6, 6, 0, 0, -1, 0, 0, 0, 0, 0) // A one short
		g('T', 6, 6, 0, 0, 0, 0, -1, 0, 0, 0) // ipiv one short
		g('N', 6, 6, 0, 0, 0, -1, 0, 0, 0, 0) // B one short
		g('N', 0, 3, 0, 0, 0, 0, 0, 0, 0, 0)  // n = 0
		g('N', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0)  // nrhs = 0
	}
	for _, p := range []struct{ prefix, kind string }{{"D", "float64"}, {"S", "float32"}} {
		// up, n, nrhs, ldaD, ldbD, dA, dB, flavor, seed, leaf
		g := func(up bool, n, nrhs, la, lb, dA, dB, flavor, lf int) {
			addL("Fuzz"+p.prefix+"potrs", boolV(up), i8(n), i8(nrhs), i8(la), i8(lb), i8(dA), i8(dB), u8(flavor), u64(n*10+nrhs+lf), u8(lf))
		}
		for _, up := range []bool{true, false} {
			g(up, 7, 3, 0, 0, 0, 0, 0, 0)
			g(up, 9, 4, 2, 1, 0, 0, 0, 2)
			g(up, 39, 3, 0, 0, 0, 0, 0, 0)
			g(up, 5, 39, 0, 0, 0, 0, 0, 1)
			g(up, 8, 2, 0, 0, 0, 0, 1, 1)
			g(up, 8, 2, 0, 0, 0, 0, 2, 1)
			g(up, -1, 3, 0, 0, 0, 0, 0, 0) // param 2
			g(up, 3, -1, 0, 0, 0, 0, 0, 0) // param 3
			g(up, 3, 3, -1, 0, 0, 0, 0, 0) // param 5
			g(up, 3, 3, 0, -1, 0, 0, 0, 0) // param 7
			g(up, 6, 6, 0, 0, -1, 0, 0, 0) // A one short
			g(up, 6, 6, 0, 0, 0, -1, 0, 0) // B one short
			g(up, 0, 3, 0, 0, 0, 0, 0, 0)
			g(up, 3, 0, 0, 0, 0, 0, 0, 0)
		}
	}
	// QR. cfg selects the block configuration (0 to 3 small ones, 4 the default,
	// the only one netlib has, so the only one NaN and Inf are compared on).
	for _, p := range []struct{ prefix, kind string }{{"D", "float64"}, {"S", "float32"}} {
		// m, n, ldaD, dA, dT, flavor, cfg
		g := func(m, n, la, dA, dT, flavor, cfg int) {
			addL("Fuzz"+p.prefix+"geqrf", i8(m), i8(n), i8(la), i8(dA), i8(dT), u8(flavor), u64(m*100+n*10+cfg+flavor), u8(cfg))
		}
		g(7, 5, 0, 0, 0, 0, 4)
		g(5, 7, 2, 0, 0, 0, 0)
		g(9, 9, 0, 0, 0, 0, 1)
		g(39, 39, 0, 0, 0, 0, 2)
		g(38, 20, 1, 0, 0, 0, 3)
		g(20, 38, 0, 0, 0, 0, 0)
		g(39, 39, 0, 0, 0, 0, 4)
		g(1, 1, 0, 0, 0, 0, 4)
		g(1, 5, 0, 0, 0, 0, 4)
		g(5, 1, 0, 0, 0, 0, 4)
		g(-1, 3, 0, 0, 0, 0, 4) // param 1
		g(3, -1, 0, 0, 0, 0, 4) // param 2
		g(3, 3, -1, 0, 0, 0, 4) // param 4
		g(6, 6, 0, -1, 0, 0, 4) // A one short
		g(6, 6, 0, 0, -1, 0, 4) // tau one short
		g(0, 5, 0, 0, 0, 0, 4)  // m = 0
		g(5, 0, 0, 0, 0, 0, 4)  // n = 0
		g(8, 8, 0, 0, 0, 1, 4)  // NaN planted
		g(8, 8, 0, 0, 0, 2, 4)  // infinity planted
		g(8, 8, 0, 0, 0, 3, 0)  // huge
		g(8, 8, 0, 0, 0, 4, 0)  // tiny
	}
	for _, p := range []struct{ prefix, kind string }{{"D", "float64"}, {"S", "float32"}} {
		// side (L R l r X), trans (N T n t C X), m, n, k, ldaD, ldcD, dA, dT, dC, flavor, cfg
		g := func(sd, tr, m, n, k, la, lc, dA, dT, dC, flavor, cfg int) {
			addL("Fuzz"+p.prefix+"ormqr", u8(sd), u8(tr), i8(m), i8(n), i8(k), i8(la), i8(lc), i8(dA), i8(dT), i8(dC), u8(flavor), u64(m*100+n*10+k+sd+tr), u8(cfg))
		}
		g(0, 0, 7, 5, 3, 0, 0, 0, 0, 0, 0, 4)
		g(0, 1, 9, 6, 6, 1, 2, 0, 0, 0, 0, 0)
		g(1, 0, 6, 9, 5, 0, 0, 0, 0, 0, 0, 1)
		g(1, 1, 5, 7, 7, 2, 0, 0, 0, 0, 0, 2)
		g(2, 2, 12, 5, 12, 0, 0, 0, 0, 0, 0, 3)
		g(3, 3, 5, 12, 12, 0, 1, 0, 0, 0, 0, 0)
		g(0, 0, 39, 10, 10, 0, 0, 0, 0, 0, 0, 4)
		g(1, 1, 10, 39, 10, 0, 0, 0, 0, 0, 0, 4)
		g(0, 0, 38, 5, 20, 0, 0, 0, 0, 0, 0, 3)
		g(4, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4)  // param 1
		g(0, 5, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4)  // param 2
		g(0, 4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4)  // param 2: C is refused for a real Q
		g(0, 0, -1, 3, 0, 0, 0, 0, 0, 0, 0, 4) // param 3
		g(0, 0, 3, -1, 0, 0, 0, 0, 0, 0, 0, 4) // param 4
		g(0, 0, 3, 3, -1, 0, 0, 0, 0, 0, 0, 4) // param 5
		g(0, 0, 3, 3, 4, 0, 0, 0, 0, 0, 0, 4)  // k above nq
		g(0, 0, 3, 3, 3, -1, 0, 0, 0, 0, 0, 4) // param 7
		g(0, 0, 3, 3, 3, 0, -1, 0, 0, 0, 0, 4) // param 10
		g(0, 0, 6, 6, 6, 0, 0, -1, 0, 0, 0, 4) // A one short
		g(0, 0, 6, 6, 6, 0, 0, 0, -1, 0, 0, 4) // tau one short
		g(0, 0, 6, 6, 6, 0, 0, 0, 0, -1, 0, 4) // C one short
		g(0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4)  // m = 0
		g(0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4)  // n = 0
		g(0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 4)  // k = 0
		g(0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 1, 4)  // NaN planted
		g(1, 1, 8, 8, 8, 0, 0, 0, 0, 0, 2, 4)  // infinity planted
	}
	for _, p := range []struct{ prefix, kind string }{{"D", "float64"}, {"S", "float32"}} {
		// m, n, k, ldaD, dA, dT, flavor, cfg
		g := func(m, n, k, la, dA, dT, flavor, cfg int) {
			addL("Fuzz"+p.prefix+"orgqr", i8(m), i8(n), i8(k), i8(la), i8(dA), i8(dT), u8(flavor), u64(m*100+n*10+k+cfg), u8(cfg))
		}
		g(7, 5, 3, 0, 0, 0, 0, 4)
		g(9, 9, 9, 0, 0, 0, 0, 0)
		g(6, 6, 6, 2, 0, 0, 0, 1)
		g(39, 39, 39, 0, 0, 0, 0, 4)
		g(39, 30, 12, 1, 0, 0, 0, 2)
		g(38, 20, 20, 0, 0, 0, 0, 3)
		g(10, 4, 0, 0, 0, 0, 0, 4) // k = 0: the identity
		g(1, 1, 1, 0, 0, 0, 0, 4)
		g(-1, 0, 0, 0, 0, 0, 0, 4) // param 1
		g(3, 4, 0, 0, 0, 0, 0, 4)  // param 2: n above m
		g(3, 3, -1, 0, 0, 0, 0, 4) // param 3
		g(3, 2, 3, 0, 0, 0, 0, 4)  // param 3: k above n
		g(3, 3, 3, -1, 0, 0, 0, 4) // param 5
		g(6, 6, 6, 0, -1, 0, 0, 4) // A one short
		g(6, 6, 6, 0, 0, -1, 0, 4) // tau one short
		g(3, 0, 0, 0, 0, 0, 0, 4)  // n = 0
		g(8, 8, 8, 0, 0, 0, 1, 4)  // NaN planted
		g(8, 8, 8, 0, 0, 0, 2, 4)  // infinity planted
	}
	for target, entries := range out {
		dir := filepath.Join(*root, "testdata", "fuzz", target)
		if rest, ok := strings.CutPrefix(target, "lapack:"); ok {
			dir = filepath.Join(*root, "lapack", "testdata", "fuzz", rest)
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		for i, e := range entries {
			if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("seed-%02d", i)), []byte(e), 0o644); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
	}
}
