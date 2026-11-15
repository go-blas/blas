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
	for target, entries := range out {
		dir := filepath.Join(*root, "testdata", "fuzz", target)
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
