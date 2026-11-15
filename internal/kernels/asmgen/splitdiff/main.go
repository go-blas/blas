// Command splitdiff proves that the GEMM-only generated assembly of this
// repository is exactly the GEMM part of go-ndarray/ndarray's mixed files:
// every TEXT block and DATA symbol of sum_<arch>.s here must appear, byte for
// byte and in the same order, in ndarray's sum_<arch>.s. It lists what the
// mixed file holds beyond that (the non-GEMM kernels, which stay in ndarray)
// and exits 1 on any difference.
//
// Run from internal/kernels/asmgen, after go run ./regen:
//
//	go run ./splitdiff <ndarray clone>/internal/kernels ..
package main

import (
	"fmt"
	"os"
	"strings"
)

var arches = []string{"amd64", "arm64", "loong64", "ppc64le", "riscv64", "s390x"}

type item struct{ key, text string }

// split cuts an assembly file into its header and its items: a TEXT block, or
// the run of DATA/GLOBL lines of one symbol.
func split(src string) (header string, items []item) {
	var cur *item
	flush := func() {
		if cur != nil {
			items = append(items, *cur)
			cur = nil
		}
	}
	for _, line := range strings.SplitAfter(src, "\n") {
		switch {
		case strings.HasPrefix(line, "TEXT "):
			flush()
			cur = &item{key: "TEXT " + symbol(line)}
		case strings.HasPrefix(line, "DATA ") || strings.HasPrefix(line, "GLOBL "):
			k := "DATA " + symbol(line)
			if cur == nil || cur.key != k {
				flush()
				cur = &item{key: k}
			}
		}
		if cur == nil {
			header += line
		} else {
			cur.text += line
		}
	}
	flush()
	return header, items
}

// symbol extracts the symbol name from a TEXT, DATA or GLOBL line.
func symbol(line string) string {
	f := strings.Fields(line)[1]
	f = strings.TrimRight(f, ",")
	if i := strings.IndexAny(f, "(+"); i >= 0 {
		f = f[:i]
	}
	return f
}

// compare checks one architecture and prints its verdict; it returns the
// number of differences.
func compare(mixedPath, gemmPath string) int {
	mixed, err1 := os.ReadFile(mixedPath)
	gemm, err2 := os.ReadFile(gemmPath)
	if err1 != nil || err2 != nil {
		fmt.Println("cannot read:", err1, err2)
		return 1
	}
	mh, mi := split(string(mixed))
	gh, gi := split(string(gemm))
	bad, same, at := 0, 0, 0
	if mh != gh {
		fmt.Println("DIFFERENT header")
		bad++
	}
	taken := make([]bool, len(mi))
	for _, g := range gi {
		for at < len(mi) && mi[at].key != g.key {
			at++
		}
		if at == len(mi) {
			fmt.Printf("MISSING from the mixed file, or out of order: %s\n", g.key)
			bad++
			break
		}
		if mi[at].text != g.text {
			fmt.Printf("DIFFERENT %s\n", g.key)
			bad++
		} else {
			same++
		}
		taken[at] = true
		at++
	}
	var rest []string
	for i, m := range mi {
		if !taken[i] {
			rest = append(rest, m.key)
		}
	}
	fmt.Printf("%s: %d blocks identical, %d differ; left in the mixed file (not GEMM): %d\n",
		gemmPath, same, bad, len(rest))
	for _, r := range rest {
		fmt.Println("  not moved:", r)
	}
	return bad
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: splitdiff <ndarray internal/kernels dir> <blas internal/kernels dir>")
		os.Exit(2)
	}
	bad := 0
	for _, a := range arches {
		f := "sum_" + a + ".s"
		bad += compare(os.Args[1]+"/"+f, os.Args[2]+"/"+f)
	}
	if bad > 0 {
		os.Exit(1)
	}
}
