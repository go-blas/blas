// Command fuzzall runs every fuzz target of the repository's root package for a
// fixed time each, one after the other (go test -fuzz takes one target), and
// fails on the first failing one. CI runs it as a smoke test.
//
//	cd internal/cmd && go run ./fuzzall -root ../.. -time 10s
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

func main() {
	d := flag.String("time", "10s", "fuzz time per target")
	root := flag.String("root", ".", "the repository root")
	flag.Parse()
	n := 0
	for _, pkg := range []string{".", "./lapack"} {
		lister := exec.Command("go", "test", "-list", "^Fuzz", pkg)
		lister.Dir = *root
		list, err := lister.Output()
		if err != nil {
			fmt.Fprintln(os.Stderr, "fuzzall: listing targets of", pkg+":", err)
			os.Exit(1)
		}
		for _, name := range strings.Fields(string(list)) {
			if !strings.HasPrefix(name, "Fuzz") {
				continue
			}
			n++
			cmd := exec.Command("go", "test", "-run", "^$", "-fuzz", "^"+name+"$", "-fuzztime", *d, pkg)
			cmd.Dir, cmd.Stdout, cmd.Stderr = *root, os.Stdout, os.Stderr
			if err := cmd.Run(); err != nil {
				fmt.Fprintln(os.Stderr, "fuzzall:", pkg, name, "failed:", err)
				os.Exit(1)
			}
		}
	}
	if n == 0 {
		fmt.Fprintln(os.Stderr, "fuzzall: no fuzz target found")
		os.Exit(1)
	}
	fmt.Printf("fuzzall: %d targets ran\n", n)
}
