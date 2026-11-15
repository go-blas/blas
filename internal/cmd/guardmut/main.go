// Command guardmut checks every argument guard of the library by mutation.
//
// A guard is a line of a non-test source file that ends with the marker
// "// guard": an `if cond {` or a tagless `case cond:`. For each one the tool
// disables it (the condition becomes `(cond) && false`), runs the tests of the
// root module except the fuzz targets, and classifies the outcome:
//
//	killed    the tests failed by an assertion: the guard is tested
//	panic     the tests died of a panic: the missing guard was found by the
//	          language, not by an assertion, and the test must be made to assert
//	survived  the tests passed: the guard is untested (or redundant)
//	invalid   the mutant did not build
//
// It exits 1 unless every guard is killed. The source is restored after each
// mutant, and on exit. Run from internal/cmd:
//
//	go run ./guardmut -root ../..
package main

import (
	"flag"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const marker = "// guard"

type guard struct {
	file string
	line int // 0-based
}

func main() {
	root := flag.String("root", ".", "the repository root")
	only := flag.String("only", "", "mutate only the guards of files whose path contains one of these comma-separated strings (default: all)")
	flag.StringVar(&testPkgs, "pkgs", "./...", "the packages to run the tests of, for each mutant")
	flag.StringVar(&testRun, "run", "", "run only the tests matching this regexp (go test -run), for each mutant; default: all but the fuzz targets")
	flag.Parse()
	guards, err := find(*root)
	if err != nil {
		fmt.Fprintln(os.Stderr, "guardmut:", err)
		os.Exit(2)
	}
	if *only != "" {
		guards = filter(guards, strings.Split(*only, ","))
	}
	if len(guards) == 0 {
		fmt.Fprintln(os.Stderr, "guardmut: no guard found")
		os.Exit(2)
	}
	if out, ok := run(*root); !ok {
		fmt.Fprintln(os.Stderr, "guardmut: the unmutated tests fail:\n"+out)
		os.Exit(2)
	}
	bad := 0
	for _, g := range guards {
		verdict, text := mutate(*root, g)
		fmt.Printf("%-9s %s:%d  %s\n", verdict, g.file, g.line+1, text)
		if verdict != "killed" {
			bad++
		}
	}
	fmt.Printf("%d guards, %d not killed by an assertion\n", len(guards), bad)
	if bad > 0 {
		os.Exit(1)
	}
}

// testPkgs is the package pattern the tests of each mutant run on.
var testPkgs, testRun = "./...", ""

// filter keeps the guards of the files whose path contains one of subs.
func filter(gs []guard, subs []string) []guard {
	var out []guard
	for _, g := range gs {
		for _, s := range subs {
			if strings.Contains(g.file, s) {
				out = append(out, g)
				break
			}
		}
	}
	return out
}

// find lists the guard lines of the non-test sources of the root module.
func find(root string) ([]guard, error) {
	var gs []guard
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		if d.IsDir() && (rel == ".git" || rel == filepath.Join("internal", "cmd") || strings.Contains(rel, "asmgen")) {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.HasSuffix(strings.TrimSpace(line), marker) {
				gs = append(gs, guard{rel, i})
			}
		}
		return nil
	})
	return gs, err
}

// disable returns the line with its condition made constant false.
func disable(line string) (string, bool) {
	body := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(line), marker))
	indent := line[:len(line)-len(strings.TrimLeft(line, "\t "))]
	switch {
	case strings.HasPrefix(body, "if ") && strings.HasSuffix(body, "{"):
		cond := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(body, "if "), "{"))
		init := ""
		if i := strings.Index(cond, "; "); i >= 0 {
			init, cond = cond[:i+2], cond[i+2:]
		}
		return indent + "if " + init + "(" + cond + ") && false {", true
	case strings.HasPrefix(body, "case ") && strings.HasSuffix(body, ":"):
		cond := strings.TrimSuffix(strings.TrimPrefix(body, "case "), ":")
		return indent + "case (" + cond + ") && false:", true
	}
	return "", false
}

func mutate(root string, g guard) (verdict, text string) {
	path := filepath.Join(root, g.file)
	orig, err := os.ReadFile(path)
	if err != nil {
		return "invalid", err.Error()
	}
	lines := strings.Split(string(orig), "\n")
	text = strings.TrimSpace(lines[g.line])
	mut, ok := disable(lines[g.line])
	if !ok {
		return "invalid", "not an if or case line: " + text
	}
	lines[g.line] = mut
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		return "invalid", err.Error()
	}
	defer os.WriteFile(path, orig, 0o644)
	out, passed := run(root)
	switch {
	case passed:
		return "survived", text
	case strings.Contains(out, "[build failed]") || strings.Contains(out, "[setup failed]"):
		return "invalid", text
	case strings.Contains(out, "\npanic: ") || strings.HasPrefix(out, "panic: ") || strings.Contains(out, "fatal error: "):
		return "panic", text
	case strings.Contains(out, "--- FAIL"):
		return "killed", text
	}
	return "invalid", text
}

// run runs the tests of the module at root, fuzz targets excluded, and reports
// the output and whether they passed.
func run(root string) (string, bool) {
	args := []string{"test", "-count=1", "-skip", "^Fuzz"}
	if testRun != "" {
		args = append(args, "-run", testRun)
	}
	cmd := exec.Command("go", append(args, testPkgs)...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}
