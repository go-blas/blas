// Command gen produces sum_loong64.s, the LASX float64 kernels for loong64, via
// go-asmgen. Run with: go run . (or `go generate` from the kernels package).
//
// Go's loong64 assembler has the LASX vector float64 add/sub/mul/div
// (XVADDD, XVSUBD, XVMULD, XVDIVD), the square root (XVFSQRTD) and the
// 256-bit loads and stores (XVMOVQ, with immediate offsets), and its
// broadcast load is the arrangement form XVMOVQ off(R), X.V4 (xvldrepl.d). It
// has no fused multiply-add: that comes from go-asmgen's XVFMADDD, which
// emits a WORD (transitional until cmd/asm names it). LASX is not on every
// LoongArch CPU, so the Go side calls these kernels only when the kernel
// reports it (AT_HWCAP).
//
// Registers: R4.. hold the arguments; R12..R15 are scratch; X0..X31 are the
// 256-bit LASX registers (four float64 lanes), F0..F31 their low lanes.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

const (
	gemmMR = 8
	gemmNR = 8
)

func main() {
	f := emit.NewFile("loong64")
	f.Add(gemmKernel())
	if err := os.WriteFile("sum_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_loong64.s")
}

func zero(b *loong64.Builder, n int) {
	for r := 0; r < n; r++ {
		b.Raw("XVXORV X%d, X%d, X%d", r, r, r)
	}
}

// gemmKernel builds gemmMicro8x8LASX(kc int, pa, pb, c *float64, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*8+r] * pb[p*8+col]   (r<8, col<8)
//
// 16 accumulators X0..X15 (row r in X(2r), X(2r+1), four columns each), the B
// row in X16, X17, and each A value broadcast (XVMOVQ .V4, xvldrepl.d) into X18..X25,
// then xvfmadd.d: 16 fused multiply-adds per k step against 10 loads. Fused,
// like the other FMA kernels; on integer-valued data it equals the scalar
// ikj oracle exactly.
func gemmKernel() *emit.Function {
	sig := loong64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]loong64.Type{loong64.Int64, loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("gemmMicro8x8LASX", sig, 0)
	b.LoadArg("kc", "R5").LoadArg("pa", "R6").LoadArg("pb", "R7").LoadArg("c", "R8").LoadArg("ldc", "R9")
	b.Raw("SLLV $3, R9") // ldc -> bytes
	zero(b, 16)
	b.Raw("BEQ R5, gstore")
	b.Label("gloop")
	b.Raw("XVMOVQ 0(R7), X16").Raw("XVMOVQ 32(R7), X17")
	for r := 0; r < gemmMR; r++ {
		b.Raw("XVMOVQ %d(R6), X%d.V4", 8*r, 18+r) // xvldrepl.d: pa[r] in all four lanes
	}
	for r := 0; r < gemmMR; r++ {
		b.XVFMADDD(2*r, 18+r, 16, 2*r)
		b.XVFMADDD(2*r+1, 18+r, 17, 2*r+1)
	}
	b.Raw("ADDV $%d, R6", 8*gemmMR).Raw("ADDV $%d, R7", 8*gemmNR).Raw("ADDV $-1, R5").Raw("BNE R5, gloop")
	b.Label("gstore")
	for r := 0; r < gemmMR; r++ {
		b.Raw("XVMOVQ 0(R8), X16").Raw("XVMOVQ 32(R8), X17")
		b.Raw("XVADDD X%d, X16, X16", 2*r).Raw("XVADDD X%d, X17, X17", 2*r+1)
		b.Raw("XVMOVQ X16, 0(R8)").Raw("XVMOVQ X17, 32(R8)")
		if r < gemmMR-1 {
			b.Raw("ADDV R9, R8")
		}
	}
	b.Ret()
	return b.Func()
}
