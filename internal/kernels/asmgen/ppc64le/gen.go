// Command gen produces sum_ppc64le.s, the VSX float64 kernels for ppc64le, via
// go-asmgen. Run with: go run . (or `go generate` from the kernels package).
//
// Go's ppc64 assembler has the VSX loads, stores and splats (LXVD2X, LXVDSX,
// STXVD2X, XXPERMDI) but no vector double arithmetic; the arithmetic comes from
// go-asmgen's VSX encoders (Builder.XVADDDP, XVMADDADP, ...), which emit WORDs.
//
// Lane order: LXVD2X/STXVD2X swap the two doublewords of a vector on
// little-endian. Every vector of one kernel is loaded and stored that way, so a
// lane-wise operation pairs the same elements in every operand and the store
// puts each result back at its own index. LXVDSX (load and splat) puts the same
// double in both lanes, so it needs no care.
//
// Registers: R3.. hold the arguments; R14..R20 hold the byte offsets that
// LXVD2X's indexed form needs (R0 reads as 0); R30/R31/R13/R2/R1 are Go's and
// untouched. VS0..VS31 are accumulators, VS32..VS43 operands.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

const (
	gemmMR = 8
	gemmNR = 8
)

func main() {
	f := emit.NewFile("ppc64le")
	f.Add(gemmKernel())
	f.Add(gemm32Kernel(f.Data("oddw", words(1, 3)), f.Data("highw", words(2, 3))))
	if err := os.WriteFile("sum_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_ppc64le.s")
}

// offsets loads the byte offsets 16, 32, 48 into R14, R15, R16 (and, with
// splat, 8..56 into R17..R20 for the A-panel splats): LXVD2X/LXVDSX take only
// a (base)(index) address.
func offsets(b *ppc64.Builder, splat bool) {
	b.Raw("MOVD $16, R14").Raw("MOVD $32, R15").Raw("MOVD $48, R16")
	if splat {
		b.Raw("MOVD $8, R17").Raw("MOVD $24, R18").Raw("MOVD $40, R19").Raw("MOVD $56, R20")
	}
}

// vidx is the index register for vector k (0..3) of a 64-byte block.
func vidx(k int) string { return [...]string{"R0", "R14", "R15", "R16"}[k] }

// gemmKernel builds gemmMicro8x8VSX(kc int, pa, pb, c *float64, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*8+r] * pb[p*8+col]   (r<8, col<8)
//
// with the scheme of OpenBLAS's POWER9 dgemm kernel: no splats. Each A pair
// (a_2p, a_2p+1) and its swap (one XXPERMDI) multiply each B column pair, so
// accumulator VS(8p+2k) holds the diagonal (C[2p][2k], C[2p+1][2k+1]) and
// VS(8p+2k+1) the anti-diagonal (C[2p+1][2k], C[2p][2k+1]); the store rebuilds
// the rows with XXPERMDI (DM=1). 32 fused multiply-adds per k step against 8
// loads and 4 swaps; two k steps per iteration, the second step's loads issued
// between the first step's FMAs. Measured on a POWER9 core (cfarm29): 16.9
// GFLOP/s, against 13.3 with LXVDSX splats, 14.8 with XXPERMDI splats and 15.4
// with those unrolled; OpenBLAS's own kernel reaches 23 there.
func gemmKernel() *emit.Function {
	name := "gemmMicro8x8VSX"
	sig := ppc64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]ppc64.Type{ppc64.Int64, ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("kc", "R3").LoadArg("pa", "R4").LoadArg("pb", "R5").LoadArg("c", "R6").LoadArg("ldc", "R7")
	b.Raw("SLD $3, R7")
	offsets(b, false)
	for r := 0; r < 32; r++ {
		b.Raw("XXLXOR VS%d, VS%d, VS%d", r, r, r)
	}
	b.Raw("CMP R3, $0").Raw("BEQ %sstore", name)
	// step uses B vectors bv..bv+3, A pairs av..av+3, swapped pairs sw..sw+3.
	loadsOf := func(bv, av int) []string {
		var l []string
		for k := 0; k < 4; k++ {
			l = append(l, fmt.Sprintf("LXVD2X (R5)(%s), VS%d", vidx(k), bv+k), fmt.Sprintf("LXVD2X (R4)(%s), VS%d", vidx(k), av+k))
		}
		return l
	}
	step := func(bv, av, sw int, pending []string) {
		for p := 0; p < 4; p++ {
			b.Raw("XXPERMDI VS%d, VS%d, $2, VS%d", av+p, av+p, sw+p)
		}
		i := 0
		for p := 0; p < 4; p++ {
			for k := 0; k < 4; k++ {
				b.XVMADDADP(8*p+2*k, av+p, bv+k)
				b.XVMADDADP(8*p+2*k+1, sw+p, bv+k)
				if i%2 == 1 && len(pending) > 0 {
					b.Raw("%s", pending[0])
					pending = pending[1:]
				}
				i++
			}
		}
		for _, l := range pending { // never leave a load out
			b.Raw("%s", l)
		}
	}
	b.Label(name+"loop2").Raw("CMP R3, $2").Raw("BLT %sone", name)
	for _, l := range loadsOf(32, 36) {
		b.Raw("%s", l)
	}
	b.Raw("ADD $64, R4").Raw("ADD $64, R5")
	step(32, 36, 40, loadsOf(44, 48))
	b.Raw("ADD $64, R4").Raw("ADD $64, R5")
	step(44, 48, 52, nil)
	b.Raw("ADD $-2, R3").Raw("BR %sloop2", name)
	b.Label(name+"one").Raw("CMP R3, $0").Raw("BEQ %sstore", name)
	for _, l := range loadsOf(32, 36) {
		b.Raw("%s", l)
	}
	step(32, 36, 40, nil)
	b.Label(name + "store")
	for p := 0; p < 4; p++ {
		for half := 0; half < 2; half++ { // row 2p, then 2p+1
			for k := 0; k < 4; k++ {
				d, a := 8*p+2*k, 8*p+2*k+1
				if half == 0 {
					b.Raw("XXPERMDI VS%d, VS%d, $1, VS%d", d, a, 56) // [diag.dw0, anti.dw1]
				} else {
					b.Raw("XXPERMDI VS%d, VS%d, $1, VS%d", a, d, 56) // [anti.dw0, diag.dw1]
				}
				b.Raw("LXVD2X (R6)(%s), VS%d", vidx(k), 57)
				b.XVADDDP(57, 57, 56)
				b.Raw("STXVD2X VS%d, (R6)(%s)", 57, vidx(k))
			}
			if !(p == 3 && half == 1) {
				b.Raw("ADD R7, R6")
			}
		}
	}
	b.Ret()
	return b.Func()
}

// gemm32Kernel builds gemmMicro8x16VSX(kc int, pa, pb, c *float32, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*8+r] * pb[p*16+col]   (r<8, col<16)
//
// with the scheme of OpenBLAS's POWER9 sgemm (kernel/power/sgemm_macros_power9.S,
// LOAD8x16): no splats. Each 4-float A vector is rotated three ways, and the
// rotations multiply each B vector, so the tile is accumulated in a rotated
// layout that the store untangles once. OpenBLAS rotates B with xxperm and
// xxpermdi; this kernel rotates A, which has two vectors per k step against
// B's four, with the cyclic word shift XXSLDWI a, a, $v: rotation v holds
// a[(i+v)%4] in lane i (lanes numbered as LXVW4X loads them, memory order).
//
// Accumulator VS(16p+4k+v) lane i is then C[4p+(i+v)%4][4k+i]: A vector p
// (rows 4p..4p+3), B vector k (columns 4k..4k+3), rotation v. Row 4p+r of
// that 4x4 block takes lane i from rotation (r-i) mod 4, so it is three
// selects: x = lanes 0,1 from rotations r, r-1; y = lanes 2,3 from r-2, r-3;
// row = x's low half and y's high half (XXSEL against the odd-lanes and
// high-lanes masks, VS48 and VS49).
//
// Per k step: 2 A loads, 6 rotations, 4 B loads, 32 XVMADDASP.
func gemm32Kernel(odd, high string) *emit.Function {
	name := "gemmMicro8x16VSX"
	sig := ppc64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]ppc64.Type{ppc64.Int64, ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("kc", "R3").LoadArg("pa", "R4").LoadArg("pb", "R5").LoadArg("c", "R6").LoadArg("ldc", "R7")
	b.Raw("SLD $2, R7")
	offsets(b, false)
	b.Raw("MOVD $%s+0(SB), R8", odd).Raw("LXVW4X (R8)(R0), VS48").
		Raw("MOVD $%s+0(SB), R8", high).Raw("LXVW4X (R8)(R0), VS49")
	for r := 0; r < 32; r++ {
		b.Raw("XXLXOR VS%d, VS%d, VS%d", r, r, r)
	}
	b.Raw("CMP R3, $0").Raw("BEQ %sstore", name)
	b.Label(name + "loop").
		Raw("LXVW4X (R4)(R0), VS32").Raw("LXVW4X (R4)(R14), VS36")
	for k := 0; k < 4; k++ {
		b.Raw("LXVW4X (R5)(%s), VS%d", vidx(k), 40+k)
	}
	for p := 0; p < 2; p++ {
		for v := 1; v < 4; v++ {
			b.Raw("XXSLDWI VS%d, VS%d, $%d, VS%d", 32+4*p, 32+4*p, v, 32+4*p+v)
		}
	}
	for p := 0; p < 2; p++ {
		for k := 0; k < 4; k++ {
			for v := 0; v < 4; v++ {
				b.XVMADDASP(16*p+4*k+v, 32+4*p+v, 40+k)
			}
		}
	}
	b.Raw("ADD $32, R4").Raw("ADD $64, R5").Raw("ADD $-1, R3").
		Raw("CMP R3, $0").Raw("BNE %sloop", name)
	b.Label(name + "store")
	acc := func(p, k, v int) int { return 16*p + 4*k + (v+4)%4 }
	for p := 0; p < 2; p++ {
		for r := 0; r < 4; r++ {
			for k := 0; k < 4; k++ {
				b.Raw("XXSEL VS%d, VS%d, VS48, VS44", acc(p, k, r), acc(p, k, r-1)).
					Raw("XXSEL VS%d, VS%d, VS48, VS45", acc(p, k, r-2), acc(p, k, r-3)).
					Raw("XXSEL VS44, VS45, VS49, VS46").
					Raw("LXVW4X (R6)(%s), VS47", vidx(k))
				b.XVADDSP(47, 47, 46)
				b.Raw("STXVW4X VS47, (R6)(%s)", vidx(k))
			}
			if !(p == 1 && r == 3) {
				b.Raw("ADD R7, R6")
			}
		}
	}
	b.Ret()
	return b.Func()
}

// words is a 16-byte mask with all-ones in the given 4-byte lanes (memory
// order, as LXVW4X loads them) and zeros elsewhere.
func words(on ...int) []byte {
	m := make([]byte, 16)
	for _, w := range on {
		for i := 0; i < 4; i++ {
			m[4*w+i] = 0xff
		}
	}
	return m
}
