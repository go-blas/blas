// Command gen produces sum_riscv64.s, the RVV (vector extension) float64
// kernels for riscv64, via go-asmgen. Run with: go run . (or `go generate`
// from the kernels package).
//
// Go's riscv64 assembler names the vector instructions these kernels need
// (VSETVLI, VLE64V/VSE64V, VFADDVV, VFSUBVV, VFMULVV, VFDIVVV, VFSQRTV,
// VFMACCVV, VFREDUSUMVS, VFMVSF/VFMVFS), so nothing is emitted as a WORD. The
// loops are vector-length agnostic: VSETVLI hands out as many elements as the
// hardware vector holds (LMUL=8 groups eight registers), including a short last
// strip, so there is no scalar tail. The V extension is optional, so the Go
// side calls these only when the kernel reports it (AT_HWCAP).
//
// Operand order, decoded from Go's own testdata encodings: VFADDVV V1, V2, V3
// computes V3 = V2 + V1 (likewise SUB and DIV: V3 = V2 - V1, V2 / V1);
// VFMACCVV V1, V2, V3 computes V3 += V2*V1; VFREDUSUMVS Vs, Vv, Vd computes
// Vd[0] = Vs[0] + sum(Vv).
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

// gemm8x16Unroll is how many k steps one pass of gemmMicro8x16RVV covers.
const gemm8x16Unroll = 2

func main() {
	f := emit.NewFile("riscv64")
	f.Add(gemmKernel())
	f.Add(gemm32Kernel())
	f.Add(gemm32Kernel8x16(gemm8x16Unroll))
	f.Add(gemmVLENBKernel())
	if err := os.WriteFile("sum_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_riscv64.s")
}

// gemmKernel builds gemmMicro6x8RVV(kc int, pa, pb, c *float64, ldc int) for
// kc >= 1: C[r][0:8] += sum_p pa[p*6+r] * pb[p*8:p*8+8] for r < 6, the rows of C
// ldc float64s apart.
//
// NR is fixed by the packing, while VLEN varies between RISC-V cores. The V
// extension guarantees VLEN >= 128, so a group of four registers (LMUL=4) holds
// at least eight float64: vl = 8 on every core. That leaves eight groups, six
// of them accumulators, one for the B row and one for C. Each step loads the B
// row once and fuses each A value in with vfmacc.vf (Vacc += f*Vb).
func gemmKernel() *emit.Function {
	sig := riscv64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]riscv64.Type{riscv64.Int64, riscv64.Ptr, riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc("gemmMicro6x8RVV", sig, 0)
	b.LoadArg("kc", "X10").LoadArg("pa", "X11").LoadArg("pb", "X12").
		LoadArg("c", "X13").LoadArg("ldc", "X14")
	b.Raw("VSETIVLI $8, E64, M4, TA, MA, X15")
	for r := 0; r < 6; r++ {
		b.Raw("VMVVI $0, V%d", 4*r)
	}
	b.Raw("loop:").Raw("VLE64V (X12), V24")
	for r := 0; r < 6; r++ {
		b.Raw("MOVD %d(X11), F%d", 8*r, r)
	}
	for r := 0; r < 6; r++ {
		b.Raw("VFMACCVF V24, F%d, V%d", r, 4*r) // acc_r += a_r * b
	}
	b.Raw("ADD $48, X11, X11").
		Raw("ADD $64, X12, X12").
		Raw("ADD $-1, X10, X10").
		Raw("BNEZ X10, loop").
		Raw("SLLI $3, X14, X14") // row stride in bytes
	for r := 0; r < 6; r++ {
		b.Raw("VLE64V (X13), V28").
			Raw("VFADDVV V%d, V28, V28", 4*r). // C row += acc_r
			Raw("VSE64V V28, (X13)")
		if r < 5 {
			b.Raw("ADD X14, X13, X13")
		}
	}
	return b.Ret().Func()
}

// gemm32Kernel builds gemmMicro6x16RVV(kc int, pa, pb, c *float32, ldc int) for
// kc >= 1: C[r][0:16] += sum_p pa[p*6+r] * pb[p*16:p*16+16] for r < 6, the rows
// of C ldc float32s apart. gemmKernel in float32: with E32 a group of four
// registers (LMUL=4) holds at least sixteen floats on any VLEN >= 128, so
// vl = 16 everywhere, and the same six accumulator groups, B group and C group
// fit.
func gemm32Kernel() *emit.Function {
	sig := riscv64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]riscv64.Type{riscv64.Int64, riscv64.Ptr, riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc("gemmMicro6x16RVV", sig, 0)
	b.LoadArg("kc", "X10").LoadArg("pa", "X11").LoadArg("pb", "X12").
		LoadArg("c", "X13").LoadArg("ldc", "X14")
	b.Raw("VSETIVLI $16, E32, M4, TA, MA, X15")
	for r := 0; r < 6; r++ {
		b.Raw("VMVVI $0, V%d", 4*r)
	}
	b.Raw("loop:").Raw("VLE32V (X12), V24")
	for r := 0; r < 6; r++ {
		b.Raw("MOVF %d(X11), F%d", 4*r, r)
	}
	for r := 0; r < 6; r++ {
		b.Raw("VFMACCVF V24, F%d, V%d", r, 4*r) // acc_r += a_r * b
	}
	b.Raw("ADD $24, X11, X11").
		Raw("ADD $64, X12, X12").
		Raw("ADD $-1, X10, X10").
		Raw("BNEZ X10, loop").
		Raw("SLLI $2, X14, X14") // row stride in bytes
	for r := 0; r < 6; r++ {
		b.Raw("VLE32V (X13), V28").
			Raw("VFADDVV V%d, V28, V28", 4*r). // C row += acc_r
			Raw("VSE32V V28, (X13)")
		if r < 5 {
			b.Raw("ADD X14, X13, X13")
		}
	}
	return b.Ret().Func()
}

// gemm32Kernel8x16 builds gemmMicro8x16RVV(kc int, pa, pb, c *float32, ldc int)
// for kc >= 1: C[r][0:16] += sum_p pa[p*8+r] * pb[p*16:p*16+16] for r < 8, the
// rows of C ldc float32s apart. It needs VLEN >= 256: with E32 and LMUL=2 a
// group of two registers holds sixteen floats only then (vl would be 8 on a
// 128-bit core), so the caller picks it by gemmVLENB. LMUL=4 with vl=16, as in
// gemm32Kernel, fills half of each group on a 256-bit core. Eight accumulator
// groups (V0..V15), `unroll` B groups from V16 and one C group (V28). Each pass
// loads the B rows and A scalars of `unroll` steps ahead of their FMAs, so the
// in-order core waits on loads once per pass; kc % unroll steps run first.
func gemm32Kernel8x16(unroll int) *emit.Function {
	sig := riscv64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]riscv64.Type{riscv64.Int64, riscv64.Ptr, riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc("gemmMicro8x16RVV", sig, 0)
	b.LoadArg("kc", "X10").LoadArg("pa", "X11").LoadArg("pb", "X12").
		LoadArg("c", "X13").LoadArg("ldc", "X14")
	b.Raw("VSETIVLI $16, E32, M2, TA, MA, X15")
	for r := 0; r < 8; r++ {
		b.Raw("VMVVI $0, V%d", 2*r)
	}
	// steps emits n steps: all loads first (B group V16+2s, A scalars F8s..),
	// then the 8n FMAs.
	steps := func(n int) {
		for s := 0; s < n; s++ {
			b.Raw("VLE32V (X12), V%d", 16+2*s).Raw("ADD $64, X12, X12")
		}
		for s := 0; s < n; s++ {
			for r := 0; r < 8; r++ {
				b.Raw("MOVF %d(X11), F%d", 32*s+4*r, 8*s+r)
			}
		}
		for s := 0; s < n; s++ {
			for r := 0; r < 8; r++ {
				b.Raw("VFMACCVF V%d, F%d, V%d", 16+2*s, 8*s+r, 2*r)
			}
		}
		b.Raw("ADD $%d, X11, X11", 32*n)
	}
	if unroll > 1 {
		b.Raw("ANDI $%d, X10, X5", unroll-1).Raw("SUB X5, X10, X10").Raw("BEQZ X5, main")
		b.Raw("tail:")
		steps(1)
		b.Raw("ADD $-1, X5, X5").Raw("BNEZ X5, tail")
		b.Raw("main:").Raw("BEQZ X10, epi")
	}
	b.Raw("loop:")
	steps(unroll)
	b.Raw("ADD $%d, X10, X10", -unroll).Raw("BNEZ X10, loop")
	b.Raw("epi:").Raw("SLLI $2, X14, X14") // row stride in bytes
	for r := 0; r < 8; r++ {
		b.Raw("VLE32V (X13), V28").
			Raw("VFADDVV V%d, V28, V28", 2*r). // C row += acc_r
			Raw("VSE32V V28, (X13)")
		if r < 7 {
			b.Raw("ADD X14, X13, X13")
		}
	}
	return b.Ret().Func()
}

// gemmVLENBKernel builds gemmVLENB() int, the vector register length in bytes:
// VSETVLI with AVL = VLMAX (rs1 = x0) at E8, LMUL=1 returns it in vl.
func gemmVLENBKernel() *emit.Function {
	sig := riscv64.Layout(nil, nil, []string{"ret"}, []riscv64.Type{riscv64.Int64})
	b := riscv64.NewFunc("gemmVLENB", sig, 0)
	b.Raw("VSETVLI X0, E8, M1, TA, MA, X5").StoreRet("X5", "ret")
	return b.Ret().Func()
}
