// Command gen produces sum_arm64.s, the NEON float64 sum-reduction kernel, via
// go-asmgen. Run with: go run gen.go (or `go generate` from the kernels
// package).
//
// NumPy's sum ufunc is single-threaded but SIMD-vectorized C with several
// accumulator lanes; Go's scalar reduction loop is a single dependency chain
// (`s += v`) the gc compiler will not auto-vectorize, because that would change
// floating-point grouping. So we hand-vectorize with NEON and multiple
// accumulators, the same parallelism numpy uses inside one core.
//
//	sumNEON(a *float64, n int) float64
//	  Sums a[0..n) with FOUR independent V.D2 accumulators (8 float64 lanes),
//	  then folds the lanes pairwise and adds a scalar tail. Floating-point
//	  addition is not associative, so this lane-parallel grouping can differ
//	  from a strictly sequential sum by a few ULP — exactly the trade-off
//	  numpy's own pairwise/SIMD summation makes. It is therefore NOT held
//	  bit-identical to the scalar oracle; the kernels package validates it to a
//	  tight relative tolerance against the oracle and against numpy.
//
// (No max-reduction kernel is generated here: Max and Min come from
// go-simd/floats, whose NEON extrema kernels (extrema_arm64.s there) use
// FMAX/FMIN, which propagate NaN and order -0 below +0 as Go's max/min do.)
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(gemmKernel())
	f.Add(gemm32Kernel())
	for mr := 1; mr <= 8; mr++ {
		f.Add(gemmUAKernel(mr, true))
		if mr <= 4 {
			f.Add(gemmUAKernel(mr, false))
		}
	}
	if err := os.WriteFile("sum_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_arm64.s")
}

// The elementwise kernels use the plain vector double ops VFADD/VFSUB/VFMUL/
// VFDIV, which Go's arm64 assembler names since Go 1.27 (this module requires
// 1.27.1). Before that only the fused VFMLA/VFMLS existed, and add/sub/mul went
// through an FMA against an exact constant (b + a*1.0, a - b*1.0, -0.0 + a*b),
// bit-identical but harder to read, while division stayed scalar. Each lane is
// one correctly rounded IEEE-754 operation, the same the scalar FADDD/FSUBD/
// FMULD/FDIVD perform, so the kernels are bit-identical to the scalar oracle
// (TestBinSIMD). Go's operand order is VFSUB Vm, Vn, Vd: Vd = Vn - Vm.
//
// The loop does 8 doubles (64 bytes) per iteration across four D2 lanes, then a
// scalar FP tail for the (n mod 8) remainder.

// gemmSig is the micro-kernel signature
//
//	gemmMicro4x8(kc int, pa, pb, c *float64, ldc int)
func gemmSig() arm64.Signature {
	return arm64.Layout(
		[]string{"kc", "pa", "pb", "c", "ldc"},
		[]arm64.Type{arm64.Int64, arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64},
		nil, nil,
	)
}

// fmlaElem returns the Plan 9 line for the by-element (indexed) double FMLA
//
//	FMLA V{acc}.2D, V{breg}.2D, V{areg}.D[idx]   =>  V{acc} += V{breg} * V{areg}[idx]
//
// which Go's arm64 assembler cannot name, emitted as its raw 32-bit encoding via
// WORD. The encoding (Advanced SIMD, FMLA by element, FP64, ARM C7-2: Q=1, sz=1,
// L=0) is
//
//	0|1|0|01111|1|1|0|M|Rm[3:0]|0001|H|0|Rn|Rd
//
// with the element index = H (sz=1 so L is unused) and Rm = M:Rm[3:0]. The result
// includes a trailing comment naming the human-readable instruction so the
// generated .s stays auditable; the encoding is cross-checked against objdump in
// the asmgen tests. areg/breg/acc are V-register numbers, idx in {0,1}.
func fmlaElem(acc, breg, areg, idx int) string {
	const base = 0x4FC01000 // Q=1, sz=1, opcode 0001, base for FMLA-by-elem .2D
	M := (areg >> 4) & 1
	Rm := areg & 0xF
	H := idx & 1
	word := base | (M << 20) | (Rm << 16) | (H << 11) | (breg << 5) | acc
	return fmt.Sprintf("WORD $0x%08X // FMLA V%d.2D, V%d.2D, V%d.D[%d]",
		word, acc, breg, areg, idx)
}

// gemmKernel builds gemmMicro4x8(kc int, pa, pb, c *float64, ldc int): the NEON
// 4x8 register-blocked GEMM micro-kernel at the heart of the packed GEMM.
//
// It computes a 4-row x 8-col tile of C from packed panels and ADDS it into C:
//
//	for p in [0,kc):  C[r][col] += pa[p*4+r] * pb[p*8+col]   (r<4, col<8)
//
// pa is the packed A panel (MR=4 contiguous A values per k step), pb the packed
// B panel (NR=8 contiguous B values per k step) — both unit-stride, so the loads
// are conflict-free regardless of the source matrices' strides. This contiguity
// is the whole point of packing: it removes the power-of-two L1 set-conflicts
// that defeated the earlier un-packed register-blocked attempt.
//
// Register map: the 4x8 C tile lives in V0..V15 (4 rows x 4 D2 = 16 accumulators);
// the per-step B row in V16..V19 (8 doubles), the four A values in V20,V21 (two
// D2 vectors, two A doubles each), and a vector of 1.0 in V31 for the final
// C += acc.
//
// Two arm64-assembler constraints shape the kernel (both noted at point of use):
//  1. THE BY-LANE FMLA. Every hand-tuned arm64 dgemm broadcasts one A lane
//     straight into the FMA with FMLA Vd.2d, Vn.2d, Vm.d[i]. Go's arm64 assembler
//     cannot *name* this indexed-element form ("illegal combination ... ELEM"),
//     so it is emitted by its raw WORD encoding (fmlaElem below: cross-checked
//     against objdump). The four packed A doubles of a k-step are loaded as two
//     D2 vectors (V20=[a0,a1], V21=[a2,a3]) and each row's FMA reads its A scalar
//     by lane — eliminating the MOVD-to-GP + VDUP round-trip the prior kernel
//     paid per A value every k-step. Measured on this Apple-silicon core this is
//     ~1.28x faster on the L1-resident micro-kernel (44 -> 58 GFLOP/s/core);
//     the earlier note claiming the indexed form is "slower" was wrong (it was
//     never actually measured against a correct encoding). See docs/perf.md.
//  2. The closing C += acc is done as C = C + acc*1.0 via VFMLA against
//     V31=[1.0,1.0], written when Go's assembler had no VFADD (it has since 1.27). Multiplying by
//     exactly 1.0 is exact and FMA(acc,1.0,C) rounds identically to C+acc (C is
//     the only inexact addend), so this is a true add — the tile result is
//     bit-identical to the scalar oracle's ikj accumulation order.
func gemmKernel() *emit.Function {
	b := arm64.NewFunc("gemmMicro4x8", gemmSig(), 0)
	b.LoadArg("kc", "R0").
		LoadArg("pa", "R1").
		LoadArg("pb", "R2").
		LoadArg("c", "R3").
		LoadArg("ldc", "R4").
		Raw("LSL $3, R4, R4"). // ldc -> bytes
		// Zero the sixteen C accumulators V0..V15.
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("VEOR V1.B16, V1.B16, V1.B16").
		Raw("VEOR V2.B16, V2.B16, V2.B16").
		Raw("VEOR V3.B16, V3.B16, V3.B16").
		Raw("VEOR V4.B16, V4.B16, V4.B16").
		Raw("VEOR V5.B16, V5.B16, V5.B16").
		Raw("VEOR V6.B16, V6.B16, V6.B16").
		Raw("VEOR V7.B16, V7.B16, V7.B16").
		Raw("VEOR V8.B16, V8.B16, V8.B16").
		Raw("VEOR V9.B16, V9.B16, V9.B16").
		Raw("VEOR V10.B16, V10.B16, V10.B16").
		Raw("VEOR V11.B16, V11.B16, V11.B16").
		Raw("VEOR V12.B16, V12.B16, V12.B16").
		Raw("VEOR V13.B16, V13.B16, V13.B16").
		Raw("VEOR V14.B16, V14.B16, V14.B16").
		Raw("VEOR V15.B16, V15.B16, V15.B16").
		Raw("CBZ R0, gstore").
		Raw("gkloop:").
		// B row: 8 contiguous doubles into V16..V19.
		Raw("VLD1.P 64(R2), [V16.D2, V17.D2, V18.D2, V19.D2]").
		// A col: 4 contiguous doubles into two D2 vectors V20=[a0,a1], V21=[a2,a3].
		// Each row's FMA reads its A scalar by lane (header note 1) — no VDUP.
		Raw("VLD1.P 32(R1), [V20.D2, V21.D2]").
		// FMLA Vacc.2D, Vb.2D, Va.D[i] => Vacc += Vb * Va[i].  row r uses A lane
		// (r&1) of V20 (r<2) or V21 (r>=2).
		Raw(fmlaElem(0, 16, 20, 0)). // row0 = a0
		Raw(fmlaElem(1, 17, 20, 0)).
		Raw(fmlaElem(2, 18, 20, 0)).
		Raw(fmlaElem(3, 19, 20, 0)).
		Raw(fmlaElem(4, 16, 20, 1)). // row1 = a1
		Raw(fmlaElem(5, 17, 20, 1)).
		Raw(fmlaElem(6, 18, 20, 1)).
		Raw(fmlaElem(7, 19, 20, 1)).
		Raw(fmlaElem(8, 16, 21, 0)). // row2 = a2
		Raw(fmlaElem(9, 17, 21, 0)).
		Raw(fmlaElem(10, 18, 21, 0)).
		Raw(fmlaElem(11, 19, 21, 0)).
		Raw(fmlaElem(12, 16, 21, 1)). // row3 = a3
		Raw(fmlaElem(13, 17, 21, 1)).
		Raw(fmlaElem(14, 18, 21, 1)).
		Raw(fmlaElem(15, 19, 21, 1)).
		Raw("SUB $1, R0").
		Raw("CBNZ R0, gkloop").
		// C += acc, row by row, via VFMLA against V31=[1.0,1.0]: exact, so
		// bit-identical to C+acc (written before Go 1.27 named VFADD).
		Raw("gstore:").
		Raw("MOVD $0x3FF0000000000000, R6").
		Raw("VDUP R6, V31.D2").
		Raw("MOVD R3, R5"). // row0
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V0.D2, V31.D2, V16.D2").
		Raw("VFMLA V1.D2, V31.D2, V17.D2").
		Raw("VFMLA V2.D2, V31.D2, V18.D2").
		Raw("VFMLA V3.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Raw("ADD R4, R3, R3"). // row1
		Raw("MOVD R3, R5").
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V4.D2, V31.D2, V16.D2").
		Raw("VFMLA V5.D2, V31.D2, V17.D2").
		Raw("VFMLA V6.D2, V31.D2, V18.D2").
		Raw("VFMLA V7.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Raw("ADD R4, R3, R3"). // row2
		Raw("MOVD R3, R5").
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V8.D2, V31.D2, V16.D2").
		Raw("VFMLA V9.D2, V31.D2, V17.D2").
		Raw("VFMLA V10.D2, V31.D2, V18.D2").
		Raw("VFMLA V11.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Raw("ADD R4, R3, R3"). // row3
		Raw("MOVD R3, R5").
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V12.D2, V31.D2, V16.D2").
		Raw("VFMLA V13.D2, V31.D2, V17.D2").
		Raw("VFMLA V14.D2, V31.D2, V18.D2").
		Raw("VFMLA V15.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Ret()
	return b.Func()
}

// gemmUAKernel builds the unpacked-A tiles (the "rv" sup kernels of BLIS,
// bli_gemmsup_rv_*): A is read in place, mr rows lda elements apart, instead
// of from a packed panel; B stays packed.
//
//	single: gemmUA<mr>x8S(kc int, a *float32, lda int, pb, c *float32, ldc int), mr <= 8
//	double: gemmUA<mr>x8D(kc int, a *float64, lda int, pb, c *float64, ldc int), mr <= 4
//
// computing C[r][col] += a[r*lda+p] * pb[p*8+col] for r < mr. Each row has its
// own pointer (R5..R12). A row is contiguous along k, so one plain vector load
// brings U consecutive k values of it (U = 4 floats or 2 doubles) into V20+r,
// and the U k steps that follow take their A value from lane s of that vector
// with the by-element FMLA (fmlaElem/fmlaElemS), as the packed kernels do.
// The kc%U remaining steps load each row's value as a scalar into lane 0.
// A first version loaded every A value with a load-and-replicate (VLD1R) and
// used the vector FMLA: on a Neoverse N1 it was 26-78% slower than packing A,
// apparently because the replicating loads compete with the FMAs for the
// vector pipes.
// Accumulators: float32 row r in V(2r), V(2r+1); float64 row r in V(4r)..V(4r+3).
func gemmUAKernel(mr int, single bool) *emit.Function {
	sig := arm64.Layout(
		[]string{"kc", "a", "lda", "pb", "ldb", "c", "ldc"},
		[]arm64.Type{arm64.Int64, arm64.Ptr, arm64.Int64, arm64.Ptr, arm64.Int64, arm64.Ptr, arm64.Int64},
		nil, nil,
	)
	name, shift, arr, vecs := fmt.Sprintf("gemmUA%dx8D", mr), 3, "D2", 4
	if single {
		name, shift, arr, vecs = fmt.Sprintf("gemmUA%dx8S", mr), 2, "S4", 2
	}
	size := 1 << shift
	bregs := make([]string, vecs)
	for v := range bregs {
		bregs[v] = fmt.Sprintf("V%d.%s", 16+v, arr)
	}
	blist := "[" + strings.Join(bregs, ", ") + "]"
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("kc", "R0").
		LoadArg("a", "R1").
		LoadArg("lda", "R2").
		LoadArg("pb", "R13").
		LoadArg("ldb", "R16").
		LoadArg("c", "R3").
		LoadArg("ldc", "R4").
		Raw(fmt.Sprintf("LSL $%d, R2, R2", shift)).
		Raw(fmt.Sprintf("LSL $%d, R16, R16", shift)). // B rows ldb elements apart (BLIS's b += rs_b)
		Raw(fmt.Sprintf("LSL $%d, R4, R4", shift))
	for r := 0; r < mr; r++ { // row pointers R5.. = a + r*lda
		if r == 0 {
			b.Raw("MOVD R1, R5")
		} else {
			b.Raw(fmt.Sprintf("ADD R2, R%d, R%d", 4+r, 5+r))
		}
	}
	for v := 0; v < mr*vecs; v++ {
		b.Raw(fmt.Sprintf("VEOR V%d.B16, V%d.B16, V%d.B16", v, v, v))
	}
	u := 16 / size // k steps per A vector
	fmla, scalar := fmlaElem, "FMOVD"
	if single {
		fmla, scalar = fmlaElemS, "FMOVS"
	}
	step := func(lane int) {
		b.Raw(fmt.Sprintf("VLD1.P (R13)(R16), %s", blist)) // post-increment by ldb
		for r := 0; r < mr; r++ {
			for v := 0; v < vecs; v++ {
				b.Raw(fmla(vecs*r+v, 16+v, 20+r, lane))
			}
		}
	}
	b.Raw(fmt.Sprintf("LSR $%d, R0, R14", shift4(u))). // whole groups of u steps
								Raw(fmt.Sprintf("AND $%d, R0, R15", u-1)). // remaining steps
								Raw("CBZ R14, uatail").
								Raw("uakloop:")
	for r := 0; r < mr; r++ {
		b.Raw(fmt.Sprintf("VLD1.P 16(R%d), [V%d.%s]", 5+r, 20+r, arr))
	}
	for s := 0; s < u; s++ {
		step(s)
	}
	b.Raw("SUB $1, R14").
		Raw("CBNZ R14, uakloop").
		Raw("uatail:").
		Raw("CBZ R15, uastore").
		Raw("uatloop:")
	for r := 0; r < mr; r++ {
		b.Raw(fmt.Sprintf("%s.P %d(R%d), F%d", scalar, size, 5+r, 20+r))
	}
	step(0)
	b.Raw("SUB $1, R15").
		Raw("CBNZ R15, uatloop").
		Raw("uastore:")
	for r := 0; r < mr; r++ {
		b.Raw(fmt.Sprintf("VLD1 (R3), %s", blist))
		for v := 0; v < vecs; v++ {
			b.Raw(fmt.Sprintf("VFADD V%d.%s, V%d.%s, V%d.%s", vecs*r+v, arr, 16+v, arr, 16+v, arr))
		}
		b.Raw(fmt.Sprintf("VST1 %s, (R3)", blist))
		if r < mr-1 {
			b.Raw("ADD R4, R3, R3")
		}
	}
	b.Ret()
	return b.Func()
}

// shift4 is log2 of u, 2 or 4.
func shift4(u int) int {
	if u == 4 {
		return 2
	}
	return 1
}

// fmlaElemS is fmlaElem for single precision:
//
//	FMLA V{acc}.4S, V{breg}.4S, V{areg}.S[idx]   =>  V{acc} += V{breg} * V{areg}[idx]
//
// The same Advanced SIMD "FMLA (by element)" encoding with sz=0 (ARM C7-2):
//
//	0|1|0|01111|1|0|L|M|Rm[3:0]|0001|H|0|Rn|Rd
//
// where the element index is H:L and, for sz=0, the register is M:Rm, so
// areg may be any of V0..V31. idx in {0,1,2,3}.
func fmlaElemS(acc, breg, areg, idx int) string {
	const base = 0x4F801000 // Q=1, sz=0, opcode 0001: FMLA by element .4S
	M := (areg >> 4) & 1
	Rm := areg & 0xF
	H, L := (idx>>1)&1, idx&1
	word := base | (L << 21) | (M << 20) | (Rm << 16) | (H << 11) | (breg << 5) | acc
	return fmt.Sprintf("WORD $0x%08X // FMLA V%d.4S, V%d.4S, V%d.S[%d]",
		word, acc, breg, areg, idx)
}

// gemm32Kernel builds gemmMicro8x8S(kc int, pa, pb, c *float32, ldc int), the
// float32 micro-kernel of the packed GEMM: an 8-row x 8-col tile,
//
//	for p in [0,kc):  C[r][col] += pa[p*8+r] * pb[p*8+col]   (r<8, col<8)
//
// the float64 kernel's structure at twice the lanes. The tile lives in V0..V15
// (row r in V(2r), V(2r+1), four floats each), the B row in V16,V17 and the
// eight A values in V18=[a0..a3], V19=[a4..a7], which the by-element FMLA
// (fmlaElemS) reads lane by lane. Sixteen independent accumulator chains cover
// the FMA latency of two 4-cycle pipes (Neoverse N1) with room to spare; each
// k step is two loads for sixteen FMAs. The closing C += acc is a VFADD.
func gemm32Kernel() *emit.Function {
	b := arm64.NewFunc("gemmMicro8x8S", gemmSig(), 0)
	b.LoadArg("kc", "R0").
		LoadArg("pa", "R1").
		LoadArg("pb", "R2").
		LoadArg("c", "R3").
		LoadArg("ldc", "R4").
		Raw("LSL $2, R4, R4") // ldc -> bytes
	for r := 0; r < 16; r++ {
		b.Raw(fmt.Sprintf("VEOR V%d.B16, V%d.B16, V%d.B16", r, r, r))
	}
	b.Raw("CBZ R0, sstore").
		Raw("skloop:").
		Raw("VLD1.P 32(R2), [V16.S4, V17.S4]"). // B row: 8 floats
		Raw("VLD1.P 32(R1), [V18.S4, V19.S4]")  // A column: 8 floats
	for r := 0; r < 8; r++ {
		areg, lane := 18+r/4, r%4
		b.Raw(fmlaElemS(2*r, 16, areg, lane))
		b.Raw(fmlaElemS(2*r+1, 17, areg, lane))
	}
	b.Raw("SUB $1, R0").
		Raw("CBNZ R0, skloop").
		Raw("sstore:")
	for r := 0; r < 8; r++ {
		b.Raw("VLD1 (R3), [V16.S4, V17.S4]")
		b.Raw(fmt.Sprintf("VFADD V%d.S4, V16.S4, V16.S4", 2*r))
		b.Raw(fmt.Sprintf("VFADD V%d.S4, V17.S4, V17.S4", 2*r+1))
		b.Raw("VST1 [V16.S4, V17.S4], (R3)")
		if r < 7 {
			b.Raw("ADD R4, R3, R3")
		}
	}
	b.Ret()
	return b.Func()
}
