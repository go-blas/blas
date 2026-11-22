// Command gen produces sum_amd64.s, the SSE2 float64 kernels, via go-asmgen.
// Run with: go run gen.go (or `go generate` from the kernels package).
//
//	sumSSE2(a *float64, n int) float64
//	  Sums a[0..n) with FOUR independent XMM accumulators (8 float64 lanes via
//	  2 doubles per register), then folds them and adds a scalar tail. SSE2 is
//	  part of the amd64 baseline (GOAMD64=v1), so the kernel is always callable
//	  with no CPU-feature branch. The lane-parallel grouping is a
//	  different-but-valid summation order, so it is validated to a tight relative
//	  tolerance against the scalar oracle (and numpy), not held bit-identical.
//
//	sqrtSSE2(dst, src *float64, n int)
//	  Writes sqrt(src[i]) into dst[i] with packed SQRTPD (2 doubles per op,
//	  unrolled 4x = 8 lanes/iter) plus a scalar SQRTSD tail. SQRTPD is the same
//	  correctly-rounded IEEE square root math.Sqrt computes, so this is
//	  bit-identical to the scalar oracle including NaN/Inf/signed-zero.
//
//	maxSSE2(a *float64, n int) float64 / minSSE2(a *float64, n int) float64
//	  NaN-propagating max/min reduction (numpy.max / Go math.Max semantics): if
//	  any element is NaN the result is NaN, else the largest/smallest. Packed
//	  MAXPD/MINPD give the extreme over non-NaN data; a parallel CMPPD-unordered
//	  ($3) OR-accumulator records whether ANY lane ever held a NaN, and the fold
//	  forces the result to NaN if so. MAXPD/MINPD's own (non-propagating) NaN
//	  rule is irrelevant because the NaN mask overrides the value in that case.
//	  Accumulators are seeded by broadcasting a[0] (MOVDDUP), so n>=1 (the
//	  caller guarantees non-empty) needs no -Inf/+Inf constant. Bit-identical to
//	  the scalar Max/Min oracle.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(gemmFMAKernel())
	f.Add(gemm32FMAKernel())
	for mr := 1; mr <= gemmMR; mr++ {
		f.Add(gemm32UAKernel(mr))
		f.Add(gemmUAKernel(mr, false))
	}
	f.Add(gemmSSE2HalfKernel())
	f.Add(amd64.FeatureProbe("hasFMA", amd64.FMA))
	// The float32 max32SSE2/min32SSE2 kernels moved to go-simd/floats (Float32Max,
	// Float32Min; go-ndarray/ndarray#98).

	if err := os.WriteFile("sum_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_amd64.s")
}

// gemmMR, gemmNR are the micro-tile the packing in gemm.go produces on amd64:
// 6 rows of packed A per k step, 8 columns of packed B.
const (
	gemmMR = 6
	gemmNR = 8
)

func gemmSig() amd64.Signature {
	return amd64.Layout(
		[]string{"kc", "pa", "pb", "c", "ldc"},
		[]amd64.Type{amd64.Int64, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
}

// gemmFMAKernel builds gemmMicro6x8FMA(kc int, pa, pb, c *float64, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*6+r] * pb[p*8+col]   (r<6, col<8)
//
// the BLIS Haswell dgemm shape (6x8, row-major here). Each k step loads the
// 8-double B row into two YMM, broadcasts each of the 6 A values with
// VBROADCASTSD straight from the panel, and issues 12 VFMADD231PD into the 12
// accumulators Y0..Y11 (row r in Y(2r), Y(2r+1)). Twelve independent chains
// cover the FMA latency on both ports (Haswell: 5 cycles x 2 ports needs >= 10),
// and 8 loads per 12 FMAs stay under the 2-loads-per-cycle budget, so the loop
// is FMA-throughput bound — the property that makes it a real dgemm kernel.
//
// Fused: each step rounds a*b+c once, as tuned BLAS (and gc's own scalar code at
// GOAMD64=v3, and arm64's) do. The SSE2 fallback below does not fuse, so on
// non-integer data the two can differ in the last place; both are the ikj sum.
// The caller gates this kernel on hasFMA (CPUID FMA+AVX+OSXSAVE and XCR0 YMM).
func gemmFMAKernel() *emit.Function {
	b := amd64.NewFunc("gemmMicro6x8FMA", gemmSig(), 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("pa", "SI") // A panel
	b.LoadArg("pb", "DI") // B panel
	b.LoadArg("c", "DX")  // C base
	b.LoadArg("ldc", "R8")
	b.Raw("SHLQ $3, R8") // ldc -> bytes
	// Prefetch the C tile now: its rows are ldc apart, so they are 6 different
	// cache lines (12 when a row straddles two), usually not in L1, and the
	// store at the end would otherwise stall on each of them in turn. Issued
	// first, they arrive while the k loop runs. The last byte of each row is
	// fetched too, for a row that does not start on a line boundary.
	b.Raw("MOVQ DX, R9")
	for r := 0; r < gemmMR; r++ {
		b.Raw("PREFETCHT0 (R9)")
		b.Raw("PREFETCHT0 63(R9)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, R9")
		}
	}
	for r := 0; r < 12; r++ {
		b.Raw("VXORPD Y%d, Y%d, Y%d", r, r, r)
	}
	// Unrolled by two k steps, with a one-step tail. The A panel streams from
	// L2 (the B micro-panel stays in L1 across the whole A block), so each pair
	// of steps prefetches the A line 8 steps ahead.
	b.Raw("MOVQ CX, R10")
	b.Raw("SHRQ $1, R10")
	b.Raw("JZ ftail")
	b.Raw("fkloop:")
	b.Raw("PREFETCHT0 %d(SI)", 8*gemmMR*8)
	fmaStep(b, 0)
	fmaStep(b, 1)
	b.Raw("ADDQ $%d, SI", 2*8*gemmMR)
	b.Raw("ADDQ $%d, DI", 2*8*gemmNR)
	b.Raw("DECQ R10")
	b.Raw("JNZ fkloop")
	b.Raw("ftail:")
	b.Raw("TESTQ $1, CX")
	b.Raw("JZ fstore")
	fmaStep(b, 0)
	b.Raw("fstore:")
	for r := 0; r < gemmMR; r++ {
		b.Raw("VADDPD (DX), Y%d, Y15", 2*r)
		b.Raw("VMOVUPD Y15, (DX)")
		b.Raw("VADDPD 32(DX), Y%d, Y15", 2*r+1)
		b.Raw("VMOVUPD Y15, 32(DX)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, DX")
		}
	}
	b.Raw("VZEROUPPER") // no AVX-SSE transition penalty in the Go code after
	b.Ret()
	return b.Func()
}

// gemm32NR is the float32 tile width on amd64: two YMM of eight floats.
const gemm32NR = 16

// gemm32FMAKernel builds gemmMicro6x16FMA(kc int, pa, pb, c *float32, ldc int),
// the float32 micro-kernel of the packed GEMM:
//
//	for p in [0,kc):  C[r][col] += pa[p*6+r] * pb[p*16+col]   (r<6, col<16)
//
// gemmFMAKernel at twice the lanes, the BLIS Haswell sgemm shape: the 16-float
// B row in Y12,Y13, each of the six A values broadcast with VBROADCASTSS, and
// twelve VFMADD231PS into Y0..Y11 (row r in Y(2r), Y(2r+1)). Gated on hasFMA.
func gemm32FMAKernel() *emit.Function {
	b := amd64.NewFunc("gemmMicro6x16FMA", gemmSig(), 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("pa", "SI")
	b.LoadArg("pb", "DI")
	b.LoadArg("c", "DX")
	b.LoadArg("ldc", "R8")
	b.Raw("SHLQ $2, R8") // ldc -> bytes
	// Prefetch the C tile: six rows of 64 bytes, each on one or two lines.
	b.Raw("MOVQ DX, R9")
	for r := 0; r < gemmMR; r++ {
		b.Raw("PREFETCHT0 (R9)")
		b.Raw("PREFETCHT0 63(R9)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, R9")
		}
	}
	for r := 0; r < 12; r++ {
		b.Raw("VXORPS Y%d, Y%d, Y%d", r, r, r)
	}
	b.Raw("MOVQ CX, R10")
	b.Raw("SHRQ $1, R10")
	b.Raw("JZ stail")
	b.Raw("skloop:")
	b.Raw("PREFETCHT0 %d(SI)", 16*gemmMR*4)
	fma32Step(b, 0)
	fma32Step(b, 1)
	b.Raw("ADDQ $%d, SI", 2*4*gemmMR)
	b.Raw("ADDQ $%d, DI", 2*4*gemm32NR)
	b.Raw("DECQ R10")
	b.Raw("JNZ skloop")
	b.Raw("stail:")
	b.Raw("TESTQ $1, CX")
	b.Raw("JZ sstore")
	fma32Step(b, 0)
	b.Raw("sstore:")
	for r := 0; r < gemmMR; r++ {
		b.Raw("VADDPS (DX), Y%d, Y15", 2*r)
		b.Raw("VMOVUPS Y15, (DX)")
		b.Raw("VADDPS 32(DX), Y%d, Y15", 2*r+1)
		b.Raw("VMOVUPS Y15, 32(DX)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, DX")
		}
	}
	b.Raw("VZEROUPPER")
	b.Ret()
	return b.Func()
}

// gemm32UAKernel builds gemmUA<mr>x16FMA(kc int, a *float32, lda int, pb, c
// *float32, ldc int): the 6x16 float32 tile with A read in place instead of
// packed, for mr rows (1..6):
//
//	for p in [0,kc):  C[r][col] += a[r*lda+p] * pb[p*16+col]   (r<mr, col<16)
//
// It is the "rv" sup kernel of BLIS (bli_gemmsup_rv_haswell_asm_s6x16m.c):
// each A value is broadcast straight from its row, r*lda bytes past the
// k-step's column, through a scaled index (lda, 2lda, 3lda in R11, 4lda,
// 5lda in R12), and the column advances 4 bytes per step. A small product
// then pays no copy of A at all. The kernels for mr < 6 cover the bottom rows
// of a block without reading past A's last row.
//
// B's rows are ldb elements apart (R13 in bytes), as BLIS's b += rs_b: a
// packed panel passes its own width (16 or 8), and a full panel of a
// row-major B is read in place with ldb its row stride. A partial last panel
// is still packed, for its zero padding.
//
// gemmUAKernel(mr, false) is the float64 counterpart over the 6x8 packing,
// gemmUA<mr>x8FMA: the same loop in PD instructions, 8-byte steps.
func gemm32UAKernel(mr int) *emit.Function { return gemmUAKernel(mr, true) }

func gemmUAKernel(mr int, single bool) *emit.Function {
	sig := amd64.Layout(
		[]string{"kc", "a", "lda", "pb", "ldb", "c", "ldc"},
		[]amd64.Type{amd64.Int64, amd64.Ptr, amd64.Int64, amd64.Ptr, amd64.Int64, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	name, size, shift, sfx, bcast := fmt.Sprintf("gemmUA%dx8FMA", mr), 8, 3, "PD", "VBROADCASTSD"
	if single {
		name, size, shift, sfx, bcast = fmt.Sprintf("gemmUA%dx16FMA", mr), 4, 2, "PS", "VBROADCASTSS"
	}
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("a", "SI")
	b.LoadArg("lda", "R8")
	b.LoadArg("pb", "DI")
	b.LoadArg("ldb", "R13")
	b.LoadArg("c", "DX")
	b.LoadArg("ldc", "R9")
	b.Raw("SHLQ $%d, R8", shift)  // lda -> bytes
	b.Raw("SHLQ $%d, R13", shift) // ldb -> bytes
	b.Raw("SHLQ $%d, R9", shift)  // ldc -> bytes
	b.Raw("LEAQ (R8)(R8*2), R11") // 3*lda
	b.Raw("LEAQ (R8)(R8*4), R12") // 5*lda
	b.Raw("MOVQ DX, R10")
	for r := 0; r < mr; r++ {
		b.Raw("PREFETCHT0 (R10)")
		b.Raw("PREFETCHT0 63(R10)")
		if r < mr-1 {
			b.Raw("ADDQ R9, R10")
		}
	}
	for r := 0; r < 2*mr; r++ {
		b.Raw("VXOR%s Y%d, Y%d, Y%d", sfx, r, r, r)
	}
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ uastore")
	b.Raw("uakloop:")
	b.Raw("VMOVU%s (DI), Y12", sfx)
	b.Raw("VMOVU%s 32(DI), Y13", sfx)
	for r := 0; r < mr; r++ {
		b.Raw("%s %s, Y14", bcast, uaRow(r))
		b.Raw("VFMADD231%s Y12, Y14, Y%d", sfx, 2*r)
		b.Raw("VFMADD231%s Y13, Y14, Y%d", sfx, 2*r+1)
	}
	b.Raw("ADDQ $%d, SI", size)
	b.Raw("ADDQ R13, DI") // next B row: ldb bytes on
	b.Raw("DECQ CX")
	b.Raw("JNZ uakloop")
	b.Raw("uastore:")
	for r := 0; r < mr; r++ {
		b.Raw("VADD%s (DX), Y%d, Y15", sfx, 2*r)
		b.Raw("VMOVU%s Y15, (DX)", sfx)
		b.Raw("VADD%s 32(DX), Y%d, Y15", sfx, 2*r+1)
		b.Raw("VMOVU%s Y15, 32(DX)", sfx)
		if r < mr-1 {
			b.Raw("ADDQ R9, DX")
		}
	}
	b.Raw("VZEROUPPER")
	b.Ret()
	return b.Func()
}

// uaRow is the address of row r of the unpacked A column at SI, rows lda
// bytes apart (R8), 3*lda in R11 and 5*lda in R12.
func uaRow(r int) string {
	return [...]string{"(SI)", "(SI)(R8*1)", "(SI)(R8*2)", "(SI)(R11*1)", "(SI)(R8*4)", "(SI)(R12*1)"}[r]
}

// fma32Step emits one k step of the 6x16 float32 FMA tile.
func fma32Step(b *amd64.Builder, step int) {
	ao, bo := 4*gemmMR*step, 4*gemm32NR*step
	b.Raw("VMOVUPS %d(DI), Y12", bo)    // B cols 0..7
	b.Raw("VMOVUPS %d(DI), Y13", bo+32) // B cols 8..15
	for r := 0; r < gemmMR; r++ {
		b.Raw("VBROADCASTSS %d(SI), Y14", ao+4*r)
		b.Raw("VFMADD231PS Y12, Y14, Y%d", 2*r)
		b.Raw("VFMADD231PS Y13, Y14, Y%d", 2*r+1)
	}
}

// fmaStep emits one k step of the 6x8 FMA tile, step reading the A and B
// values at that many steps past SI and DI.
func fmaStep(b *amd64.Builder, step int) {
	ao, bo := 8*gemmMR*step, 8*gemmNR*step
	b.Raw("VMOVUPD %d(DI), Y12", bo)    // B cols 0..3
	b.Raw("VMOVUPD %d(DI), Y13", bo+32) // B cols 4..7
	for r := 0; r < gemmMR; r++ {
		b.Raw("VBROADCASTSD %d(SI), Y14", ao+8*r)
		b.Raw("VFMADD231PD Y12, Y14, Y%d", 2*r)
		b.Raw("VFMADD231PD Y13, Y14, Y%d", 2*r+1)
	}
}

// gemmSSE2HalfKernel builds gemmMicro6x4SSE2(kc int, pa, pb, c *float64, ldc
// int), the baseline (GOAMD64=v1, no FMA) fallback over the same 6x8 packing:
// it computes the 6-row x 4-col half of the tile whose B columns start at pb,
// so the Go side calls it twice (pb, c) and (pb+4, c+4). A whole 6x8 tile in
// XMM would need 24 accumulators; the half needs 12 (X0..X11, row r in X(2r),
// X(2r+1)) plus the B half-row X12,X13, the broadcast A X14 and a product X15.
// The B stride is still the full NR=8 doubles per k step.
//
// SSE2 has no fused multiply-add, so each step is MULPD then ADDPD: the
// running sum is the plain dst += a*b in ikj order, like the scalar oracle at
// GOAMD64=v1.
func gemmSSE2HalfKernel() *emit.Function {
	b := amd64.NewFunc("gemmMicro6x4SSE2", gemmSig(), 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("pa", "SI")
	b.LoadArg("pb", "DI")
	b.LoadArg("c", "DX")
	b.LoadArg("ldc", "R8")
	b.Raw("SHLQ $3, R8")
	for r := 0; r < 12; r++ {
		b.Raw("XORPS X%d, X%d", r, r)
	}
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ hstore")
	b.Raw("hkloop:")
	b.Raw("MOVUPD (DI), X12")
	b.Raw("MOVUPD 16(DI), X13")
	for r := 0; r < gemmMR; r++ {
		b.Raw("MOVDDUP %d(SI), X14", 8*r) // X14 = [a,a]
		b.Raw("MOVAPS X12, X15")
		b.Raw("MULPD X14, X15")
		b.Raw("ADDPD X15, X%d", 2*r)
		b.Raw("MULPD X13, X14") // the broadcast is spent: reuse it for the product
		b.Raw("ADDPD X14, X%d", 2*r+1)
	}
	b.Raw("ADDQ $%d, SI", 8*gemmMR)
	b.Raw("ADDQ $%d, DI", 8*gemmNR)
	b.Raw("DECQ CX")
	b.Raw("JNZ hkloop")
	b.Raw("hstore:")
	for r := 0; r < gemmMR; r++ {
		b.Raw("MOVUPD (DX), X14")
		b.Raw("ADDPD X%d, X14", 2*r)
		b.Raw("MOVUPD X14, (DX)")
		b.Raw("MOVUPD 16(DX), X15")
		b.Raw("ADDPD X%d, X15", 2*r+1)
		b.Raw("MOVUPD X15, 16(DX)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, DX")
		}
	}
	b.Ret()
	return b.Func()
}
