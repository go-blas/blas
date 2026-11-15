// Command gen produces sum_s390x.s, the vector-facility float64 kernels for
// s390x, via go-asmgen. Run with: go run . (or `go generate` from the kernels
// package).
//
// Go's s390x assembler names the instructions (VL/VLM/VSTM, VFADB, VFSDB,
// VFMDB, VFDDB, VFSQDB, VFMADB, VREPG, WFADB), so nothing is a WORD. A vector
// register holds two float64; each loop does eight (four registers) and a
// scalar tail does the rest. The Go side checks the vector facility in
// AT_HWCAP.
//
// Operand order, measured by running each instruction on known inputs under
// s390x emulation (the IBM manuals give the other order): VFSDB V1, V2, V3 is
// V3 = V2 - V1, VFDDB V1, V2, V3 is V3 = V2 / V1, VFMADB V1, V2, V3, V4 is
// V4 = V1*V2 + V3, and FSUB F1, F0 is F0 = F0 - F1.
//
// R1..R4 hold arguments and pointers (R0 cannot be a base register); V16..V19
// are the accumulators; F0..F15 alias the first element of V0..V15.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	f.Add(gemmKernel())
	f.Add(gemm32Kernel())
	if err := os.WriteFile("sum_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_s390x.s")
}

// gemmKernel builds gemmMicro4x8VX(kc int, pa, pb, c *float64, ldc int) for
// kc >= 1: C[r][0:8] += sum_p pa[p*4+r] * pb[p*8:p*8+8] for r < 4, the rows of C
// ldc float64s apart. Sixteen two-lane accumulators V0..V15 (row r in
// V4r..V4r+3); each step loads the B row in one VLM, replicates each A value
// across a register with VLREPG, and fuses it in with VFMADB.
func gemmKernel() *emit.Function {
	sig := s390x.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]s390x.Type{s390x.Int64, s390x.Ptr, s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc("gemmMicro4x8VX", sig, 0)
	b.LoadArg("kc", "R5").LoadArg("pa", "R2").LoadArg("pb", "R3").
		LoadArg("c", "R1").LoadArg("ldc", "R4")
	for r := 0; r < 16; r++ {
		b.Raw("VZERO V%d", r)
	}
	b.Raw("loop:").Raw("VLM (R3), V16, V19") // the B row
	for r := 0; r < 4; r++ {
		b.Raw("VLREPG %d(R2), V%d", 8*r, 20+r) // a_r in both lanes
	}
	for r := 0; r < 4; r++ {
		for j := 0; j < 4; j++ {
			b.Raw("VFMADB V%d, V%d, V%d, V%d", 20+r, 16+j, 4*r+j, 4*r+j) // acc += a_r * b
		}
	}
	b.Raw("ADD $32, R2").Raw("ADD $64, R3").Raw("SUB $1, R5").
		Raw("CMP R5, $0").Raw("BNE loop").
		Raw("SLD $3, R4") // row stride in bytes
	for r := 0; r < 4; r++ {
		b.Raw("VLM (R1), V24, V27")
		for j := 0; j < 4; j++ {
			b.Raw("VFADB V%d, V%d, V%d", 4*r+j, 24+j, 24+j) // C row += acc_r
		}
		b.Raw("VSTM V24, V27, (R1)")
		if r < 3 {
			b.Raw("ADD R4, R1")
		}
	}
	return b.Ret().Func()
}

// gemm32Kernel builds gemmMicro4x16VXE(kc int, pa, pb, c *float32, ldc int)
// for kc >= 1: C[r][0:16] += sum_p pa[p*4+r] * pb[p*16:p*16+16] for r < 4, the
// rows of C ldc float32s apart. gemmKernel at twice the lanes: sixteen
// four-lane accumulators V0..V15 (row r in V4r..V4r+3), the B row in one VLM,
// each A value replicated across a register with VLREPF, and VFMASB, the
// vector-enhancements facility 1 (z14) fused multiply-add, which go-asmgen
// encodes because Go's assembler has no short-format vector float op.
func gemm32Kernel() *emit.Function {
	sig := s390x.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]s390x.Type{s390x.Int64, s390x.Ptr, s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc("gemmMicro4x16VXE", sig, 0)
	b.LoadArg("kc", "R5").LoadArg("pa", "R2").LoadArg("pb", "R3").
		LoadArg("c", "R1").LoadArg("ldc", "R4")
	for r := 0; r < 16; r++ {
		b.Raw("VZERO V%d", r)
	}
	b.Raw("loop:").Raw("VLM (R3), V16, V19") // the B row: 16 floats
	for r := 0; r < 4; r++ {
		b.Raw("VLREPF %d(R2), V%d", 4*r, 20+r) // a_r in all four lanes
	}
	for r := 0; r < 4; r++ {
		for j := 0; j < 4; j++ {
			b.VFMASB(4*r+j, 20+r, 16+j, 4*r+j) // acc += a_r * b
		}
	}
	b.Raw("ADD $16, R2").Raw("ADD $64, R3").Raw("SUB $1, R5").
		Raw("CMP R5, $0").Raw("BNE loop").
		Raw("SLD $2, R4") // row stride in bytes
	for r := 0; r < 4; r++ {
		b.Raw("VLM (R1), V24, V27")
		for j := 0; j < 4; j++ {
			b.VFASB(24+j, 24+j, 4*r+j) // C row += acc_r
		}
		b.Raw("VSTM V24, V27, (R1)")
		if r < 3 {
			b.Raw("ADD R4, R1")
		}
	}
	return b.Ret().Func()
}
