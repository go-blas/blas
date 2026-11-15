package kernels

import "os"

// hwcapV is COMPAT_HWCAP_ISA_V, the bit of 'V' in the letter-indexed riscv
// AT_HWCAP (Linux arch/riscv/include/asm/hwcap.h: 1 << ('V' - 'A')). The kernel
// sets it only when it also saves vector state across context switches. The
// vector extension is optional on RISC-V, so the RVV GEMM kernels run only when
// the kernel reports it.
const hwcapV = 1 << ('V' - 'A')

// useRVV selects the RVV kernels. A var so tests can force the scalar path on
// hardware that has V (never the reverse: that would fault).
var useRVV = rvvFromAuxv()

func rvvFromAuxv() bool {
	auxv, err := os.ReadFile("/proc/self/auxv")
	return err == nil && hwcapFrom(auxv)&hwcapV != 0
}
