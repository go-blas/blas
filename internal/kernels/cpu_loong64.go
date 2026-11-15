package kernels

import "os"

// hwcapLASX is HWCAP_LOONGARCH_LASX (Linux arch/loongarch/include/uapi/asm/hwcap.h).
const hwcapLASX = 1 << 5

// useLASX selects the LASX kernels. LASX (256-bit, four float64 lanes) is not on
// every LoongArch CPU, so the GEMM micro-kernel runs only when the kernel
// reports it in AT_HWCAP. A var so tests can force the scalar path on hardware
// that has LASX (never the reverse: that would fault).
var useLASX = lasxFromAuxv()

func lasxFromAuxv() bool {
	auxv, err := os.ReadFile("/proc/self/auxv")
	return err == nil && hwcapFrom(auxv)&hwcapLASX != 0
}
