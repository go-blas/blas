package kernels

import "os"

// hwcapVX is HWCAP_S390_VXRS (Linux arch/s390/include/asm/elf.h: bit 11).
const hwcapVX = 1 << 11

// useVX selects the vector kernels. Every s390x Go runs on (z13 and later) has
// the facility, but a kernel can still disable it, so the GEMM kernels run only
// when AT_HWCAP reports it. A var so tests can force the scalar path on hardware
// that has the facility (never the reverse: that would fault).
var useVX = vxFromAuxv()

func vxFromAuxv() bool {
	auxv, err := os.ReadFile("/proc/self/auxv")
	return err == nil && hwcapFrom(auxv)&hwcapVX != 0
}

// hwcapVXE is HWCAP_S390_VXRS_EXT (bit 13), the vector-enhancements facility 1
// of the z14, which the float32 GEMM kernel needs.
const hwcapVXE = 1 << 13

// useVXE selects the float32 kernel. A var so tests can force the scalar path.
var useVXE = vxeFromAuxv()

func vxeFromAuxv() bool {
	auxv, err := os.ReadFile("/proc/self/auxv")
	return err == nil && hwcapFrom(auxv)&hwcapVXE != 0
}
