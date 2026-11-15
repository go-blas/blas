package kernels

import (
	"os"
	"testing"
)

// TestRVVWhenRequired fails when BLAS_REQUIRE_RVV is set and the CPU does
// not report V. CI sets it on the riscv64 qemu lane, which runs with V
// enabled: without this check, a qemu whose CPU lost V would take the scalar
// paths and the RVV kernels would go untested behind a green tick, as they
// did until 2026-10-26.
func TestRVVWhenRequired(t *testing.T) {
	t.Logf("useRVV = %v", useRVV)
	if os.Getenv("BLAS_REQUIRE_RVV") != "" && !useRVV {
		t.Fatal("BLAS_REQUIRE_RVV is set but the CPU does not report V: the RVV kernels did not run")
	}
}
