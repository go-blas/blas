// Separate module: the go-asmgen generators are build-time tools only, so the
// library keeps no dependency on go-asmgen; go generate regenerates the .s.
module github.com/go-blas/blas/internal/kernels/asmgen

go 1.27.3

require github.com/go-asmgen/asmgen v0.23.2
