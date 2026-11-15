# Design

The decisions of this repository, agreed on
[go-embedded-ruby/ruby#918](https://github.com/go-embedded-ruby/ruby/issues/918)
(read the last two comments there) and recorded here so that the code can be
judged against them. julia binds libblastrampoline to this module
([go-embedded-julia/julia#67](https://github.com/go-embedded-julia/julia/issues/67));
go-ndarray will consume it in a later pull request.

## What this is

The fleet's pure-Go BLAS, and later LAPACK, with the Fortran reference
signatures, so that a Fortran ABI shim can call it exactly: column-major
storage, leading dimensions, increments, the reference's scalar rules and quick
returns.

## Decisions

1. **Reference signatures and semantics.** One Go routine per Fortran routine,
   parameter for parameter, named with the s/d/c/z prefix convention
   (`Dgemm`, `Sgemv`, `Idamax`, and later `Zgemm`, `Zdotc`, `Zgetrf`).
   `beta = 0` never reads C; `alpha = 0` never reads A or B; `m = 0`, `n = 0`
   and the netlib quick returns return at once.
2. **Complex and LAPACK come next, without breaking anything.** This release is
   `float32` and `float64`. The API is shaped so that c and z follow as pure
   additions: `Transpose` has `NoTrans`, `Trans` and `ConjTrans` from the start
   (for real types `ConjTrans` behaves like `Trans`, as in netlib); the
   routines carry the s/d/c/z prefix; `ArgError` and the `Uplo`, `Diag` and
   `Side` enums are type-independent. The complex routines (`Cgemm`, `Zgemm`,
   `Zgemv`, `Zdotc`/`Zdotu`, ...) take `complex64` and `complex128` slices,
   and LAPACK (`zgetrf`, `zheevd`, `zgesdd`, ...) follows the BLAS.
3. **Argument errors never panic.** Each routine returns `error`: nil, or
   `*ArgError{Routine, Param}` carrying xerbla's parameter number (LAPACK
   `info = -Param`). Beyond the reference's tests, every slice is checked
   against what the call needs (`lda*(cols-1)+rows`, `1+(n-1)*|inc|`) with
   overflow-safe arithmetic that holds on 386 and arm. A slice is checked
   only when the call reads or writes it. The checks live in
   `internal/check` and return a parameter number; `blas` and `netlib` wrap it,
   so both report the same error.
4. **The contract is an error bound, not bit identity.** Each routine states a
   bound, asserted against a `math/big` oracle of 256 bits
   (`internal/oracle`); the bounds are in the package comment of `blas`. FMA,
   blocking, packing and reordered sums are allowed and used inside it. For
   gemm the bound is `gamma_{k+2} (|alpha||op(A)||op(B)| + |beta||C|)`,
   elementwise, `gamma_k = ku/(1-ku)`. The `beta` term is inside the `k+2`
   because C is scaled first and the product accumulated into it, which is what
   the reference's own NN and NT loops do and what every blocked BLAS does; a
   bound of `gamma_1 |beta||C|` alone would need the product formed apart and
   combined at the end, at the price of an `m*n` scratch matrix on every call
   with `beta != 0`.
5. **No new kernel.** The column-major product is the row-major product
   `C^T = op(B)^T op(A)^T` over ndarray's strided GEMM kernel. The only change
   to the moved driver is that its destination takes a row stride
   (`MatMulLDStridedP`: `dst += a*b`, rows `ldc` apart) instead of assuming
   `ldc = n`; the micro-kernels already took `ldc`. The kernel has no alpha:
   with `alpha != 1` the product is formed in a pooled scratch matrix and added
   with alpha. `Sgemm` without a float32 kernel (CPUs and targets that have none)
   runs the reference loops.
6. **`netlib` is a public exact option.** Plain-Go ports of reference BLAS 3.12
   (Reference-LAPACK/lapack, tag `v3.12.0`, `BLAS/SRC`), in netlib's loop order
   and without FMA: every product is wrapped in an explicit conversion, which
   the Go specification says forbids fusing (arm64, ppc64le, riscv64 and s390x
   fuse `x*y+z` otherwise); `TestNetlibIsNotFused` pins it with an operand pair
   on which fused and unfused differ. Ported files: `dgemm.f`, `sgemm.f`,
   `dgemv.f`, `sgemv.f`, `ddot.f`, `sdot.f`, `daxpy.f`, `saxpy.f`, `dscal.f`,
   `sscal.f`, `dnrm2.f90`, `snrm2.f90`, `dasum.f`, `sasum.f`, `idamax.f`,
   `isamax.f`, and (decision 9) `dtrsm.f`, `strsm.f`, `dtrmm.f`, `strmm.f`,
   `dsyrk.f`, `ssyrk.f`, `dsyr2k.f`, `ssyr2k.f`, `dsymm.f`, `ssymm.f`. Double and single precision share one generic implementation
   (`internal/ref`), which differ only in the type and the four constants of
   nrm2. They are the equality oracle of the tests while porting, and the
   fallback of `Sgemm`.
7. **Level 1 with unit increments calls go-simd/floats** (`Dot`, `Axpy`,
   `Scale`, `Float32Dot`) instead of duplicating it; `Gemv` is one `Axpy` or
   `Dot` per column. Strided calls, nrm2, asum and iamax run the reference
   loops.
8. **A general-stride entry, `DgemmStrided` / `SgemmStrided`, beside the
   column-major one.** go-ndarray's arrays are row-major views with an arbitrary
   row and column stride (`a[:, ::2]`, a transposed or a sliced matrix), which
   `Dgemm`'s leading dimension cannot express: the only way through it was to
   copy the operand first, measured at 1.1x (n = 256) to 1.5x (n = 1024)
   slower than the packing that reads the strides in place. The precedent is
   BLIS's typed API, `bli_?gemm(transa, transb, m, n, k, alpha, a, rsa, csa, b,
   rsb, csb, beta, c, rsc, csc)`, where "all matrix arguments have associated
   row and column strides arguments", row stride first, and "unit row stride
   implies column-major storage and unit column stride implies row-major
   storage"
   ([BLISTypedAPI.md](https://github.com/flame/blis/blob/master/docs/BLISTypedAPI.md),
   read 2026-11-15).

   The shape chosen:
   `DgemmStrided(m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) error`,
   and `SgemmStrided`. Element (i, j) of an operand is `x[i*rs+j*cs]`; A is
   m x k, B k x n, C m x n. There are no transpose flags (a transpose is a
   stride swap, so a flag would be a second spelling), and a Go `error` replaces
   BLIS's `bli_check`.

   *Parameters*, numbered from 1 as in the signature, for `*ArgError`
   (`Routine` is `DGEMMSTRIDED` or `SGEMMSTRIDED`): 1 m, 2 n, 3 k, 4 alpha, 5 a,
   6 rsa, 7 csa, 8 b, 9 rsb, 10 csb, 11 beta, 12 c, 13 rsc, 14 csc. Negative
   dimensions are 1, 2, 3. Every stride of A and B must be at least 1 (6, 7, 9,
   10): zero and negative strides, which BLIS accepts, are refused, because they
   alias elements or run backwards past the start of the slice. C must be
   row-major (`csc = 1`, `rsc >= n`) or column-major (`rsc = 1`, `csc >= m`):
   those are the two destinations the kernel writes (the second through the
   transpose identity of decision 5), and C is written, so overlapping or
   doubly-strided C is refused: `rsc < n`, `rsc < 1` and neither stride 1 are
   13; `csc < 1` and a column-major C with `csc < m` are 14. A slice must hold
   `1+(rows-1)*rs+(cols-1)*cs` elements (0 for an empty matrix), computed
   without overflow (`check.NeedStrided`), against its own number (5, 8, 12),
   only when the call reads or writes it, after the quick returns. A caller with
   a stride-0 broadcast, a reversed view or an extent-1 dimension with an
   odd stride normalizes it first.

   *Semantics and bound* are `Dgemm`'s: the same quick returns, `beta = 0`
   never reads C, `alpha = 0` never reads A or B, and the elementwise bound
   `gamma_{k+2}(|alpha||A||B| + |beta||C|)`, asserted against the oracle on
   every storage of A, B and C. It runs the same packed kernel through
   `MatMulLDStridedP`: no new kernel. Where `Dgemm` builds a `Mat` from `lda`
   and the transpose flag, this builds it from the two strides; for the same
   matrices the result is bit for bit that of `Dgemm` (float64,
   `TestGemmStridedSameAsGemm`; float32 where there is a kernel). Without a float32 kernel `SgemmStrided` runs
   plain loops over the strides. `HasSgemmKernel` reports whether it has one.

9. **The rest of level 3, `Strsm`/`Dtrsm`, `Strmm`/`Dtrmm`, `Ssyrk`/`Dsyrk`,
   `Ssyr2k`/`Dsyr2k`, `Ssymm`/`Dsymm`, built on the one GEMM.** They are the
   reference signatures (`side`, `uplo`, `transA`, `diag` as `Side`, `Uplo`,
   `Transpose`, `Diag`, case-insensitive), with xerbla's numbers: trsm and trmm
   side 1, uplo 2, transA 3, diag 4, m 5, n 6, alpha 7, a 8, lda 9, b 10, ldb 11;
   syrk uplo 1, trans 2, n 3, k 4, alpha 5, a 6, lda 7, beta 8, c 9, ldc 10;
   syr2k the same with b 8, ldb 9, beta 10, c 11, ldc 12; symm side 1, uplo 2, m
   3, n 4, alpha 5, a 6, lda 7, b 8, ldb 9, beta 10, c 11, ldc 12. The checks are
   in `internal/check` (`Tr`, `TrLens`, `Syrk`, `Syr2k`, `Symm` and their `Lens`),
   shared by `blas` and `netlib`; the quick returns are the reference's (trsm and
   trmm: `m = 0` or `n = 0`; syrk and syr2k: `n = 0`, or `alpha = 0` or `k = 0`
   with `beta = 1`; symm: `m = 0`, `n = 0`, or `alpha = 0` with `beta = 1`), taken
   before the slice-length checks, and a slice a call does not read (A and B for
   `alpha = 0`) is not checked.

   *Algorithms*, as in GotoBLAS and BLIS: a blocked algorithm whose bulk is GEMM
   and whose diagonal is a small triangular kernel, here the reference loops of
   `internal/ref` (ports of `dtrsm.f`, `dtrmm.f`, `dsyrk.f`, `dsyr2k.f`,
   `dsymm.f` of Reference-LAPACK 3.12.0, in loop order, no FMA, each product
   wrapped in a conversion; they are `netlib`'s routines). No micro-kernel was
   written. `blockSize` (64) is the order of a diagonal block, a variable only so
   that the tests can make 40-row matrices span many blocks.
   * *trsm* solves block row by block row (Left) or block column by block column
     (Right), in the order the triangle dictates: B is scaled by alpha once, a block
     is B minus the GEMM of the blocks already solved, then substitution on the
     diagonal block. *trmm* is the same in the opposite order, so that the blocks
     a GEMM reads are still the original ones: the diagonal block in place, plus
     the GEMM of the other blocks of the row or column.
   * *syrk*, *syr2k* cut the triangle of C into tiles: a tile off the diagonal is
     a GEMM (two for syr2k), a diagonal tile is one GEMM into a pooled scratch
     tile whose triangle is merged into C (for syr2k the tile product A B**T, and
     the merge takes `t + t**T`). The other triangle of C is never written.
   * *symm* copies the referenced triangle, exactly, into a dense s x s scratch
     matrix, s the order of A (at most the size of A, which the length check
     guarantees the caller holds), and is one GEMM.
   Without a float32 kernel (and for `alpha = 0`, and syrk/syr2k with `k = 0`) the
   routine runs the reference loops.

   *Bounds*, asserted against the 256-bit oracle for every combination of side,
   uplo, trans and diag, padded leading dimensions, zero dimensions, `alpha = 0`,
   `beta = 0` (C holding NaN), three block sizes, `float32` and `float64`, with NaN
   in every element the routine must not read (the other triangle, the diagonal of
   a unit triangle) so a routine that reads one fails by its result; the other
   triangle of C is asserted untouched. With `u` the unit roundoff, `gamma_k =
   ku/(1-ku)` and `s` the order of A:
   * syrk: `gamma_{k+2}(|alpha||op(A)||op(A)^T| + |beta||C|)`, gemm's bound. A tile
     off the diagonal is a GEMM; on the diagonal the tile has `k` terms, then
     `alpha*t`, `beta*c` and their sum: `k+3` roundings on the one term and 2 on
     the other, inside `k+2`.
   * syr2k: `gamma_{2k+2}(|alpha|(|op(A)||op(B)^T| + |op(B)||op(A)^T|) + |beta||C|)`.
     Two GEMMs accumulated into C cost at most `k+3` on any term (`k+3 <= 2k+2` for
     `k >= 1`), and the reference's own loops `2k+2`.
   * symm: `gamma_{s+2}(|alpha||A||B| + |beta||C|)`, gemm's with `k = s`, the copy
     being exact.
   * trmm: `gamma_{s+3}|alpha||op(A)||B|`: on one element the reference loops of a
     diagonal block contribute at most `ib+1` roundings, the GEMM `k` for its
     product and 2 for `alpha` and the accumulation, `ib + k + 3 <= s + 3`.
   * trsm, a **backward error (residual) bound**, not a forward one:
     `|op(A) X^ - alpha B| <= gamma_{s+4}(|op(A)||X^| + |alpha B|)`, elementwise.
     It is the componentwise backward stability of substitution (Higham,
     *Accuracy and Stability of Numerical Algorithms*, ch. 8: the computed
     solution of a triangular system satisfies `(T + dT) x^ = b`, `|dT| <=
     gamma_n |T|`, for any order of the summations), which is what a blocked
     solve inherits: each unknown is obtained from `b` by the same chain of at
     most `s` additions and one division (the GEMM sums are of the solved
     unknowns, in any order, and the update's own scaling adds two roundings, the
     scaling of B by alpha one more, hence `s+4`). It was chosen over the forward
     form `|x - x^| <= gamma_n cond(T, x)|x|` because the latter needs the exact
     solution, and so a condition number that a test must construct and a fuzz
     target cannot, whereas the residual is computed exactly from the computed
     `X^`, with no assumption on the conditioning of `A`: the fuzz targets run
     on arbitrary, ill-conditioned triangles. The forward bound follows from it
     by the usual `|x - x^| <= |T^-1||dT||x^|`. *Higham's text was not open in this
     session*: the statement above is the standard result, given here with its
     derivation sketched, and it is the tests, not the citation, that assert
     it. A zero on the diagonal is not tested for, as in the reference.

   *Not promised*: bit identity with netlib (the order of the sums differs, and
   the reference skips a product when a factor is exactly zero where GEMM forms it,
   so that an infinity times such a zero is NaN here); special values travel the
   same way when the data are finite apart from a planted NaN or infinity, which
   the fuzz targets compare by class. Each routine is not safe with an output that
   overlaps an input.

   *Fuzz*: one target per routine and type (ten), each comparing blas with netlib
   (same `*ArgError`, same class of NaN or infinity, an error or a quick return
   leaves the outputs as they were) and, on finite data of moderate magnitude, both
   with the oracle's bound; the input chooses the block order, so that matrices
   below 40 rows span several blocks. For trsm the finite entries of A are scaled
   to a well-conditioned triangle before the class comparison, since a random
   triangle overflows in one order of summation and not in another.

## What was moved from go-ndarray

From `go-ndarray/ndarray` at `main` (v0.41.3, commit `8370f0f`):

* `internal/kernels/gemm.go` (the packed, cache-blocked, multicore driver),
  `gemm_<arch>.go`, `gemm32_*.go`, `gemmua_other.go`, `gemm_generic.go`;
  from `parallel.go` only `spinUntil`, from `hwcap.go` the AT_HWCAP parser, and
  the CPU-feature variables of `reduce_<arch>.go` (`cpu_<arch>.go` here). The
  mat-vec and dot code of `gemm.go`, the thread pool and the elementwise
  fan-out stay in ndarray, which has no GEMM left to share with them.
* The GEMM micro-kernels of the six architectures (amd64 AVX2/FMA 6x8, arm64
  NEON 4x8, ppc64le VSX 8x8, loong64 LASX 8x8, riscv64 RVV 6x16 and 8x16, s390x
  4x8) with their tests and benchmarks.
* The go-asmgen generators, `internal/kernels/asmgen/<arch>/gen.go`, which in
  ndarray write GEMM and non-GEMM kernels into one `sum_<arch>.s`. Here each
  emits only the GEMM kernels, to the same file names. `asmgen/splitdiff` proves
  that the output is exactly the GEMM part of ndarray's file: every TEXT block and
  DATA symbol is byte-identical and in the same order, and the headers are
  identical. The kernels left in ndarray's files (`dot*`, `blocks8*`, `*32*`
  elementwise, `sqrt32*`, `max*`/`min*`) are not GEMM.
* Their tests: per-architecture kernels, one worker (`TestGemmOneWorkerSameBits`),
  bits, `packRows8`, the fences that fault on a one-element overrun, and the
  `BLAS_REQUIRE_RVV`/`_LASX`/`_VXE` lane guards (`NDARRAY_REQUIRE_*` in ndarray).

Switching ndarray to this module is a later pull request.

## Testing the guards

Every argument guard is a line ending in `// guard`. `internal/cmd/guardmut`
disables each in turn and requires the tests to fail by an assertion: a
mutant that survives is an untested guard, and one that ends in a panic is a
guard the language found for us. `-only a,b` restricts the mutants to the files
whose path contains one of the strings, and `-pkgs` the packages the tests of each
mutant run on (CI runs them all). The routines never panic, and a test turns a
panic of the routine under test into a failed assertion (`noPanic`).
