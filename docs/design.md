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
   `beta = 0` (C holding NaN), two block sizes, `float32` and `float64`, with NaN
   in every element the routine must not read (the other triangle, the diagonal of
   a unit triangle) so a routine that reads one fails by its result; the other
   triangle of C is asserted untouched. With `u` the unit roundoff, `gamma_k =
   ku/(1-ku)` and `s` the order of A:
   * syrk: `gamma_{k+2}(|alpha||op(A)||op(A)^T| + |beta||C|)`, gemm's bound. A tile
     off the diagonal is a GEMM; on the diagonal the tile has `k` terms, then
     `alpha*t`, `beta*c` and their sum: `k+2` roundings on the product term (`k`,
     the product by alpha, the sum) and 2 on the `beta` term (the product, the
     sum): exactly `k+2`.
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
     **Source, opened 2026-11-15:** J. W. Demmel and N. J. Higham, *Stability
     of Block Algorithms with Fast Level 3 BLAS*, LAPACK Working Note 22 (July
     27, 1990), http://www.netlib.org/lapack/lawnspdf/lawn22.pdf. Page 4, eq.
     (4): for the substitution algorithms for `T X = B`, `T` of order `m`,
     `T X^ = B + dB`, `|dB| <= (m+1) u |T||X^| + O(u^2)`. That is a first-order
     componentwise residual bound with constant `m+1`. Page 3 adds the caveat
     that matters here: for block algorithms with conventional BLAS3, "it is
     generally accepted that the same stability results are valid, although we
     are not aware of any detailed proofs". So for this BLOCKED trsm the bound
     is asserted by the tests against the 256-bit oracle, not by a proof.
     The rounding of `alpha*B` adds `u |alpha B|` (absent for `alpha = 1`). The
     asserted bound, `gamma_{s+4}` on both terms, is the first-order constant
     `s+1` widened by 3, in the non-asymptotic `gamma` form; it is conservative,
     not the tightest. The constant did not change when the source was read.
     Higham's book (*Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
     ch. 8, Theorem 8.5 and Lemma 8.4), which states the non-asymptotic
     `gamma_n` form for substitution, was NOT opened; it is named, not cited. It was chosen over the forward
     form `|x - x^| <= gamma_n cond(T, x)|x|` because the latter needs the exact
     solution, and so a condition number that a test must construct and a fuzz
     target cannot, whereas the residual is computed exactly from the computed
     `X^`, with no assumption on the conditioning of `A`: the fuzz targets run
     on arbitrary, ill-conditioned triangles. The forward bound follows from it
     by the usual `|x - x^| <= |T^-1||dT||x^|`. A zero on the diagonal is not tested for, as in the reference.

   *Not promised*: bit identity with netlib (the order of the sums differs, and
   the reference skips a product when a factor is exactly zero where GEMM forms it,
   so that an infinity times such a zero is NaN here). OpenBLAS sides with this
   package, not with netlib: measured by go-embedded-julia on 2026-11-15 with
   julia 1.12.6 over libopenblas64_ (ILP64, one thread, 4×4 so the BLAS path is
   taken), `zeros(4,4) * fill(Inf,4,4)` and `zeros(4,4) * fill(Inf,4)` give NaN.
   What all three agree on: beta = 0 never reads C (or y), and alpha = 0 reads
   neither A nor B (julia's `gemm!` with alpha = 0 over an Inf-filled A returns
   beta·C); the tests hold both for every routine, with NaN planted in C under
   beta = 0 and nil operands under alpha = 0. Special values travel the
   same way when the data are finite apart from a planted NaN or infinity, which
   the fuzz targets compare by class. Each routine is not safe with an output that
   overlaps an input.

   *Measured* against netlib's ports of the reference (not against OpenBLAS or
   MKL, which this repository has no run of), `level3_bench_test.go`, a
   cross-compiled test binary on cfarm420 (AMD EPYC 7773X, linux/amd64),
   `taskset -c 9`, `GOMAXPROCS=1`, 1-minute load 2.4 before and after, 10
   iterations, `blockSize` 64, 2026-11-15. Gflop/s counts `n^3` flops for trsm,
   trmm and syrk and `2n^3` for syr2k and symm (the flops of the operation, not
   of what the blocked code does); one run, no confidence interval, and one core
   of a shared machine:

   | n = 512 | float64 blas | float64 netlib | float32 blas | float32 netlib |
   |---|---|---|---|---|
   | trsm | 10.4 | 2.3 | 11.4 | 2.3 |
   | trmm | 11.7 | 2.5 | 14.2 | 2.3 |
   | syrk | 21.3 | 1.9 | 36.5 | 2.0 |
   | syr2k | 23.3 | 2.8 | 38.4 | 3.0 |
   | symm | 30.1 | 3.2 | 63.9 | 3.7 |

   trsm and trmm run at about half of syrk's rate: the diagonal blocks are the
   reference's scalar loops (`blockSize/m`, an eighth of the flops at n = 512,
   at scalar speed). A smaller diagonal kernel, or recursion into it, is the
   obvious next step and would be a measurement away from a kernel of its own.

   *Fuzz*: one target per routine and type (ten), each comparing blas with netlib
   (same `*ArgError`, same class of NaN or infinity, an error or a quick return
   leaves the outputs as they were) and, on finite data of moderate magnitude, both
   with the oracle's bound; the input chooses the block order, so that matrices
   below 40 rows span several blocks. For trsm the finite entries of A are scaled
   to a well-conditioned triangle before the class comparison, since a random
   triangle overflows in one order of summation and not in another.

10. **LAPACK, first part (v0.4.0): package `lapack`, `Sgetrf`/`Dgetrf`,
    `Sgetrs`/`Dgetrs`, `Spotrf`/`Dpotrf`, `Spotrs`/`Dpotrs`.** julia's
    libblastrampoline needs them for `\`, `lu` and `cholesky`
    ([go-embedded-julia/julia#67](https://github.com/go-embedded-julia/julia/issues/67)).
    They are LAPACK 3.12's signatures, parameter for parameter, column-major, with
    the conventions of decisions 1 to 4: flags case-insensitive (`Transpose`,
    `Uplo`; for real types `ConjTrans` is `Trans`), `*ArgError` with xerbla's
    numbers, overflow-safe length checks in `internal/check` (`Getrf`, `Getrs`,
    `Potrf`, `Potrs` and their `Lens`, shared by `lapack` and `netlib`), the
    reference's quick returns before the slice checks.

    *Parameter numbers*: getrf m 1, n 2, a 3, lda 4, ipiv 5; getrs trans 1, n 2,
    nrhs 3, a 4, lda 5, ipiv 6, b 7, ldb 8; potrf uplo 1, n 2, a 3, lda 4; potrs
    uplo 1, n 2, nrhs 3, a 4, lda 5, b 6, ldb 7. The reference's own tests are on m,
    n, nrhs, lda, ldb and the flags (`INFO` -1, -2, -3, -4, -5, -7, -8 as in
    `dgetrf.f`, `dgetrs.f`, `dpotrf.f`, `dpotrs.f`); a slice too short is reported
    against its own number, `ipiv` against `min(m, n)` (getrf) or `n` (getrs).
    Getrs also refuses a pivot outside `1..n` as parameter 6: the reference would
    read outside the array, and a pivot is data from a caller that may have
    corrupted it.

    *The shape of `info`.* A positive `INFO` is a result, not an error: a singular
    `U` (getrf; the factorization is complete, a solve with it divides by zero) or a
    matrix that is not positive definite (potrf). So the factorizations return
    **`(info int, err error)`**, `info` being exactly what LAPACK leaves in `INFO`:
    0, `i > 0`, or `-p` for an illegal parameter `p`, in which case `err` is the
    `*ArgError` of parameter `p` (`err.Info() == info`). `err != nil` if and only if
    `info < 0`. A Fortran shim reads `info` alone; a Go caller tests `err`, then
    `info`. The solves have no positive `INFO` and return `error` only. The same
    shape for every later factorization (`geqrf`, `sytrf`, ...): the second result is
    the xerbla error, the first is `INFO`. `ipiv` is `[]int`, 1-based, as in LAPACK
    (a shim converts to its integer width).

    *Algorithms.* No kernel was written. Getrf and potrf are recursive, as the
    reference's `dgetrf2.f` and `dpotrf2.f` (Toledo's recursive LU), over the
    blocked `blas.Dtrsm`, `Dgemm` and `Dsyrk` of decision 9, down to a panel of
    `leaf` = 16 columns factored by plain loops (`getf2`, `potf2`: right-looking
    LU with partial pivoting, the pivot being the first entry of largest magnitude
    as `IDAMAX`, the multipliers formed with the reciprocal or, below the smallest
    normal number, by division as the reference; Cholesky by inner products for
    Upper and by a rank-1 update for Lower). The recursion is the same as
    go-ndarray's `linalg.luRec` (read for it; ndarray's code is row-major and
    generic, this is column-major with LAPACK's interchanges applied lazily, as
    `dlaswp.f`). Getrs and potrs are two `Dtrsm` calls and the interchanges.
    `leaf` is a variable only so that the tests can make 5-column matrices span
    every level of the recursion (1, 2, 3 and the default are run).

    *`netlib`* has the exact option, in the reference's order and block size: ports
    of `dgetrf.f`, `dgetrf2.f`, `dgetrs.f`, `dlaswp.f`, `dpotrf.f`, `dpotrf2.f`,
    `dpotrs.f` and the s twins of Reference-LAPACK 3.12.0 (`SRC`, opened), with the
    block size `ilaenv.f` answers for `GETRF` and `POTRF`, 64. They call the BLAS
    ports of `internal/ref`, so they are without FMA, and they are the equality
    oracle of the tests. The ports skip a BLAS call the reference makes with `k = 0`
    (a `GEMM` or `SYRK` over no columns), since the ports leave the BLAS quick
    returns to their callers.

    *Bounds*, asserted against the 256-bit oracle (`oracle.Prec`) with `u` the unit
    roundoff, `gamma_k = ku/(1-ku)`, `r = min(m, n)`, `L^`, `U^` the computed
    factors, `C^` the computed Cholesky factor, `X^` the computed solution, all
    elementwise, and `P` the operator of the row interchanges of `ipiv` applied in
    order as `DLASWP` does (`P X` is `X` with them applied, so `P A = L U`; it is
    the transpose of the permutation matrix of LAPACK's `A = P L U`):
    * getrf: `|P A - L^ U^| <= gamma_{3r+2} |L^||U^|`, for `INFO > 0` too (the
      factorization as completed);
    * getrs: `|P B - L^ U^ X^| <= gamma_{2n+8} |L^||U^||X^|`, and for Trans
      `|B - U^T L^T P X^| <= gamma_{2n+8} |U^T||L^T||P X^|` (`P X^` the forward
      interchanges applied to `X^`);
    * potrf: `|A - C^T C^| <= gamma_{3n+3} |C^T||C^|` (Upper; `C^ C^T` for Lower) on
      the triangle referenced, for `INFO > 0` on the leading block of order
      `INFO - 1`;
    * potrs: `|B - C^T C^ X^| <= gamma_{2n+8} |C^T||C^||X^|` (Upper; Lower likewise).

    **Sources, opened 2026-11-17:** J. W. Demmel and N. J. Higham, *Stability of
    Block Algorithms with Fast Level 3 BLAS*, LAPACK Working Note 22 (the copy of
    decision 9). §2.1, "Error Analysis" of block LU, pp. 4 to 8: p. 4, eq. (3), a
    product by a conventional BLAS3 satisfies `|C^ - AB| <= n u |A||B| + O(u^2)`,
    and eq. (4), substitution gives `T X^ = B + dB`, `|dB| <= (m+1) u |T||X^| +
    O(u^2)`, componentwise; p. 5 to 7, the induction on the blocks of the outer
    product form, eqs. (5) to (14), ending in eq. (15), `|| dA || <= u (delta(n,r)
    ||A|| + theta(n,r) || L^ || || U^ ||) + O(u^2)`, with `delta(n,r) <= n/r` (eq. 7
    and the text after eq. 15) and, with partial pivoting, the same for `P A` ("(14)
    (with A replaced by P A) and (15) remain valid, with minor changes in the
    recurrences", p. 7 to 8); p. 8, eqs. (17) and (18), the solve by the computed LU
    and the residual for several right-hand sides, with the closing remark that the
    analysis "adapts in a straightforward way to ... block Cholesky factorization of
    (banded) symmetric positive definite matrices". **What the source gives and
    what it does not.** Its bounds are first-order and *normwise* (a max-entry
    norm) for the block factorization; only (3) and (4), for the BLAS operations,
    are componentwise. The bounds above are componentwise and non-asymptotic: the
    `gamma` form with constants `3r+2`, `3n+3`, `2n+8` is **my own widening** of
    those first-order results. The constants are not derived from a proof and not
    taken from a source: I chose them with the pieces in mind (the `gemm` bound
    `gamma_{k+2}` and the `trsm` residual bound `gamma_{s+4}` of decisions 4 and 9,
    each applied to the blocks of every level of the recursion, plus the panel's
    own sums of at most `r` terms) and with room, without summing them, and they are
    *tested*, over every panel width and shape of the test suite, not proved. The reference list of LAWN 22 names Demmel,
    *On floating point errors in Cholesky* (LAWN 14, its ref. [8]) and Higham's
    book; **neither was opened**, they are named here, not cited, and no Cholesky
    constant of this package comes from them. Page 3 carries the caveat that
    matters for the blocked algorithms: for conventional BLAS3 "it is generally
    accepted that the same stability results are valid, although we are not aware
    of any detailed proofs". The blocked and recursive algorithms here rest on
    that statement plus the bounds of the BLAS they call; where it fails, the tests
    (against the oracle, on random mixed-magnitude matrices, singular and not
    positive definite ones, and the fuzz targets) are what stands, as for `trsm`
    in decision 9.

    *Quick returns that carry no `// guard` marker.* The `m = 0 || n = 0` return of
    `getrf` and the `n = 0` return of `potrf` in `lapack`: disabling them changes
    nothing (the recursion does nothing on an empty matrix and returns INFO 0), so
    the mutant is equivalent to the original and no test can kill it. The same two
    lines in `netlib` are unmarked for another reason: the ported `dgetrf2.f` and
    `dpotrf2.f` recurse without end on an empty matrix (the reference never gets
    there, its callers return first), and a mutant that overflows the stack is found
    by the runtime, not by an assertion. Both are tested by the quick-return cases of
    `TestArgErrors` (nil slices, `m = 0`, `n = 0`). The quick returns of `getrs` and
    `potrs` are guards, and are killed.

    *Testing.* Every bound above for both implementations, `float32` and `float64`,
    padded `lda` and `ldb`, every `Trans`/`Uplo` flag in both cases, panel widths 1,
    2, 3 and 16, shapes up to 130 (past netlib's block of 64: 63, 64, 65, ragged last
    blocks, wide and tall); zero columns (exact `INFO`), a negative, zero or NaN
    diagonal for potrf (exact `INFO` in every block position); a subnormal pivot
    (division, as the reference); NaN and Inf in the data (no panic, no error, a
    leading NaN is the pivot as in `IDAMAX`; the factors of `lapack` and `netlib`
    are of the same class at every element, in several positions of the bad value
    and for every panel width; a zero pivot skips the interchange and the scaling
    but not the update, as `DGETF2`, so that `0 * Inf` is NaN in both: review of
    this PR found `lapack` skipping the update, and the test now fails without it); every parameter number, every short
    slice, the quick returns; `lapack` against `netlib` on `INFO` and `*ArgError`
    over a grid of arguments; a ref-level test of `Laswp` (increments 1, -1, 0,
    columns on both sides of the 32-column unroll) and of `Getrf`/`Potrf` at block
    sizes other than 64. Eight fuzz targets (`FuzzDgetrf` ... `FuzzSpotrs`), with the
    committed corpus `lapack/testdata/fuzz` (written by `fuzzcorpus`, run by
    `fuzzall`), compare `lapack` with `netlib` (same `*ArgError` and `INFO`,
    outputs untouched on an error or a quick return) and, on finite data of
    moderate magnitude, both with the bound; INFO is compared for equality on
    finite data only where rounding cannot change it (potrf, whose matrix is
    strictly diagonally dominant; getrf only as "a zero pivot at or before the
    exactly zero column", since a pivot that cancels to zero in one order of
    evaluation and not in the other is legitimate); the fuzzer found two bugs of the
    harness itself, kept in the corpus. Whole module block coverage 100%
    (`covblocks`); `guardmut` on the LAPACK files, 47 guards, none survives.

    *Measured* against netlib's ports (not against OpenBLAS or LAPACK proper, of
    which this repository has no run), `lapack/bench_test.go`, a cross-compiled
    test binary on cfarm420 (AMD EPYC 7773X, linux/amd64), `taskset -c 9`,
    `GOMAXPROCS=1`, `-test.benchtime 300ms -test.count 3`, files in
    `~/nd-bench/lapack1/`, 2026-11-17. The run was deferred until the machine's
    1-minute load fell under 8 (it was 40 to 50 shortly before): **1-minute load 5.73
    at the start, 2.64 after**. Median of three runs, Gflop/s, `lapack / netlib`;
    flops counted are `2n^3/3` (getrf), `n^3/3` (potrf), `2 n^2 nrhs` (the solves,
    `nrhs` = 64), the operation's own, not the blocked code's; one core of a shared
    machine, three runs, no confidence interval (the spread of the three is up to
    about 15%, `potrs` float64 at n = 512, and mostly under 3%):

    | | n = 64 | 128 | 256 | 512 | 1024 |
    |---|---|---|---|---|---|
    | getrf float64 | 4.06 / 2.10 | 5.16 / 2.22 | 7.36 / 2.27 | 10.84 / 2.37 | 15.55 / 2.54 |
    | getrf float32 | 4.08 / 2.05 | 5.45 / 2.00 | 8.07 / 2.29 | 13.21 / 2.42 | 21.22 / 2.49 |
    | potrf float64 | 2.93 / 1.89 | 3.48 / 2.06 | 5.52 / 2.16 | 8.58 / 2.24 | 13.30 / 2.38 |
    | potrf float32 | 3.13 / 1.91 | 3.59 / 2.03 | 5.53 / 2.19 | 9.79 / 2.34 | 16.21 / 2.37 |
    | getrs float64 | 2.08 / 2.05 | 3.81 / 2.27 | 6.79 / 2.31 | 10.55 / 2.35 | 16.53 / 2.40 |
    | getrs float32 | 2.04 / 2.02 | 3.92 / 2.19 | 7.46 / 2.22 | 13.37 / 2.33 | 21.28 / 2.38 |
    | potrs float64 | 1.88 / 1.85 | 3.50 / 2.10 | 6.24 / 2.19 | 10.23 / 2.24 | 15.49 / 2.29 |
    | potrs float32 | 1.83 / 1.89 | 3.63 / 2.03 | 6.80 / 2.18 | 12.06 / 2.20 | 19.60 / 2.24 |

    Every size from 128 up is 1.7 to 8.5 times netlib's. At n = 64 the solves
    are level with it (two `Trsm` of 64 rows are one diagonal block of the scalar
    reference loops, `blockSize` of decision 9), and `potrs` float32 is 3% under
    (1.83 against 1.89, inside the spread of the runs). The rates grow with n
    because more of the work is in the GEMM; potrf trails getrf by about 15% at
    n = 1024 (13.3 against 15.6 in float64), which this run does not explain (it
    did not profile). The scalar panel and diagonal blocks are what holds trsm at
    half of syrk's rate in the table above, and a smaller diagonal kernel behind
    `Trsm` is the obvious next step, a measurement away.

11. **LAPACK, QR (v0.5.0): `Sgeqrf`/`Dgeqrf`, `Sormqr`/`Dormqr`,
    `Sorgqr`/`Dorgqr`** (requested by go-embedded-julia, julia#67), in
    package `lapack`, with ports of the reference (3.12.0) in `netlib`. The
    algorithms are the reference's: xGEQRF (block 32, crossover 128) over xGEQR2,
    xLARFG, xLARF and, for the block, xLARFT and xLARFB; xORMQR over xORM2R;
    xORGQR over xORG2R. One implementation serves both packages, `internal/ref`,
    parameterised by the four level 2 and 3 kernels it calls (`Gemm`, `Trmm`,
    `Gemv`, `Ger`): `netlib` passes the reference loops, `lapack` the packed
    `Gemm` and `Trmm` of package `blas`, `Gemv`, and the rank-1 update as an
    `Axpy` per column. The workspace of the reference is allocated, not passed, so
    the numbering of an `*blas.ArgError` is the reference's with the workspace
    parameters left in place.

    *The bound.* Per column, against the exact product of the stored reflectors
    (`Q = H(1) ... H(k)` formed in 256-bit arithmetic): `||A(:,j) - (QR)(:,j)||_2
    <= g ||A(:,j)||_2`, `||Q**T Q - I||_F <= g`, and the analogues for Orgqr
    and Ormqr (package comment of `lapack`), `g = gamma_{8(m+n)+16}`. The
    only source opened for it is LAWN 22 (Demmel and Higham, "Stability of
    block algorithms with fast level 3 BLAS"), section 3, equation (30): it shows
    that a product of Householder transformations applied through the WY
    representation is an exact orthogonal update of a perturbation bounded by
    first-order terms in `u`, and concludes that the analysis of the unblocked
    algorithm carries over with changed constants. It does not give this constant.
    **`8(m+n)+16` is a tested constant, not a proved one**: the tests assert it
    in both precisions on random data (and, for Geqrf, extreme-scale data), for
    five block configurations (block, crossover, Ormqr block): (2, 0, 2), (3, 2, 3),
    (4, 5, 4) and (8, 0, 7) on shapes up to 65 x 30 (Ormqr cases up to k = 31),
    where every panel path of the blocked code runs on a few dozen columns, and
    the default (32, 128, 32) on those shapes plus, for that configuration only
    (the exact products would cost minutes under the others), a 160 x 140
    Geqrf and Orgqr (k = 140 above the crossover 128, so the default blocked path
    runs) and an Ormqr case with k = 40 (above its block 32). That is the native
    lanes; the emulated ones (qemu, `go test -short`) run (3, 2, 3) and the
    default on the small shapes only: with every configuration, ppc64le under qemu
    ran past go test's ten minutes. It is a stated bound
    the code is asserted against, and a counterexample would be a defect of the
    bound's constant, to be raised and stated again. The Geqrf test also asserts
    the contract of xLARFG on every tau: 0, or in [1, 2] up to rounding.

    *Tests.* The oracle is the exact `Q` of the stored reflectors in 256-bit
    arithmetic, not agreement with netlib. Where NaN and Inf are compared, it is
    the blocked `lapack` against the `netlib` port at the default block sizes,
    and both run the same `internal/ref` algorithms with different kernels, so
    that comparison shows the kernels agree, not that either matches the
    Fortran reference. Fuzz targets for the six routines
    with a committed corpus; the coverage gate and `guardmut` apply.

    *Measured* against netlib's ports (not against OpenBLAS or LAPACK proper), as
    in decision 10: `lapack/bench_test.go` (`BenchmarkGeqrf`, `BenchmarkOrgqr`,
    `BenchmarkOrmqr`), cross-compiled test binary on cfarm420 (AMD EPYC 7773X,
    linux/amd64), `taskset -c 9`, `GOMAXPROCS=1`, `-test.benchtime 300ms
    -test.count 3`, head 2ddb654, 2026-11-22. cfarm420's 1-minute load was 1.13
    before and 1.97 after. Median of three runs, Gflop/s, `lapack / netlib`, the
    number of runs in parentheses; flops counted are `4n^3/3` (geqrf, orgqr) and
    `2n^3` (ormqr, `nrhs` = n, left, Trans), the operation's own, not the blocked
    code's; one core of a shared machine, no confidence interval:

    | | n = 64 | 128 | 256 | 512 | 1024 |
    |---|---|---|---|---|---|
    | geqrf float64 | 3.84 / 2.33 | 5.15 / 2.34 | 7.24 / 1.97 | 10.70 / 1.98 | 14.35 / 2.20 |
    | geqrf float32 | 2.26 / 2.38 | 2.66 / 2.27 | 6.50 / 2.03 | 11.81 / 2.12 | 18.32 / 2.24 |
    | orgqr float64 | 4.11 / 2.47 | 5.54 / 2.27 | 8.51 / 1.68 | 11.47 / 2.14 | 15.00 / 2.23 |
    | orgqr float32 | 2.34 / 2.50 | 2.68 / 2.24 | 7.56 / 1.56 | 12.36 / 2.12 | 19.21 / 2.17 |
    | ormqr float64 | 2.26 / 1.59 | 4.44 / 1.86 | 7.50 / 2.06 | 10.83 / 2.13 | 13.08 / 2.21 |
    | ormqr float32 | 2.39 / 1.68 | 5.11 / 1.97 | 9.22 / 2.11 | 14.08 / 2.13 | 19.80 / 2.20 |

    From n = 256 up every cell is 3.2 to 9 times netlib's. Two cells
    are behind it: float32 `geqrf` and `orgqr` at n = 64 (2.26 against 2.38 and
    2.34 against 2.50, 5% and 6% under), an open defect. Float32 is also slow for
    its size at n = 64 and 128 in all three routines (about 2.3 to 2.7 Gflop/s,
    against 3.8 to 5.5 in float64), which this run does not explain: it did not
    profile and the spread of the three runs was not tabulated. A likely cause,
    not measured, is that below the crossover of 128 the work is the level 2
    panel (`Gemv` and `Axpy` per column), which is not blocked; a faster panel
    is the next step, a measurement away. This table is against netlib's ports,
    not a claim against OpenBLAS, which this repository has not run.

12. **The symmetric eigensolver (unreleased): `Ssyev`/`Dsyev` and what they
    need, ported from reference LAPACK 3.12, bounds stated as tested constants.**
    The root package gains `Ssymv`/`Dsymv`, `Ssyr2`/`Dsyr2`, `Srot`/`Drot`,
    `Sswap`/`Dswap`, `Scopy`/`Dcopy`; `lapack` gains `Syev` over `sytrd` (blocked
    `latrd` panel plus the packed `Syr2k`, `sytd2` below the crossover), `orgtr`
    (`orgql`/`org2l`; the upper form is the reference's), `steqr` (implicit QL/QR
    with `lasr`) and `sterf` (root-free QL/QR), and the auxiliaries `lamch`,
    `lartg`, `lae2`, `laev2`, `lasr`, `lasrt`, `lascl`, `laset`, `lanst`,
    `lansy`. As for decision 10, the workspace arguments do not exist, and a
    positive `info` (an unconverged eigenvalue) is a result with `err == nil`.
    `netlib` holds the eigen routines on the unblocked reference kernels; the
    BLAS routines above have no `netlib` twin, as they are the reference loops
    already.

    *Bounds.* With `u` the unit roundoff, `gamma_k = ku/(1-ku)`, each constant is
    tested, not proved (the analyses of Demmel, *Applied Numerical Linear
    Algebra*, section 5, and LAWN 22 give the shape `O(n) u ||A||`, not the
    constant of this code): `A - Q T Q^T` in Frobenius norm at most
    `gamma_{16n+32} ||A||_F` (`sytrd`); `Q^T Q - I` the same constant over
    `sqrt(n)` (`orgtr`); `T Z - Z L` at most `gamma_{30n} ||T||_F` and the
    eigenvalues of `steqr`/`sterf` within `gamma_{30n} ||T||_2` of an exact
    Sturm-sequence oracle in 256-bit arithmetic; `Syev` residual `A Z - Z L`,
    each column's residual, `Z^T Z - I` and the eigenvalues against the exact
    ones, all at `gamma_{16n}`. `Symv` is within `gamma_{n+2}(|alpha||A||x| +
    |beta||y|)` per element and `Syr2` within `gamma_4(|A| + |alpha|(|x||y|^T +
    |y||x|^T))` (the reference loop is four operations deep, not three: the
    first version of the test said gamma_3 and failed).

    *Calibration.* Worst measured `value / (gamma_k * norm)` over every shape,
    block-size configuration and matrix kind of the tests (dense spectra
    random, clustered, repeated, graded; tridiagonals random, clustered,
    repeated, graded, tiny off-diagonal, Wilkinson, Clement, zero
    off-diagonal; scales that put the data near the overflow and underflow
    thresholds of `lascl`), float64 / float32: sytrd 0.064 / 0.198, orgtr 0.025 / 0.067, steqr
    `TZ-ZL` 0.035 / 0.042, steqr `Z^T Z - I` 0.069 / 0.036, eigenvalues against
    the Sturm oracle 0.044 / 0.033, `Syev` `AZ-ZL` 0.153 / 0.135, column
    residual 0.135 / 0.130, `Z^T Z - I` 0.137 / 0.115, eigenvalues 0.120 / 0.087
    (`N` only 0.068 / 0.089). The first run used `8n+16` for sytrd and the
    float32 worst case was 0.396, only a 2.5 times margin, so the constant was
    doubled (0.198, 5 times): every constant now has at least a 5 times margin.
    A counterexample would be a defect of the constant, to be raised and
    stated again.

    *Tests.* The oracle is exact arithmetic, not netlib: the exact `Q`, `T`
    and Sturm counts in `math/big`. Blocked `lapack` and `netlib` run the same
    `internal/ref` algorithms on different kernels, so comparing them (used for
    NaN, Inf and the argument errors) shows the kernels agree, not that either
    matches the Fortran. `FuzzSyev` has a committed corpus and ran 60 s
    (1.29 million executions) without a failure before this change; the
    coverage gate, `guardmut` and the fuzz lane apply.

    *Measured* (2026-11-23, cfarm420, AMD EPYC 7773X, one core:
    `taskset -c 9`, `GOMAXPROCS=1`; 1-minute load 3.6 to 4.3 before and after,
    below the 8 limit). Go: `-test.benchtime 300ms -test.count 3`, median of
    three. NumPy: `numpy.linalg.eigh` / `eigvalsh` (`UPLO='L'`, OpenBLAS with
    one thread, so LAPACK `syevd`), median of three 300 ms runs, same
    symmetric matrix (seed 3); the grid was run twice and the second round
    differs from the first by at most 10 % in every cell (the A/A control).
    Times in microseconds per call, `V` = eigenvectors, `N` = values only.
    Go and NumPy ran back to back, not interleaved cell by cell.

    | type | n | mode | lapack | netlib | NumPy (rounds 1 / 2) | lapack / NumPy |
    |---|---|---|---|---|---|---|
    | float64 | 64 | N | 231 | 318 | 169 / 163 | 1.4 |
    | float64 | 64 | V | 851 | 982 | 349 / 348 | 2.4 |
    | float64 | 128 | N | 1183 | 1759 | 645 / 607 | 1.8 |
    | float64 | 128 | V | 4737 | 6302 | 1487 / 1563 | 3.2 |
    | float64 | 256 | N | 6728 | 12274 | 2771 / 2825 | 2.4 |
    | float64 | 256 | V | 31248 | 43465 | 7088 / 7036 | 4.4 |
    | float64 | 512 | N | 40100 | 66801 | 14766 / 15296 | 2.7 |
    | float64 | 512 | V | 218503 | 319415 | 39045 / 41203 | 5.6 |
    | float32 | 64 | N | 243 | 271 | 163 / 179 | 1.5 |
    | float32 | 64 | V | 786 | 783 | 355 / 355 | 2.2 |
    | float32 | 128 | N | 1218 | 1576 | 647 / 614 | 1.9 |
    | float32 | 128 | V | 4896 | 5733 | 1512 / 1440 | 3.2 |
    | float32 | 256 | N | 6799 | 9718 | 3365 / 3405 | 2.0 |
    | float32 | 256 | V | 30121 | 37726 | 8126 / 8404 | 3.7 |
    | float32 | 512 | N | 38092 | 61896 | 17848 / 17811 | 2.1 |
    | float32 | 512 | V | 189793 | 293382 | 46480 / 43533 | 4.1 |

    Reading: the blocked `lapack` path beats `netlib` in every cell (up to
    1.7 times at n = 512, the one tie being float32 n = 64 V), but **every
    row is slower than NumPy, by 1.4 to 5.6 times: an open defect**, not
    accepted. The gap grows with n and is largest with vectors. The cause is
    not profiled yet; the hypotheses are the `steqr` rotations applied to `Z`
    at level 1 (`lasr`) where `syevd` uses divide and conquer (`stedc`) with
    level 3 products, and the `sytrd` / `orgtr` trailing updates not using the
    packed GEMM driver. Profiling and those two are the next slices; the
    contract (the error bounds above) is unchanged by them.

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
mutant run on, `-run` the tests (CI runs them all). The routines never panic, and a test turns a
panic of the routine under test into a failed assertion (`noPanic`).
