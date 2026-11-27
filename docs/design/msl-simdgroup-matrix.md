# MSL SIMD-group matrices and the Metal 4 tensor path: what the specification says (facts, not a design)

> **Condition, to be read with everything in this file about tile statements (`TileLoad`, `TileMma`, `TileStore`):**
> lifted by CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance; no attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read.
> This document says nothing about infringement and never calls anything cleared or non-infringing.

**Status (2026-11-27):** reference only; still true on main 0.31.2: unpopped 0.61.0 has no `TileLoad`, `TileMma` or `TileStore` statement (0 files in its source; the control `ProxyFence` is present), so the decline-by-name arms wait on their publication. **No emitter code is started** (pm-kernels, 2026-11-26): `vulkane-msl-emit` is the first consumer of Unpopped's B1 opaque-tile statements, and those statements are not
specified yet. This file records what Apple's specification says so the B1 plan and the emitter row can use it.

**Source and method.** *Metal Shading Language Specification, Version 4.1* (Apple, dated 2026-06-04, 383 pages), downloaded from
`https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf` into a scratch directory and text-extracted with pymupdf (`python -I`).
Quotes are verbatim from that text; page numbers are the PDF page. A cell marked **UNVERIFIED** was not read or not measured. The *Metal Feature Set Tables* (GPU families) are an
external document that was **not read**.

## 1. The SIMD-group matrix type (section 2.4, p. 38)

* "All OS: Metal 2.3 and later support SIMD-group matrix types." Type `simdgroup_matrix<T,Cols,Rows>` in `<metal_simdgroup_matrix>`.
* "Operations on SIMD-group matrices are executed cooperatively by threads in the SIMD-group. Therefore, all operations must be executed only under uniform control-flow within the SIMD-group or the behavior is undefined."
* Types: "`T` is `half`, `bfloat` (in Metal 3.1 and later) or `float` and `Cols` and `Rows` are 8": `simdgroup_half8x8`, `simdgroup_bfloat8x8` (Metal 3.1 and later), `simdgroup_float8x8`.
  **Only 8x8.** No 16x16 and no integer element type appears in sections 2.4 or 6.8.
* "The mapping of matrix elements to threads in the SIMD-group is **unspecified**." The text of 4.1 has **no per-element access** (`thread_elements`: 0 occurrences in the 383 pages;
  control: `make_filled_simdgroup_matrix` is found). A consumer therefore cannot address an element: a matrix is opaque and moves only by load, store, fill and multiply.

## 2. Functions (section 6.8, p. 212-213)

Section 6.8 opens: "Instead of using simdgroup matrix multiplication, consider using Tensors (section 2.22) and Metal Performance Primitives (section 7). Together, they provide a performant and portable
library supporting more general matrix and tensor operations across a wider variety of dimensions and extents." (A recommendation; the word *deprecated* does not appear next to these functions.)

| function (Table 6.9 / 6.10) | what the text says |
|---|---|
| `simdgroup_matrix<T,Cols,Rows>(T dval)` | "Creates a diagonal matrix with the given value." |
| `make_filled_simdgroup_matrix(T value)` | "Initializes a SIMD-group matrix filled with the given value." (a zero matrix is `make_filled_simdgroup_matrix(T(0))`) |
| `simdgroup_load(thread simdgroup_matrix<T,Cols,Rows>& d, const threadgroup T *src, ulong elements_per_row = Cols, ulong2 matrix_origin = 0, bool transpose_matrix = false)` | "Loads data from threadgroup memory into a SIMD-group matrix. The `elements_per_row` parameter indicates the number of elements in the source memory layout." |
| the same with `const device T *src` | "Loads data from device memory into a SIMD-group matrix." |
| `simdgroup_store(thread simdgroup_matrix<T,Cols,Rows> a, threadgroup T *dst, ulong elements_per_row = Cols, ulong2 matrix_origin = 0, bool transpose_matrix = false)` and the `device T *dst` form | "Stores data from a SIMD-group matrix into threadgroup / device memory." |
| `simdgroup_multiply_accumulate(d, a, b, c)` with `d: <T,Cols,Rows>`, `a: <T,K,Rows>`, `b: <T,Cols,K>`, `c: <T,Cols,Rows>` | "Returns d = a * b + c" |
| `simdgroup_multiply(d, a, b)` | "Returns d = a * b" |

Facts that follow from the signatures (not extra claims): load and store take only the **`threadgroup` and `device`** address spaces (no `thread`, `constant`); there is a **leading dimension** (`elements_per_row`),
an **origin** (`matrix_origin`, a `ulong2`) and a **transpose flag** on both load and store; `d`, `a`, `b` and `c` share **one `T`**, so the spec's `simdgroup_multiply_accumulate` has **no mixed-precision
overload in the text** (half inputs with a float accumulator is not a form in 6.8; but see section 5: Apple's compiler accepted that call). **Accumulate precision inside the instruction is not stated: UNVERIFIED.**

## 3. The Metal 4 tensor path (sections 2.22, 7.1, 7.2.1; p. 350-354)

* "All OS: Metal 4 and later support tensor types" (2.22): `tensor<...>` (address space constant, device, threadgroup or thread) and `cooperative_tensor<...>` ("whose storage is in thread and pre-partitioned among a set of participating threads").
* "All OS: Metal 4 and later support Metal Performance Primitives" (7): namespace `mpp`, `tensor_ops::matmul2d`, tuned "for Apple silicon GPUs"; "For a list of supported GPU families, refer to the Metal Feature Set Tables" (not read: UNVERIFIED).
* Execution scopes (7.1): `execution_thread`, `execution_simdgroups<N>` (`execution_simdgroup` for N == 1; "TensorOp support N with a value of 1 or `simdgroups_per_threadgroup`").
* `matmul2d` (7.2.1): "multiplies an M×K tensor A by a K×N tensor B and accumulates the result into an M×N tensor C"; descriptor `matmul2d_descriptor(M, N, K = dynamic_length_v<int>, transpose_left = false, transpose_right = false, relaxed_precision = false, mode = mode::multiply)` (Table 7.4); "`relaxed_precision`: ... can use relaxed precision for float data type. Relaxed precision allows the operation to truncate the mantissa before the multiplication. The default is false."
* **Mixed precision exists here** (Table 7.3, p. 351), Metal 4: `half x half -> half` and `half x half -> float`, `float x float -> float`, `char x char -> int`, `char x half -> half or float`, ...; Metal 4 and OS 26.1: `bfloat x bfloat -> bfloat or float`, and more; Metal 4.1 adds `metal_fp8_e4m3_format` / `metal_fp8_e5m2_format` combinations with `half`/`float` (rows seen, not all read).
* "All threads must call the method within that scope, or the result is undefined" (7.2): the same uniformity rule as 2.4, at the execution scope.
* "TensorOps may use a barrier at the level of the execution scope"; a TensorOp that writes a `device` or `threadgroup` tensor needs the caller to insert a barrier before reading it; none for `thread` memory or a `cooperative_tensor`.

**So the 4.1 tensor path does not replace the 8x8 functions in the text; Apple recommends it** (6.8), it needs Metal 4 (OS 26 class), it takes dynamic dimensions and mixed precision, and its operand objects are `tensor` / `cooperative_tensor`
(a different shape of operand from an opaque 8x8 matrix). Which one a given Unpopped tile statement should lower to is **not decided here**.

## 4. Availability

| form | stated availability (this spec) | GPU family |
|---|---|---|
| `simdgroup_half8x8`, `simdgroup_float8x8` and 6.8 functions | Metal 2.3 and later, all OS (2.4) | **UNVERIFIED** (Feature Set Tables not read) |
| `simdgroup_bfloat8x8` | Metal 3.1 and later (2.4) | UNVERIFIED |
| tensors, MPP, `matmul2d` | Metal 4 and later (2.22, 7) | UNVERIFIED |
| `matmul2d` with `bfloat` operands | Metal 4 and OS 26.1 (Table 7.3) | UNVERIFIED |
| FP8 combinations | Metal 4.1 (Table 7.3, 2.14) | UNVERIFIED |

## 5. What the hosted Apple device accepts (measured, compile only)

Probe: `.github/metal/simdgroup_matrix.metal` and `.github/metal/matmul2d_tensor.metal` plus a "records, never fails" step in `.github/workflows/metal-probe.yml` on branch `macos-metal-probe`
(the spec's own examples and negative controls), compiled with Apple's `xcrun -sdk macosx metal` on a GitHub-hosted `macos-latest` runner. **Compile only: nothing was run, so nothing here says the hosted
paravirtual device executes these instructions.**

Environment (from the run log): GitHub-hosted `macos-latest`, macOS 26.7.2, Xcode 26.6 (build 17F113), `Apple metal version 32023.883 (metalfe-32023.883)`; run `45910703000`, commit `0b7c139` of `macos-metal-probe`. Exit code of `metal -c` (0 = the compiler accepts the source):

| variant (flags) | rc | what the compiler said |
|---|---|---|
| `simdgroup_float8x8` multiply-accumulate (the spec's own example), `-std=metal3.2` | 0 | accepted |
| the same, `-std=metal3.0` and `-std=metal4.0` | 0 | accepted |
| `simdgroup_half8x8`, `-std=metal3.2` | 0 | accepted |
| `simdgroup_bfloat8x8`, `-std=metal3.2` | 0 | accepted |
| `simdgroup_bfloat8x8`, `-std=metal3.0` | 1 | `unknown type name 'simdgroup_bfloat8x8'; did you mean 'simdgroup_float8x8'?` (consistent with "Metal 3.1 and later") |
| **MIXED: `a`, `b` `simdgroup_half8x8`, `c`, `r` `simdgroup_float8x8`**, `-std=metal3.2` | **0** | **accepted** (I expected a refusal from the single-`T` signature in 6.8) |
| `simdgroup_matrix<float, 16, 16>`, `-std=metal3.2` | 1 | refused (error raised in the `metal_simdgroup_matrix` header; message truncated in the summary line) |
| `matmul2d` TensorOp (the spec's `matrixMultiply` example), `-std=metal4.0` | 0 | accepted |
| the same, `-std=metal3.2` | 1 | `error: expected namespace name` (`mpp` is not available below Metal 4) |

Reading, with its limits:

* The 8x8 float and half forms and the Metal 4 `matmul2d` compile on this toolchain, and the controls that should fail do fail for the stated reason (bfloat below 3.1, 16x16, tensors below Metal 4).
* **A discrepancy between the specification text and the compiler:** the mixed half-in, float-accumulator call compiles, although Table 6.10 shows one `T`. The compiler may have an overload the spec does not list, or may convert the operands: **which one is UNVERIFIED**, and **the accumulate precision of that call is UNMEASURED** (compile only). The specification text alone does not support a mixed 8x8 form; this measurement shows the toolchain accepts the call.
* **Not measured:** execution on the hosted device (it is a paravirtual device, not Apple hardware; the existing device probe in the same workflow records what it returns), results, throughput, and every GPU family. Nothing here says the instructions run.

## 6. Open questions for the B1 plan (none answered here)

* Which lowering a `TileMma` takes (8x8 `simdgroup_multiply_accumulate`, or `matmul2d` on tensors) and the precision rule when the IR asks for half inputs with a float accumulator: only the tensor path has that mixed form in the text.
* How a tile larger than 8x8 is composed from 8x8 matrices (not described in 6.8), and whether the opaque element-to-thread mapping ("unspecified") is compatible with an IR statement that names elements.
* Address spaces: the 8x8 functions load and store `threadgroup` and `device` only; a tile in registers is not a memory operand.
* Uniform control flow: both forms are undefined otherwise; the 506 uniformity guard (unpopped 0.59.1 `check_kernel`) is about subgroup collectives and is **not shown to cover** these statements.
* Whether any target below Metal 4 matters (the probe's Metal 3.x results above are compile-only and say nothing about GPUs).
