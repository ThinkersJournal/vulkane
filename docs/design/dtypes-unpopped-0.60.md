# The seven dtypes unpopped 0.60 added to `ElementKind`: what each target has, and what the emitters do

**Scope.** unpopped 0.60.1 (published 2026-11-26 22:43Z; E1a, KISS 0.24.0) adds `F128`, `I128`, `U128`, `I256`, `U256`, `F8E3M4`, `F8UE4M3` to
`ElementKind` and `KernelError::DtypeNotComputed { dtype }`: "declined by name, never run" (the oracle holds no value of them). Taken together with fishstringer 0.10.0
(which requires `unpopped ^0.60.1`; the lock holds one copy of every unpopped and fishstringer crate), so `vulkane-hlsl-emit` and the spirv-emit oracle test compile.

## What the emitters do (measured, unpopped 0.60.1, `cargo test`, no device)

* **As a kernel's compute dtype, and as a buffer's element dtype under a `U32` kernel:** `vulkane-glsl-emit`, `-wgsl-emit` and `-spirv-emit` return the typed
  `LowerError::UnsupportedKernel` with `check_kernel refused kernel ...: DtypeNotComputed { dtype: X }` for each of the seven (the control, a `U32` kernel, lowers).
  The refusal is **unpopped's checker's**, reached before any emitter match, and it names the dtype.
* **`Backend::supports_dtype`:** each emitter says `Unsupported` for all seven, by an allow-list with a wildcard arm (`GLSL: _ => Unsupported`; `WGSL: wgsl_type ... _ => None`;
  `SPIR-V: Family::of ... _ => None` then `dtype_ok`). No emitter has an exhaustive `match` on `ElementKind`, which is why the workspace compiled unchanged against 0.60.1
  (`cargo check` of kernel-text, glsl-emit, wgsl-emit, hlsl-parse and spirv-emit libs, and the tests of kernel-text/glsl/wgsl: clean). Every helper read
  (`buffer_elem_ok`, `float_elem`, `type_name`, `cast_to_compute`, `Family::of`, `storage_ok`) returns `None`/`false` for an unknown dtype: none mis-sizes one.
* **Tests (`tests/dtype_decline.rs` in glsl-, wgsl-, spirv- and hlsl-emit; 3 each):** the seven are refused naming the dtype and `DtypeNotComputed`; `supports_dtype` is `Unsupported`
  for the seven and `Supported` for `U32` (control); every `ElementKind::ALL` (35 at 0.60.1) lowers or declines without panicking and the seven are among the refused.
* **Vacuity mutants (all caught by name, source restored, control green):** add `F128` to GLSL's supported list; add `F8E3M4` to WGSL's `wgsl_type`; map `I128` to `Family::I32` in SPIR-V's
  `Family::of`; make HLSL's `supports_dtype` say `Supported` for `F128` instead of delegating to Slang: each fails `each_new_dtype_is_unsupported_to_supports_dtype`. A first SPIR-V mutant (`I128` to `Family::F16`) was NOT caught: an **equivalent mutant**, since `dtype_ok`
  excludes half-width families; it proved nothing and was replaced.
* **What the tests do not pin:** the first test pins unpopped's checker text; an emitter-side change cannot change it (the checker refuses first). The emitter's own decline is pinned by the
  `supports_dtype` test only. `vulkane-msl-emit` has no such test yet (it is on the unmerged stack). For HLSL the emitter's `supports_dtype` delegates to fishstringer's Slang emitter, so the test pins the delegation, not a list of Vulkane's own.

## What each language has (source confidence stated per cell; "unread" is not "absent")

| type | SPIR-V | GLSL | WGSL | HLSL | MSL |
|---|---|---|---|---|---|
| `F128` (binary128) | unread (instruction text 3.3.6 never reached; capability rows read have Int8/16/64, Float16/64 only; core validation rules, summary-derived: 32-bit "plus any additional sizes enabled by capabilities") | absent from the extensions read (`GL_EXT_shader_explicit_arithmetic_types`); GLSL 4.60 core summary-derived | **absent** (spec: "The scalar types are bool, AbstractInt, AbstractFloat, i32, u32, f32, and f16.") | **absent** (HLSL spec 3.9.1: half, float, double, int16/32/64_t, uint16/32/64_t, min16*) | **absent** (MSL 4.1 Table 2.1, verbatim: bool, char, uchar, short, ushort, int, uint, long, ulong, half, bfloat, float, size_t, ptrdiff_t, void; "Metal does not support the double, long long, unsigned long long, and long double data types") |
| `I128`, `U128` | unread (as above) | absent from the extensions read | **absent** | **absent** | **absent** (MSL 4.1 Table 2.1, verbatim: bool, char, uchar, short, ushort, int, uint, long, ulong, half, bfloat, float, size_t, ptrdiff_t, void; "Metal does not support the double, long long, unsigned long long, and long double data types") |
| `I256`, `U256` | unread (as above) | absent from the extensions read | **absent** | **absent** | **absent** (MSL 4.1 Table 2.1, verbatim: bool, char, uchar, short, ushort, int, uint, long, ulong, half, bfloat, float, size_t, ptrdiff_t, void; "Metal does not support the double, long long, unsigned long long, and long double data types") |
| `F8E3M4` | **absent** (FP Encoding table 3.2.55, verbatim: BFloat16KHR, Float8E4M3EXT, Float8E5M2EXT, Float6E2M3EXT, Float6E3M2EXT, Float4E2M1EXT, Float8UnsignedE8M0EXT, MXInt8EXT) | **absent** (`GL_EXT_float8_e5m2_e4m3`: e4m3, e5m2; `GL_EXT_ocp_microscaling_types`: e2m1, e3m2, e2m3, ue8m0, mxint8) | **absent** (no 8-bit float read) | not found: HLSL proposals list E4M3 and E5M2 (signed) as DXIL/linalg component types only, not as scalar types (0029 "Rejected", 0035 "Accepted", planned SM 6.10: not verified as shipped) | **absent** (MSL 4.1 Table 2.14 format types, verbatim: int4b, uint4b, int2b, uint2b, fp4_e2m1, fp8_e4m3, fp8_e5m2, fp8_ue8m0; section 2.21: only as `packed_numeric_type` / tensor element types, not scalars) |
| `F8UE4M3` | **absent** (same table: the only unsigned 8-bit float is UE8M0, a different format) | **absent** (only `floatue8m0_t`) | **absent** | not found (same proposals) | **absent** (MSL 4.1 Table 2.14 format types, verbatim: int4b, uint4b, int2b, uint2b, fp4_e2m1, fp8_e4m3, fp8_e5m2, fp8_ue8m0; section 2.21: only as `packed_numeric_type` / tensor element types, not scalars) |

Notes: (1) A target has the *signed* `E4M3`/`E5M2` 8-bit floats (SPIR-V `SPV_EXT_float8`; GLSL `GL_EXT_float8_e5m2_e4m3`, Draft; HLSL as DXIL component types), but **no emitter here lowers any 8-bit
float**; `F8E3M4` and `F8UE4M3` are different formats from those. (2) MSL was read from the Metal Shading Language Specification version 4.1 (dated 2026-06-04; 383 pages) downloaded to a scratch directory and text-extracted with pymupdf: Table 2.1 (page 25-26), section 2.21 and Table 2.14 (page 79-80). That confirms what the `vulkane-msl-emit` docs record (FP8 only as `packed_numeric_type` and tensor format tags).
**MSL Table 2.16 (FP8 format type properties), page 80 of 383, dated 2026-06-04: a typo in the specification, read from the rendered page image (not an extraction artefact).** For `metal_fp8_e5m2_format` the page prints Infinities `S 11111 00₂` and, directly below, NaN `S 11111 {00, 10, 11}₂`: the two cells overlap at mantissa `00`. The extracted text said the same; the rendered image shows the identical cell. Which pattern Apple intends is **not stated by the specification**: an IEEE-style E5M2 would have NaN on `{01, 10, 11}`, but that is a reading, so the NaN set of `metal_fp8_e5m2_format` is **UNVERIFIED**. The other cells read cleanly on the image: `metal_fp8_e4m3_format` bias 7, no infinities, NaN `S 1111 111₂`, max normal `±448`; `metal_fp8_ue8m0_format` bias 127, NaN `11111111₂`; `metal_fp4_e2m1_format` (Table 2.15) bias 1, no infinities or NaN, max normal `±6.0`. (3) Reading limits: WGSL scalar types read verbatim (spec source `wgsl/index.bs`, section "Scalar Types"); HLSL 3.9.1 read in full; MSL 4.1 Tables 2.1 and 2.14 verbatim from the extracted PDF; SPIR-V FP Encoding table and the
`SPV_EXT_float8` page verbatim, the width list in OpTypeInt/OpTypeFloat NOT reached; GLSL 4.60 core summary-derived. (4) This document says nothing about patents.

## Still to do

msl-emit (on the unmerged stack): the same `dtype_decline.rs` on the stack tip after rebase, with its own mutant.
