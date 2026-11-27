# vulkane-msl-emit

The Metal Shading Language compute-kernel emitter for [Unpopped](https://crates.io/crates/unpopped) (ROADMAP row 8o).
`Msl::lower_kernel` lowers a 32-bit integer kernel-layer kernel to MSL through `vulkane-kernel-text`: a `kernel void`
entry with `device` buffers bound by `[[buffer(n)]]`, `threadgroup` arrays, the thread and threadgroup position
builtins, `threadgroup_barrier`, and structured control flow. Anything it cannot spell is a typed `UnsupportedKernel`
naming it; the NVIDIA-only statements of board 434 C are refused by name.

**Subgroup collectives (board 506 A), integer kernels only.** Board 506 A (narrowed; CireSnave 2026-11-26): lifted by CireSnave as his risk call; "outside as read" is a reading of public documents, not a clearance; no attorney; findings, not advice. Lifted: subgroup-scope `Reduce` (Sum, Prod, Max, Min, And, Or, Xor), `Shuffle`, `Vote` and `Broadcast`. Still declined by name with the 434 C marker: `Scan`, `ReduceWith`, `ScanWith`, `CollectiveTuple`, every block-scope collective, and every collective on GLSL and SPIR-V. Lowered to `simd_sum`, `simd_shuffle` and the other SIMD-group functions of MSL 4.1 6.10.2. `Ballot` and every float kernel stay declined. The text is pinned by tests; no device has run it.

**Not validated by a Metal compiler yet.** The text is pinned by tests; compiling it needs `xcrun metal` on macOS,
which this repository's runners do not have. Treat the output as unproven until that oracle exists.

Licensed under MIT OR Apache-2.0.
