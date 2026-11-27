# vulkane-msl-emit

The Metal Shading Language compute-kernel emitter for [Unpopped](https://crates.io/crates/unpopped) (ROADMAP row 8o).
`Msl::lower_kernel` lowers a 32-bit integer kernel-layer kernel to MSL through `vulkane-kernel-text`: a `kernel void`
entry with `device` buffers bound by `[[buffer(n)]]`, `threadgroup` arrays, the thread and threadgroup position
builtins, `threadgroup_barrier`, and structured control flow. Anything it cannot spell is a typed `UnsupportedKernel`
naming it; the NVIDIA-only statements of board 434 C are refused by name.

**Not validated by a Metal compiler yet.** The text is pinned by tests; compiling it needs `xcrun metal` on macOS,
which this repository's runners do not have. Treat the output as unproven until that oracle exists.

Licensed under MIT OR Apache-2.0.
