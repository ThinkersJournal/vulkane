# vulkane-glsl-emit

The GLSL compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped). `Glsl::lower_kernel` lowers an
integer kernel-layer kernel to GLSL 4.50 through `vulkane-kernel-text`: `std430` storage buffers, `shared` arrays,
`barrier()` with the memory barriers, `Fence`, and structured control flow. Anything it cannot spell is a typed
`UnsupportedKernel` naming it; the NVIDIA-only statements of board 434 C are refused by name.

## Declined: possible patent exposure (board 434 C and its extension)

**Builders of non-NVIDIA emitters: lowering any statement below to a hardware operation risks infringing NVIDIA patents; do not.**
CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for other manufacturers' hardware know about that and we consider ourselves not infringing."* and, for the second patent, *"I think we treat that patent like the others that are Nvidia-only. Add a note and specifically notify our non-Nvidia-emitter builders about the potential for infringement and not to."*

`vulkane-glsl-emit` targets non-NVIDIA hardware and does **not** lower any of them: `Glsl::lower_kernel` refuses them by name through `vulkane-kernel-text` before it lowers anything else.

The thirteen statements refused by name (the same list as `vulkane-kernel-text`'s `src/decline.rs`, which has the basis):

- Board 434 C (CireSnave 2026-11-13, "434: C."): `CopyAsync`, `AsyncCommit`, `AsyncWait` (NVIDIA US11294713B2 and US12118382B2, an API that blocks a thread's instructions around an asynchronous shared-memory copy; `unpopped` `docs/PATENT-SWEEP.md` hit 22), `Collective` and `CollectiveTuple`.
- Board 434 C extended (CireSnave 2026-11-16): `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` and `Tcgen05Commit` (NVIDIA US11803380B2, a synchronization barrier with an arrival counter, a phase and a hardware copy operator), `WgmmaWait` (NVIDIA US12204897B2 and its continuation US12710961B2, a wait on a count of pending matrix-multiply operations) and `CopyUnpack` (unpopped 0.58, covered by the `CopyBulk` note).

The refusal is a typed `UnsupportedKernel` whose text carries `declined: possible exposure, board 434 C` (the five) or `declined: possible exposure, board 434 C extended` (the eight), at the top level and inside `If`, `For` and `While` at any depth. This note is pinned by `tests/readme_note.rs`.

Licensed under MIT OR Apache-2.0.
