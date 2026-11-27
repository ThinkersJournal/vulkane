# vulkane-kernel-text

The shared text walker for Vulkane's kernel emitters. Given an Unpopped kernel-layer `KernelFn` and a `Dialect`
(`vulkane-glsl-emit`, `vulkane-wgsl-emit`), it lowers integer kernels to shader source: buffers, static workgroup
arrays, locals, `Load`/`Store`, barriers and fences, and structured control flow. What a dialect cannot spell is a
typed `UnsupportedKernel` naming it, never an approximation. The NVIDIA-only statements of board 434 C are refused by
name before anything else.

## Declined: possible patent exposure (board 434 C and its extension)

**Builders of non-NVIDIA emitters: lowering any statement below to a hardware operation risks infringing NVIDIA patents; do not.**
CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for other manufacturers' hardware know about that and we consider ourselves not infringing."* and, for the second patent, *"I think we treat that patent like the others that are Nvidia-only. Add a note and specifically notify our non-Nvidia-emitter builders about the potential for infringement and not to."*

`vulkane-kernel-text` is the walker every Vulkane text emitter (`vulkane-glsl-emit`, `vulkane-wgsl-emit`) shares: it refuses them by name before it lowers anything else, so no dialect built on it lowers them to a hardware operation. **A dialect author must not add a lowering for any of them.**

The thirteen statements refused by name (the same list as `vulkane-kernel-text`'s `src/decline.rs`, which has the basis):

- Board 434 C (CireSnave 2026-11-13, "434: C."): `CopyAsync`, `AsyncCommit`, `AsyncWait` (NVIDIA US11294713B2 and US12118382B2, an API that blocks a thread's instructions around an asynchronous shared-memory copy; `unpopped` `docs/PATENT-SWEEP.md` hit 22), `Collective` and `CollectiveTuple`.
- Board 434 C extended (CireSnave 2026-11-16): `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` and `Tcgen05Commit` (NVIDIA US11803380B2, a synchronization barrier with an arrival counter, a phase and a hardware copy operator), `WgmmaWait` (NVIDIA US12204897B2 and its continuation US12710961B2, a wait on a count of pending matrix-multiply operations) and `CopyUnpack` (unpopped 0.58, covered by the `CopyBulk` note).

The refusal is a typed `UnsupportedKernel` whose text carries `declined: possible exposure, board 434 C` (the five) or `declined: possible exposure, board 434 C extended` (the eight), at the top level and inside `If`, `For` and `While` at any depth. This note is pinned by `tests/readme_note.rs`.

## Subgroup collectives (board 506 A)

Board 506 A (narrowed; CireSnave 2026-11-26): lifted by CireSnave as his risk call; "outside as read" is a reading of public documents, not a clearance; no attorney; findings, not advice. Lifted: subgroup-scope `Reduce` (Sum, Prod, Max, Min, And, Or, Xor), `Shuffle`, `Vote` and `Broadcast`. Still declined by name with the 434 C marker: `Scan`, `ReduceWith`, `ScanWith`, `CollectiveTuple`, every block-scope collective, and every collective on GLSL and SPIR-V. The walker's by-name decline takes a `lift_subgroup` flag (`declined_by_name_with`); a dialect that says `subgroup_collectives()` gets the ruled kinds as `Dialect::subgroup_collective` calls and keeps every other kind declined. A collective acts on the *active* invocations; the uniformity guard is Unpopped's `check_kernel`, which refuses a collective under divergent control or after a divergent return, so only uniform placements reach a dialect.

Licensed under MIT OR Apache-2.0.
