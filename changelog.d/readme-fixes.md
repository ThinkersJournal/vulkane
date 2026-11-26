kind: Fixed: four emitter READMEs corrected before their first publish
---
`vulkane-hlsl-emit` and `vulkane-wgsl-emit` no longer say "does not lower kernel-layer kernels yet" (false: wgsl lowers integer kernels through
`vulkane-kernel-text`; hlsl refuses float kernels and passes the rest to the Slang path, whose scope is stated UNVERIFIED) and now count the
thirteen declined statements (`CopyUnpack` was missing from the count). `vulkane-glsl-emit` and `vulkane-kernel-text` gain the infringement note
CireSnave asked for on 2026-11-16 (the thirteen names, the patents, the 434 C basis), pinned by `tests/readme_note.rs` in each.
`vulkane-wgsl-parse` ships a `THIRD-PARTY-NOTICES.md` (no third-party source in the package). Docs, one test file per crate, no code change.
