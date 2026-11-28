# Changelog

All notable changes to vulkane will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [kiss-vulkan-vocab 0.6.0] — vocabulary version 7 (BREAKING)

`kiss-vulkan-vocab` 0.6.0 (0.5.0 is the vocabulary-6 release): `VOCABULARY_VERSION` is 7 and the `<arith>` field gains three names, `a64` (`shaderBufferInt64Atomics` AND `shaderSharedInt64Atomics`), `af32` (`shaderBufferFloat32AtomicAdd`) and `af64` (`shaderBufferFloat64AtomicAdd`), per KISS#837 (`spec/namespaces/vulkan.md`, V-16, KISS 0.19.0). `vulkane`'s optional dependency now requires `^0.6.0`.

**Breaking, and why:** all three feature bits predate the registry baseline (`VK_HEADER_VERSION` 348), so naming them changes the token of every device that has them. **Migration:** a device that spelled `arith-dot8-f16-i8` at version 6 spells `arith-a64-dot8-f16-i8` at version 7 if it has 64-bit atomics; a consumer that cached or matched a version 6 token for such a device must re-derive it, and one that asserts `VOCABULARY_VERSION` (KISS-CLASSIFY-6.8-0009) must move to 7. Measured on this repo's Windows GPU: its arith field went from `dot8-f16-f64-i16-i64-i8-st16-st8` to `a64-dot8-f16-f64-i16-i64-i8-st16-st8`. The committed §6.8-0008 manifest moves in exactly three lines (`vocabulary_version`, `generated_from`, `arith_names`); no vector or digest changed. Additive in the crate API (`Arith::ATOMIC64`, `ATOMIC_ADD_F32`, `ATOMIC_ADD_F64`).

## [kiss-vulkan-vocab 0.5.0] — vocabulary version 6 (BREAKING)

`kiss-vulkan-vocab` 0.5.0 (0.4.9 is served on crates.io with different source): the `x<n>` escape is removed and `VOCABULARY_VERSION` is 6. `vulkane`'s optional dependency now requires `^0.5.0`. See the vocab-6 fragment for the full change.

## [vulkan_gen 0.5.2] — vulkan_gen version bump (row 9a source change)

`vulkan_gen` 0.5.2: the row 9a module-cycle fix changes its source, and 0.5.1 is already served on crates.io with different source, so the version-divergence gate requires an unpublished version. No other crate version was bumped. The same 0.5.2 (still unpublished) also carries the `vk.xml` 368 generator fix: a mutable `void*` parameter is no longer turned into a by-value `-> c_void` safe method (`vkGetLatencyTimingsLegacyNV`, new in 368).

## [0.34.0] — **BREAKING**: take unpopped ~0.64 and fishstringer ~0.14; `Stmt::Assume` is omitted

**Dependencies (the rule: most recent versions).** `unpopped`, `unpopped-vocab`, `unpopped-cpu-c` and `unpopped-convert` move from `~0.63.0` to `~0.64.0` (the lock resolves **0.64.3**, the published line; 0.64.1 and 0.64.2 are additive inside it), and `fishstringer-slang-emit`, `-slang-parse` and `-kernels-source` from `~0.13.0` to `~0.14.0` (lock 0.14.0: it requires unpopped 0.64). One copy of `unpopped-vocab`: `cargo tree -i unpopped-vocab` prints one line and `cargo tree -d` lists no `unpopped`, `fishstringer` or `fireline` duplicate. Vulkane has no `fireline` dependency (0 hits in every manifest and in `Cargo.lock`). The lock changes only those seven packages. Breaking because unpopped's types are in the public surface and `Stmt` is exhaustive on purpose.

**`Stmt::Assume` (unpopped 0.64.0, ROADMAP U50) is OMITTED, by ruling.** `Assume { cond }` states that `cond` holds for each thread that reaches it: undefined behaviour on a device when false, a typed trap in the oracle, no memory read or written, nothing ordered. The 0.64.0 CHANGELOG entry (read at the release tag, not paraphrased): "an emitter may lower it to an optimiser hint or ignore it (`unpopped-cpu-c` ignores it) and must not rely on it otherwise". So omitting it cannot change defined behaviour, and it keeps every kernel a front end gives a `multiple_of` fact (a decline would refuse working kernels for nothing; pm-kernels first said decline, then ruled ignore after this was put to them: `[FIX]` of 2026-11-28). Omitted in the shared walker (`vulkane-kernel-text`: GLSL, WGSL, HLSL text, MSL) and in the SPIR-V emitter; `coverage()` says lowered-by-omission; every exhaustive `Stmt` match names it (kernel-text `coverage` and `decline`, the SPIR-V and HLSL `patent_decline` lists). **No optimiser hint is emitted:** SPIR-V `OpAssumeTrueKHR` needs `SPV_KHR_expect_assume`, which no emitter here enables, and nothing in this change read its documented use; a hint is a later row. HLSL goes through `fishstringer-slang-emit`, which omits it since 0.14.

**Tests.** `vulkane-test-support::stmt_kernel::assume_pairs()` gives the same kernel with and without the statement in three placements (first, between and last, alone). `an_assume_never_changes_the_emitted_code_of_the_rest_of_the_kernel` (GLSL, WGSL, MSL, HLSL) and `an_assume_never_changes_the_emitted_module_of_the_rest_of_the_kernel` (SPIR-V) assert byte-equal emission with and without it, the with-assume kernel checker-valid, and a control that two different kernels do NOT emit equal text. **Red first:** with the arm disabled, the with-assume kernel does not lower (the walker and SPIR-V refuse it as "not lowered"). **Mutants, each caught by name:** the walker emitting a comment for it, emitting `return;` for it, and the arm disabled; SPIR-V with the arm disabled and with it lowered as a `Return`. The HLSL test has no mutant of its own: its emission is fishstringer's, which this repository does not mutate.

**A second change in what the checker declines, found by the tests (the first was 0.62's wide integers).** unpopped 0.63/0.64 computes `F128` too (ROADMAP E3, the soft binary128 of `oracle/kernel_f128.rs`): `check_kernel` no longer answers `DtypeNotComputed` for it. The `dtype_decline` tests of the GLSL, WGSL, HLSL and SPIR-V emitters asserted the checker's refusal for `F128` and failed on the first full run after the take (4 failures, every other test passed). `F128` now joins the four wide integers in the group whose refusal must come from the EMITTER's own dtype gate (naming the dtype, not the checker); `F8E3M4` and `F8UE4M3` stay checker-declined. Every backend refuses `F128` today ("compute dtype is F128"), so no behaviour changes; what changed is who says no.

**Read, not assumed.** `Numerics` in 0.64.3 is unchanged from 0.63: default `round: NearestEven, ftz: false, approx: false, spelling: None`, so the doc line in `vulkane-spirv-emit/src/lib.rs` stays true; this repository has no `Numerics { .. }` literal in `*.rs`. The eight cited sites stay at `0.33` here: the number is stamped by the gate, and the citation moves with the stamp (citing 0.34 before it made the live gate print eight STALE: the own-version rule compares to the `Cargo.toml` version at the checked commit, 0.33.2).

## [0.33.2] — tooling: a citation of the workspace's own unpublished version is accepted (cited_versions.py)

**The defect class.** `cited_versions.py --gate` compared a citation only to the registry's max stable version, so the release PR of X.Y.0 could not cite X.Y: crates.io serves X.(Y-1).z until the PM publishes, the citation was STALE at CI time, and the publish waits on a green main. The red was unfixable inside the PR that caused it, and moved to a separate PR after every publish (#203, #205). PM ruling, option (a), narrowest form (relayed by pm-assist): compare to the `Cargo.toml` version AT THE CHECKED COMMIT, never to a flag.

**The rule.** A citation is `ok` if it accepts the served version (as before), or if it is a citation of THIS crate and accepts the crate's own `Cargo.toml` version while that version is AHEAD of what the registry serves (`accepts_own_unserved`). Everything else is as today: older than served is STALE, a version ahead of the tree's own is STALE, another crate's number is not borrowed (the own version is looked up by the cited crate's own name, `verdict_for`), once the registry serves it the served-version rule decides again, a never-published crate is NEVER-PUBLISHED, and an unparsed requirement or an unparsed own version is not guessed ok. An accepted own-version row prints a `^ accepted: ...` note so the acceptance is visible. No flag, no allow.

**Tests, RED first.** In the tool's own `--self-test`: the four cases of the ruling (own unpublished ok; older than served STALE; ahead of the workspace STALE; another crate's version STALE) and five controls (today's rule; the registry serving the own version; a registry AHEAD of the tree gets no own-version pass; never-published; unparsed). With `verdict_for` stubbed to today's behaviour, only the own-unpublished case failed (the other three already expect STALE); after the change all pass. Four mutants, each caught by a named arm: no `accepts(req, own)` check, the own version borrowed from `vulkane` for every crate, the registry-ahead guard dropped, the own version never consulted. Replayed against the real manifests (`cargo metadata`: 17 members, all at 0.33.0): citing 0.33 while the registry serves 0.32.2 is `ok` with the note; citing 0.34 is STALE. Note that this workspace gives every crate one version number, so within it the own-version rule applies to each member at 0.33.0; the "another crate" case is for a member whose version differs.

## [0.33.1] — docs: the version cited for consumption moves from 0.32 to 0.33 (8 sites)

Publishing 0.33.0 (the take of unpopped ~0.63 and fishstringer ~0.13, BREAKING) made every `vulkane = { version = "0.32", ... }` a citation that no longer accepts the served version, so `python .github/cited_versions.py --gate` (required check "Local gate harness self-test") is red on main until this merges. The tool lists the same eight sites as for 0.32: `README.md` lines 87, 178, 261 and 289, `vulkane/src/lib.rs:15`, `vulkane/src/safe/naga.rs:8`, `vulkane/src/safe/shaderc.rs:10` and `vulkane_derive/README.md:13`. Each moves to `0.33`. The gate lists all eight STALE (0.32 against the served 0.33.0) before and prints all eight `ok` after. No code and no version edit: the gate stamps the number. This is the release-checklist step after a breaking publish.

## [0.33.0] — **BREAKING**: take unpopped ~0.63 and fishstringer ~0.13

**Dependencies (the rule: most recent versions).** `unpopped`, `unpopped-vocab`, `unpopped-cpu-c` and `unpopped-convert` move from `~0.62.0` to `~0.63.0` (the lock resolves 0.63.8, the newest published at 02:40Z; the first gate of this branch ran on 0.63.5 and was repeated on 0.63.8, so the evidence is at the head), and `fishstringer-slang-emit`, `-slang-parse` and `-kernels-source` from `~0.12.0` to `~0.13.0` (0.13.0 requires `unpopped ^0.63.3`, read from crates.io). One copy of `unpopped-vocab` in the tree: `cargo tree -i unpopped-vocab` prints one line and `cargo tree -d` lists no `unpopped` or `fishstringer` duplicate. Breaking because `unpopped`'s types are in this repository's public surface, as for 0.62.

**What 0.63 changes for the emitters, read and not assumed.** `Numerics` gains a `spelling: Option<ApproxSpelling>` field (ROADMAP U37, a hint that never changes the meaning of the node). A `Numerics { .. }` struct literal without `..Numerics::default()` stops compiling; this repository has none (`Numerics\s*\{` 0 hits in `*.rs` and `*.md`; the same grep finds the `Stmt::Load {` literals, so the pattern works). The doc line of `vulkane-spirv-emit/src/lib.rs` that says `Numerics::default()` is `ftz: false` stays true: the 0.63.5 default is `round: NearestEven, ftz: false, approx: false, spelling: None`. No source edit was needed: every crate builds `--all-targets --locked` on 0.63.5 as it stands.

**Lock.** The lock changes only `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-convert` and the three `fishstringer-*` crates; a full `cargo update` also moved `ahash` 0.8.14 to 0.8.15, which is not in this graph's resolved tree (`cargo tree -i ahash`: nothing to print), so it was put back with `--precise` to keep the diff to what this PR is about.

## [0.32.2] — docs: the version cited for consumption moves from 0.31 to 0.32 (8 sites)

Publishing 0.32.0 made every `vulkane = { version = "0.31", ... }` a citation that no longer accepts the served version, so `python .github/cited_versions.py --gate` (required check "Local gate harness self-test") went red on main and refused the publish of 0.32.1. The tool lists eight sites, not the five first named: `README.md` lines 87, 178, 261 and 289, `vulkane/src/lib.rs:15`, `vulkane/src/safe/naga.rs:8`, `vulkane/src/safe/shaderc.rs:10` and `vulkane_derive/README.md:13`. Each moves to `0.32`, which accepts 0.32.0 and 0.32.1. The gate prints all eight `ok` after, and still lists the same eight as STALE before (read at origin/main 3c53b8c). No code and no version edit: the gate stamps the number.

## [0.32.1] — `vulkane::kiss`: a cooperative-matrix entry of a non-subgroup scope is omitted whole and reported as `vulkan:scope:<n>` (KISS V-18)

**KISS 0.24.5 (KISS#853, V-18) rules that a `cm-` tuple names a Subgroup-scope shape**: the tuple carries no scope coordinate, so an entry of any other `VkScopeKHR` (workgroup 2, device 1, queue family 5, or a value the registry does not give us) must be omitted whole, never spelled as a subgroup tuple, and the device must report it in `unnamed_types` as `vulkan:scope:<n>`. `DeviceCapabilities::of` read every `VkCooperativeMatrixPropertiesKHR` entry and never looked at `scope` (the answer given to KISS on 2026-11-28 after a grep that first looked in the wrong crate: the producer is `vulkane/src/kiss.rs`, not `kiss-vulkan-vocab`). It now keeps only `VK_SCOPE_SUBGROUP_KHR` entries as shapes; each other scope is appended to `unnamed_types` after the `vulkan:component_type:<n>` entries, sorted by `n` and unique.

**Nothing moves for the devices we have seen.** The committed device fixture has two devices (AMD Radeon 610M, 11 entries; NVIDIA RTX 4070 Laptop, 15), every one of the 26 entries scope 3 (subgroup), so their tokens and `unnamed_types` are unchanged; no non-subgroup entry has been seen on any device (lavapipe, NVIDIA coopmat2-class, Intel and Apple unchecked). No stored kernel key contains a `cm-` tuple from this repository, and KISS states no stored byte moves and no vocabulary bump. The change is to the token SET of a device that does advertise such entries (it no longer over-claims).

**Two readings, stated rather than assumed.** (1) An omitted entry is omitted before its component types are considered, so a non-subgroup entry holding an unnamed component type reports its scope but NOT `vulkan:component_type:<n>` (V-18 says the entry is omitted whole; it does not say whether its types are still reported). (2) The order of `unnamed_types` is the component-type entries as before, then the scope entries; V-18 fixes the spelling, not the order across the two families. Both are asked of KISS in the PR.

**Tests.** `subgroup_shapes` is a pure function, so the rule is tested without a device: only the subgroup entry becomes a shape and each other scope (including `1000`, which the registry does not name) is reported once, by number, sorted; the all-subgroup control reports nothing; a workgroup-only device yields `CoopMatrix::None`; the registry's scope numbers (1, 2, 3, 5) are asserted against the generated binding. Mutants, each caught by name: every entry kept, the dedup removed, the sort removed, the reported number shifted.

## [0.32.0] — **BREAKING**: take unpopped ~0.62 and fishstringer ~0.12; the four Tile statements and the Tile memory space are declined by name; the wide integers are now an emitter's own refusal

**Dependencies (the rule: most recent versions).** `unpopped`, `unpopped-vocab`, `unpopped-cpu-c` and `unpopped-convert` move from `~0.61.0` to `~0.62.0`, and `fishstringer-slang-emit`, `-slang-parse` and `-kernels-source` from `~0.11.0` to `~0.12.0` (live on crates.io 2026-11-27, the fishstringer take of unpopped 0.62). One copy of `unpopped-vocab` in the tree (`cargo tree -i unpopped-vocab`: one line). Breaking because `unpopped`'s `Stmt` and `MemorySpace` are in this repository's public surface and `Stmt` is exhaustive on purpose.

**Tile statements declined by name (ROADMAP row 8j).** unpopped 0.62 adds `Stmt::TileLoad`, `TileMma`, `TileFill`, `TileStore` and `MemorySpace::Tile`. Every exhaustive match on `Stmt` here (`vulkane-kernel-text` coverage and decline, the SPIR-V and HLSL patent-decline lists) names them, with `Coverage::Row` pointing at row 8j and the patent hold written into the note (no transpose option, no saturating variant: US12405770B2). The walker and the SPIR-V checker refuse a `Tile` allocation by an explicit arm that names the space, instead of the generic "not a static shared or private array" text a wildcard would give. **The 519 B text on the tile statements is CireSnave's risk call, not a clearance.** Nothing lowers a tile yet: a Vulkan cooperative-matrix lowering (SPIR-V, GLSL) and simdgroup matrices (Metal) are later rows.

**A change in what the checker declines, found by the tests.** Since 0.62 `check_kernel` COMPUTES the four wide integers (`I128`, `U128`, `I256`, `U256`, ROADMAP E2) and no longer refuses them with `DtypeNotComputed`; `F128`, `F8E3M4` and `F8UE4M3` are still checker-declined. An integer constant of a wide type is now an illegal leaf. The `dtype_decline` tests of the GLSL, WGSL, HLSL and SPIR-V emitters asserted the checker's `DtypeNotComputed` for all seven; they now use a load-and-store kernel and assert, for the four wide ones, that the EMITTER's own gate names the dtype (not the checker, not a lowering): each backend refuses them today (probe: GLSL, "compute dtype is I128"). The old assertion would have passed a wrong lowering of a wide integer as long as the checker said no; this one does not.

**Tests.** `the_tile_statements_and_a_tile_space_are_refused_by_name` (GLSL, SPIR-V, MSL: the refusal names the Tile space and is not the checker's; WGSL and HLSL refuse an F32 kernel before the tile space, so they pin only that a tile kernel is never lowered) and a coverage unit test for the row and the patent note. Mutants, each caught by name: the walker's Tile arm disabled, the SPIR-V checker's Tile arm disabled, the patent note dropped from the coverage text.

**The OOB check against KISS 0.24.4 (6.11-0004, -0005, -0015, -0016, 6.13-0009: reads decline, writes skip).** Those clauses bind `gather`, `scatter`, `scatter_add` and `index_select`. Nothing in this repository lowers those ops: `gather|scatter|oob_policy|index_select|OobPolicy` has 0 hits in the sources and tests outside the generated `vulkan_gen` and `vulkane/src/raw` (the same grep finds `MaskedLoad`, which is present, so the query works). What this repository does lower is the kernel layer's `MaskedLoad`/`MaskedStore`, whose masked-off lane does not touch memory (a write that is skipped, a read that takes the else value), which agrees with "writes skip" for a mask but is not the index policy of those clauses. A kernel `Load`/`Store` with an out-of-range index is the IR's contract and the device's robust-buffer-access behaviour, unchanged here. **No code change follows; the first op-level lowering of `gather` or `scatter` has to implement them.**

## [0.31.7] — Six decline pins revived from the old patent-audit branch, and a test of the byte-addressed `Place::Raw`/`AllocRaw` refusal (ROADMAP 9h, row 8y)

**Declines that no test noticed removing are pinned (the decline-test audit of 2026-11-25, ported onto current main).** The audit found six refusals whose removal every suite passed: the walker's integer unary-op decline and the cast guard (GLSL `kernel_lowering.rs`), a checker refusal of a parameter-dependent grid (it asserts the refusal is the CHECKER's: the walker's own is masked by unpopped's seam rule, an equivalent mutant), a fence at cluster scope in the GLSL dialect, a fence at cluster scope, an unspellable execution id and a clamping integer cast in the SPIR-V emitter, and an atomic at cluster scope in SPIR-V (`vulkane-spirv-emit/src/kernel_tests.rs`), plus a WGSL and an HLSL row. Ported by cherry-pick; the SPIR-V cast pin was rewritten for main (the old refusal text is gone and an identity U32 to U32 cast is spelled as the operand, so the clamp is only reached across types: an inner wrapping U32 to I32 cast and an outer clamping one), the old `reverse` call took the new `SharedLen`. **The audit's `patent_decline.rs` rewrites are NOT revived:** they are superseded by the `unpopped::fixtures` tests (#176, #180), which give checker-valid kernels and assert emitter-owned refusals.

**Re-mutated on main, each caught by name:** the walker's integer unary decline removed (GLSL pin); the GLSL dialect spelling a cluster fence as `memoryBarrier`; the SPIR-V emitter approximating a cluster fence, a cluster atomic and an unsupported execution id by a device scope or a subgroup builtin; the SPIR-V clamping-cast guard removed. **New test (a gap the row-8y work found):** `Place::Raw` and `Place::AllocRaw` (unpopped 0.57) have no refusal test on main; `vulkane-test-support::stmt_kernel::raw_place_kernels` (recovered from the deleted `take-unpopped-0.57.0` branch) and `a_byte_addressed_place_is_refused_by_name_not_lowered_to_a_wrong_address` in GLSL, WGSL, SPIR-V and MSL `stmt_coverage.rs` assert a checker-valid kernel is refused by name by the emitter, not the checker; with `place_ok` made to accept every place the walker hits its own `unreachable!` and the test fails. Tests and test-support only: nothing in the published crates changes.

## [0.31.6] — Pin the wide-kernel declines the changelog promises, and name the real reason for a 64-bit specialization-constant extent (review of #170)

**Tests pin three GLSL declines and one WGSL decline that the 0.25.6 entry promised and no test exercised (review of #170, Low).** A 64-bit GLSL kernel holding an integer `Div`, an atomic, or a specialization-constant extent is declined by name (`vulkane-glsl-emit/tests/kernel_wide.rs`; the shared wide kernel there is truncated before its `Div` columns), and an `I64` kernel is declined by the WGSL dialect for having no 64-bit integer spelling (`vulkane-wgsl-emit/tests/kernel_lowering.rs`), so a later rung cannot start lowering one silently. **The decline for a specialization-constant extent in a 64-bit kernel now says why** (nit): it read "parameter `n` is not a flat buffer of the compute dtype", which was wrong for a scalar parameter of the compute dtype; it now reads "a specialization-constant extent (parameter `n`) is not lowered for a 64-bit kernel". Control: with the old message the new test fails on exactly that text. No lowering changes.

## [0.31.5] — The architecture declaration names components, layers, volatile namespaces and budgets: seven rules that never ran now run (ROADMAP 9c)

**`.coderipper.toml` declares the workspace's architecture at crate granularity, so ARC-003, ARC-004, ARC-006, ARC-007, ARC-012, EFX-001 and ERR-006 run instead of reporting "not run" (the portfolio sweep's "a rule did not run" gate).** Every declaration is a fact read from `cargo metadata` or from the sweep's own counts: 17 components with a role (`policy` for the target-neutral walker `vulkane-kernel-text`; `adapter` for the per-target emitters and the parsers; `infrastructure` for `vulkane`, `vulkan_gen`, `vulkane_derive`, `vulkane-kernels-source` and `kiss-vulkan-vocab`; `test-support`) and a volatility; three layers that match the real dependency graph (the old declaration put `vulkane-kernel-text` in one "leaf" layer with the five emitters that depend on it); the volatile namespaces `vulkane::`, `vulkan_gen::` and `ash::` that the stable policy crate must not name; ARC-006 budgets at today's measured public size (items plus members, a ratchet and not a design claim); `vulkane-kernel-text` tagged `pure = true` (EFX-001); and `none = ["owns_data", "composition"]`, two stated facts (no database, shared table or topic; no executable that wires these crates together).

**Checked, not assumed.** The gate's own "not run" grep finds none (seven before). Each rule that has something to judge was shown to catch a deliberate violation in a scratch commit, reset afterwards: an upward edge (ARC-001), a `vulkane::raw` name in the policy crate (ARC-003), a `vulkane::raw` type in its public signature (ARC-004), `std::fs` in the declared-pure crate (EFX-001), an error type naming `vulkane::raw` (ERR-006), and a wrong budget (ARC-006). **No control possible:** ARC-007 and ARC-012 report "0 sites by declaration" by design, and a dependency cycle makes `cargo metadata` fail before ARC-002 reads it. **Dispositions, not opt-outs:** ARC-006 reports two Info findings: `vulkane-slang-emit` and `vulkane-slang-parse` re-export a library they do not own, which is true and by design (they are deprecated shims of the fishstringer crates); they stay reported. ARC-001 does not judge the 15 dev-dependency edges. Nothing in the build, the tests or the published crates changes.

## [0.31.4] — Docs: the board 506 A subgroup-collectives plan and verified function facts (vote and ballot rows withheld)

**`docs/design/506-subgroup-collectives.md` lands on main as facts and a plan (no code):** the board 506 A ruling as relayed (lifted: subgroup-scope reductions, shuffles and broadcasts; still declined by name with the 434 C marker: scans, block scope, GLSL and SPIR-V), the ruled set by IR kind, the per-dialect functions with verbatim specification quotes (MSL 4.1 section 6.10.2, WGSL, HLSL), the edge semantics where the IR (the oracle) differs from the target, the HLSL-to-SPIR-V guard design measured with dxc 1.9.0.5347, and the uniformity guard. It is the reference for the HLSL lowering, which waits on a published plain-HLSL entry in `fishstringer-slang-emit`. **The vote and ballot rows are withheld** (pm-kernels, 2026-11-27, under the PM's interim vote hold: the shipped Vote statement is recorded NEEDS MORE against a different patent; nothing new for votes): the document carries a pointer to that effect, and the full original is kept on the branch `day-2026-11-26-506-prep` and in a dated scratch copy; the rows return in a docs PR when the hold lifts. Any patent named is cited as findings, not advice; "outside as read" is a reading of public documents, not a clearance; no attorney. Nothing in the build, the tests or the published crates changes.

## [0.31.3] — Docs: the MSL simdgroup-matrix facts document, and ROADMAP rows 8w, 8x, 8y

**One design document lands on main, as facts (no code):** `docs/design/msl-simdgroup-matrix.md` (what the Metal Shading Language specification says about the 8x8 SIMD-group matrix and the Metal 4 tensor path, with the board 519 B condition at its top: lifted by CireSnave 2026-11-26 as his risk call, not a clearance, no attorney; its status line records that unpopped 0.61.0 still has no Tile statements). The board 506 A plan document stays on its branch for now: it carries vote and ballot facts and the PM's interim vote hold (2026-11-27) says nothing new for votes. **ROADMAP rows opened:** 8w (the shared `coverage()` table calls every `Stmt::Collective` patent-declined although WGSL and MSL lower the ruled kinds), 8x (the stamp tool leaves workspace path-dependency requirements at the old minor) and 8y (36 old remote branches with no pull request, to be verified one by one before deletion). Nothing in the build, the tests or the published crates changes.

## [0.31.2] — MSL lowers F32 subgroup Sum, Max and Min (Max and Min with an explicit NaN select); board 506 A float ruling (ROADMAP 8s, 8t)

**`vulkane-msl-emit` lowers `Reduce(Sum)`, `Reduce(Max)` and `Reduce(Min)` over an `F32` kernel's subgroup (board 506 A float ruling, pm-kernels 2026-11-27).** Sum is `simd_sum`; Max and Min are `simd_max` / `simd_min` wrapped in `select(.., NAN, simd_sum(isnan(v) ? 1.0f : 0.0f) > 0.0f)`, a NaN indicator via a Sum reduction because the bare `simd_max`/`simd_min` IGNORE a NaN operand (measured on the hosted macOS paravirtual device, vulkane#189 run 45972534688) and KISS-OPS-6.11-0002 requires a float reduce(max|min) to propagate it (the indicator count is exact while the lane count is below 2^24; the hosted device reports a SIMD-group width of 32, and the MSL text read states no maximum). The `Dialect` hook is the existing `subgroup_collective`; the float dialect says `subgroup_collectives()` for `F32` only, so half and bfloat kernels stay declined by name. `Prod`, `Mean`, shuffles, votes, broadcasts and scans stay declined (unmeasured or outside the ruling); the refusal text names the three that are lowered.

**Tests and measurements.** Text pins for the three spellings and the declines; seven vacuity mutants (NaN select removed from Max, from Min, the indicator sum removed, the `> 0.0f` miscounted as `> 1.0f`, Max spelled as Min, half collectives enabled, Prod accepted as a sum), each caught by name. `vulkane-test-support::float_collective_kernel` holds the kernel, an eight-subgroup input (finite; NaN in the first, last and a middle lane; all NaN; NaN with +inf; NaN with both infinities; signed zeros), the oracle (Unpopped's CPU oracle, which propagates NaN) and the comparator: Max and Min exact-byte with a NaN lane required to give NaN, the +-0 cells excluded (KISS does not pin them), Sum under a tolerance that is OURS, NOT KISS's (the clauses KISS-OPS-6.0-0004, 6.17-0005 and 6.17-0008 state no bound): `(n-1) * 2^-24 * sum|x_i|`. The hosted-device cases include a vacuity mutant, the lowered Max kernel with the select removed, which the comparator must reject. **Limits:** measured on the hosted paravirtual device, not Apple hardware; safe math only, fast-math not measured (a kernel compiled with fast-math may differ). Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call; not a clearance; no attorney; findings, not advice.

**Measured (hosted paravirtual device, run 45985572933, the Sum-indicator wrapper):** the three device cases pass and the three mutants are each rejected by value on the device; fast-math not measured.

## [0.31.1] — Hosted-device MEASURE-ONLY probe of float simd_sum, simd_max and simd_min: NaN, infinity, signed zero and addition order (ROADMAP 8t, board 506 A)

**A hand-written float subgroup-collective case joins the hosted macOS device probe, measure only (pm-kernels 2026-11-27).** `raw_float_collective_measure` (in `vulkane-msl-emit/tests/metal_device.rs`, not emitted; `math_mode(safe)` as the float emitter writes) runs `simd_sum`, `simd_max` and `simd_min` over 12 input patterns of one 32-lane subgroup (an all-ones control; a NaN in the first, a middle and the last lane; both infinities and one; signed zeros in several positions; and a sum whose result depends on the order of addition) and prints every result as `MEASURED float_collective ...`, which the job summary carries. It never fails on a value; it fails only if the all-ones control is wrong. **No float collective is lowered**: they stay declined by name until the measured NaN, infinity and signed-zero behaviour is in hand and KISS states a rule for a float reduction (pm-kernels). Measured (run 45972534688): `simd_sum` propagates NaN; `simd_max` and `simd_min` IGNORE a NaN operand (not NaN-propagating); signed-zero ties went to the lowest lane's zero; the addition order was not determined; all on the paravirtual hosted device in safe math (full values in ROADMAP 8t). Lifted by CireSnave 2026-11-26 (board 506 A) for integer kernels as his risk call; not a clearance; no attorney; findings, not advice.

## [0.31.0] — **BREAKING**: take unpopped ~0.61 and fishstringer ~0.11 (dependency floor); `UnaryOp::{Popc, Clz, Brev}` declined by name in every emitter

**BREAKING (a dependency floor): every crate that names `unpopped`, `unpopped-vocab`, `unpopped-convert` or `unpopped-cpu-c` now requires `~0.61.0`, and the Slang crates re-export `fishstringer-slang-emit`, `fishstringer-slang-parse` and `fishstringer-kernels-source` `~0.11.0`.** unpopped 0.61.0 is BREAKING upstream (`UnaryOp` gains `Popc`, `Clz` and `Brev`, so every exhaustive `UnaryOp` match breaks); this workspace had none (the trial compile at 0.61 is clean, the new `bitop_kernel` tests are the control: they use variants that exist only in 0.61), so the change is in the manifests and the lock. A consumer that names unpopped types through this workspace's backends (`impl Backend for Glsl` and the others) must be on unpopped 0.61 too. `cargo tree -d` shows one copy of `unpopped-vocab` (0.61.0); the published fishstringer 0.11.0 requires `^0.61.0`.

**The three new integer bit operations are declined by name, not lowered.** `Popc`, `Clz` and `Brev` pass `check_kernel` at an `I32` or `U32` kernel, so the emitter's own refusal is the only guard: GLSL, WGSL, MSL and the HLSL kernel dialect refuse through the shared walker (`unary op Popc is not lowered (integer kernels)`), SPIR-V through its unary gate (`Unary(Popc): v1 unary is float/double; integer dtypes have no unary math`), and HLSL through fishstringer-slang-emit 0.11's own refusal. `vulkane-test-support::bitop_kernel` builds the kernel; each of the five emitters has `tests/bitops_decline.rs` asserting a typed refusal that names the op and is not `check_kernel`'s; vacuity mutants (the walker's decline removed for GLSL, WGSL and MSL; the whole SPIR-V unary function bypassed) each fail by name, and the SPIR-V case caught a silently emitted module. **Limit:** the HLSL refusal is fishstringer's, so no mutant of it is possible here. The facts for the later lowering (the IR rule and each target's text, quoted with sources and checksums) are `docs/design/unaryop-bitops-0.61.md`; the patent wording in it is RELAYED from unpopped-patents-2 (not on main): not a clearance; no attorney; findings, not advice. unpopped's `cfamily::unary_f32`/`unary_f64` panic on these ops, and no emitter here calls them (only `dtype_tag`, `assert_no_int_div_or_const` and `params_used` are used).

## [0.30.9] — Advisory CI job validates the emitted WGSL subgroup collectives with Dawn's tint (ROADMAP 8t)

**A new advisory job, `Tint validate` (`.github/workflows/tint-validate.yml`), feeds the WGSL that `vulkane-wgsl-emit` writes for the 14 ruled board 506 A subgroup collectives to Dawn's `tint`.** naga 30.0.2, the latest published, rejects `enable subgroups;` (a test pins that), so until now the WGSL subgroup lowering was checked against the WebGPU specification by text tests only. New test `vulkane-wgsl-emit/tests/tint_dump.rs` writes the 14 collective kernels plus two controls (a plain kernel Tint must accept, and a collective kernel with one function name damaged that Tint must reject) and a manifest of the expected answer for each; the job checks every answer against it.

**Provenance, stated in the workflow:** the `tint` binary is the `bin/tint` of the official google/dawn GitHub release `v20261126.172545` (asset `Dawn-421e5b0ef6ec76ee8f49daa97e760ba8b42852fc-ubuntu-latest-Release.tar.gz`, built from google/dawn commit 421e5b0e), pinned by tag and by the tarball's sha256, which is checked before extraction; it is a nightly build of a Google project, extracted into a fresh directory and run only on the hosted runner. The first run (45971814219) recorded: Tint accepted all 14 collective kernels and the plain control and rejected the damaged control (`unresolved call target`), 16 of 16 as expected; that is Tint's front end only, no device ran the WGSL (ROADMAP 8t). Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call; not a clearance; no attorney; findings, not advice.

## [0.30.8] — Hosted macOS CI job compiles the emitted MSL with xcrun metal and probes for a Metal device (advisory; ROADMAP 8o)

**An advisory hosted-macOS job, `metal` (its own `.github/workflows/metal-probe.yml`), compiles the MSL `vulkane-msl-emit` emits with Apple's compiler and probes for a Metal device (ROADMAP 8o).** New test `vulkane-msl-emit/tests/metal_dump.rs` lowers the 27 shared kernels the MSL tests use (integer, flow, atomic, private, subgroup, masked, f32/f16/bf16, vector, narrow) and, when `VULKANE_MSL_DUMP_DIR` is set, writes each to a `.metal` file plus a `MANIFEST.txt`. The `metal` job on `macos-latest` runs it, asserts the manifest's count equals the files, compiles each with `xcrun -sdk macosx metal -std=metal3.2 -c` (a failure is a red job), links each with `metallib`, then records, without failing, whether Metal accepts a threadgroup array whose extent is a function constant (`.github/metal/function_constant_extent.metal`, the MSL 5.8 claim behind the scalar-parameter blocker) and what `MTLCreateSystemDefaultDevice` returns (`.github/metal/probe.swift`). Not a required check; no new library dependency; actions pinned by full SHA as in `ci.yml`. The README's "not validated by a Metal compiler" line now says what the job does and does not establish. The job also DISPATCHES 32 cases on the runner's Metal device (`.github/metal/run.swift`, `vulkane-msl-emit/tests/metal_device.rs`) and checks each output against the shared oracles used by the Vulkan device tests, with mutant and control cases that must differ. Bf16 gets an oracle (`vulkane-test-support::bf16_kernel`) and its own device cases (ROADMAP 8p); the hosted run (42 of 42) also measured that the device flushes F32 and Bf16 subnormals to zero (MSL 8.1 allows it) and preserves F16 ones. The MSL arm also lowers `I64`/`U64` kernels (`long`/`ulong`, total integer `Div` for every width) and runs the shared 64-bit kernel on the device exactly (ROADMAP 8q); 64-bit atomics are declined by name because Apple's compiler rejects every `atomic_ulong` operation under all five probed flag combinations. The MSL decline tests are audited (checker-valid probe kernels, the refusal asserted to be the emitter's own, decline-removed mutants): the F64 and `Sqrt` declines in `what_is_not_spelled_is_declined_by_name` were passing for the wrong reason (the shared kernel is named `float`, an MSL type, so the reserved-name decline answered); they now assert their own text, a cluster-scope fence decline has its first test, and the `stmt_coverage` suites of every emitter require a checker-valid kernel.

**Board 506 A device cases (ROADMAP 8t).** The hosted-device case table gains one case per ruled subgroup collective (14: seven reductions, All and Any votes, a constant broadcast, and the Idx, Up, Down and Xor shuffles) over a 128-thread block with a stated subgroup size of 32, checked word for word against Unpopped's own CPU oracle (`run_launch_with_subgroup`), not a second implementation; `vulkane-test-support::collective_kernel` holds the kernels, the input (four subgroups: all non-zero, one zero, all zero, one non-zero) and a test that no two kinds agree on it. A device whose subgroup width is not 32 reports the case as not applicable. Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call; not a clearance; no attorney; findings, not advice. Measured on the hosted paravirtual device (run 45970669904): 14 of 14 equal to the oracle; not Apple hardware, and no device-side mutant for these cases (ROADMAP 8t). The workflow's push trigger is dropped (pull_request and workflow_dispatch remain). `Cargo.lock` takes unpopped 0.60.32, whose oracle now traps a racy named-barrier kernel as `DataRace`.

## [0.30.7] — CI: the local-gates job installs the pinned toolchain before the harness probes it by name

**The `Local gate harness self-test` job installs the toolchain pinned in `rust-toolchain.toml` explicitly.** The job went red on main from #181's merge on: the ubuntu image of 2026-11-26 (20261116.202.1, Rust 1.100.0 preinstalled) does not carry 1.98.0 (the CodeRipper job of the same run logs that rustup auto-installed the missing 1.98.0), and the harness probes the pinned channel by name (`cargo +1.98.0 --version`) before any plain `cargo` could trigger rustup's install from the file; the `dtolnay/rust-toolchain` step there installs `stable`, not the pin. Reproduced locally by pinning a channel that is not installed: exactly the one control `the toolchain probe answers for an installed toolchain` fails and the rest pass; with the pin installed the self-test passes. The new step reads the channel from the file, runs `rustup toolchain install <channel> --profile minimal` and probes it; `tools/local_gates.py` already classifies `rustup toolchain install` as skipped locally. No Rust code changes.

## [0.30.6] — Subgroup collectives lower on WGSL and MSL (integer kernels): board 506 A, narrowed (ROADMAP 8c, 8s, 8t, 8u)

**Subgroup-scope `Reduce` (Sum, Prod, Max, Min, And, Or, Xor), `Shuffle` (Idx, Up, Down, Xor), `Vote` (All, Any) and `Broadcast` now lower on `vulkane-wgsl-emit` and, for integer kernels, `vulkane-msl-emit`.** Board 506 A, narrowed (lifted by CireSnave 2026-11-26 as his risk call; "outside as read" is a reading of public documents, not a clearance; no attorney; findings, not advice). `vulkane-kernel-text` gains `declined_by_name_with(lift_subgroup, ..)`, `collective_is_ruled`, `SubgroupCollective` and the `Dialect` hooks `subgroup_collectives`, `subgroup_collective` and `prologue_for` (all with defaults, so every other dialect is unchanged); the by-name decline still refuses `Scan`, `ReduceWith`, `ScanWith`, `CollectiveTuple`, every block-scope collective, and every collective on GLSL and SPIR-V, the ruled kinds included.

WGSL spells `subgroupAdd` and the rest under `enable subgroups;` with `subgroup_invocation_id` and `subgroup_size` builtins declared only when a collective is present; MSL spells `simd_sum`, `simd_shuffle` and the others (MSL 4.1 6.10.2). Where the IR defines an edge the target leaves undefined, the text is explicit: `Idx` is `operand mod n` (Euclidean for signed), `Up`, `Down` and `Xor` keep a lane's own value out of range, `Broadcast` and the shuffle delta or mask must be constants below the launch's stated subgroup size, votes return 0 or 1, and a WGSL `Ballot` is limited to 32 lanes. MSL `Ballot` and every MSL float collective stay declined. The collective acts on active invocations; Unpopped's `check_kernel` (taken at 0.60.26 in the lock) refuses a collective under divergent control or after a divergent return, and both emitters pin those refusals.

**Limits, stated.** The WGSL and MSL text is pinned by tests; no WGSL compiler accepts `enable subgroups;` here (naga 30.0.2 rejects it, pinned by a test) and no Metal device has run the MSL, so the lowering is UNVALIDATED (ROADMAP 8t). HLSL waits on a published plain-HLSL entry in `fishstringer-slang-emit`. ROADMAP 8u records the held US9324175B2 working hold.

## [0.30.5] — `vulkane-kernel-text` classifies every `Stmt` variant; a statement with no lowering says why (ROADMAP 9f)

**`vulkane_kernel_text::coverage` names where every `unpopped::kernel::Stmt` variant stands, and the shared walker's and the SPIR-V emitter's "not lowered" refusals now carry the reason (ROADMAP 9f).** `coverage(&Stmt)` is one match with no wildcard arm over every variant, so a statement added to `unpopped` fails to compile in this crate until someone says which of four it is: `Lowered` (16: `Let`, `Load`, `Assign`, `Store`, `MaskedLoad`, `MaskedStore`, `If`, `For`, `While`, `Break`, `Continue`, `Return`, `Barrier`, `Fence`, `AtomicRmw`, `AtomicCas`), `Row` (6, with the ROADMAP row they wait for: `Call`, `CallTuple`, `ExternCall` under 8g; `Mma`, `MatrixLoad`, `MatrixStore` under 8j), `Patent` (the twelve of board 434 C, still refused by `declined_by_name` first) or `Nvidia` (7, each with its reason: `Wgmma` and `WgmmaCommit`; `Tcgen05Mma` and `Tcgen05Ld`; `GridDepWait` and `GridDepLaunchDependents`, CUDA programmatic dependent launch; `MulticastLoadReduce`). `not_lowered_note` is the sentence the GLSL, WGSL, MSL (through the walker) and SPIR-V emitters append, so `Stmt::GridDepWait` now reads "is not lowered (...): NVIDIA-specific: CUDA programmatic dependent launch ..." instead of the generic text. `vulkane-test-support::stmt_kernel` builds a one-statement kernel; `tests/stmt_coverage.rs` in each of the four emitters asserts the named refusal for the operand-free NVIDIA statements, and a mutant that rewrites a reason fails the unit test and all four. The reasons are this repository's reading (no target language here has a construct with those semantics), not a ruling; HLSL through fishstringer's Slang path is not this repository's lowering. No lowering changes.

## [0.30.4] — New crate vulkane-msl-emit (Metal Shading Language kernels: integer, float, half, Bf16), masked loads and stores, and DeviceInner as a leaf module (ROADMAP 8c, 8e, 8o, 9, 9e)

**`vulkane-msl-emit` (new) lowers a 32-bit integer kernel to MSL (ROADMAP 8o, first slice).** `Msl::lower_kernel` walks the kernel through the shared `vulkane-kernel-text` walker and writes `#include <metal_stdlib>`, `using namespace metal;` and one `kernel void <name>(...)`: a `device const T* b<N> [[buffer(N)]]` per `Read` buffer (`device T*` otherwise), `uint3 tid [[thread_position_in_threadgroup]]` and `uint3 gid [[threadgroup_position_in_grid]]`, then the `threadgroup T a<K>[N];` arrays at kernel scope (Metal requires it), per-thread arrays and locals. The workgroup size is a dispatch fact in Metal, so the text carries it as a `// threadsPerThreadgroup:` comment. `Barrier { Block }` is `threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)`; a block or device `Fence` is `atomic_thread_fence(..., memory_order_seq_cst, ...)` (every `MemoryOrder` collapses to the strongest: stronger than asked, never weaker). Declined by name (typed `UnsupportedKernel`/`UnsupportedOp` from the walker): float, 64-bit, 8/16-bit integer and vector kernels, atomics, scalar parameters, subgroup and `BlockDim`/`GridDim` ids, a device barrier; the twelve NVIDIA-only statements of board 434 C are refused by name first (`tests/patent_decline.rs`, six nesting shapes). Tests pin the text of the shared reverse, flow, grid and `Private`-array kernels; a mutant of the barrier spelling fails the barrier test. **Not compiled by a Metal compiler:** that needs macOS (`xcrun metal`), which no runner here has; the README says so and the oracle is the next slice. No new `#[allow]` and no `#[path]` include (the kernels come from `vulkane-test-support`).

**Second slice: `F32` kernels (ROADMAP 8o).** The source opens with `#pragma METAL fp math_mode(safe)` and `#pragma METAL fp contract(off)` (Metal's default fast math assumes no NaN, INF or signed zeros and allows reassociation and fusion; `safe` alone leaves contraction on), every float operation is pinned in a `const` temporary, unary ops are the exact ones only (`fabs`, `floor`, `ceil`, `trunc`, `rint` for half-to-even), max/min propagate NaN, and float-to-integer conversions clamp and map NaN to 0. `F64`, `Div`, `Sqrt` and the transcendentals stay declined by name. Text pinned, not Metal-compiled. Unpopped named one further Metal construct for rows 71/72: the Apple 8x8 simdgroup MMA atom (theirs to type; this crate consumes it).

**Third slice: atomics (ROADMAP 8o).** `AtomicRmw` (Add, Min, Max, And, Or, Xor, Exchange) and `AtomicCas` on `I32`/`U32` buffers and `threadgroup` arrays lower to `atomic_int`/`atomic_uint` targets and `atomic_*_explicit(..., memory_order_relaxed)` (spec 6.16.4, Metal 4.1). Metal has only `atomic_compare_exchange_weak_explicit`, which can fail spuriously and writes the value it saw into `expected`; the IR's compare-exchange cannot express that, so it is retried while the value seen still equals `expected`. Block and Device scope only; a non-`Relaxed` order and float atomics are declined by name. A mutant removing the retry condition fails the atomics test. Text pinned, not Metal-compiled. The Apple simdgroup MMA waits on Unpopped's tile statements (Q1): MSL spec 2.4 leaves the element-to-thread mapping unspecified, so no per-lane `MmaAtom` exists.

**WGSL plan-path header note (ROADMAP 8m).** `DENORM_NOTE` said only that subnormals are the device's; WGSL 15.7.5 also says "An implementation may reassociate operations" and "An implementation may fuse operations if the transformed expression is at least as accurate as the original formulation", so a multiply feeding an add may round once. The header of every `f32` WGSL kernel, the crate doc and the README now say so. No behaviour change; no golden carries the line (checked: the only other hit is the `validity.rs` assertion, extended red-first).

**Correction to the F32 paragraph above:** `math_mode(safe)` does not preserve subnormals or fix the rounding mode; it only stops compiler transformations (and leaves contraction on, hence `contract(off)`). **Hardware caveat, stated not hidden:** MSL 8.1 says "Denormalized single-precision, half-precision, or brain floating-point numbers passed as input to or produced as the output of ... arithmetic operations may be flushed to zero" and 8.2 says "Either round ties to even or round toward zero rounding mode may be supported", so no source pragma makes a Metal float kernel bit-exact IEEE; a device oracle must treat subnormals and the last bit as the device's, as the WGSL arm does.

**Fourth slice: `F16` kernels (ROADMAP 8e, 8o).** `half` buffers, locals and temporaries, literals `1.5h` (the walker declines a constant not exact at half), the same `safe`/`contract(off)` pragmas and exact unary ops; half to or from `F32` is `static_cast` (round to nearest even, MSL 2.23); half to or from an integer and `Bf16` are declined by name. **Defect found and closed:** a kernel named `float` (the shared test kernel's name) emitted `kernel void float(`, which cannot compile; names that are MSL keywords or types (`float`, `kernel`, `tid`, ...) are now a typed decline telling the author to rename. A mutant forcing the `f` literal suffix on half fails its test (it survived the first set of tests, which pinned no literal, so one was added). Text pinned, not Metal-compiled.

**Fifth slice: subgroup ids and the subgroup barrier (ROADMAP 8c, 8o).** `LaneId`, `SubgroupId` and `SubgroupSize` lower to `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup` and `threads_per_simdgroup` (MSL 4.1 section 5.2.3.6, kernel function input attributes), declared in the entry signature only when the body reads them (the walker lowers the body before the prologue, so the dialect records what it spelled); a subgroup `Barrier` is `simdgroup_barrier(...)` (MSL 4.1 section 6.10.1). Metal cannot request a subgroup size, so a launch's required range is written into the source as `// requires a subgroup size in a..=b` instead of being dropped. `BlockDim`/`GridDim` need nothing: the shared walker already folds them to launch constants (the earlier claim that they were declined was stale; a test pins the folding). Public type change: `Msl` is now only the `Backend`; the dialect behind it is crate-private (it records state while lowering) and no longer implements `Dialect`. Mutants of the subgroup-id mapping and of the barrier spelling each fail a test. Text pinned, not Metal-compiled.

**Sixth slice: 8- and 16-bit integer storage (ROADMAP 8e, 8o).** `I8`/`U8`/`I16`/`U16` buffers and locals in an `I32`/`U32` kernel lower to Metal's native `char`, `uchar`, `short` and `ushort`; the kernel computes at 32 bits and a C-style cast widens (sign- or zero-extending) and narrows (modular), as in the GLSL and HLSL arms. An atomic beside a narrow buffer stays a decline. Tests pin all four types' declarations and both cast directions; mutants of the `char` spelling and of the cast target each fail. Narrowing a signed value is implementation-defined before C++20 and two's complement in practice: stated, not compiled.

**The shared walker gets one default-false hook, `Dialect::bfloat()`, and the MSL arm uses it to lower `Bf16` kernels (ROADMAP 8e, 8o).** `vulkane-kernel-text`: `Dialect::bfloat` (default `false`) selects a `Bf16` compute dtype the way `half` selects `F16` (`scope::dtype`), and a float constant must then be exactly a `bfloat16` (an `f32` whose low 16 bits are zero). No existing dialect sets it, so the SPIR-V, GLSL, HLSL and WGSL emitters are unchanged: the six emitter crates' tests give the same 54 results and 234 passes before and after this commit (see the PR body for the per-crate list), and no golden file changed. `vulkane-msl-emit`: `Bf16` kernels are `bfloat` (Metal 3.1 and later; MSL 4.1 2.1) with explicit `static_cast<bfloat>(...)` literals, because MSL 2.24 allows no implicit float-to-bfloat conversion; the unary ops (`abs`, `floor`, `ceil`, `trunc`, `round`) go through `float` because the math functions of MSL 6.6 take `half` and `float` only, which is exact (widening is exact and the result needs no more bits than the input). `isnan` and the comparisons are defined on `bfloat` (6.5). A kernel with packed (vector) locals already lowered, scalarized lane by lane by the walker with the same bits; a test now pins that and that no `float2`/`float4` is spelled. FP8 stays declined by name: MSL 4.1 2.21 has FP8 only as `packed_numeric_type` and tensor format tags. Mutants of the bfloat exactness test and of the through-float spelling each fail a test. Text pinned, not Metal-compiled; whether a device rounds one bfloat add or multiply once is the device's.

**`vulkane-msl-emit` does not lower any `Stmt::Collective`, and a test now pins every kind (ROADMAP 8c, 8o).** The task was to map the SIMD-group functions of MSL 4.1 6.10.2 (`simd_shuffle*`, `simd_all`/`any`/`ballot`, `simd_sum`/`product`/`max`/`min`/`and`/`or`/`xor`, `simd_prefix_*`, `simd_broadcast`) to the IR's subgroup statements. Those statements are `Collective` and `CollectiveTuple`, two of the twelve of board 434 C, which CireSnave ruled on 2026-11-13 (verbatim, *"emitters may lower AsyncWait/CopyAsync/AsyncCommit/Collective for NVIDIA devices ... and decline by name for non-NVIDIA devices"*); Metal is not an NVIDIA device, so the shared walker already refuses them first. `tests/patent_decline.rs` now lists every `CollectiveKind` (four shuffle modes, three votes, seven reductions, inclusive and exclusive scans, broadcast, both user-combine forms) beside the MSL function that would spell it, and asserts each is refused with `Stmt::Collective` and the 434 C marker at all six nesting shapes; a mutant that stops declining `Collective` fails it. The crate doc records the mapping as what is NOT lowered, and why a lifted ruling would still not be one to one (the MSL functions act on active threads, `simd_ballot` zeroes inactive bits, `T` excludes `bool`, `bfloat`, `long`, `ulong`). Lowered and unaffected: `LaneId`, `SubgroupId`, `SubgroupSize` and `simdgroup_barrier`. No behaviour change.

**`Stmt::MaskedLoad` and `Stmt::MaskedStore` lower on the SPIR-V, GLSL, WGSL and MSL emitters (and the HLSL kernel dialect) (ROADMAP 9e).** No emitter lowered them and no ROADMAP row named them. `MaskedLoad` is `if (mask) { l = element; } else { l = fill; }` and `MaskedStore` is `if (mask) { element = value; }`, each through the plain `Load` / `Store` path (so an atomically typed buffer and a pinned float value behave as there): the guard is exactly the IR's meaning, the memory is not touched where the mask is false (no bounds check, no race record). A `select` would still execute the load, which is why it is not used. Tests: `vulkane-test-support::masked_kernel` (256 threads, a load whose index is in range only where the mask holds, a plain store then a masked store over it); a text test per walker emitter (GLSL, WGSL, MSL) pins the guard structure; device tests on GLSL (glslang) and WGSL (naga) run it and check every cell against the host reference. Mutants of the load guard and of the store guard each fail both device tests (the text test alone passes them, so the device tests are what pin the meaning). SPIR-V, which has its own builder, lowers them as the structured `If` it already has (`OpSelectionMerge` and a conditional branch, the access in the taken arm; the load's fill is the else arm), with a live device test through `spirv-val` and a mutant that runs the access unconditionally failing it. HLSL through fishstringer's Slang path still declines the statements (not this repository's lowering). Other emitters' output for kernels without these statements is unchanged: the WGSL and SPIR-V goldens are byte-identical and the per-result test counts of the other crates match the parent's gate.

**`crate::safe::device_inner` (new, crate-private) holds `DeviceInner`, dissolving the `crate::safe` module cycle (ROADMAP row 9).** Every resource module held `Arc<DeviceInner>` through `use super::device::DeviceInner` while `device.rs` names their types, which CodeRipper's MOD-002 reads as one 14-module, 38-edge cycle. `DeviceInner` and its `Send`/`Sync`/`Drop` impls move unchanged to `safe/device_inner.rs`, a leaf (it reads only `crate::raw` and `InstanceInner`); the 15 `use` lines, the one path string in the build-time handle generator (`vulkan_gen/.../safe_handles_gen.rs`, which writes `auto_handles_generated.rs` into `OUT_DIR` on every build, so nothing generated is checked in and a rerun reproduces it) and the comment in `safe/auto.rs` now name the new path. No public path, type or behaviour changes (`DeviceInner` is `pub(crate)`). Measured with `coderipper check MOD-002 --profile extended` on `a224561`: 41 modules, 107 edges, 1 cycle; after: 42 modules, 109 edges, 0 cycles (the committed tree of this branch is re-measured in the PR body).

## [0.30.3] — Test support is one crate, not 142 `#[path]` includes (ROADMAP 8n)

**`vulkane-test-support` (new, `publish = false`, test-only) holds the GPU lock witness and the shared kernels and oracles (ROADMAP 8n).** The helpers that test files used to `#[path]`-include (`vulkane/tests/common/mod.rs`, `vulkane-spirv-emit/tests/support/*.rs`, `vulkane-glsl-emit/tests/support/glslang.rs`) are now modules of one crate that the five test-bearing crates take as a dev-dependency and use by `use vulkane_test_support::{common, gpu, ...}`. 142 `#[path]` includes (CodeRipper `SUP-004`) and the per-file `#[allow(dead_code, unreachable_pub, missing_docs)]` that went with them (`GOV-010`) are gone; the crate carries no `#[allow]` and every public item is documented. `reverse_kernel::reverse` takes a `SharedLen` (`Static` or `Spec`) in place of its second `bool`. No test body changed. The crate and `vulkane`/`vulkane-spirv-emit` form a dev-dependency cycle on purpose (the harness needs the device wrapper and the emitter's workgroup size; Cargo allows it, and nothing is published). Red first: `tests/no_path_includes.rs` scans every workspace crate for a `#[path]` include of `common/mod.rs` or a `support/` file and failed on all 142 before the move; it carries a positive control for its own predicate. The `common` unit tests that each test binary used to repeat now run once.

## [0.30.2] — GLSL 8/16-bit integer storage, HLSL 16-bit integer kernels, WGSL/HLSL 8/16-bit declines pinned (ROADMAP 8e, 8n)

**GLSL kernels hold 8- and 16-bit integer buffers and locals (ROADMAP 8e, GLSL integer storage).** An `I32`/`U32` kernel through `vulkane-glsl-emit` may now have `I8`/`U8`/`I16`/`U16` buffers and locals (`int8_t`/`uint8_t`/`int16_t`/`uint16_t`, with `GL_EXT_shader_8bit_storage` and `GL_EXT_shader_explicit_arithmetic_types_int8`, or the 16-bit pair, declared only for the widths the kernel uses), reached by wrapping `Cast`s the shared walker now spells for such a kernel (`Dialect::narrow_ints` / `convert`; default off, so WGSL and HLSL are unchanged): widening sign-extends a signed type and zero-extends an unsigned one, narrowing wraps. The SPIR-V emitter had this; the kernels are its `narrow_kernel` support. An atomic in a kernel holding a narrow buffer is a typed decline (the target would have to be proved 32-bit). Tests on a device through glslang and `spirv-val` (`shaderInt8`/`storageBuffer8BitAccess` or `shaderInt16`/`storageBuffer16BitAccess` requested, the module's `required_features` checked, a `SKIP (unsupported)` line otherwise): all four types widen with the right extension and wrap at the maximum (127 to -128, 255 to 0, 32767 to -32768, 65535 to 0), each with its mutant (the other extension, no wrap) asserted different; the extensions appear only for the widths used and a 32-bit kernel declares none.

**HLSL and WGSL 8/16-bit integer buffers: declines pinned (ROADMAP 8e).** WGSL has no 8- or 16-bit integer types, so a kernel with an `I8`/`U8`/`I16`/`U16` buffer or local is now a typed decline that says so (new `Dialect::narrow_refusal`, default none: GLSL and the others are unchanged), and a test reads naga to confirm it rejects `i8`/`u8`/`i16`/`u16` while accepting `u32` in the same module. HLSL hands such a kernel to the Slang emitter, which already declines each element type by name ("not lowered yet", `fishstringer-slang-emit`'s wording); what `dxc` 1.9.0.5347 can do is pinned separately: it compiles `int16_t`/`uint16_t` buffers with `-enable-16bit-types` and has no 8-bit integer type (`int8_t` is undeclared), and the test fails the day either changes. (The 16-bit arm is written: next paragraph.) No new `#[allow]` and no `#[path]` include: the two test files build their small kernel inline.

**HLSL kernels hold 16-bit integer buffers and locals (ROADMAP 8e, HLSL 16-bit arm).** An `I32`/`U32` kernel through `vulkane-hlsl-emit` may now have `I16`/`U16` buffers and locals. `fishstringer-slang-emit` declines them, so `Hlsl::lower_kernel` lowers such a kernel through its own dialect of `vulkane-kernel-text` (new dependency; `Cargo.lock` gains the edge): `StructuredBuffer<int16_t>`/`RWStructuredBuffer<uint16_t>`, `int16_t` locals, wrapping C-style casts (widening sign-extends `I16` and zero-extends `U16`, narrowing wraps), `[[vk::binding(n, 0)]]`, a `[numthreads]` entry over `SV_GroupThreadID`/`SV_GroupID`, `AllMemoryBarrierWithGroupSync()`. `dxc` needs `-enable-16bit-types` (without it `int16_t` is undeclared): new `vulkane_hlsl_emit::kernel_dxc_args(&kernel)` returns it and the text names it on a `// dxc:` line; the module needs `shaderInt16` and `storageBuffer16BitAccess` (`required_features` says so). Other kernels still go through the Slang path unchanged. `I8`/`U8` stay a typed decline naming the element and the reason (`dxc` has no 8-bit integer type); atomics, subgroup ids, scalar parameters, float and 64-bit in a 16-bit kernel are typed declines too. Tests (all pass on the harness device through `dxc` and `spirv-val`): `I16` and `U16` widen with the right extension and wrap at the maximum (32767 to -32768, 65535 to 0), bit-exact against the host oracle; the mutant with the other extension, applied to the emitted text, runs on the device and equals the oracle's wrong-extension value; the flagless compile is rejected by `dxc`; text pins for the 16-bit declarations and no `int64`/`ConstantBuffer`. Added test-include pattern, as the other test files: one `#[allow]` and three `#[path]` includes in `tests/kernel_narrow.rs`, retired by ROADMAP row 8n.

## [0.30.1] — GLSL kernels lower F16 compute (ROADMAP 8e, GLSL half arm)

**GLSL kernels lower `F16` compute (ROADMAP 8e).** `vulkane-glsl-emit`'s `lower_kernel` now lowers `F16` kernels through the shared walker: `float16_t` buffers, locals and arithmetic (`GL_EXT_shader_explicit_arithmetic_types_float16`, `GL_EXT_shader_16bit_storage`; glslang declares `Float16` and `StorageBuffer16BitAccess`), every local and stored value `precise` (each multiply, add and subtract `NoContraction`), and `DenormPreserve` and `SignedZeroInfNanPreserve` at 16 bits through `GL_EXT_spirv_intrinsics`, as the SPIR-V emitter does. The ops are the SPIR-V accuracy gate's exact set (`Add`/`Sub`/`Mul`, compares, NaN-propagating `Max`/`Min`, `Neg`/`Abs`/`Sqr`/`Floor`/`Ceil`/`Trunc`/`Round`, `Select`) and, with the float control flow of the previous cut, `If`/`For`/`While`. A float literal is lowered only when it is exactly a half (`0.5hf`); `0.1` is declined by name rather than rounded. `Div`, `Sqrt`, `Recip`, `Rsqrt` and the transcendentals, a `Cast` to or from a half, an `F32` or `F64` buffer in a half kernel and a scalar parameter are typed declines. `vulkane-kernel-text::Dialect` gains `half` (default false: WGSL and HLSL unchanged). The host needs `shaderFloat16` and `storageBuffer16BitAccess` and the 16-bit float controls (`required_features` on the module says so). Tests (all pass on the harness device through glslang and `spirv-val`): half multiply-add rounding after each step with a subnormal result (2^-24) and the fusion pair `(1+2^-10)^2 + (1+2^-10)`, NaN-propagating `Max`/`Min` and `Floor` over NaN, infinities and signed zeros, each bit-exact against the host half oracle with its mutant (fused, flushed, NaN-ignoring max, nearest-rounding floor) asserted different; the module declares `Float16`, `StorageBuffer16BitAccess`, `DenormPreserve`/`SignedZeroInfNanPreserve` at 16 and two `NoContraction`s, and the `F32` control declares none of the half capabilities; Div/Sqrt/Exp, an inexact literal and an `F32` buffer are declines. Not tested on a device: `NoContraction` (instruction count only).

**SPIR-V kernels lower `Cast` between integers and `F16` (ROADMAP 8e, int<->half).** Integer (`I32`/`U32`) to half is rounded to nearest even at 16 bits and declares `RoundingModeRTE` at 16 only where a conversion rounds; it goes through `f32` and then `OpFConvert` (exact: an integer below 2^24 is exact in `f32`, and one at or above 65520 is infinity in half whichever way `f32` rounded it), because on the harness device a direct `OpConvertSToF` to half rounds toward zero (2051 gave 2050, not 2052) although the mode is declared and the device reports `shaderRoundingModeRTEFloat16`; the `OpFConvert` narrowing honours it. Half to `I32`/`U32` rounds per `RoundMode` (`RoundEven`, `Trunc`, `Floor`, `Ceil`), maps NaN to 0 and clamps; a half tops out at 65504, below every 32-bit range end, so the bounds are the finite extremes (`-65504` or `0`, `65504`, inclusive) and only an infinity is out of range. Tests: `i32`/`u32` to `f16` over the ties at 2049 and 2051 and the overflow edge (65519 vs 65520, `i32::MAX`, `u32::MAX`), and `f16` to `i32` (nearest even, toward zero) and `u32` over halves, negatives, the maxima, both infinities and NaN, bit-exact on a device through `spirv-val`, each with its mutant (toward zero, wrap instead of clamp) asserted different; a unit test asserts the rounding mode is declared only where a conversion rounds. GLSL, HLSL and WGSL half casts stay declined.

**Kernels lower packed values (`vec2`/`vec4`) on SPIR-V and GLSL, by scalarization (ROADMAP 8e, vectors).** The kernel IR has packed locals (`LocalDecl::packed`), `Pack` and `Extract`, and lane-wise `Add`/`Sub`/`Mul`/`Div`/`Min`/`Max`/`Select`/`Cast`; `unpopped`'s shared `lower_dag` has no seam for them ("packed values have no lowering seam yet"), so an emitter cannot spell `OpTypeVector` or `vec4` through it. New `vulkane_kernel_text::scalarize` (called first by `vulkane-spirv-emit` and the shared text walker, which makes `vulkane-spirv-emit` depend on `vulkane-kernel-text`; `Cargo.lock` gains that edge) rewrites a kernel with packed values into scalars: a packed local of `n` lanes becomes `n` scalar locals, `Pack(xs)` lane `k` is `xs[k]`, `Extract(v, k)` is lane `k`, a lane-wise operator applies to each lane, and an assignment to a packed local evaluates every lane into a fresh temporary before writing any, so `v = Pack(v[1], v[0])` swaps rather than copies one lane. The bits are those of a native vector (the float controls are per dtype). The output has no vector type; native `OpTypeVector`/`vec`, vector buffers and `Schedule::Vectorized` wait on the `unpopped` seam. A packed `Bitcast`, an operator that is not lane-wise on a packed value (`Unary`) and a packed `Qualified` operation are typed declines naming them; HLSL and WGSL are not wired. Tests on SPIR-V and GLSL (device, bit-exact against native `f32`, one run per lane over NaN, infinities, signed zeros, a denormal and the fusion pair): `vec2` `v = v * v + v` then a lane swap, `vec4` `Pack(x, y, x + y, x - y)`, lane-wise NaN-propagating `Max` and a two-pair swap; the fused, no-swap and swap-in-place (both lanes end as the old second lane) mutants are each asserted different from the reference somewhere; the GLSL source has no `vec` type and goes through temporaries.

## [0.30.0] — **BREAKING** (dependency floor): take unpopped 0.60.1 and fishstringer 0.10.0 (seven new ElementKind dtypes, declined by name)

**The emitters move to unpopped, unpopped-vocab, unpopped-convert and unpopped-cpu-c `~0.60.1` (the `fixtures` dev-dependency too) and to fishstringer-slang-emit, fishstringer-slang-parse and fishstringer-kernels-source `~0.10.0`** (fishstringer 0.10.0 requires `unpopped ^0.60.1`). The lock holds one copy of each (`cargo tree -d` shows no duplicate `tree-sitter`, `unpopped` or `fishstringer` crate) and `cargo check --workspace --all-targets --locked` is clean. fireline is not pinned by this repository (no `Cargo.toml` or `Cargo.lock` entry). **Breaking, and why:** the dependency floor of nine crates moves a minor line, so a consumer on unpopped 0.59 or fishstringer 0.9 cannot resolve these versions; `cargo-semver-checks` against `origin/main` passes 202 of 202 on the eight emitter, parser and walker crates (it does not compare the internals of the fishstringer items `vulkane-slang-emit` and `vulkane-slang-parse` re-export: UNVERIFIED for those). **What unpopped 0.60 adds (E1a, KISS 0.24.0):** seven `ElementKind` variants (`F128`, `I128`, `U128`, `I256`, `U256`, `F8E3M4`, `F8UE4M3`) and `KernelError::DtypeNotComputed`. Measured: `check_kernel` refuses each of them first (as a kernel dtype and as a buffer dtype), so `lower_kernel` returns the typed `UnsupportedKernel` naming the dtype; every emitter's `supports_dtype` says `Unsupported` for all seven (wildcard allow-lists; no emitter matches `ElementKind` exhaustively, which is why nothing needed changing). New `tests/dtype_decline.rs` in the glsl, wgsl, spirv and hlsl emitters (3 tests each, walking all 35 `ElementKind` values) with four vacuity mutants caught by name; they pin the emitter's own `supports_dtype`, not the checker's text. `docs/design/dtypes-unpopped-0.60.md` records what each target has for these dtypes with a confidence per cell (SPIR-V 128/256-bit and HLSL shipped 8-bit floats UNREAD; MSL read from the 4.1 specification, whose Table 2.16 has a typo in the `fp8_e5m2` NaN cell). Not covered: `vulkane-msl-emit` (on the unmerged stack).

## [0.29.0] — **BREAKING** (dependency floor): take unpopped 0.59.1 and fishstringer 0.8.0 (the uniformity guard, U63)

**The emitters move to unpopped, unpopped-vocab, unpopped-convert and unpopped-cpu-c `~0.59.1` (the `fixtures` dev-dependency too) and to fishstringer-slang-emit, fishstringer-slang-parse and fishstringer-kernels-source `~0.8.0`.** The lock holds one `unpopped` (0.59.1; `cargo tree --workspace -i unpopped` shows a single copy, and no duplicate `tree-sitter`, `unpopped` or `fishstringer` crate in `cargo tree -d`): fishstringer 0.8.0 is the first release on unpopped 0.59, and the 0.7.0 releases it replaces required `unpopped ^0.58.0`, which with the vulkane crates on 0.59 left two `unpopped` copies and failed to compile (`vulkane-hlsl-emit`: `Slang` implemented the 0.58.5 `Backend`; `vulkane-spirv-emit` oracle test: two `ElementKind`s). `cargo check --workspace --all-targets --locked` is clean. **Breaking, and why:** the dependency floor of every emitter crate moves a minor line, so a consumer on unpopped 0.58 or fishstringer 0.7 cannot resolve these versions; no vulkane API changed. **Behaviour that comes with unpopped 0.59 (U63, from its notice):** `check_kernel` refuses a `Subgroup` `Collective`/`CollectiveTuple` after a lane-divergent `Return` (`KernelError::CollectiveAfterReturn`) and admits control from `threadIdx.y/.z` when the launch states one subgroup size dividing `block.x`; no vulkane emitter lowers a collective yet, so no emitted output changes. Not exercised here: the seven dtype tokens KISS 0.24.0 added (nothing in these crates matches `ElementKind` exhaustively, which is why it compiles; a test for the typed decline of each token is not part of this change).

## [0.28.1] — Fixed: a closure variable named like a Vault token tripped CodeRipper SEC-002 on main (ROADMAP 9g)

**`PhysicalDevice` property readers rename a closure variable so the hosted portfolio job stops reporting a committed secret (ROADMAP 9g).** `vulkane/src/safe/physical.rs` read two property flags through a one-letter closure variable; gitleaks' `vault-service-token` rule is `s.` plus 24 characters, and one of the field names is exactly 24 long, so SEC-002 reported a Critical secret that is a Rust field access. The variable is now `f` in both closures; behaviour is identical and the current tree scans clean. An older commit still holds the pattern, so CodeRipper keeps reporting it from history until a CodeRipper-side waiver lands (CireSnave allowed one, board 510; a `.gitleaksignore` entry does not clear CodeRipper's SEC-002, measured); no waiver or ignore entry is added here.

## [0.28.0] — **BREAKING** (vulkane-kernel-text only): GLSL kernels lower F32 compute with an accuracy-gated op set (ROADMAP 8e, GLSL float arm)

**GLSL kernels lower `F32` and `F64` compute (ROADMAP 8e).** `vulkane-glsl-emit`'s `lower_kernel` now lowers `F32` kernels (and `F64` ones, spelled `double` with `#extension GL_ARB_gpu_shader_fp64 : require`, literals `1.5lf`, float controls at 64 bits; the host needs `shaderFloat64`, as for SPIR-V) through the shared walker: buffers and workgroup arrays of `float`, locals declared `precise`, every stored value held in a `precise` temporary so glslang decorates each contributing add, subtract and multiply `NoContraction` (no FMA the source does not ask for; the Unpopped oracle rounds a statement once at f64 and cannot tell fused from unfused, so the evidence is the instruction count and a host `f32` reference). The source declares `DenormPreserve` and `SignedZeroInfNanPreserve` at 32 bits through `GL_EXT_spirv_intrinsics` (`SPV_KHR_float_controls`, capabilities 4464 and 4466), as the SPIR-V emitter does. Lowered: `Add`/`Sub`/`Mul`, the six compares (`1.0`/`0.0`, `!=` unordered), NaN-propagating `Max`/`Min`, `Neg`/`Abs`/`Sqr`/`Floor`/`Ceil`/`Trunc`/`Round` and `Select`; indices are the `u32` execution ids. Float `Cast` lowers (`I32`/`U32` and `F32`/`F64`, as `vulkane-spirv-emit`'s): integer to float rounds to nearest even with `RoundingModeRTE` declared (through `GL_EXT_spirv_intrinsics`, only where a conversion rounds), float to integer rounds per the mode, maps NaN to 0 and clamps (the conversion only sees an in-range value), integer to integer wraps; a float kernel may hold `I32`/`U32` buffers and locals. A conversion between `F32` and `F64`, or any kernel using both widths, is a typed decline: glslang keeps one `spirv_execution_mode` per kind (measured with glslang 16.2: `DenormPreserve 32` then `64` leaves only 64), so the float controls could not be declared at both widths. **`If`, `While` and `For` lower in float kernels on SPIR-V and GLSL.** A condition is a float expression (tested with an unordered `!= 0.0`, so a NaN is true, as `Select` has it; GLSL holds it in a `precise` temporary first so its adds and multiplies are not fused) or an integer leaf (a loop counter, an execution id); a `For` counts with an `I32` or `U32` local (`ConstInt` bounds of those types are now lowered in a float kernel), `Break` and `Continue` work. Integer arithmetic and integer comparisons inside a float kernel remain a typed decline (the shared `lower_dag` has no integer seam there). Device tests on both arms (all pass): the `y * w > 0` / `acc + y * w` / `acc > 100` break / `while acc > 8` halving kernel over NaN, a denormal, 2^24+1, signs and zeros, `U32` counting up and `I32` counting down, bit-exact against native `f32`; the fused-multiply-add and the ignored-`break` kernels each differ from the reference somewhere (asserted), and the GLSL module's `NoContraction` count is 0 for the same source without `precise`. **WGSL declines float kernels by name.** The shared walker's decline of a float compute dtype now carries the dialect's reason (`Dialect::float_refusal`): WGSL has no spelling for `precise`, `NoContraction` or `SignedZeroInfNanPreserve`, and its floating point evaluation rules let an implementation fuse a multiply and an add (a body of correctly rounded primitives is `correctly_rounded`, 0 ulp, in unpopped's precision contract) and leave NaN and infinity indeterminate, so a float kernel could not be held to that tier; nothing is lowered. This is the kernel path only: `lower_kernel` already declined every float dtype on main ("integer kernels only"), so no capability is removed, and the plan path `Wgsl::lower` still spells `f32` unchanged. A test reads the SPIR-V naga writes for `a * b + a` (one `OpFMul`, one `OpFAdd`, `LocalSize` as the control) and asserts no `NoContraction` and no float-control mode, failing the day naga writes one. **HLSL declines float kernels by name.** `vulkane-hlsl-emit`'s `lower_kernel` now refuses any kernel touching `F16`/`Bf16`/`F32`/`F64` (compute dtype, buffer, scalar or local): the Slang path it delegates to wrote such a kernel as plain HLSL with no `precise` and no float controls, so a multiply-add could be fused and NaN/infinity/signed-zero handling was the driver's choice. Measured with `dxc` 1.9.0.5347: `precise` becomes `NoContraction` and `-denorm preserve` becomes `DenormPreserve`, but nothing requests `SignedZeroInfNanPreserve` (`[[vk::ext_execution_mode]]` is a parse error). Two tests pin that against `dxc` itself (with a no-`precise` control) and fail the day `dxc` can spell it. Declined by name, each with its reason: `Div` (`OpFDiv` is 2.5 ULP), `Sqrt`, `Recip`, `Rsqrt`, the transcendentals (the Vulkan bounds of `vulkane-spirv-emit`'s accuracy table), a `Cast` outside those types and modes, control flow (its test and counter are integer ones), a scalar parameter, atomics and `F16`. `vulkane-kernel-text::Dialect` gains `float`, `lit_float`, `float_unary`, `float_minmax`, `type_name` and `temp_typed` (defaults: not lowered), so WGSL and HLSL are unchanged. Tests: a multiply-add, `Max`/`Min` and `Floor` bit-exact on a device over NaN, infinities, signed zeros, a denormal, 2^24+1 and 3.4e38*10; the `NoContraction` count is 2, and 0 for the same source without `precise` (mutant); the float-control execution modes are present in the glslang module and absent from an integer kernel; Div, Sqrt and Exp are typed declines. Not tested on a device: `NoContraction` is evidenced by the instruction count (a driver that fused would only be seen on one that does), and the float controls are read from the device and skip loudly when absent.

**Breaking, and why (measured):** `vulkane-kernel-text` was first published at 0.27.1 and this change adds public fields to two of its exhaustively constructible structs: `Decl.elem`, `Layout.local_elems` and `Layout.rte`. `cargo-semver-checks` 0.51.0 against the published 0.27.1: `vulkane-kernel-text` 201 of 202 pass, 1 fails (`constructible_struct_adds_field`: "semver requires new major version"); `vulkane-glsl-emit`, `vulkane-hlsl-emit`, `vulkane-spirv-emit` and `vulkane-wgsl-emit` pass 202 of 202. The new `Dialect` methods have defaults (not breaking). **Migration:** only code that builds a `Layout` or `Decl` with a struct literal is affected, and these structs are handed to a `Dialect` by the walker; the only implementors known to us are the emitters in this repository (`vulkane-glsl-emit`, `vulkane-wgsl-emit`, `vulkane-msl-emit` on the unpublished stack), so a consumer outside it has nothing to change.

## [0.27.2] — Changed: decline tests use unpopped 0.58.5 fixtures (ROADMAP 9h); the version stamp moves the KISS vocabulary manifest

The four patent-decline test files (glsl, hlsl, spirv, wgsl emitters) now take their statements from the published `unpopped::fixtures` (unpopped ~0.58.5, feature `fixtures`, dev-dependency only) instead of hand-built ones: every fixture and every nested shape is asserted checker-valid first, so a refusal is the emitter's own text. `CopyUnpack` is now covered by a checker-valid fixture (it was the one interim hand-built case). New test: every `Collective` kind, including the exclusive Max/Min/Prod scans of unpopped 0.58.4, is refused by name; nothing is emitted (board 506). New control: every non-declined statement's fixture never carries the declined marker. Verified with a vacuity-mutant matrix (13 name removals and 4 nesting mutants, run against kernel-text via glsl and wgsl, and against spirv and hlsl; all caught, unmutated control passes). No library code changed.

`tools/stamp-extra.txt` (new) lists the `generated_from` line of `kiss-vulkan-vocab/manifest/vulkan-vocabulary.json`, so `tools/stamp-version.py` stamps the crate version it names. Since WSP-010 (#173, #174) put `kiss-vulkan-vocab` on the project version line, every stamp changed the version the manifest was emitted by, and `vocabulary_manifest.rs` failed byte-for-byte until the manifest was regenerated by hand. Verified by stamping a scratch copy (0.27.0 to 0.27.1) and running that test with and without the line. No library code and no script change.

## [0.27.1] — Fixed: four emitter READMEs corrected before their first publish

`vulkane-hlsl-emit` and `vulkane-wgsl-emit` no longer say "does not lower kernel-layer kernels yet" (false: wgsl lowers integer kernels through
`vulkane-kernel-text`; hlsl refuses float kernels and passes the rest to the Slang path, whose scope is stated UNVERIFIED) and now count the
thirteen declined statements (`CopyUnpack` was missing from the count). `vulkane-glsl-emit` and `vulkane-kernel-text` gain the infringement note
CireSnave asked for on 2026-11-16 (the thirteen names, the patents, the 434 C basis), pinned by `tests/readme_note.rs` in each.
`vulkane-wgsl-parse` ships a `THIRD-PARTY-NOTICES.md` (no third-party source in the package). Docs, one test file per crate, no code change.

## [0.27.0] — **BREAKING** (dependency floor): take unpopped 0.58.0 and fishstringer 0.7.0 (`Stmt::CopyUnpack` is refused by name)

**The emitters move to unpopped, unpopped-vocab, unpopped-convert and unpopped-cpu-c `~0.58.0` and to fishstringer-slang-emit, fishstringer-slang-parse and fishstringer-kernels-source `~0.7.0`.** The lock holds one `unpopped` (0.58.0; `cargo tree -i unpopped` shows no second version): fishstringer 0.7.0 is built on unpopped 0.58, which the 0.6.0 line was not (the hlsl-emit `Slang` wrapper failed to compile against two `unpopped` versions until it was published). unpopped 0.58.0 adds the statements `Tcgen05Alloc`, `Tcgen05Dealloc`, `Tcgen05RelinquishPermit` and `CopyUnpack` and the trap `KernelTrap::TmemMisuse` (not referenced here), and new fields on `Wgmma` (`a_regs`) and `Tcgen05Mma` (`m`, `d_lane`, `b_ty`, `negate_a`, `negate_b`), which this repository matches only with `{ .. }`. The three exhaustive `Stmt` walks (`vulkane-kernel-text::decline::nested` and the SPIR-V and HLSL emitters' `patent_decline::nested`) name the new variants, so no wildcard absorbs them: `CopyUnpack` is refused by name, because its own doc says the `CopyBulk` patent note applies and emitters for other manufacturers decline it (fishstringer classifies it the same way, kernel.rs:1095); the three `Tcgen05` allocation statements go on the unnamed side with `Tcgen05Mma` and `Tcgen05Ld` (the board 434 named list is a ruling and is not widened). **Interim:** the new `CopyUnpack` rows in the four `tests/patent_decline.rs` files are not checker-valid and have no vacuity mutant, so they prove the by-name refusal only; checker-valid fixtures wait on `unpopped::fixtures` (ROADMAP 9h). Verified: workspace builds with all targets; fmt, clippy, doc and the device tests under the GPU lock are in the PR body.

## [0.26.1] — kiss-vulkan-vocab, vulkan_gen and vulkane_derive join the project version line (number change, no API change; WSP-010)

**`kiss-vulkan-vocab` (was 0.6.0), `vulkan_gen` (was 0.5.2) and `vulkane_derive` (was 0.1.3) now carry the project version (0.26.0 at this base), and `vulkane` requires them as `0.26` (ROADMAP 9g).** This is CireSnave's one-version-per-project rule and CodeRipper WSP-010. It is a number change with no API change, checked two ways: the three crates' only manifest edit is the `version` line (their source, build script, features and dependencies are byte-identical to main), and `cargo semver-checks --baseline-rev` against main at `--release-type patch` passes 229 of 229 checks for each of `vulkan_gen` and `kiss-vulkan-vocab` (re-run on main 1f5ad80). `vulkane_derive` is a proc-macro crate, which that tool cannot check, so its evidence is the identical source. The three crates stay listed in `changelog.d/self-versioned.txt` for this PR only: `tools/check-fragment.py` counts a version change in an unlisted manifest as a bump and refuses it beside a fragment, so the list is emptied in the next PR (the stamp does not read the list). Consumers of the published 0.6.0, 0.5.2 and 0.1.3 are unaffected until the next publish; the new numbers are higher, so no version goes down.

## [0.26.0] — Take unpopped 0.57.0 and fishstringer 0.6.0 (Stmt::ProxyFence is NVIDIA-only and refused by name)

**The emitters move to unpopped, unpopped-vocab, unpopped-convert and unpopped-cpu-c `~0.57.0` and to fishstringer-slang-emit, fishstringer-slang-parse and fishstringer-kernels-source `~0.6.0`.** unpopped 0.57.0 adds `Stmt::ProxyFence` (PTX `fence.proxy.async`); it is named in the three statement walks that list every `Stmt` variant (`vulkane-kernel-text::decline::nested`, and the SPIR-V and HLSL emitters' `patent_decline::nested`), so no wildcard absorbs it, and a kernel holding one is refused as before (no target language here has an asynchronous proxy). The byte-addressed `Place::Raw` and `Place::AllocRaw` are new since 0.56.5 (the version the lock held) but already in 0.56.12, so they are not a 0.57.0 change (0.57.0 itself adds only `Stmt::ProxyFence` among the variants a vulkane match names); `Place` is `#[non_exhaustive]` and each emitter already ends its place match in a typed refusal, so they stay refused. The lock holds one `unpopped` (0.57.0) and one `fishstringer-slang-emit` (0.6.0). The next rungs of the kernel-emission stack add the by-name refusal tests for the new variants.

## [0.25.7] — SPIR-V kernels lower F32/F64/F16 compute dtypes with an accuracy-gated op set, float Cast, and F16 storage (ROADMAP 8, 8e float slice)

**SPIR-V kernels lower F32 and F64 compute dtypes, with an accuracy-gated op set (ROADMAP 8).** `vulkane-spirv-emit`'s `lower_kernel` now lowers `F32`/`F64` kernels (the plan path is unchanged). The module declares `SignedZeroInfNanPreserve` and `DenormPreserve` at the kernel's width (the oracle is IEEE: NaN, infinities, signed zeros and denormals kept), decorates every float multiply, add and subtract `NoContraction` (no FMA the oracle does not do, fuel#568), and uses no `RelaxedPrecision`. `Float64` is declared only at `F64`; every 32-bit and 64-bit integer kernel is byte-identical (golden). Only ops whose Vulkan accuracy bound is no looser than the contract tier are lowered: `Add`/`Sub`/`Mul`, `Neg`, `Abs`, `Floor`/`Ceil`/`Trunc`/`Round`, `Sqr`, `Relu`/`Sign`/`Step`, `Max`/`Min`/`NMax`/`NMin`, compares, and `Rsqrt` at `F32`; float `Div`/`Recip` (`OpFDiv` is 2.5 ULP, the oracle divides correctly rounded), `Sqrt`, and the transcendentals are declined by name, with the Vulkan spec section cited in the crate-doc accuracy table. `required_features` now also names `shader_float64` and the four float-controls modes; new `vulkane::safe::PhysicalDevice::float_controls_properties()` reads `VkPhysicalDeviceFloatControlsProperties` so a host can check them; the driver is not a guard. Float atomics (needs an `af32` device) and float `Cast` remain todo. Tests: `F32` and `F64` multiply-add, NaN-propagating `Max`/`Min` and `Floor` bit-exact on a device over 0.1, 1e-30*1e-10 (a denormal), 2^24+1, 3.4e38*10, NaN, infinities and signed zeros; an f32-for-f64 mutant oracle that must differ; a control that the comparison rejects wrong values, flipped zero signs and flushed denormals. The accuracy table is filled from the Vulkan spec source (`appendices/spirvenv.adoc`, read 2026-11-23): no transcendental's bound is within its contract tier (`Exp` 3 + 2|x| ULP vs 2, `Log` 3 ULP vs 1, `Sin`/`Cos` absolute error only, `Atan` 4096 ULP), `sqrt()` is "inherited from 1.0 / inversesqrt()" and `OpFDiv` is 2.5 ULP, so all stay declined with the bound cited. The float controls are justified from the oracle (`Numerics` default `ftz: false`; `Select`/`Max`/`Min` move NaN payloads and signed zeros verbatim); `NoContraction` is evidenced by an instruction-count unit test only.

**SPIR-V kernels lower Cast between integers and floats and between F32 and F64 (ROADMAP 8).** `vulkane-spirv-emit`'s `lower_kernel` lowers `ScalarExpr::Cast` against `unpopped`'s `cast_ref.rs`: integer to integer (wrapping), integer to float and `f64` to `f32` rounded to nearest even (the module declares `RoundingModeRTE` at the result's width only when such a conversion is present), `f32` to `f64` exact, and float to integer as the oracle defines it: rounded per `RoundMode` (`RoundEven`, `Trunc`, `Floor`, `Ceil`), NaN to 0, clamped to the destination (`OpConvertFToS/U` are undefined out of range, so the clamp is composed with selects and only an in-range value is converted). Declined by name: `NearestAway` and non-nearest int-to-float rounding, `ClampFinite`, a non-`Clamp` float-to-int, and a 64-bit integer destination of a float cast. A float kernel may now hold `F32`/`F64` and 32-bit integer buffers and locals (the operands and results of a conversion); `Float64` and the float controls are declared for each float width actually used. `required_features` also names `rounding_mode_rte_f32`/`_f64`; `vulkane::safe::FloatControlsProperties` gained `rounding_mode_rte_f32`/`_f64`. Tests on a device: i32/u32 to f32 (ties at 2^24+1 and 2^24+3), i32 to f64, f32 to i32 (nearest even and toward zero) and u32 with NaN, infinities, out-of-range and negative inputs, f64 to f32 (ties, overflow to infinity, denormals, NaN), f32 to f64; each case proves its mutant (truncating, wrapping, through-f32, sign dropped) differs from the expected output.

**SPIR-V kernels hold IEEE half (`F16`) buffers and locals, converted to and from `F32`/`F64` (ROADMAP 8e, half slice).** An `F16` is a storage and local type of a float kernel, never a compute dtype and never a plan dtype; conversions to and from `F32`/`F64` are the `Cast`s above (narrowing `NearestEven` with `Wrap` under `RoundingModeRTE` at 16 bits, widening exact; an integer to or from a half is declined by name). `Float16`, `StorageBuffer16BitAccess` and `SPV_KHR_16bit_storage` are declared only when a half is used, and the float controls at width 16; `required_features` names `shader_float16` and the 16-bit float controls, and `FloatControlsProperties` gained the 16-bit fields. Device tests (all pass; ties and one double-rounding trap): `f32` to `f16` over 1+2^-11 (tie), 65519/65520 (overflow edge), 2^-25 and 3·2^-25 (subnormal ties), NaN, infinities, signed zero; `f16` to `f32` over subnormals, infinities, NaN, signed zero; `f64` to `f16` over a value just above a tie that rounds up in one rounding but down when rounded through `f32`. Mutants (toward-zero rounding, flushed subnormals, rounding through `f32`) are asserted different. F16 arithmetic as a compute dtype is the next paragraph.

**SPIR-V kernels compute at `F16` (ROADMAP 8e, half compute).** `lower_kernel` accepts a kernel whose compute dtype is `F16`: `Add`/`Sub`/`Mul` and `Neg`, `Abs`, `Sqr`, `Floor`/`Ceil`/`Trunc`/`Round`, `Relu`/`Sign`/`Step`, `Max`/`Min`, `FmaxIeee`/`FminIeee` and the comparisons, under the same accuracy gate as `F32` (the Vulkan bound no looser than the contract tier); `Div`, `Sqrt`, `Rsqrt` and the transcendentals are declined by name. `Float16` is declared once, with `DenormPreserve`/`SignedZeroInfNanPreserve` at 16 bits and `NoContraction` on each multiply, add and subtract. A float literal is lowered only when it is exactly a half; one that is not (0.1) is declined rather than rounded, and a scalar extent parameter is not lowered at `F16`. Device tests (all pass): a multiply feeding an add rounds after each step and does not fuse (`(1+2^-10)^2 + (1+2^-10)` differs fused), a subnormal result (`2^-24`) survives, NaN, infinities and signed zeros are IEEE, and NaN-propagating `Max`/`Min` and `Floor` match; each case asserts its mutant (fused, NaN-ignoring max, nearest-rounding floor) differs from the expected output. Not tested on a device: `Rsqrt`/`Div` (declined), integer to or from half `Cast` (declined by name).

**`vulkane-spirv-emit`: no module cycle (ROADMAP row 9d, CodeRipper MOD-002).** `accuracy.rs` used `ops::Family` while `ops.rs` used `accuracy::{arith_refusal, binary_refusal, unary_refusal}`, a 2-module cycle this PR added (`main` has none). `Family` and its `impl` move unchanged to a new leaf module `family.rs` (it reads only `unpopped_vocab`) and the six `use` lines name it; nothing public changes (`Family` is `pub(crate)`). `coderipper check MOD-002 --profile extended` on the committed tree: `vulkane-spirv-emit` 12 modules, 15 edges, 1 cycle before; after, see the PR body.

## [0.25.6] — GLSL, HLSL and WGSL kernel arms: dynamic extent, 32-bit atomics, ExecId, Private arrays, GLSL I64/U64 (ROADMAP 8a, 8d, 8c, 8k, 8d wider dtypes)

**GLSL and WGSL lower a shared array sized by a scalar parameter (ROADMAP 8a).** `vulkane-kernel-text` lowers a `Shared` allocation with `Extent::Param(local)` when `local` is a scalar parameter of the compute dtype: GLSL declares `layout(constant_id = <param index>) const T sN = 1;` and `shared T aK[sN];`, WGSL `@id(<param index>) override sN: T = 1;` and `var<workgroup> aK: array<T, sN>;`; the parameter's local starts at the constant. The extent is fixed per pipeline, not per dispatch (as on SPIR-V). Tests: the block-reverse kernel through each on a device. GLSL sets the spec constant at pipeline creation; WGSL fixes the override with naga's `process_overrides` first, since naga's SPIR-V writer refuses an unresolved override. HLSL still declines a dynamic extent. `Decl` and `Layout` of the (unpublished) kernel-text crate gain fields.

**GLSL kernels lower 32-bit integer atomics (ROADMAP 8d, GLSL arm).** `vulkane-kernel-text` gains `Dialect::atomic_rmw` / `atomic_cas` (default: no spelling, a typed decline) and lowers `AtomicRmw`/`AtomicCas` through them; GLSL spells `atomicAdd/Min/Max/And/Or/Xor/Exchange/CompSwap`. GLSL atomics carry no memory order, so a non-`Relaxed` order is a typed decline naming it rather than a silent drop. WGSL and HLSL (slang) keep declining atomics by name. Tests: the SPIR-V slice's contended kernel through glslang on a device (1024 adds exact; the plain-increment control reached 1 of 1024), the spelling counts with the racy control, and the order decline.

**HLSL kernels: 32-bit integer atomics verified on a device (ROADMAP 8d, HLSL arm).** `fishstringer-slang-emit` already lowers `AtomicRmw`/`AtomicCas` to `InterlockedAdd/Min/.../CompareExchange`; `Hlsl::lower_kernel` passes them through. New tests: the contended atomic kernel through dxc on a device (1024 adds exact, one CAS winner; the plain-increment control reached 1 of 1024) and the spelling counts with the racy control.

**WGSL kernels lower 32-bit integer atomics (ROADMAP 8d, WGSL arm).** A buffer or workgroup array that an `AtomicRmw`/`AtomicCas` touches is declared `array<atomic<u32|i32>>` and its plain loads and stores become `atomicLoad`/`atomicStore`; every other kernel lowers to byte-identical WGSL (golden test). Compare-exchange is `atomicCompareExchangeWeak` retried in a loop (the weak form can fail spuriously; the IR contract cannot). Relaxed order and block/device scope only: stronger orders and subgroup/cluster scope are typed declines. `vulkane-kernel-text` gains `Dialect::typed_atomics/atomic_scope_ok/atomic_load/atomic_store/atomic_cas_lines` (defaults leave GLSL unchanged). Tests: the contended kernel through naga on a device (1024 adds exact, one CAS winner, control reached 1 of 1024), a naga-rejected plain-read mutant, goldens.

**GLSL kernels lower ExecId LaneId, SubgroupId and SubgroupSize (ROADMAP 8c, GLSL arm).** `vulkane-glsl-emit` spells the ids `gl_SubgroupInvocationID`, `gl_SubgroupID` and `gl_SubgroupSize` and adds `#extension GL_KHR_shader_subgroup_basic : enable` for a launch that names a subgroup size range. `vulkane-kernel-text` refuses a range with no power of two up to 128 by name (as the SPIR-V emitter does). WGSL and HLSL arms follow. Tests: ids match the device through glslang on a device (128 threads; reading SubgroupId as the lane id fails it), the spellings, the impossible-range declines with a control.

**HLSL kernels: ExecId subgroup ids verified on a device; WGSL declines them by name (ROADMAP 8c).** HLSL: `fishstringer-slang-emit` already lowers LaneId/SubgroupId/SubgroupSize to `WaveGetLaneIndex()` / `gi_ / WaveGetLaneCount()` / `WaveGetLaneCount()`; new dxc device test (128 threads, size 64) and a mutant that must be caught. WGSL: the ids need `enable subgroups;`, which naga 30 (the validity oracle) rejects as not yet implemented, so they stay a typed decline; a test pins the naga message so the decline is lifted when naga accepts it.

**GLSL kernels lower `I64`/`U64` (ROADMAP 8d, GLSL 64-bit arm).** `vulkane-glsl-emit`'s `lower_kernel` accepts a 64-bit integer kernel through the shared `vulkane-kernel-text` walker, which gained `Dialect::wide` and `Dialect::cast_to_compute` (defaults: no, so the WGSL dialect and any other still decline a 64-bit kernel by name) and a wrapping integer `Cast` seam for a wide dialect. The GLSL text is `int64_t`/`uint64_t` under `GL_ARB_gpu_shader_int64`, shift counts masked at 63, execution ids kept `u32` and carried by a `Cast`, buffer indices narrowed to `int`. 64-bit atomics, a specialization-constant extent and integer division stay typed declines. Device tests through `glslangValidator` and `spirv-val` (both pass): add carrying into the high word, a multiply whose high word depends on the low, a shift past 32, a count masked mod 64, and arithmetic vs logical `>>`, each cell compared at 64 bits; a unit test shows a 32-bit lowering would differ in every checked column. A 32-bit GLSL kernel is unchanged.

## [0.25.5] — SPIR-V kernels lower I64/U64, 64-bit atomics with required_features, and 8/16-bit integer buffers (ROADMAP 8d, 8e integer slice)

**SPIR-V kernels lower I64 and U64 compute dtypes (ROADMAP 8d, the road to 64-bit atomics and 8h).** `vulkane-spirv-emit`'s `lower_kernel` now lowers kernels at `I64` and `U64`: 64-bit buffers (stride 8), locals, add/sub/mul, the ruled total `Div`/`Rem` (zero divisor gives the dividend, `MIN / -1 = MIN`, `Rem` as the `Select` composition), shifts masked mod 64 and the arithmetic/logical `>>` by signedness. The `Int64` capability and 64-bit types are declared only for such a kernel: every 32-bit kernel lowers to byte-identical SPIR-V (golden test over seven kernels written before the change). `ExecId` is a `U32` in the IR, so a 64-bit kernel reaches its thread index through a new integer `Cast` (wrapping, to the compute dtype only; float and clamping casts still decline). Plan lowering (`Backend::lower`/`supports_dtype`) still declines 64-bit integers; F32/F64 kernels still decline ("integer kernels only"); a 64-bit atomic is a typed decline naming `Int64Atomics` until its slice. The host must enable `shaderInt64`; the emitter does not see the device. Tests: nine results per thread across the 32-bit boundary at I64 and U64 on a device (shaderInt64=true on the harness GPU), a host oracle at 32 bits that differs in every column (the truncating-lowering mutant), module checks under spirv-val.

**SPIR-V kernels lower 64-bit integer atomics; required_features names the device features a module needs (ROADMAP 8d).** `vulkane-spirv-emit` lowers `AtomicRmw`/`AtomicCas` in `I64`/`U64` kernels to the 64-bit `OpAtomic*` instructions (signed or unsigned min/max by the dtype) and declares `Int64Atomics` once, only for a kernel that has one. New `vulkane_spirv_emit::required_features(words) -> RequiredFeatures { shader_int64, int64_atomics }` reads a generated module's capabilities: **enabling `shaderInt64` (and `shaderBufferInt64Atomics` or `shaderSharedInt64Atomics`) at device creation is the host's job**, the emitter cannot see the device, and on the harness GPU a device created without `shaderInt64` ran an `Int64` module with no error, so the driver is no guard (a test reports this). The kernel `Cast` added for 64-bit kernels is **wrapping only**: an integer cast to the compute dtype, never a saturating cast; float and clamping casts still decline. Tests: 1024 contended 64-bit adds, a signed max and a compare-exchange exact past 2^32 on a device (a plain 64-bit add reached 1 of 1024), the capability declared once and only with an atomic, signed/unsigned instruction choice, spirv-val. Float atomics and GLSL/HLSL/WGSL 64-bit atomics remain todo.

**SPIR-V kernels lower 8- and 16-bit integer buffers and locals (ROADMAP 8e, integer slice).** `vulkane-spirv-emit`'s `lower_kernel` accepts `I8`/`U8`/`I16`/`U16` buffer elements and locals inside an `I32`/`U32` kernel (they are not compute dtypes: the IR admits `ConstInt` at I32/I64/U32/U64 only), reached through the wrapping integer `Cast` (sign-extending a signed source, zero-extending an unsigned one, truncating when narrowing; it is no saturating cast). Capabilities `Int8`/`Int16` and `StorageBuffer8BitAccess`/`StorageBuffer16BitAccess` (with `SPV_KHR_8bit_storage`/`SPV_KHR_16bit_storage`) are declared only for the widths a kernel uses; 32/64-bit kernels are unchanged (golden). `required_features` now also names `shader_int8`, `storage_buffer_8bit`, `shader_int16` and `storage_buffer_16bit`: the host must enable them at device creation and **the driver is not a guard**. The plan path still declines these dtypes. Tests: per type on a device (inputs include each type's extremes; sign vs zero extension and wrap at the maximum, `127 -> -128`, `255 -> 0`; the wrong-extension and no-wrap mutants must differ), capability-only-when-used checks under spirv-val. F16/Bf16/FP8/vectors need float kernels first and stay todo.

**CI: the advisory CodeRipper portfolio job (CireSnave 2026-10-23, "Roll out advisory-only, non-blocking CodeRipper checks everywhere.").** `.github/workflows/ci.yml` gains a `portfolio:` job calling `ciresnave/coderipper/.github/workflows/portfolio.yml` pinned to `56b8982c4c8105866db4e972335588a2e362cc69` (installs CodeRipper 0.48.0, the CODERIPPER_VERSION at this SHA; coderipper Cargo.toml says 0.48.1, unpublished), `permissions: contents: read`, nothing silenced. It is not a required check and branch protection is unchanged; a new finding is a disposition, not a reason to switch the job off.

## [0.25.4] — SPIR-V kernels lower Private arrays, 32-bit integer atomics and the subgroup ids (ROADMAP 8k, 8d, 8c: the SPIR-V integer slice); Fixed: HLSL float kernels are declined

**Fixed: `Hlsl::lower_kernel` no longer writes float kernels as plain HLSL.** `vulkane-hlsl-emit`'s `lower_kernel` (added in #166, 2026-11-21, in no tagged release and in a crate that is not on crates.io: the newest tag is v0.22.0) handed every kernel to the Slang emitter, so an `F32` kernel such as `a * b + a` was written with no `precise` qualifier and no float controls: a driver could contract the multiply and add into an FMA the host `f32` reference does not do, and NaN, infinity, signed-zero and denormal handling was the driver's choice, which breaks the bit-exactness the SPIR-V emitter guarantees. It now refuses any kernel touching `F16`/`Bf16`/`F32`/`F64` (compute dtype, buffer, scalar or local) with a typed `UnsupportedKernel` naming the missing control. Measured with `dxc` 1.9.0.5347: `precise` becomes `NoContraction` and `-denorm preserve` becomes `DenormPreserve`, but nothing requests `SignedZeroInfNanPreserve`; `[[vk::ext_execution_mode]]` is a parse error. Tests: float kernels at `F16`/`F32`/`F64` decline with an integer-kernel control; two `dxc` tests pin the behaviour (a probe gives `NoContraction` 2, `DenormPreserve`, no `SignedZeroInfNanPreserve`; the same probe without `precise` gives 0) and fail the day `dxc` can spell the mode, so the HLSL float arm gets written then. The plan path (`Backend::lower`) is unchanged.

**SPIR-V kernels lower Private arrays; Constant allocations decline (ROADMAP 8k, first slice).** `SpirV.lower_kernel` now lowers a static-extent `MemorySpace::Private` allocation of the compute dtype to a module-scope `OpVariable Private` array (one instance per invocation) and indexes it with the same `Place::Alloc` path as shared memory, including data-dependent indices. A `MemorySpace::Constant` allocation is a typed decline naming ROADMAP 8k: `AllocDecl` carries no initialiser, so a constant table has no data to hold. Kernel locals (`Let/Assign`) were already lowered. Tests: unit tests for the storage class (with a Workgroup control) and the decline; a live device test where each thread fills its own array and reads it at `in[t]`. Not done: GLSL, WGSL and HLSL arms, `Private` with a parameter extent, an oversized-constant negative (needs a constant source).

**SPIR-V kernels lower 32-bit integer atomics (ROADMAP 8d, first slice).** `SpirV.lower_kernel` lowers `Stmt::AtomicRmw` (`Add`, `Min`, `Max`, `And`, `Or`, `Xor`, `Exchange`; signed or unsigned form from the compute dtype) and `Stmt::AtomicCas` on a storage-buffer or workgroup-array element at subgroup, block or device scope. `Relaxed` carries no ordering bits; acquire, release and acq-rel/seq-cst add the pointer's memory class (seq-cst collapses to acquire-release, as the fence does). A `Cluster` scope is a typed decline; an atomic on a `Private` array is refused by `check_kernel`, with a matching guard in the emitter. Tests: unit tests for the instructions and a racy-control module; a live device test where 1024 contended adds from 16 workgroups sum exactly, the unsigned minimum and exactly-one compare-exchange winner hold, and the non-atomic control lost updates (reached 1 of 1024 on the Windows device; reported, not asserted). Not done: GLSL, WGSL and HLSL arms, 64-bit and float atomics (capability-gated).

**SPIR-V kernels lower ExecId LaneId, SubgroupId and SubgroupSize (ROADMAP 8c, ids only).** `vulkane-spirv-emit` lowers the three subgroup ids to the `SubgroupLocalInvocationId`, `SubgroupId` and `SubgroupSize` builtins (capability `GroupNonUniform`, declared only for a launch that names a subgroup size range, which `check_kernel` requires). SPIR-V for Vulkan cannot pin the size, so the launch range is the contract the host meets with `required_subgroup_size` (or the device must run a size inside it); a range no Vulkan device can have (no power of two up to 128) is a typed decline. `Stmt::Collective` is unchanged: still declined by name (ROADMAP 8l). Tests: ids match the device on a device (swapping SubgroupId for the lane id fails it), the device-size-outside-range decline, the impossible-range declines, the capability and builtins in the module.

## [0.25.3] — 3-D launch device tests on all four kernel emitters (ROADMAP 8f)

Adds a live device test for multi-dimensional dispatch on `SpirV`, `Glsl`, `Wgsl` and `Hlsl` `lower_kernel`: a 4x3x2 block in a 3x2x5 grid where every thread writes its packed `ThreadIdx`/`BlockIdx` to the cell that `BlockDim`/`GridDim` place it at, each of the 720 cells checked against the host value, one shared kernel (`vulkane-spirv-emit/tests/support/grid_kernel.rs`). A control that points one `GridDim` at the wrong axis fails the SPIR-V test (run once, by hand, not kept as a test). Adds `Gpu::run_grid`/`run_grid_params` to the test support, and a unit negative that a `Scope::Cluster` barrier on a cluster launch is a typed decline on SPIR-V. Test-only: no emitter code changes. The HLSL delegate needs the buffer rank declared (`Some(1)`); without it it declines "a buffer whose rank is not declared". Not covered: a cluster negative on GLSL, WGSL and HLSL.

## [0.25.2] — logo replaced (FLUX.1 schnell, Apache-2.0; CireSnave's pick 2026-11-21), repository moved to ThinkersJournal/vulkane

`vulkan.jpeg` (Microsoft Designer, personal-use-only terms) is removed from the working tree and replaced by `vulkane-logo.png`, generated with ComfyUI 0.43.0 and FLUX.1 [schnell] (Apache-2.0), seed 11; tool, model, licence, seed, prompt and CireSnave's pick are recorded in `PROVENANCE.md`. History is not rewritten. `.github/nul_gate.py` now lists `.png` as the binary extension. Every `github.com/ciresnave/vulkane` link (crate `repository`/`homepage`, README badge, docs, CI probe user agents) now points at `github.com/ThinkersJournal/vulkane`, which is where the repository was transferred on 2026-11-21.

Two new crates at the project version: `vulkane-kernel-text` (the shared text walker: scope checks, expressions through `lower_dag`, statements and structured control flow, the by-name patent refusals, a `Dialect` trait for the per-language spellings) and `vulkane-glsl-emit` (`Glsl::lower_kernel`, GLSL 4.50: `std430` buffers, `shared` arrays, `memoryBarrierShared(); memoryBarrierBuffer(); barrier();`, `groupMemoryBarrier()`/`memoryBarrier()` fences with every `MemoryOrder` collapsed to the one full fence, the entry is `main`). `Wgsl::lower_kernel` now lowers through the walker too (`var<storage>`, `var<workgroup>`, `storageBarrier(); workgroupBarrier();`; a `Fence`, a subgroup barrier and a device barrier are typed declines), and its duplicate decline module is gone. v1 scope for both: integer kernels, flat buffers, static shared arrays, `Let`/`Load`/`Assign`/`Store`, `Barrier`, `Fence`, `If`/`For`/`While`/`Break`/`Continue`/`Return`, comparisons, bit ops, masked shifts, `Select`; everything else, floats, dynamic extents, division and atomics included, is a typed decline naming it. Verified on the device (Fuel's gpu-run mutex): GLSL (glslangValidator + spirv-val) and WGSL (naga) each reverse a 256-element block exactly through shared memory (without the barrier 128 of 256 cells differ on both, reported, not asserted) and match a host reference on the nested break/continue/early-return kernel. **Measured against the ROADMAP's assumption:** naga 30.0 accepts a `workgroupBarrier()` under thread-dependent control flow (Tint would not), so the guard is `check_kernel`'s `BarrierDivergent`, tested. CI: the pinned-dxc Lavapipe job now also installs `glslang-tools` and `spirv-tools` and runs the three kernel emitters' tests and the SPIR-V kernel device tests with `VULKANE_REQUIRE_DEVICE`, `VULKANE_REQUIRE_GLSLANG` and `VULKANE_REQUIRE_SPIRV_VAL` set.

`Hlsl::lower_kernel` (it was only the patent-decline stub) now lowers through `fishstringer-slang-emit`'s kernel lowering, whose Slang dialect is HLSL, with the header relabelled (reuse over a third walker; `fishstringer-slang-emit` is `0.4.3`). Its coverage is Slang's: buffer *views* (base and strides in a `ConstantBuffer`, `int64_t`, so `shaderInt64`), `groupshared` arrays, `GroupMemoryBarrierWithGroupSync()`, control flow, atomics; no bitwise `BinaryOp` in a kernel expression yet. Verified: `dxc -T cs_6_2 -spirv` compiles the block-reverse kernel and `spirv-val` accepts it; on the device it reverses a 256-element block exactly through `groupshared` memory (the barrier-less control differed in 0 of 256 cells there, so the race is not shown by this device for this form; reported, not asserted); the patent declines still come first. The shared `Gpu` test helper gained `open_with` (device features) and `run_params` (a uniform buffer after the output).

`vulkane/vk.xml` is Khronos Vulkan-Docs tag `v1.4.368` (header 368; fetched 2026-11-19 from `raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/v1.4.368/xml/vk.xml`, copyright header unchanged), so the generated bindings and safe wrappers cover the new extensions (among them `VK_EXT_shader_ocp_microscaling_types`). Two things the new spec required: `VkMetalSurfaceCreateInfoEXT::pLayer` became non-const upstream, so `Surface::from_metal` casts with `cast_mut().cast()` (same representation, constness only); and `vulkan_gen` no longer emits a by-value `-> c_void` safe method for a mutable `void*` parameter (the new `vkGetLatencyTimingsLegacyNV` made rustc warn, and clippy `-D warnings` would have failed), which `vulkan_gen` 0.5.2 (unpublished) carries. Verified: `cargo build -p vulkane` with no warnings, clippy `-D warnings` on `vulkane` and `vulkan_gen`, and the `vulkane`, `vulkan_gen` and `kiss-vulkan-vocab` test suites, the `vulkane` integration tests under the GPU mutex. Not changed: the vocabulary (`kiss-vulkan-vocab`) does not yet name the new microscaling types (separate row), and the doc comments that say "measured at header 348" keep that ref.

## [0.25.1] — `vulkane-spirv-emit`: `Backend::lower_kernel` first slice (workgroup memory, barriers, fences, structured control flow)

`SpirV::lower_kernel` lowers integer (`I32`/`U32`) kernel-layer kernels: buffer parameters (one storage buffer each, `Read` buffers `NonWritable`), static `Shared` arrays as `OpVariable Workgroup`, thread, block, `BlockDim` and `GridDim` ids, locals, `Let`/`Load`/`Assign`/`Store`, `Barrier { Block | Subgroup, Full }` as `OpControlBarrier` (semantics `AcquireRelease | WorkgroupMemory | UniformMemory`, the buffer class added to the ROADMAP's text because a CUDA `__syncthreads()` orders global writes too) and `Fence` as `OpMemoryBarrier` (`SeqCst` collapses to `AcquireRelease`). `If`, `For`, `While`, `Break`, `Continue` and `Return` lower to the structured SPIR-V block patterns (`OpSelectionMerge`, `OpLoopMerge`; a zero `For` step leaves the loop at once, since an invocation cannot trap). A dynamic shared extent (`Extent::Param`) is an `OpSpecConstant` (SpecId = the scalar parameter's index, default 1), so it is fixed per pipeline, not per dispatch. A barrier under thread-divergent control is refused by `check_kernel` (`BarrierDivergent`) and surfaces as a typed decline. Everything else (floats, atomics, device or named barriers, scalar parameters) is a typed `UnsupportedKernel` naming it; the by-name patent declines still come first. ROADMAP rows 8a, 8b and 8g are partly done (see their rows). Verified: `spirv-val --target-env vulkan1.1` on every emitted module (it caught a doubly decorated stride and block while this was written), a structural control (the module without the barrier has no `OpControlBarrier`), and a device test that reverses a 256-element block through shared memory exactly with the barrier; without it 128 of 256 cells differed on the test device (reported, not asserted: a missing barrier is a race and a device may still come out right). Row 8g adds a nested break/continue/early-return kernel checked per thread against a host reference on the device, and a seeded invalid-merge mutant (merge and continue swapped) that `spirv-val` rejects.

## [0.25.0] — **BREAKING**: vulkan vocabulary version 7 (`a64` `af32` `af64`, kiss-vulkan-vocab 0.6.0), repin to unpopped ~0.56.2 / fishstringer ~0.4.0, by-name patent declines

Three changes in one PR (one PR per day, PM 2026-11-16).

**1. Vocabulary 7.** Adds the atomic capability names `a64`, `af32` and `af64` to the `vulkan:` `<arith>` field (KISS#837, KISS 0.19.0): `vulkane::safe::PhysicalDevice::shader_atomic_features` / `ShaderAtomicFeatures` read `VkPhysicalDeviceShaderAtomicInt64Features` and `VK_EXT_shader_atomic_float` (chained only when the device lists the extension; a failed extension enumeration declines), and `vulkane::kiss::DeviceCapabilities` derives the names from them. Breaking for tokens: a device with 64-bit atomics now spells `a64`. `kiss-vulkan-vocab` moves to 0.6.0 on its own version line (CHANGELOG section above, with the migration); the manifest changes in three lines. ROADMAP row 8d-cap.

**2. Repin.** `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-convert` `~0.47.0` -> `~0.56.2` and `fishstringer-*` `~0.2.0` -> `~0.4.0` (0.4.0 requires `unpopped ^0.56.1`; the `Backend` trait must be the same one on both sides), across the emit/parse crates and `vulkane`'s dev-dependency on `fishstringer-kernels-source`. Source compiled unchanged.

**3. Patent declines by name, twelve statements (board 434 C and its extension; NVIDIA US11294713B2, US12118382B2, US11803380B2, US12204897B2).** `vulkane-spirv-emit`, `vulkane-hlsl-emit` and `vulkane-wgsl-emit` override `Backend::lower_kernel` and refuse `CopyAsync`, `AsyncCommit`, `AsyncWait`, `Collective` and `CollectiveTuple` (board 434 C) and `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk`, `Tcgen05Commit`, `WgmmaWait` (extension) by name with `LowerError::UnsupportedKernel`, at the top level and at any `If`/`For`/`While` depth, via an exhaustive `Stmt` match (no wildcard: a new statement fails to compile until classified). Every other kernel gets the same typed refusal as before. `tests/patent_decline.rs` in each crate pins all twelve names at six nesting shapes with a control; a mutation that drops one name fails it; `CollectiveTuple` is the tuple form of `Collective`: 434 C read as covering it; not named in the ruling's list. ROADMAP row 8l; the READMEs carry the builder warning.

**4. Dependency-security gate.** Adds `deny.toml` and a `cargo deny` CI job (advisories, licences, bans, sources; FishStringer's template). The one advisory is RUSTSEC-2026-0192 (`ttf-parser`, unmaintained, dev-only via `winit`), ignored with its reason and a 30-day expiry (2026-12-18, PM ruling 2026-11-17). ROADMAP row 10.

## [0.24.2] — docs: document the vulkan_gen and vulkane::raw::version public items (ROADMAP row 9a)

Documents the 72 undocumented public items of `vulkan_gen` (generator `new` functions, the `generate_*` entry points and their stats structs, logging, the `vk_types` data model and its aliases and JSON wrappers, the two top-level modules) and the 3 `vulkane::raw::version::Version` items coderipper MOD-001 listed. Doc comments only, plus one `#![allow(missing_docs)]` with a reason on `parser/vk_types.rs`; no behaviour or API change. The `vulkan_gen` public surface is kept by PM ruling (a narrowing would be breaking); the measurement (384 / 282 fields / 29 variants / 72 items) is in ROADMAP row 9a. Verified: `cargo doc -p vulkan_gen --no-deps` with `-D warnings`; the missing_docs lint now reports 312 (fields, variants, crate root) against 384.

## [0.24.1] — docs: decline mbarrier and bulk-copy statements by name (patent, board 434 C extended)

The three emitter READMEs and ROADMAP row 8l record CireSnave's 2026-11-16 ruling on NVIDIA US11803380B2: `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` and `Tcgen05Commit` are declined for these non-NVIDIA targets, and so is `WgmmaWait` (NVIDIA US12204897B2, CireSnave's second 2026-11-16 ruling). Docs only: no code or version-bearing file changes; the by-name refusal and its tests wait on an `unpopped` release that has the variants (0.47.0 has none).

## [0.24.0] — **BREAKING**: repin unpopped to ~0.47.0 (sk8) and fishstringer to ~0.2.0; `unpopped::convert` is now the `unpopped-convert` crate

`unpopped`, `unpopped-vocab`, `unpopped-cpu-c` 0.43.2 -> ~0.47.0 and `fishstringer-slang-emit`/`-slang-parse`/`-kernels-source` -> ~0.2.0 across the emitter crates. `unpopped`s `convert` feature moved to the `unpopped-convert` crate, so `vulkane-hlsl-emit` and `vulkane-hlsl-parse` depend on it and import `unpopped_convert::`. Verified: `cargo check --workspace --all-targets --all-features`; tests under gpu-run (see PR).

## [0.23.0] — **BREAKING**: `vulkan:` vocabulary version 6, the `x<n>` escape removed (KISS 0.18.0, V-7)

`VOCABULARY_VERSION` 5 -> 6. A component type the vocabulary does not name is never spelled: every `cm-`/`cv-` tuple holding one is omitted whole (`CoopMatrix::from_shapes` / `CoopVector::from_combos` drop it; `canonical_enumeration` filters a hand-built value too), so the token under-claims. `VulkanTarget::parse` declines `x<n>` as unknown. `ComponentType::spelling()` now returns `Option<String>` (`None` for `Other(n)`); new `is_named()` on `ComponentType`, `CoopShape`, `CoopVecCombo`; new `kiss_vulkan_vocab::unnamed_types(shapes, combos)` yields the sorted `vulkan:component_type:<n>` identifiers for the Announce `unnamed_types` record (KISS-ANNOUNCE-6.6-0011), and `vulkane::kiss::DeviceCapabilities` gains the matching `unnamed_types` field. Digests (`cm-`/`cv-fnv1a64-`) and the §6.8-0008 manifest move only for inputs that held an unnamed type or whose digest ran over one; a device whose types are all named spells what it spelled at version 5. The manifest drops `unnamed_component_escape` and `_order` (replaced by `unnamed_component_type`) and gains three `unnamed_omitted_*` vectors; the manifest's narrow/wide threshold families now vary over named types. Verified: `cargo test -p kiss-vulkan-vocab` (all green, committed manifest byte-identical to a fresh emission), `cargo check --workspace --all-targets`.

## [0.22.10] — Fixed the `parser::tree_parser` <-> `vulkan_spec_parser` module cycle in `vulkan_gen` (ROADMAP row 9a)

`VulkanSpecification` moves to `parser::vk_types`, which both modules already read; `vulkan_spec_parser` re-exports it, so `vulkan_gen::VulkanSpecification` and `vulkan_gen::vulkan_spec_parser::VulkanSpecification` are unchanged. Removes the edge `tree_parser.rs:8` that closed the cycle. Verified: `cargo clippy -D warnings`, `cargo test -p vulkan_gen` (68 + 3 pass), docs with `-D warnings`. The rest of row 9a (vulkan_gen documentation and visibility) is a separate PR.

## [0.22.9] — Added `[architecture]` declaration in `.coderipper.toml` (ROADMAP row 9c, docs/config only)

Declares the 13 workspace crates as components in two layers (12 leaf crates below, `vulkane` on top). ARC-001 and ARC-002 now reach a verdict instead of "not run": 3 edges judged, 0 findings, 0 cycles (coderipper 0.43.x, `--profile extended`, at this branch). ARC-003/004/006/007/012 stay "not run": they need policy/volatile/composition/budget declarations that this PR does not invent; they follow with row 9 (module groups inside `vulkane`, where the crate::safe cycle lives). No code changed.

## [0.22.8] — docs: ROADMAP rows 9-9c from the ARC/MOD census

Schedules the vulkane findings of the portfolio ARC/MOD census (coderipper 0.43.54, `493cc106`): the two non-Info findings (MOD-001 on `vulkan_gen`, MOD-002 on the `crate::safe` cycle and the `tree_parser` pair), the Info findings, and the `[architecture]` declaration. Docs only; no code change.

## [0.22.7] — tooling: CHANGELOG fragments and version stamping

Adopts the portfolio fragment-and-stamp kit (PM 2026-11-12). A PR now adds one `changelog.d/<topic>.md` fragment and never edits a `Cargo.toml` version, `Cargo.lock` or `CHANGELOG.md`; the gate runs `python tools/stamp-version.py <version>`. Added `tools/stamp-version.py`, `tools/check-fragment.py`, `tools/stamp-current.txt` (`vulkane/Cargo.toml`, so the ten crates at the project version are stamped and `kiss-vulkan-vocab`, `vulkan_gen` and `vulkane_derive` keep their own versions), `.github/workflows/changelog-fragment.yml` and `changelog.d/README.md`. Verified on a scratch copy: a real stamp moved all nine crates then at the project version (ten with wgsl-emit, dry-run verified) and their `Cargo.lock` entries and left no old version in any tracked `Cargo.toml`. Version number allocated by the PM at gate time.

## [0.22.6] - unreleased

### Added - `vulkane-wgsl-emit` (ROADMAP 3d step 3)

The WGSL compute emitter for Unpopped: `Wgsl` lowers the scalar contiguous elementwise path at `f32`/`i32`/`u32` to a `@compute @workgroup_size(256, 1, 1)` module with `var<storage>` operands, through `lower_dag` with WGSL's own spellers (no ternary, `select`, `u32` shift counts). Dtypes and ops WGSL cannot express exactly are typed declines naming why, never approximations. Verified: every emitted module validates in naga and lowers to SPIR-V (9 tests, incl. a negative control); on a device, 4182 in-domain cells over 61 kernels agree with `unpopped-cpu-c`, and a seeded Relu mutant fails that oracle. One WGSL compiler only (naga). `Rem`/`RemTrunc` rows where the quotient is within a few ULP of an integer are reported, not failed: WGSL bounds `/` at 2.5 ULP and the test device returned `1.5 % 1.5 = 1.5`. Unpublished. Not measured: the cause of the `x % x` result (the float-controls explanation is a hypothesis), Tint, a second device.

## [0.22.4] - unreleased

### Documentation - ROADMAP rows 8-8k

Rows for comprehensive kernel emission (`Backend::lower_kernel`) on the SPIR-V, GLSL, HLSL and WGSL emitters, one per construct family, each with owner, size, the typed decline where a target cannot express it, and the test that will show it. Docs only; no code. Version number is a placeholder for the PM to allocate (0.22.3 is the rustls bump, #103).

## [0.22.3] - unreleased

### Changed - dependencies

`rustls` 0.23.37 to 0.23.46 in `Cargo.lock` (Dependabot #103; CireSnave's latest-versions rule). No source change. Workspace version 0.22.3 allocated by this lane at the PM's instruction; the PM confirms the number at gate.

## [0.22.2] - unreleased

PM-allocated (2026-11-11): `vulkane-wgsl-parse` (#150) merged without a bump at 0.21.9 and is brought to the workspace number here, so every crate moves to 0.22.2.

### Added - `vulkane-spirv-emit` spells integer `Div`/`Rem` and opts in to Unpopped's total integer rule (Unpopped ROADMAP E16)

`SpirV` now returns `true` from `Backend::total_integer_ops`, so Unpopped's plan gate admits integer `Div`, `ConstInt`, `Select` and the six comparisons at `i32`/`u32` for it. Division follows `docs/int-div-shift-semantics.md` rules 1 and 2: the divisor is `select(bad, 1, b)` where `bad` is `b == 0` (and, signed, `a == MIN && b == -1`), so a zero divisor and `MIN / -1` give the dividend (SPIR-V 3.3.13 leaves them undefined but does not trap, so an unconditional select is allowed, unlike C). Integer `Rem` is Unpopped's `Select(b == 0, 0, a - (a / b) * b)` composition over it. Comparisons give `1`/`0` in the compute dtype (`OpSLessThan` family at `i32`, `OpULessThan` family at `u32`); `Select` tests `c != 0`; a float `Const` in an integer kernel is now a typed decline (the `assert_no_int_div_or_const` panic backstop is gone with the gate it backstopped). Shifts were already masked in 0.20.2.

Checked three ways. `integer_div_divides_by_one_on_a_zero_divisor_and_on_min_over_minus_one` pins the guard's structure; two seeded mutants each turn it red (drop the guard; drop only the signed `MIN / -1` arm). `tests/int_div_live.rs` runs the emitted `Div` and the `Rem` composition on a device over the document's vectors (9 dividends x 8 divisors x 2 dtypes x 2 kernels = 288 results, all equal to the host's rule on an NVIDIA device). That live test could not tell a guarded kernel from an unguarded one if the device does not misbehave on the unguarded forms (the same limit as `int_shift_live.rs`), so it guards the device path, not the guard. Not done: 64-bit integers (the backend declines them as a dtype), and the same vectors on a second implementation (Lavapipe on a Linux job). Version allocated by the PM.

## [0.22.0] - unreleased

### Added - redirect crates `vulkane-slang-emit` / `vulkane-slang-parse` (ROADMAP 6)

Thin `pub use fishstringer_slang_*::*` crates (deprecated, `fishstringer-slang-*` 0.0.17, `convert` feature forwarded on emit) so a dependent migrates by renaming. The 0.21.1 emit tests (3 files, 12 tests) were run unchanged against the redirect and pass; unpublished, the PM publishes. Version stays at the workspace number for the PM to allocate.


### Removed (BREAKING) - the Slang crates and kernels (Slang switch, step 2 of 2)

FishStringer owns Slang (CireSnave 2026-11-06) and has published `fishstringer-slang-emit`, `fishstringer-slang-parse` and `fishstringer-kernels-source` (0.0.14; re-read on crates.io 2026-11-10, unyanked). **Removed from this workspace:** `vulkane-slang-emit`, `vulkane-slang-parse` (both ARE published on crates.io, through 0.21.1, and those releases stay; a redirect release is arranged separately), the 147 `.slang` kernels and `hoist_extinst_imports.py` with its test from `vulkane-kernels-source`, the `slang-emit` and `kernels-scripts` CI jobs, and the Slang branch of `compile.sh`. `vulkane-kernels-source` keeps the 27 `.glsl` kernels and a GLSL/WGSL `compile.sh`; its `THIRD-PARTY-NOTICES.md` is rewritten (the llama.cpp, webgpu-blas and candle dispositions went with the Slang files, and FishStringer's notices carry the licence texts). The dependents' pin moves `~0.0.12` to `~0.0.14`. Diffs of the published packages: 0.0.12 to 0.0.13 is only `kernels/no_contraction.py` (new `--out-dir` handling, plus its test) and manifests; 0.0.13 to 0.0.14 is only the new `kernels/spv_equiv.py` with its tests, README and `lib.rs` text, and manifests. No `.slang` file or emitter source changed in either step. **Breaking, hence 0.22.0 (pre-1.0 the second number is the major): `vulkane-kernels-source` 0.21.1 is on crates.io with the `.slang` files, and this release no longer ships them. Anyone who took them from `vulkane-kernels-source` must take them from `fishstringer-kernels-source`.**

## [0.21.9] - unreleased

### Changed - dependents move to FishStringer's published Slang crates (Slang switch, step 1 of 2)

FishStringer owns Slang (CireSnave 2026-11-06) and published `fishstringer-slang-emit`, `fishstringer-slang-parse` and `fishstringer-kernels-source` 0.0.12 (re-read from crates.io 2026-11-07: unyanked, MIT OR Apache-2.0). `vulkane-hlsl-emit` and `vulkane-spirv-emit` (dev) now depend on `fishstringer-slang-emit = "~0.0.12"` (tilde: hlsl-emit asserts the emitter's output shape), and `softmax_f64_live` reads `softmax_f64.slang` from `fishstringer-kernels-source` (dev-dependency, located with `cargo metadata`). Checked against the published crates on a device: HLSL oracle, SPIR-V oracle and Slang-vs-SPIR-V cross-check each 4324 in-domain cells over 61 kernels agree; `softmax_f64_live` 21576 outputs, worst relative error 3.2e-15. Nothing is deleted here: the in-repo Slang crates and `.slang` files go in step 2, after this is green.

## [0.21.8] - unreleased

### Docs - WGSL parser measurement (ROADMAP 3d step 1)

`docs/design/wgsl-parser-emitter.md` W1 decided by measurement over the 5 graphics examples plus 3 new compute fixtures (`docs/design/wgsl-measurement/`): `wgsl-parse` 0.6.0 parses 8/8; `tree-sitter-wgsl-bevy` 0.1.4 fails the 2 compute fixtures (no `workgroupBarrier();`, `>>=`, `requires`). No library change.

## [0.21.7] - unreleased

### Added - HLSL GPU oracle (ROADMAP 3c, H4)

`vulkane-hlsl-emit/tests/oracle_live.rs`: the same cases, tolerances and domains as the SPIR-V emitter's G4.2 oracle, with the HLSL compiled by `dxc` and run on a live device against `unpopped-cpu-c`. Under `gpu-run.ps1`: 4324 in-domain cells over 61 kernels agree. `f64` is not run (the shared test device does not request `shaderFloat64`). Test-only; no library change. CI job `hlsl-oracle` runs it on Lavapipe with a pinned `dxc` (v1.9.2609, sha256-checked) and `VULKANE_REQUIRE_DXC`/`VULKANE_REQUIRE_DEVICE` set, so a missing dxc or device is a failure; a control step proves the oracle goes red without dxc. The test also asserts it compared at least one cell.

## [0.21.6] - unreleased

### Added - `vulkane-hlsl-emit` (ROADMAP 3c step 3)

New crate: `Hlsl`, an Unpopped `Backend` for the scalar elementwise path, built as a thin layer over `vulkane-slang-emit` (the Slang text for that path is already HLSL): it changes the entry prefix (`unpopped_hlsl_`), the header and the compiler-argument line (`required_dxc_args`: `-denorm preserve` for f32), and the tests assert the bodies are byte-identical to the Slang emitter's. No `register(..)` clauses, because under `dxc -spirv` `t0`/`t1`/`u0` collide on binding 0. Checked by `dxc -T cs_6_2 -spirv` over f32/f64/i32/u32/i64 (with a rejected-file control; needs `dxc`, skips without it unless `VULKANE_REQUIRE_DXC` is set, so CI does not run it yet) and by re-lifting through the HLSL frontend. Not yet published. Not yet done: the G4.2-style GPU oracle against `unpopped-cpu-c`.

## [0.21.5] - unreleased

### Added - `vulkane-hlsl-parse` (ROADMAP 3c step 2)

New crate: `parse_hlsl` (tree-sitter-hlsl 0.2), `HLSL_RESIDUE` and the `HLSL` `Frontend`, so Unpopped's `lift_*` walkers read HLSL kernels. Five hand-written pure-HLSL fixtures (each compiles with dxc) cover `register`, `cbuffer`, `groupshared`, semantics and `tbuffer`. Decision: the grammar gaps are tolerated, not patched. New measurement: `tbuffer N : register(tK)` has the same `MISSING ;` gap as `cbuffer`. Both gaps are pinned by tests so an upstream fix is noticed. Not yet published.

### Changed - HLSL design note records the H1 measurement (docs only)

`docs/design/hlsl-parser-emitter.md` now holds the `tree-sitter-hlsl` 0.2.0 result over the DXC-accepted kernels: 131 of 131 parse clean once the Slang-only `[shader("compute")]` line is removed (0 of 131 as they are), and `cbuffer N : register(..) { .. }` reports `MISSING ;`, a grammar gap. No code changes. Every crate moves to 0.21.5 with the series.

## [0.21.4] - unreleased

### Added - `vulkane-slang-emit` spells the `u32` bitwise and shift ops

`BitAnd`, `BitOr`, `BitXor`, `Shl` and `Shr` at `ElementKind::U32` were typed declines (`slang_binary` routed only `I32`/`I64` to the integer speller, the gap the 0.21.3 cross-check found); they now lower through it, so a `u32` shift count is masked to 5 bits (Unpopped integer rule 3) and `>>` on `uint` is logical. Not breaking: a typed decline becomes a spelling. The cross-check in `vulkane-spirv-emit/tests/oracle_live.rs` now compares all 61 cases (4324 in-domain cells agree, Slang against SPIR-V and against the host C) and fails on any Slang decline. Every crate moves to 0.21.4 with the series.

## [0.21.3] - unreleased

### Added - `vulkane-spirv-emit` behaviour oracle against `unpopped-cpu-c` (G4.2)

`tests/oracle_live.rs` lowers the same `OpDef` with `unpopped-cpu-c` (C, compiled and run on the host) and with the SPIR-V emitter (run on a live device) and compares the result bits: 4324 in-domain `f32`/`i32`/`u32` cells over 61 kernels agree (integer ops, comparisons, min/max and rounding bit-exact; transcendentals within a stated tolerance). `Pow` (x<=0), `Atan2` at (0,0) and with a `-0` y, and the sign of a zero `RemTrunc` result are unspecified by SPIR-V and are reported, not failed (33 cells here). The same file holds the Slang-vs-SPIR-V cross-check: the cases run through `vulkane-slang-emit` (compiled by `slangc`) and the SPIR-V emitter on the device; 3919 in-domain cells over 56 kernels agree, Slang against SPIR-V and Slang against the host C. Finding: the Slang emitter declines every `u32` bitwise and shift op (`BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr`; its integer speller covers only `i32`/`i64`), which the SPIR-V emitter spells; the test pins that decline set. Test-only: no library code changes. Every crate moves to 0.21.3 with the series.

## [0.21.2] - unreleased

### Fixed - `vulkane-slang-emit` masks the `Shl`/`Shr` count (Unpopped integer rule 3)

`Shl` and `Shr` on `i32`/`i64` were spelled with the raw count (`a << b`). Once Slang lowers to SPIR-V a count of the operand width or more is poison (SPIR-V 3.3.14), so the result for such counts, and for negative counts, was whatever the device chose. They now spell `a << (b & 31)` for `i32` and `a << (b & 63)` for `i64`, as `docs/int-div-shift-semantics.md` rule 3 requires (count bits read as unsigned, modulo the width). Checked by `int_shift_count_is_masked_to_the_width`, which pins the exact spelling at both widths with `BitAnd` beside it unchanged as the control. Slang integer `Div`/`Rem` were already declined, so the plan gate for them stays shut. This mirrors the `vulkane-spirv-emit` fix in 0.20.2. Every crate moves to 0.21.2 with the series.

## [0.21.1] - 2026-11-05

### Fixed - docs and package metadata

The seven cited dependency lines (README x4, `lib.rs`, `safe/naga.rs`, `safe/shaderc.rs`) cite `vulkane = "0.21"`, which crates.io now serves (the `cited_versions` gate). `vulkane-kernels-source` declares `rust-version = "1.85"` (edition 2024's own floor; it has no dependencies and compiles there, and the minimality gate rejected 1.88 because it also compiles on 1.87). Every crate moves to 0.21.1 with the series.

## [vulkane_derive 0.1.3] - 2026-11-05

`vulkane_derive` ships a README (`readme = "README.md"`). It keeps its own number and moves 0.1.2 -> 0.1.3, because a README changes the package contents of a version crates.io already serves (the version-collision gate).

## [0.21.0] - 2026-11-05

### Changed (Breaking) - `unpopped` 0.43.2

`vulkane-slang-emit`, `vulkane-slang-parse` and `vulkane-spirv-emit` move from `unpopped`/`unpopped-vocab`/`unpopped-cpu-c` 0.40.0 to 0.43.2 (public dependencies, so Breaking; unblocks Baracuda's move to 0.43 with one vocab copy). Our code needed no change: no exhaustive `ReduceOp` or `Stmt` match (`ReduceOp` is now `#[non_exhaustive]`; `Stmt::ExternCall`, `AtomicCas`, `Fence` and `Call` are refused typed by Unpopped until an emitter implements them). Consumers building `Stmt::Barrier { scope }` must use `Stmt::barrier(scope)`; our `lower_kernel_default_is_the_typed_decline` test does, and its `CpuC` refusal cell now uses a barrier without a launch, since 0.43 `CpuC` lowers a barrier inside a launch. Every crate moves to 0.21.0 with the series.

## [0.20.2] - unreleased

### Fixed - `vulkane-spirv-emit` masks the `Shl`/`Shr` count (Unpopped integer rule 3)

`Shl` and `Shr` on `i32`/`u32` emitted the shift with the raw count. SPIR-V makes a count of 32 or more poison (3.3.14), so the result for such counts, and for negative counts, was whatever the device chose. They now emit `count & 31` first, as `docs/int-div-shift-semantics.md` rule 3 requires (count bits read as unsigned, modulo 32; `<<` defined for any `a`; `>>` arithmetic on `i32`, logical on `u32`). Checked two ways: `shift_counts_are_masked_to_five_bits_before_the_shift` (the shift's count operand is an `OpBitwiseAnd` with the constant 31; killed by a mask of 30 and by dropping the mask) and `tests/int_shift_live.rs`, which runs the emitted kernels on a device over 9 values x 12 counts x 4 kernels (432 results, all equal to Rust's `wrapping_shl`/`wrapping_shr` on an NVIDIA device). The live test could NOT tell a masked kernel from an unmasked one on that device (dropping the mask still passed 432/432: the hardware masks), so it guards the device path on other implementations, not the mask itself.
Not done: integer `Div`. Unpopped's `Backend::lower` still panics on integer `Div` before the emitter sees it (the plan gate opens when the emitters adopt the guards, Unpopped ROADMAP E16), so the guarded `Div`/`Rem` spelling cannot be reached or tested here yet. `vulkane-spirv-emit` gains `vulkane` as a dev-dependency for the live test. Every crate moves to 0.20.2 with the series.

## [0.20.1] - unreleased

### Added - NUL-byte gate (`.github/nul_gate.py`, CI job `nul`)

Fails when any tracked text file contains a NUL byte (invisible in review, makes git treat the file as binary). Enumerates from `git ls-files`, prints the population (405 tracked, 395 scanned as text, 10 binary by extension `.spv`/`.jpeg`, 0 with NUL at main c759fb6), refuses an empty population and a stale binary-extension entry. `--self-test` has five controls over throwaway git trees (clean passes; NUL in `.rs` fails; UTF-16 `.md` fails; stale entry fails; too few text files fails); a NUL appended to the real `README.md` made the gate exit 1 naming the file. Added to `.github/local_gates.py` as two RUN rows. Every crate moves to 0.20.1 with the series.

## [0.20.0] - unreleased

### Changed (Breaking) - `unpopped` 0.40.0

`vulkane-slang-emit`, `vulkane-slang-parse` and `vulkane-spirv-emit` move from `unpopped`/`unpopped-vocab`/`unpopped-cpu-c` 0.38.0 to **0.40.0**. Breaking because `unpopped` types are part of these crates' public API. 0.40.0 adds `Stmt::AtomicRmw`; no Vulkane `src/` names `Stmt`, and the one test that does (`lower_kernel_default_is_the_typed_decline.rs`) only constructs `Stmt::Barrier`, so no new match arm is needed. Built `cargo build --workspace` (and `--locked`) against 0.40.0; tests of `vulkane-slang-emit`, `vulkane-slang-parse`, `vulkane-spirv-emit`, `vulkane-spirv-parse` pass (CPU-side; the GPU suite runs in CI). Every crate moves to 0.20.0 with the series.

## [0.19.5] - unreleased

### Added - ROADMAP row 1e: issue #130 bisect result (docs only)

A plain C program with no Vulkane code reproduces the Linux 26.04 SIGSEGV: 8 threads sharing one `libvulkan.so.1`, each looping create instance, enumerate, create device, destroy, crash (25 of 40 runs) only when each iteration does both `vkGetInstanceProcAddr(instance, name)` lookups and `vkGetDeviceProcAddr(device, name)` over 626 names; either alone, or neither, gives 0 of 40. The same program against Vulkan-Loader main (`aa6dee83`, built in CI) gave 0 of 40. The CI pin to ubuntu-24.04 (0.19.1) stays. ROADMAP row 1c now records #132 as merged. No code change; every crate moves to 0.19.5 with the series.

## [0.19.4] - unreleased

### Fixed - `softmax_f64.slang` and `unary_f64.slang` `exp` returned garbage for results below 2^-1022 (issue #106, ROADMAP 1c)

The `GLSL.std.450 Exp` validation error in #106 was already fixed by #119 (`poly_exp_d`). Running the kernel for the first time against a CPU f64 softmax (new `vulkane/tests/softmax_f64_live.rs`) found a second defect in that `poly_exp_d`: for `x` in about `[-745, -708.4]` the range reduction gives `k < -1022`, `pow2_int(k)` builds its exponent field from `uint(k + 1023)`, which wraps, and the "power of two" is garbage. In softmax one such element (an input 709 to 745 below the row max) made the row sum negative, so every output of that row was `-0`. Measured before the fix: 2758 of 3001 sweep points wrong, all `-0`. Both kernels now scale in two exact power-of-two steps when `k < -1021`; the result is subnormal there and rounds once. `unary_f64`'s `exp`, `sigmoid`, `silu` and `tanh` share the function, so they had the same defect for large-magnitude arguments.

After: the 3001-point sweep over `x - max` in `[-750, 0]` and 21,576 softmax outputs (rows of 1 to 4097 columns, including a spread wide enough to underflow) agree with the CPU f64 reference within 1e-13 relative; worst measured 3.2e-15 on an NVIDIA device (Windows). Both modules pass `spirv-val --target-env vulkan1.1`. A third defect surfaced on CI's Lavapipe (subgroup size 8, the first non-NVIDIA/AMD device to run it): `softmax_f64` combined the per-subgroup partials with `WaveActiveSum`/`WaveActiveMax` inside subgroup 0, which sees only that subgroup's lanes, so it needed at least 256/size partials to fit in one subgroup (size >= 16). With size 8 it summed 8 of 32 partials; outputs were off by about 1e-7 relative to a row summing to 5.01. A probe shader on Lavapipe showed the wave reductions themselves are right on doubles and the two-level combine covered 64 of 256 elements. `softmax_f64.slang` now combines the partials serially on thread 0 (and its shared arrays are sized for subgroup size 4). The same pattern is in 16 other kernels (`softmax`, `softmax_{f16,bf16}`, `reduce_last_dim*`, `layer_norm_last_dim_backward*`, `rms_norm_last_dim_backward`, `softmax_last_dim_backward*`): not changed here, not run on a small-subgroup device; tracked in an issue.

Not run on a device: `unary_f64` itself (same edit, same function; only validated). CI runs the new test on Lavapipe with a pinned slangc and both `VULKANE_REQUIRE_*` variables set.

## [0.19.3] - unreleased

### Added - upstream contribution for tree-sitter-glsl, prepared not posted (ROADMAP 3b, docs only)

`docs/design/upstream-glsl/` holds a verified patch for `tree-sitter-grammars/tree-sitter-glsl` (qualifier-only declarations such as `layout(local_size_x = 64) in;`: grammar change plus a corpus test that fails before and passes after), its PR body, and a separate issue body for the unfixed `(float(1) - 1.0)` gap. Run with `tree-sitter` CLI 0.24.7 and `tree-sitter-c` 0.24.2, whose regeneration of unmodified upstream is byte-identical to its committed `src/`. The 9.7 MB regenerated `parser.c` is not stored; the README gives the steps and expected SHA-256 of each generated file. No code change; every crate moves to 0.19.3 with the series.

## [0.19.2] - unreleased

### Changed - GLSL design note: measured patch results replace the sketches (ROADMAP 3b, docs only)

`docs/design/glsl-parser-emitter.md` replaces the untested patch sketches and a wrong cause ("C `cast_expression`") with results run on `tree-sitter` CLI 0.27.1: patch 1 (a qualifier-only `declaration`, for `layout(local_size_*) in;`) takes the corpus from 0 to 26 of 29 clean; gap 2 (`(float(1) - 1.0)`) has no working patch (the tried `call_expression` alternative fixes 3 files and regresses 5). No code change; every crate moves to 0.19.2 with the series.

## [0.19.1] - unreleased

### Fixed - CI: Linux build & test pinned to `ubuntu-24.04` (no code change)

`ubuntu-latest` moved to 26.04 (Mesa 26.0.8, from 25.2.8 on 24.04). On it the live-device tests die with SIGSEGV intermittently, in a different test each run (`generator_live_device_test`, `extension_pnext_test`), on a docs-only diff; the backtrace is inside the Vulkan loader's `vkGetDeviceProcAddr` (issue #130), and it is not established whether the loader or Vulkane's use of it is at fault. The job is pinned to 24.04 while keeping its required-check name `Build & test (ubuntu-latest)`. Every crate moves to 0.19.1 with the series.

## [0.19.0] - unreleased

### Changed (Breaking) - `unpopped` 0.38.0 (ROADMAP 3a, one dependency move)

`vulkane-slang-emit` and `vulkane-slang-parse` move from `unpopped`/`unpopped-vocab`/`unpopped-cpu-c` 0.36.0, and `vulkane-spirv-emit` from 0.37.1, to **0.38.0**. Breaking because `unpopped` types (`Backend`, `GeneratedKernel`, `Frontend`) are part of these crates' public API. `unpopped` 0.38.0 keeps `GeneratedKernel::with_words`, which `vulkane-spirv-emit` returns; `ArtifactKind::SpirV` belongs to the `Compiler` path and does not apply to a `Backend`.

`vulkane-slang-emit`: the round-trip test imports the Slang frontend from `vulkane_slang_parse::SLANG` (new path dev-dependency; `unpopped::convert::SLANG` left `unpopped` in 0.37.0). No change to emitted Slang.

Crates whose code or dependencies changed: `vulkane-slang-emit`, `vulkane-slang-parse`, `vulkane-spirv-emit`. `vulkane-spirv-parse`, `vulkane-kernels-source` and `vulkane` are unchanged and move to 0.19.0 with the series. MSRV unchanged (`unpopped` 0.38.0 declares 1.85).

## [0.18.4] - unreleased

### Added - `vulkane-spirv-emit` (ROADMAP 3a step 4)

New crate: the `SpirV` `unpopped::Backend`. Lowers the scalar elementwise path (f32, f64, i32, u32; single output, no runtime params) to a binary SPIR-V 1.3 module for Vulkan 1.1, returned as `GeneratedKernel::words()` (needs `unpopped` 0.37.1) with the disassembly as `source`. Body math runs through `lower_dag`; the spellers emit `rspirv` instructions and the `temp` seam maps hoisted names to result ids. Declares `DenormPreserve` for float kernels. Same declined set as `unpopped-slang`, plus the transcendentals at f64 (GLSL.std.450 does not define them at 64 bits). Every op is validated with `spirv-val --target-env vulkan1.1` (fatal in CI); behaviour against `unpopped-cpu-c` on a device is a follow-up.

## [0.18.3] - unreleased

### Changed - GLSL design note (ROADMAP 3b, docs only)

`docs/design/glsl-parser-emitter.md` gains the G1 measurement of `tree-sitter-glsl` 0.2.0, `glsl-lang` 0.8.1 and naga `glsl-in` over the 29-file corpus, and the minimal repros plus patch sketches for the two `tree-sitter-glsl` grammar gaps. No code change; every crate moves to 0.18.3 with the series.

## [0.18.2] - unreleased

### Changed - `vulkane-slang-emit` slangc test (ROADMAP 2c, part)

`emitted_slang_compiles_with_slangc` writes its sources into a private `tempfile::tempdir()` (removed on drop) instead of the shared `std::env::temp_dir()`, which Codacy flags. New dev-dependency `tempfile` (workspace version). Measured: the test and the rest of the crate's tests pass with `VULKANE_REQUIRE_SLANGC=1` and `--features convert`; clippy clean. The complexity items of row 2c stay deferred to the emitter work (2a). Version allocated by the PM (all crates move to 0.18.2).

## [0.18.1] - unreleased

### Changed - `hoist_extinst_imports.py` (ROADMAP 1d step 2)

`hoist_imports` (Codacy CC 34, 115 lines) is split into five helpers; no behaviour change. Measured: output identical to the previous function on all 178 modules of Fuel's `fuel-vulkan-kernels/spv` (disassembled with `spirv-dis`) and on 7 synthetic inputs including both error paths; note the corpus never triggers a hoist (0 of 178 changed text), so the synthetic cases carry that path. 2 tests added (alias prefix ordering, OpMemoryModel-only fallback); all 9 pass on the old function and the new. Version allocated by the PM (all crates move to 0.18.1).

## [0.18.0] - unreleased

### Changed (breaking) - `unpopped` 0.36.0

`vulkane-slang-emit` and `vulkane-slang-parse` now depend on `unpopped` / `unpopped-vocab` / `unpopped-cpu-c` 0.36.0 (unpopped-slang removed upstream; public dependency versions change, hence the minor bump). No source change was needed. `tree-sitter` stays 0.27 with one copy in the tree (`cargo tree -i tree-sitter`: `unpopped` 0.36.0 and `vulkane-slang-parse`). Measured: emit and parse tests pass with the `convert` feature (17 emit unit tests incl. the `slangc` test, 7 + 3 cells/parse tests, 8 parse unit, corpus and doctests); `rust-version` floors unchanged (emit 1.85, parse 1.90). Version allocated by the PM.

## [0.17.3] - unreleased

### Added - `vulkane-spirv-parse` spirv-val tests (ROADMAP 3a)

`tests/spirv_val.rs`: `spirv-val --target-env vulkan1.3` gives the same verdict on each of the 9 committed modules before and after parse/encode (with a guard that some are accepted), and a well-formed module with its capabilities removed still parses and re-encodes unchanged while `spirv-val` rejects it (control: the unmodified module passes). New CI job `vulkane-spirv-parse (spirv-val)` sets `VULKANE_REQUIRE_SPIRV_VAL`. Version allocated by the PM (all crates move to 0.17.3).

## [0.17.2] - unreleased

### Added - `vulkane-spirv-parse` crate (ROADMAP 3a step 2)

New workspace member: `Module::parse` / `to_words` over `rspirv` 0.13, with reflection (`entry_points` with `LocalSize`, `bindings`, `capabilities`). Parsing does not validate. Header word 2 (generator) is kept because `rspirv` re-encodes its own. Measured: all 9 committed modules and all 178 modules of Fuel's `fuel-vulkan-kernels/spv` (via `VULKANE_SPV_CORPUS`) re-encode byte-identically. Version allocated by the PM (all crates move to 0.17.2). Open: `spirv-val` verdict comparison and the #106 negative case (ROADMAP 3a).

## [0.17.1] - unreleased

### Fixed

- `vulkane-kernels-source`: `softmax_f64.slang` computed f64 `exp` with GLSL.std.450 `Exp` (16/32-bit only), so `spirv-val` rejected it (#106). It now uses `poly_exp_d` from `unary_f64.slang`. `spirv-val --target-env vulkan1.1` passes; no GPU numerical comparison yet. Version allocated by the PM.

## [0.17.0] - unreleased

### Changed (breaking) - `unpopped` 0.35.0, `tree-sitter` 0.27

`vulkane-slang-emit` and `vulkane-slang-parse` now depend on `unpopped` / `unpopped-vocab` / `unpopped-cpu-c` 0.35.0 (breaking in Unpopped: tree-sitter 0.27),
and `vulkane-slang-parse` on `tree-sitter` 0.27 (the same copy, so `Frontend::parse` still returns the one `Tree` type). No source change was needed. Measured:
all emit tests (17 unit incl. the `slangc` test, 12 cells) and parse tests (8 unit, doctest, corpus 53/94/0 over the 147 kernels) pass; one tree-sitter in the tree.
`vulkane-slang-emit`'s `convert` feature is documented as needing rustc 1.90. Version allocated by the PM.

## [0.16.14] - unreleased

### Added - `vulkane-slang-parse` crate (Slang front end, from Unpopped)

New workspace member: `parse_slang`, `SLANG_RESIDUE` and the `SLANG` `Frontend`, moved from `unpopped::convert` (Unpopped `213df4f`,
v0.34.11), built on `unpopped` 0.34.11's public `Frontend::new` and the `convert` walkers. Measured: 8 unit tests + 1 doctest pass;
`tests/corpus.rs` reproduces Unpopped's baseline over the 147 `.slang` kernels (53 residue, 94 unrecognized, 0 accepted). `tree-sitter`
is pinned to 0.26 to match published `unpopped`; it moves to 0.27 with Unpopped's step-4 release. Version allocated by the PM.

## [0.16.13] - unreleased

### Added - characterisation tests for `hoist_extinst_imports.py`

`vulkane-kernels-source/kernels/test_hoist_extinst_imports.py`: 7 text-in/text-out tests pinning what `hoist_imports` does today (ROADMAP 1d step 1),
so the Codacy-flagged function (CC 34) can be split without changing behaviour; new CI job `kernels-scripts`. Mutation control: removing
both word-boundary guards (the sort order and the lookahead) fails exactly the prefix test; removing either alone does not (they are redundant).

## [0.16.12] - unreleased

### Added - `vulkane-slang-emit`: the Slang conformance cells

The three Slang cells (`cell_token_is_not_a_computation`, `dtype_lowering_coverage`, `lower_kernel_default_is_the_typed_decline`; 12 tests)
move from Unpopped's `unpopped-conformance` (Unpopped `5500fd1`) into `vulkane-slang-emit/tests/`, run against the published `unpopped` /
`unpopped-cpu-c` 0.34.11 (new dev-dependency), per D3 of the Slang handover note. Only the `Slang` import path changed; rustfmt applied.
Unpopped removes its copies after this merges.

## [0.16.11] - unreleased

### Changed - `vulkane-slang-emit`: the `slangc` test runs in CI

`emitted_slang_compiles_with_slangc` is no longer `#[ignore]`d: it returns early when `slangc` is absent, and fails if
`VULKANE_REQUIRE_SLANGC` is set. A new CI job installs a pinned, sha256-verified `slangc` (2026.20) and runs the crate's tests with
`--features convert`, which no other job built. Origin recorded in the crate's `THIRD-PARTY-NOTICES.md`.

## [0.16.10] - unreleased

### Added - `vulkane-slang-emit` crate (moved from Unpopped)

New workspace member: the Slang compute-shader emitter, moved from Unpopped's `unpopped-slang` (Unpopped `213df4f`, v0.34.11)
with its git history (165 commits; tree blob-identical to the source). Depends on the published `unpopped` / `unpopped-vocab`
0.34.11. Unchanged code; new manifest, README and `THIRD-PARTY-NOTICES.md`. Measured: 15 unit tests pass (16 with feature
`convert`), the `slangc`-gated test passes locally (`#[ignore]`d in CI: no `slangc` on GitHub runners), clippy `-D warnings`,
rustdoc `-D warnings`, `cargo package --list` ok. Design: `docs/design/slang-parser-emitter-handover.md` step 2.

## [0.16.9] - unreleased

### Added - WGSL parser/emitter design note

`docs/design/wgsl-parser-emitter.md` (no code): options W1-W4 for `vulkane-wgsl-parse` / `vulkane-wgsl-emit`. `ROADMAP.md` row 3d.
Version allocated by the PM.

## [0.16.8] - unreleased

### Added - HLSL parser/emitter design note

`docs/design/hlsl-parser-emitter.md` (no code): options H1-H4 for `vulkane-hlsl-parse` / `vulkane-hlsl-emit`. `ROADMAP.md` row 3c.
Version allocated by the PM.

## [0.16.7] - unreleased

### Added - GLSL parser/emitter design note

`docs/design/glsl-parser-emitter.md` (no code): options G1-G5 for `vulkane-glsl-parse` / `vulkane-glsl-emit`. `ROADMAP.md` row 3b.
Version allocated by the PM.

## [0.16.6] - unreleased

### Changed - `naga` 29 to 30, `shaderc` 0.8 to 0.10 (latest-versions rule)

No public API change: `shaderc` 0.10 returns `Result` from `Compiler::new` / `CompileOptions::new`, mapped onto the
existing `ShadercError::CompilerInit` / `OptionsInit`. Measured on this branch with `--features naga,shaderc`: lib tests (82), `naga_test`,
`shaderc_test`, `ci_coverage` pass; clippy `--all-targets -D warnings` clean. Version allocated by the PM.

## [0.16.5] - unreleased

### Added - SPIR-V parser/emitter design note

`docs/design/spirv-parser-emitter.md` (no code): options S1-S5 for `vulkane-spirv-parse` / `vulkane-spirv-emit`.
`ROADMAP.md` rows 3a (note) and 3e (stale `naga`/`shaderc` pins). Version allocated by the PM.

## [0.16.4] - unreleased

### Added - Slang parser/emitter handover design note

`docs/design/slang-parser-emitter-handover.md` (no code): how `unpopped-slang` and the Slang front end move from
Unpopped to Vulkane (`vulkane-slang-emit`, `vulkane-slang-parse`). `ROADMAP.md` gains row 1d. Version `0.16.4`
allocated by the PM.

## [0.16.3] - unreleased

### Added - `vulkane-kernels-source` crate and `ROADMAP.md`

New workspace member holding the GPU kernel sources (147 Slang, 27 GLSL, `compile.sh`,
`hoist_extinst_imports.py`) moved from Fuel's `fuel-kernels-source` (Fuel `e3ff3427`) with their git history,
including the history from before they were renamed out of `fuel-graph-executor/src/shaders/`. The imported tree is
blob-identical to Fuel's at that commit. `compile.sh` takes `--out-dir DIR` (default `spv/` inside the crate)
instead of writing into Fuel's `fuel-vulkan-kernels/spv/`. Third-party origins are traced in the crate's
`THIRD-PARTY-NOTICES.md` (llama.cpp for `get_scale_min_k4`; webgpu-blas acknowledged, no code taken).
Version `0.16.3` is provisional; the PM allocates at gate time. `vulkane` itself is unchanged apart from its version.

Known, carried over from Fuel and not fixed here: `softmax_f64.slang` compiles to SPIR-V that `spirv-val`
rejects (`GLSL.std.450 Exp` on a 64-bit float); Fuel's committed `softmax_f64.spv` fails identically.

## [0.16.2] - unreleased

### Added - `PROVENANCE.md` records where `vulkan.jpeg` came from

Documentation only; no source or asset changed. The logo's embedded C2PA manifest (read-only parse, not
cryptographically validated) names Microsoft's Image Creator from Designer; CireSnave's statement and the open
licence-terms question are recorded in `PROVENANCE.md`. Bumped because every pushed change takes a version.

### Changed — CI hardening (no packaged source touched)

Every `uses:` in `.github/workflows/ci.yml` is pinned to a full commit SHA (tag in a trailing comment),
the workflow declares `permissions: contents: read`, and `.github/CODEOWNERS` names `@ciresnave`.
No crate version changes: all four members already sit on unpublished versions and no packaged file changed.

## [vulkane_derive 0.1.2] — unreleased

### Changed — bumped because the licence sweep edits PACKAGED source

`vulkane_derive 0.1.1` is served by crates.io, and this PR adds an
`SPDX-License-Identifier` line to `src/lib.rs`, which ships in the package.
Leaving the version alone would put changed content under a string the
registry already serves: one version string naming two artifacts.

⚠️ **The collision would exist from the MERGE, not from the next publish.**
The divergence gate armed in #91 runs on main, so the sweep landing without
this bump would turn main red and block every subsequent PR.

Found by that gate on this PR — the first time it fired on an input nobody
wrote it for. `vulkane_derive` was deliberately left on its served string in
#89, on the reasoning that the armed probe guards the moment it starts to
matter rather than a preemptive bump having to be remembered. It did.

The other three members needed nothing: all were already on unpublished
versions after #89.

## [kiss-vulkan-vocab 0.4.9] — unreleased

### Changed — bumped immediately AFTER publishing 0.4.8, which is the point

0.4.8 went to crates.io and the tree moved off that string in the same
sitting. Nothing is wrong with 0.4.8 and nothing is being fixed here: the
bump happens **because** the tree and the artifact still agree, so that the
next edit to this crate cannot put changed content under a string the
registry already serves.

⚠️ **The old habit was to bump when preparing a release.** That leaves the
whole interval between publishes as a period in which the declared version
is a false claim about the tree — invisible, because the version field reads
identically in both worlds. Bumping after publishing makes the tree's
declared version *normally unpublished*, which is a state that cannot
collide.

The heading says `unreleased` rather than a date for the same reason.

## [vulkan_gen 0.5.1] — unreleased

### Fixed — `0.5.0` was one version string naming two artifacts

`vulkan_gen` sat on `0.5.0`, which crates.io serves, while its `src/`
differed from the served tarball by **36 hunks across 8 files** (−289/+284).
A consumer resolving `0.5.0` got one artifact and this repository's own CI
tested another, and both were green about different objects.

⚠️ **The remedy is a BUMP, NOT A PUBLISH, and that was measured rather than
assumed.** The two versions' *generated output* is byte-identical —
2,982,463 bytes, 69,634 lines, 0 diff hunks — so there is nothing
consumer-visible to ship. Publishing would have been the wrong fix: it
would ship a release whose only content is a false claim being corrected.

Found by the divergence probe added in #85, which is the first thing in this
repository that could see the defect at all.

## [0.16.1] — unreleased

### Fixed — the install lines cited requirements that resolve to old releases

⚠️ **Seven sites told users to depend on `vulkane = "0.1"`, `"0.4"` or `"0.10"`
while crates.io served `0.16.0`.** Four in `README.md`, and three inside the
crate's own rustdoc — `src/lib.rs`, `src/safe/naga.rs`, `src/safe/shaderc.rs` —
the last of which is the docs.rs landing page, the most-read copy of all.

⚠️ **Every one of them RESOLVED, which is why nothing ever complained.**

| cited | where | resolves to | published minors newer |
| --- | --- | --- | --- |
| `"0.1"` | `src/safe/naga.rs` | `0.1.0` | 13 |
| `"0.4"` | `README.md` ×3, `src/safe/shaderc.rs` | `0.4.5` | 10 |
| `"0.10"` | `README.md` ×1, `src/lib.rs` | `0.10.1` | 4 |

A `cargo add` following the README **succeeds** and silently hands a stranger an
old published release, with feature flags documented against 0.16. **A
requirement that fails to resolve at least tells the user something. This one
has no complainant** — the dangerous form is not the error, it is the plausible
answer.

⚠️ **An earlier draft of this entry said "a version twelve releases old", and
was wrong twice.** It assigned ONE distance to THREE different requirements —
caught in review — and the number itself was wrong for every one of them,
because it came from subtracting minor numbers (16 − 4) when the published
minors are `1..10, 13, 14, 15, 16`. **0.11 and 0.12 were never released, so a
difference between version numbers is not a count of releases.** Recorded rather
than edited away, since this entry is about citing versions accurately.

**The rule, written down because nothing enforced it:** a version cited for
CONSUMPTION must be one the registry currently serves; a version cited to
DESCRIBE THIS TREE must be the tree's own. Today a single string does both jobs,
and only because the two happen to coincide.

**Two sites deliberately keep their old numbers.** `CHANGELOG.md` and
`CONTRIBUTING.md` state what was true at a past release; rewriting those would
falsify a record rather than fix a pointer. The distinction is the whole point —
citing for consumption and citing to describe history are different acts.

### Changed — bumped, because three of the corrections ship inside the crate

`src/lib.rs`, `src/safe/naga.rs` and `src/safe/shaderc.rs` are packaged, so
leaving the tree at `0.16.0` would put edited source under a version string
crates.io already serves: **one version string naming two artifacts.** `0.16.1`
is unpublished and remains so until a release.

⚠️ **The heading says `unreleased` rather than a date on purpose.** The remedy
for that class is to bump immediately AFTER publishing rather than before, so a
tree's declared version is normally NOT yet on the registry — and a dated
heading would claim a release that has not happened.

## [kiss-vulkan-vocab 0.4.8] — 2026-09-09

### Fixed — ⚠️ RETRACTED: `coverage_note` published a trend inferred from lower bounds

0.4.7's `coverage_note` told foreign readers that a residue series of **8 → 6 → 3**
showed the surface being *refined rather than consumed*. **That claim is not
supported and it was served.**

⚠️ **A residue counts what ONE READER NOTICED CHOOSING — never what the prose
leaves open.** It is a lower bound whose tightness varies with attention, so a
falling series is evidence about **readers**, not about this document. You cannot
infer a trend from bounds that tighten by an unknown and changing amount.

**baracuda refuted their own numbers** by finding a gap their count had missed in
a release they had already scored — and they checked whether the 0.4.7 fix had
*created* it (which would have confirmed their published prediction) rather than
assuming. It had not: the `ops`/`arith` notes are **byte-identical across 0.4.6
and 0.4.7**.

**My failure, not theirs:** I elevated their inference into the artifact without
asking what their own `LOWER BOUND` header did to the trend claim. The field
whose purpose is telling a reader what *not* to conclude carried a conclusion
that could not be drawn.

### Added — a repeated set member is absorbed, and a vector proves it

`field_spec` said *"sorted and deduplicated"* for `<coop>`/`<coopvec>` and **said
nothing about `<ops>`/`<arith>`** — and **no vector carried a duplicate**, so a
producer that never deduplicated passed all seventeen. Found by running exactly
that producer: **17 pass / 0 fail, with a control mutation firing at 14/3**, so
the zero is a measurement rather than a non-run.

`ops=["b","b","w"]` → `ops-bw`. Both notes now state that a repeat is **absorbed**
because the field is a **set**.

⚠️ **This is the only set vector whose input is NOT derived from its token — and
the safety property that prevents one class of error is what prevented this
vector from existing.** Every other set vector derives input *from* the token so
the halves cannot disagree; a token spells each member once, so a duplicate
cannot be recovered from it.

### ⚠️ The finding that outlives both

**A DECLARED LEDGER CANNOT RECORD A CHOICE YOU DID NOT NOTICE MAKING.** Declaring
the population fixed *"a path no vector reaches"* and does nothing for *"a
decision I made without seeing it as one"* — their producer wrote `set(vals)`,
which felt like typing rather than deciding, and so never reached the ledger.

**Their proposed practice, which generalises past this manifest:** after building
a producer, list the stdlib calls that impose semantics — `set`, `sorted`,
`strip`, `dict`, float `==` — and ask what the spec says about each. **Every
silence is a residue entry nobody wrote down.**

`vectors_digest` moves to `fnv1a64-212b0cbfc1d2a4bb`; the array gained a member.

## [kiss-vulkan-vocab 0.4.7] — 2026-09-09

### Added — `input_shape`: what a caller HANDS a producer

⚠️ `field_spec` named **concepts** — *"M-N-K plus four component types"* — and the
input JSON uses `m`, `n`, `k`, `a`, `b`, `c`, `result`, `saturating`. **Measured:
zero of those key names appeared delimited anywhere in the manifest.**

**No vector could ever have caught it.** A producer reading different keys
**crashes rather than mismatching** — the corpus pins what a token *spells* and
said nothing about what a caller must *hand* a producer, which is exactly where
a real driver integration lives.

Every `field_spec` entry now carries `input_shape`, and the gate reads the
declared key list **out of it** rather than restating it — the defect that let
`empty_set_spelling` ship contradicting its own corpus.

### Fixed — dedup equality, dissolved rather than decided

`tuple_sort_key` says deduplication happens after sorting and **never says what
makes two tuples EQUAL**. Spelled-dedup and structural-dedup diverge exactly when
a key is **absent versus explicitly default**, and every vector carried every key
so nothing discriminated it. A producer choosing either reading passes all 19.

⚠️ **Declaring every input key REQUIRED dissolves the question instead of picking
a side**: with no absent case the two readings coincide, which is what this
implementation already does — `CoopShape::saturating` is a `bool`, not an
`Option<bool>`. Born-red by omitting the exact key that makes them diverge.

**baracuda's distinction is the useful half and it is new:** this gap was merely
**UNCOVERED** — a discriminating input exists and one vector closes it — where
`ops_alphabet`/`arith_names` are **UNDISCRIMINABLE**, no vector can ever exist.
⚠️ **The two look identical from inside a passing corpus and only one is fixable.**

### Added — `coverage_note` states what four rounds of reproduction did NOT establish

    residue reported by a foreign producer:  8  ->  6  ->  3
    new questions created by the fixes:      4, one per round

Flags → how they sort → where escapes sort → whether dimensions compare as text
→ what makes two tuples equal. ⚠️ **The residue is falling and the discovery rate
is not, which says this surface is being REFINED rather than consumed:** each
statement made precise enough to check exposes the next thing it does not say.

**A reader is now told so.** *Treat a passing reproduction as evidence about the
vectors, never as evidence that the prose is complete.*

`vectors_digest` unchanged: `field_spec` and `coverage_note` are not in the
`vectors` array.

## [kiss-vulkan-vocab 0.4.6] — 2026-09-06

### Added — the two orderings no vector can ever test

⚠️ **`ops_alphabet` and `arith_names` are THEMSELVES in lexicographic order**, so
no input — however chosen — can distinguish *"sort by the declared array's
index"* from *"sort the spellings as text"* for those fields. **A producer
implementing the wrong rule passes every vector that could ever be written.**

`component_types` is **not** alphabetical (`f64` precedes `bf16`; four adjacent
pairs disagree) and **is** discriminable — which is exactly why it is the field
that produced two ordering defects and now carries two deliberate vectors.
⚠️ **The field that was testable is the field that got tested; the two that were
not, were not, and nothing said so.**

Found by baracuda against published 0.4.5.

**This is a third category.** An unstated rule is *undiscriminated*; an
incidentally-witnessed one is *fragile*; this one is **stated, right, and
unfalsifiable** — the corpus cannot test it even in principle. It is this
repository's own banked rule pointed one level up: *a machine-checkable value
that is wrong is a bug; an unfalsifiable value that is right is a worse bug.*
Here the **value** is right and the **rule about it** is unfalsifiable.

⚠️ **And unlike a fragility, it has a TRIGGER.** It activates the day either
array stops being alphabetical — add an op class whose letter is appended rather
than inserted, and every text-sorting producer silently diverges from every
index-sorting one. **The corpus was built while the two rules agreed, so it
carries no memory of the distinction and will not object.**

`the_alphabets_that_cannot_discriminate_their_own_order_still_coincide` fires at
exactly that moment, and says what to do about it. Its own control is
`component_types` **not** coinciding: if every array were sorted, the comparison
would distinguish nothing and its passing would mean nothing.

`coverage_note` states the limit for foreign readers, since a gate in this
repository is invisible to them — and a reader whose producer passes every
vector is entitled to know which of their choices that does **not** vouch for.

`vectors_digest` is unchanged: `coverage_note` is not part of the `vectors`
array.

## [kiss-vulkan-vocab 0.4.5] — 2026-09-06

### Fixed — the sort rule reached the components and never the DIMENSIONS

`tuple_sort_key` named components, escapes, flags and dedup, and said nothing
about the three leading M, N and K of a `<coop>` tuple. They compare as
**integers**; a reader had to infer it, and ⚠️ **the rule's own precision made
the omission read as deliberate scope rather than a gap.**

It now states them. Measured by baracuda against published 0.4.4.

### Added — two vectors, because the evidence was a POPULATION OF ONE

⚠️ Integer ordering was pinned by **exactly one vector in the whole corpus, and
that vector's `pins` field says `threshold`.** Four other multi-tuple `<coop>`
vectors agree under numeric and lexicographic order and discriminate nothing;
only the 25-tuple threshold vector separates them — **incidentally to the reason
it exists.**

**So an editor changing that vector's shape set for threshold reasons — the
whole reason to touch it — could silently delete the only evidence of integer
ordering, with nothing to say so.** `integer_order` now pins it deliberately:
M values of 9 and 10, given larger-first, where text order and numeric order
disagree.

⚠️ **And measuring afterwards found the same defect in the field the fix did not
sweep.** With `<coop>` at two discriminators, the same count over `<coopvec>`
returned **one** — again a threshold vector, again incidental. `component_order`
closes it: `u8` against `u32`, which sort oppositely as text and as ordinals.

**Fixing the instance that taught the class, and not the class, is the error
this repository has now made six times.** The count is the only thing that
catches it.

### Changed — the execution gate covers both fields and counts PER FIELD

An earlier version checked `<coopvec>` only, on the judgement that `<coop>`'s
leading integers "would say nothing new". ⚠️ **That judgement was wrong, and a
foreign reader found exactly what it would have said.** It now ranks integers
numerically and components by ordinal across both fields.

Its discriminator guard is **per field**, not overall: ⚠️ an overall count of two
is satisfied by two discriminators in one field and none in the other — **which
is the precise state this repository was in.**

`vectors_digest` moves to `fnv1a64-45447bbf4e06e6fa`; the array gained two
members.

## [kiss-vulkan-vocab 0.4.4] — 2026-09-06

### Fixed — `tuple_sort_key` said LEXICOGRAPHIC and the vocabulary sorts by ORDINAL

The field added in 0.4.3 read *"Tuples sort **lexicographically by their spelled
components**…"*. The code sorts by each component's position in
`component_types`, and the two disagree:

    lexicographic   'u32…' < 'u8…'        would put u32 FIRST
    array index     u8 = 8  <  u32 = 10   puts u8 FIRST
    vectors[13]     spells u8 FIRST       -> ordinal, not lexicographic

**A producer following the stated rule emits a different enumeration and —
above the 512-byte threshold — a different digest.**

⚠️ **The field has two clauses and only one was wrong.** The *flag-as-last-key*
half is correct and is what closed the previous gap; **the justification
attached to it was not**. A correct action carrying a wrong reason survives
because the action is right, so nobody reads the reason adversarially.

Measured by baracuda against published 0.4.3, from the manifest alone.

### Added — a vector for where an unnamed escape sorts against a named type

`tuple_sort_key` claims `x<n>` escapes sort after every named type. That is true
in the code — `Other(u32)` is the last variant — but ⚠️ **zero vectors put a
named type and an escape at the same tuple position**, so no reader could check
the claim and no producer could be caught getting it wrong.

**Stating an unexercised rule is exactly how the lexicographic defect shipped.**
`vectors[4]` now carries `…-f16` beside `…-x5`, escape-first in the input.

`vectors_digest` moves to `fnv1a64-8263e55f94b959e1` — correctly, the vectors
array gained a member.

### Added — a gate that EXECUTES the stated rule

⚠️ **The composition gate cannot reach this class.** It substitutes into
**templates**; `tuple_sort_key` is prose describing an **algorithm**, and the
only instrument that finds a wrong algorithm is running it and comparing.

`executing_the_stated_sort_rule_reproduces_the_token_order` ranks every
`<coopvec>` component by its `component_types` index, re-sorts each multi-tuple
field, and asserts the result is what the token already spells. Its control is
the part that matters: **at least one vector must actually separate ordinal from
lexicographic order**, or the test passes under either rule and says nothing
about which is stated.

⚠️ **And it asserts against the prose as well as the corpus.** Executing a
hardcoded ordinal rule proves what the vocabulary does and nothing about what
the manifest claims — the same blindness that let `empty_set_spelling` ship
wrong one release earlier. Restoring either the lexicographic wording or a
swapped tuple now reds.

## [kiss-vulkan-vocab 0.4.3] — 2026-09-06

### Fixed — nothing said the angle brackets were placeholders

Several values carry `<...>` — `digest_marker` (`fnv1a64-<hex16>`),
`unnamed_component_escape` (`x<n>`), `empty_set_spelling` (`<prefix>-none`),
`unnamed_component_escape_order`, and the top-level `grammar` — and **the
manifest never said the brackets were not literal text.**

⚠️ **Found by baracuda rebuilding `vectors_digest` from the published 0.4.2
manifest alone.** They read `digest_marker` as a **prefix** rather than a
template and emitted `fnv1a64-<hex16>-5fb4518c42b73202`. It was the **only**
error in an otherwise byte-exact reproduction — the hex digits were right first
try — and the field name argues for their reading: a thing called a *marker*
sounds like literal text while its value is a format string.

Renaming that one field would have fixed the instance. `declarative.notation`
states the convention once and governs every value, which is the class. Gated,
with a positive control asserting real placeholders exist for it to govern —
a convention statement that outlives the thing it explains is decoration.

### Fixed — the `vectors_digest` pin buried its most important rule in a parenthetical

*"exactly as this manifest renders it"* is the clause that forces a **raw-text**
read, and a producer that parses and re-serializes emits its own key order and
whitespace and gets different bytes. It was a parenthetical; it is now the
opening instruction. The pin also now states that the `note` member's end must
be found by **JSON string-scanning** rather than by searching for `, `, because
a note's own prose contains that sequence — baracuda hit this and called it an
implementation hazard rather than a spec gap, which is right, and it costs one
sentence to remove.

`vectors_digest` itself is **unchanged** at `fnv1a64-5fb4518c42b73202`: the
construction's wording moved, the `vectors` array did not.

### Verified

baracuda re-ran their **merged, unmodified** 0.4.1 producer against served
0.4.2: **13 of 16 pass, and the three failures are exactly `transpose`,
`subgroup` and `saturating`** — GUESS-7, GUESS-4 and GUESS-5, the three
vector-closable items, each now caught against the producer that exploited the
gap. Nothing regressed.

⚠️ And they rebuilt `vectors_digest` from the pinned sentence alone, having
never seen this code: **`fnv1a64-5fb4518c42b73202`, and their notes-kept control
matched ours too.** The pin says what it means.

⚠️ **That is not the same claim as "a fresh reader's residue is smaller."** The
old producer failing proves the manifest now *discriminates* where it could not;
only a producer written from 0.4.2 alone measures what a reader arriving today
still has to supply. That run is outstanding, and the digest half already has a
fresh residue of three, one of them real — this entry.


### Fixed — `empty_set_spelling` was WRONG, not merely unpinned

It read **`<prefix>-none`**. `ops` declares prefix `ops-` and `coop` declares
`cm-`, so composing the rule against the manifest's own prefixes yields
**`ops--none`** and **`cm--none`** — a double dash, for four of the five fields —
while the tokens spell `ops-none` and `cm-none`. **A reader following the
declarative half against the declared prefixes emits bytes that match nothing**,
and under §6.8-0002 that is a different cell rather than an error.

Raised by baracuda as *unpinned*; measured here it is *contradictory*, which is
the stronger fault and the one no additional vector could surface — every vector
spells the right bytes, so only **composing the stated rule** reveals it
disagrees.

⚠️ **The first repair replaced it with a correct SENTENCE, and the drill caught
that.** Restoring the defect left the new gate **GREEN**, because prose cannot be
composed and the test had quietly hardcoded the rule instead of following the
stated one. It is now the template `<prefix>none`, and the gate substitutes each
declared prefix into **the value it reads from the manifest**.

> **A machine-checkable value that is wrong is a bug. An unfalsifiable value
> that is right is a worse bug — the next wrong one arrives with nothing able to
> say so.**

### Added — the sort key, which "sorted and deduplicated canonically" never gave

⚠️ baracuda's producer ranked the components and **omitted the flag**, so two
tuples differing only in `-sat` or `-t` compared **equal** and a stable sort
emitted them in input order. **Their output was not a decision at all** — it was
an artifact of the sort being stable.

**The gap could not have existed at 0.4.1**: with no saturating and no
transposing vector there was no pair differing in nothing but a flag. **Closing
GUESS-5 and GUESS-7 created this question**, and the vectors answered it while
the prose did not. `declarative.tuple_sort_key` now states that the flag is the
**last key** and that the unflagged form orders first; the gate checks the prose
*and* measures the tokens, so the two cannot drift apart.

### Added — what the threshold actually measures

`enumeration_bytes` named the quantity, but nothing said the **field prefix is
excluded**. Three or four bytes decide a borderline field — the exact case where
enumerate-versus-digest is hardest to test and a disagreement is a different
cell. `declarative.measured_length_excludes_prefix` says so.

### Fixed — the `coopvec` prose still did not spell its flag

The `coop` entry was corrected in 0.4.2 to spell `-sat`; **its sibling still
read "five component types plus a transpose flag" and never spelled `-t`.**
⚠️ That is the `sgdyn`/`transpose` per-instance-versus-per-class error a third
time, **in the same file, in the commit that fixed the first instance** — caught
by a foreign reader rather than here.


### Added — every rule-shaped value is now composed against the corpus

⚠️ **baracuda's class, and it caught this repository a fourth time.** After
`empty_set_spelling` turned out to contradict the vectors, they named the shape:

> **A declarative rule that contradicts the vectors is invisible to every
> vector-based check, because the vectors are all correct.** A vector suite
> validates the vectors against a producer; it never validates a **rule** against
> the vectors. Only composing the stated rule and comparing its output to the
> corpus finds it.

Measured here: **five values are rule-shaped and exactly one was gated** — the
one they found. Per-instance again, with the class handed over explicitly in the
message beforehand.

Two gates close it. `every_rule_shaped_value_is_on_the_checked_list` **declares**
the rule keys rather than accumulating them, so a future placeholder-bearing
value that nothing composes **fails** instead of passing silently — the
`[NOT EXERCISED]` argument in a different medium, since an accumulating check
cannot report a rule it never reached. `the_composable_rules_produce_what_the_tokens_spell`
composes `grammar` against every token's field count, `digest_marker` against
every digest, and `unnamed_component_escape` against the unnamed spellings.

⚠️ Both gates failed on first run, and both failures were the gates working: one
demanded that `coverage_note` — prose that *mentions* placeholders — be
explicitly accounted for, and the other caught the digest scan matching **the
rule stating itself** (`fnv1a64-<hex16>` inside `digest_marker`, and quoted again
inside `notation`). The scan is now scoped to where a digest actually lives
rather than filtered by the shape of whatever prose happens to exist today.

## [kiss-vulkan-vocab 0.4.2] — 2026-09-06

### Added — `vectors_digest`, so a demonstration can go stale (KISS-Classify §6.8-0017)

The manifest now carries a top-level **`vectors_digest`** and pins its input
construction in **`declarative.vectors_digest_input`**. The clause requires the
namespace to pin the construction; carrying the digest is the `MAY` half, taken
here because 24 bytes is the difference between a gate that compares a number
this repository computed against a number this repository computed, and one a
second party can check.

⚠️ **The pin is the load-bearing half.** KISS pins the algorithm and constants
and deliberately does *not* define this input: §6.8-0007's input is *"the
canonical enumeration string it **replaces**"*, and the `vectors` array replaces
nothing. An unpinned digest is a number every party computes differently while
each compares only against itself — **vacuously green for its author and
guaranteed red between two parties.**

The construction is defined over **this manifest's own rendering**, not a
re-serialization: a reader already holds the bytes, and asking them to
re-serialize reintroduces every question canonical JSON leaves open — key order,
whitespace, number formatting, `\u` escaping — none of which any clause settles.
It excludes every `note` (required by the clause), states **UTF-8** explicitly,
and is defined over **whatever members an element actually carries** rather than
a fixed template, because a `digest_input` vector carries no `token` at all.

Verified by rebuilding the digest **from the pinned sentence alone**, in another
language, outside this repository — it reproduced the emitted value byte for
byte. `vectors_digest_is_rebuildable_from_the_pinned_construction` does the same
rebuild by hand in-tree and deliberately does not call the emitter's own
helpers, which are in scope: calling them would compare the emitter against
itself and pass for any construction whatsoever.

⚠️ A version bump moves `generated_from` and **does not move `vectors_digest`** —
checked, because a contract digest that drifts on a release number would fire on
healthy input.

### Fixed — the changelog gate could only see one of four published crates

The gate added in #50 reads `vulkane`'s version and asserts a matching `## [...]`
section. **This repository publishes four crates.** A `kiss-vulkan-vocab`-only
release — like this one — shipped to crates.io with no changelog section and the
gate stayed green, which is #41's defect exactly ("the repository denying the
existence of a version that is on crates.io") reached through a different crate.

It now checks **every publishable workspace member**. Co-release facts were taken
from the registry rather than reconstructed: `vulkan_gen 0.5.0` and
`kiss-vulkan-vocab 0.3.0` were published within twelve seconds of `vulkane
0.13.0`; `vulkane_derive 0.1.1` within a minute of `vulkane 0.14.0`. Those two
headings are annotated accordingly.

### Added — `kiss-vulkan-vocab` declares `sufficiency`, and closes all eight gaps a reproduction found

**KISS-Classify §6.8-0017 requires a `sufficiency` object with a `status` of
exactly `demonstrated` or `unexercised`.** This manifest ships **`unexercised`
despite a real demonstration having happened** — baracuda reproduced the
twelve-vector 0.4.1 manifest byte-identically from the manifest alone, archive
deleted, no source and no docs.rs. §6.8-0013 makes the vectors array the
normative contract, and it has moved since; **the artifact being shipped is not
the artifact that was reproduced**, and carrying the result forward would credit
a run nobody performed.

**The reproduction's value was never the pass.** It was the list of things they
had to supply from outside. All eight are now closed — seven below, and the eighth (`saturating`) under its own heading, because it turned out to be a defect rather than a gap:

| Gap | Closed by |
|---|---|
| FNV-1a-64 offset basis and prime | `declarative`, **emitted from the crate constants** so the manifest cannot drift from the digest it describes |
| hex padding and `x<n>` ordering | `declarative` — every pinned digest happens to have no leading-zero nibble and every sampled run is equal-width, so no vector could ever discriminate |
| the digest's input **encoding** | `declarative` `digest_encoding` — every spelling this vocabulary can produce is ASCII, so the question is outside what a vector can ask |
| `transpose` | a vector carrying `transpose: true` beside a non-transposing combination |
| `sgdyn` | a vector whose **input** is the string `"dynamic"` — the gap was in the input half, which no vector passing an integer width could reach |
| the `<coop>` tie-break | two distinct shapes agreeing on m, n and k, **swapping `a` and `b`** — the only input shape that can separate `(a, b, c, result)` from any other permutation |

⚠️ **A vector pins the OUTPUT, not the algorithm** (baracuda). That sorts a gap
list into vector-closable and prose-only, and the two halves need different
fixes — which is why the digest facts are stated rather than sampled.

### Fixed — the `-sat` suffix was in NEITHER half of the manifest

**`<coop>` tuples take a trailing `-sat` when the shape saturates, and nothing
said so.** `field_spec` read "M-N-K plus four component types" and stopped, and
all fifty-six shapes across the vectors carried `saturating: false` — so the
suffix appeared in **no vector and no description**. A reader building a producer
from this manifest could not have emitted it by any route, and under §6.8-0002's
byte-exact matching that is a different cell, not a near miss.

⚠️ **Reported as "`saturating` is not spelled into the coop tuple", which is
wrong about the code — but the reasoning was sound and called for the right
fix.** The observation was that every vector carries `saturating: false`, so a
producer that appended the field would still match all twelve. That holds whether
the suffix exists or not, which is exactly why the vectors could not answer it.

### Fixed — the `sufficiency` note went stale about which artifact it describes

**It said "This manifest has thirteen" and was wrong one commit later**, when
three vectors landed and `vocabulary_version` did not move. **The field whose
entire purpose is to say "what ships here is not what was reproduced" had itself
gone stale about what ships here** — the same currency gap reported upstream
against §6.8-0017, arriving from inside the field meant to prevent it. The count
is now derived from the vector list, so there is nothing to update.

### Added — gates, because §6.8-0017 is the only §6.8 row in KISS's `UNBACKED.tsv`

Nothing in the conformance suite catches a malformed `sufficiency` block, so
these are local:

- `flag_vectors_pin_the_spellings_their_notes_describe` — a vector is
  self-consistent **by construction** here (the emitter derives the token from
  the same code that spells it), which is deliberate and is also why a vector
  cannot catch a wrong **note**. Born-red on all three arms: collapsing the
  saturating tuples, swapping the tie-break order, and spelling the subgroup
  input as a number.
- `sufficiency_is_declared_per_6_8_0017` gained a fourth arm requiring the note
  to **name** the manifest's own vector count. ⚠️ Its first draft asserted
  `n <= vectors` over each digit run, which a stale `13` satisfies as
  comfortably as a correct `16` — a gate that cannot fail the defect it is named
  after. Born-red on a stale count and on an absent one.

## [0.16.0] — 2026-09-06  (kiss-vulkan-vocab 0.4.1)

### Fixed — `kiss-vulkan-vocab`'s published grammar described a FOUR-field token

⚠️ **The module doc is the only artifact describing the `vulkan:` token grammar
to anyone outside this repository, and it was wrong.** `to_token` has emitted
**five** fields since the v4 bump —
`subgroup.ops.arith.coop.coopvec` — while the doc's grammar line, its "Four
fields in fixed positions" sentence and its field table all described four,
omitting `coopvec` entirely. Live on docs.rs for **0.4.0**.

**Not cosmetic, because KISS-Classify §6.8-0002 matches byte-exact with no
tolerance.** A second implementer reading the published doc emits a four-field
token that can never match anything this crate produces — and `parse` rejects
it outright, since `parse` requires exactly five. The code was right, the
contract description was wrong, and the description is what an outside
implementer has.

Found while gathering citations to file this crate's pinned decisions as KISS
issues: the summary said "five-field token", the module doc said four, and the
disagreement was the finding.

`the_grammar_doc_matches_what_to_token_emits` now pins it, checking three
artifacts against one source of truth because the failure was that two of them
disagreed silently: the `format!` in `to_token`, the `//!` grammar line, and the
field table's data rows. Born red against the real historical wording —
*"`to_token` emits 5 fields but the grammar line declares 4"*.
### Removed — the `slang` feature, and runtime Slang compilation with it

⚠️ **Breaking for anyone enabling `features = ["slang"]`.** The feature, the
optional `shader-slang` dependency, `safe::slang` (`SlangSession`,
`SlangModule`, `compile_slang_file`), the `Error::SlangCompile` variant and the
CI job are all gone. `naga` (WGSL/GLSL) and `shaderc` (GLSL/HLSL) are
unaffected.

**Vulkane takes SPIR-V. Producing it is somebody else's job**, and that was
already true of every consumer: measured across the portfolio, **zero** crates
enabled this feature, against a control of 78 manifests that mention vulkane at
all. Fuel — the only project shipping Vulkan kernels — says so in its own
manifest: *"compiled ahead of time … vulkane does NOT need its naga / shaderc /
slang features at build or run time. Fuel stays toolchain-free for downstream
consumers."*

**The ecosystem is not ready and we were carrying the cost of pretending it
was.** Surveyed 2026-09-05, from the published tarballs rather than from docs,
because neither `shader-slang` nor `shader-slang-sys` builds on docs.rs at all:

| crate | in-memory source? | needs at runtime |
|---|---|---|
| `shader-slang` 0.1.0 | **no** — `load_module` only | `slang.dll` |
| `shader-slang-sys` 0.1.0 | raw C API, docs build fails | `slang.dll` |
| `slang` 0.0.2 | has it, but **not `pub`** | `slang-sys` |
| `minislang` 0.4.0 | **no** — writes a `TempDir`, compiles by path | `slang.dll` |
| `concinnity-slang` 0.19.1 | n/a — `slangc` subprocess | `slangc` |

The upstream repo's `main` DOES have `load_module_from_source_string`, so the
limitation this crate documented was **unreleased, not unimplemented** —
verified in the published 0.1.0 source: `load_module` 1, source-string variants
0. Reaching it needs a git dependency, which crates.io forbids for a published
crate, or vendoring.

**And the alternative was never a different crate.** Unpopped, the kernel IR
hub, intakes Slang (tree-sitter frontend), optimizes, and emits Slang — but its
artifact type is SOURCE TEXT, recorded as an open gap in its own
`docs/deferred.md`, and its test that checks the emitted Slang compiles shells
out to `slangc`. Every route ends at the same CLI, which is where Fuel already
is.

`Shader::from_spirv` and the rest of the SPIR-V surface are unchanged; the
`shader` module already told callers to *"use any other SPIR-V producer
(rust-gpu, slang, etc.) — vulkane takes SPIR-V"*, and that sentence is now the
whole story rather than one option among three.

## [0.15.0] — 2026-09-03

**If you call `device_identity()` and match on `device_uuid`, read this one.**
A minor bump rather than a patch, deliberately: the callers most likely to exist
are the ones correlating by UUID, and that is exactly the population the defect
hurt. A patch release would have fixed it and hidden the fix from them.

⚠️ **The symptom was a wrong answer, not a missing one.** On any instance below
Vulkan 1.1 — including the `api_version` **default** — the call returned
`Some(...)` with an all-zero `device_uuid`, which is not an obvious absence but
a valid-looking value that **every device shares**. Two physically different
GPUs compared **equal**. Live from 0.8.3 through 0.14.1.

So a caller who saw `Some(...)` and matched on it did not get a failure to
correlate; they got a correlation to the wrong device, or to all of them. If
that is your code, the thing to audit is your GPU selection, not your error
handling.
### Fixed — `device_identity()` returned a zeroed join key that made every device compare equal

⚠️ **Behaviour change, and it is a breaking one for a caller on a 1.0 instance.**
`PhysicalDevice::device_identity()` now returns `None` when
`effective_api_version()` is below 1.1, where it previously returned `Some`.

`VkPhysicalDeviceIDProperties` is Vulkan 1.1 core. On a 1.0 implementation the
driver skips the unrecognized chained struct and leaves it exactly as we
allocated it — all zeros — and `PNextChain::get` hands that back
indistinguishably from a driver answer. Since
`InstanceCreateInfo::api_version` **defaults to `V1_0`**, this was the default
path, not an exotic one.

**An all-zero UUID is not a conservative reading of a join key.** The
documentation calls `device_uuid` the join key for correlating a
`VkPhysicalDevice` with NVML / CUDA / OpenGL, and zeros are a *valid-looking*
UUID that every device shares. Measured on one machine with two GPUs
(2026-09-03): on a `V1_0` instance an AMD Radeon 610M and an NVIDIA RTX 4070
**both** reported `00000000000000000000000000000000`, so they compared
**equal**; at `V1_1` they are distinct. A caller matching by UUID would have
selected the wrong GPU, or matched all of them.

Callers who need the identity should raise `InstanceCreateInfo::api_version`
to `V1_1` or higher — the same remedy `subgroup_properties()`,
`driver_properties()` and `shader_arithmetic_features()` already document,
each of which gates on the effective version for exactly this reason. This
method did not, and was the only one of the five that didn't.

Two details worth recording, because neither is visible from the fix:

- **The hazard was already understood three lines above the unguarded read.**
  `VK_EXT_pci_bus_info` is deliberately not chained unless the device
  advertises it, with a comment saying an unimplemented extension "leaves the
  struct untouched, so chaining it unconditionally would report a bogus
  `0000:00:00.0` instead of an honest `None`." The same reasoning, by the same
  hand, applied to the optional field and missed on the mandatory one.
- **The existing test named the exact failure signature and declined to assert
  it.** Its comment read "a real device never reports the all-zero UUID;
  software rasterizers may, so we only print rather than assert on it," and it
  ended "the only structural invariant to check is that the call returned." It
  also ran on the shared `V1_0` bootstrap, so it had been printing the zeroed
  struct as a result. It now builds a `V1_1` instance and actually exercises
  the surface.

The new regression test asserts the **gate**, not the UUID's content, so it is
independent of what any driver reports and a software rasterizer cannot make it
flaky. It carries its own positive control: a `V1_1` instance must produce at
least one identity, or the `V1_0` null proves nothing and the test skips
instead of passing.
## [0.14.1] — 2026-09-03

A patch release for one reason: **0.14.0's changelog describes a repair its
published artifact does not contain.** A consumer whose build hits the
no-spec path on 0.14.0 gets an unreadable error and a changelog telling them it
is fixed. That is a defect in the record as much as in the code, and it costs
one patch release to remove.

### Fixed — the build-failure message reached dependents as one escaped line

Found by running the error path rather than reading it. `fn main() -> Result<_, E>` reports with
`{:?}`, and a `Box<dyn Error>` built from a `String` debug-prints it **quoted, with every newline
escaped**. The two-audience message written in 0.14.0 was formatted, compiled, and then flattened
to a single unreadable line at the last step — for the reader who has nothing else to go on.

Nothing failed, because a build script that errors is already failing; only legibility was lost.
A `PlainError` newtype whose `Debug` is its `Display` restores it: 18 lines instead of 1.

**A substring check cannot see this bug** — the text is present either way. The CI assertion
therefore requires the numbered option to occupy a line of its own, which is only true when the
newlines are real, and the path is now executed rather than merely compiled.

### Changed (no behaviour) — `slang.rs` uses `as_chunks` instead of `chunks_exact`

Converting a SPIR-V byte blob to words now uses `as_chunks::<4>` rather than `chunks_exact(4)`
with `c[0]..c[3]` indexing. **The output is byte-identical** — verified across five inputs
including the SPIR-V magic word, an empty blob and a 64-byte run — so nothing a consumer can
observe changes. Recorded because it is a change to a shipped file, and a reader diffing
0.14.0 against 0.14.1 would otherwise find it unexplained.

⚠️ **One consequence is worth knowing if you reason about our MSRV.** `slice::as_chunks` was
stabilised in **1.88**, so it is now an independent driver of this crate's declared floor
alongside the let-chains and `libloading 0.9`. Anyone who concluded from the let-chains alone that
1.88 could be lowered by rewriting them now has a wrong model. The floor is unchanged at 1.88 and
the `msrv-minimality` job continues to require this crate to *fail* to compile on 1.87.

## [0.14.0] — 2026-09-02  (kiss-vulkan-vocab 0.4.0, vulkane_derive 0.1.1)

### Fixed — the normative vectors pinned no set-valued field

Every vector in the vocabulary manifest carried `ops-none` and `arith-none`. So the manifest
pinned both fields' **alphabets** while pinning nothing about **how a set with more than one
member is written down** — and those are different questions.

Found by a consumer, not here. Unpopped asked whether two arithmetic capabilities spell
`arith-f16i8`, `arith-f16-i8`, or a repeated field. The answer is normative in the namespace
document — *"This field uses explicit `-` separators and MUST NOT juxtapose"* — but **the
machine-readable artifact, which is the thing a producer validates against, could not answer.**
They declined to infer it from the single-member example in their tree, which was the right call.

Two vectors added, one per set-valued field, because **the two fields do not spell alike**:
`<ops>` juxtaposes single letters, `<arith>` joins variable-length names with `-`. A vector for
one says nothing about the other, and an implementer generalising from `<arith>` alone would emit
`ops-a-b-r`.

```
arith   input ["i8", "f16"]        ->  arith-f16-i8
ops     input ["w", "l", "b"]      ->  ops-blw
```

Each input is **non-canonical on purpose**, so one vector pins two things a parser cannot infer
from an alphabet: the join, and the canonical order. Order is load-bearing because §6.8-0002
matching is byte-exact — `arith-i8-f16` is a different cell, not a differently-written same one.

**The input members are derived from the token rather than passed beside it.** A first draft took
them as a separate argument and I immediately wrote members that did not match the flags: the
vector would have taught a reader that `{r,a,b}` spells `blw`. Deriving them makes the two halves
of a vector incapable of disagreeing, and a `> 1` assertion refuses a "set" vector carrying one
member, which would pin nothing.

**No `vocabulary_version` bump.** No token changes, and no producer acquires an obligation it did
not have — V-6 already required this. The vectors make existing non-conformance detectable rather
than creating new non-conformance.

### Fixed — a capability could be NAMED without being derivable, and the test that said otherwise

The vocabulary's obligation runs two ways and only one was checked:

- **Direction A** — every capability the device reports is named in the token.
- **Direction B** — every name the vocabulary offers is one some device can produce.

Nothing checked B, and its absence is invisible by construction: a phantom name spells,
parses and round-trips correctly, so every parse, format and manifest test passes. That is
the defect vocabulary version 4 shipped with for the packed component types.

**Demonstrated before it was fixed.** Adding an arith name `PHANTOM` — a capability no
Vulkan device can report — and regenerating the manifest left the freshness gate **green**:
it compares the manifest against the emitter that produced it, and a phantom is perfectly
consistent with its own source.

`kiss-vulkan-vocab` gains `Arith::all()` and `OpClasses::all()`, derived from the existing
name tables rather than restated. `vulkane::kiss` lifts its derivation into tables it
actually iterates, so what the deriver COVERS is a value, and a new unit test compares that
value against `all()`. Neither side is a copy, so neither can drift, and no source scanning
is involved.

**A test was renamed because it asserted the guarantee it did not provide.**
`v5_arith_names_are_derivable_from_the_device_not_merely_spellable` walks the *device's*
feature bits — direction A — and passed with a phantom in the vocabulary, because a phantom
bit is never set and no iteration looks at it. It is now
`v5_arith_names_the_device_reports_are_spelled_in_the_token`. A test named for a guarantee it
does not give is worse than an absent one: nobody reading the name goes looking for the gap.

### Changed — clause citations in `kiss-vulkan-vocab` name their spec

Every `§` citation in the exported crate now reads `KISS-Classify §…` rather than a bare
clause id. **A bare id does not identify a clause.** `6.8-0002` is *defined* in at least three
KISS specs and means something different in each — token matching in Classify, the ULP oracle
in Conform, a typed decline in Emit. So a reader resolving a bare cite can land on a real
clause that says something unrelated, which is worse than failing to find one.

**Two of the twenty-three were in `Display` output, not comments**, and those are the ones
that mattered most: an error message is where a reader with no context meets a citation.
`ParseError::Colons` and `ParseError::Charset` now name the spec in their messages. No test
asserted on either string, and the emitted manifest is byte-identical.

Scoped to the exported crate deliberately. The superseded namespace proposal keeps its bare
cites — it already carries a banner saying not to build against it — and **the CHANGELOG's own
historical entries are left verbatim**, because a changelog records what was said at the time
and correcting a citation inside one rewrites history rather than fixing a document.

### Changed (breaking) — `vulkan:` vocabulary version 5 names three arithmetic capabilities

`<arith>` gains `i16`, `i64` and `f64`, naming the core `VkPhysicalDeviceFeatures` bits
`shaderInt16`, `shaderInt64` and `shaderFloat64`. `Arith` gains `INT16`, `INT64` and `FLOAT64`,
and `vulkane::kiss` derives all three from `supported_features()`.

**This changes the bytes of almost every token in the wild.** Unlike FP8 or the packed types, these
are not exotic — an RTX 4070 and an AMD Radeon 610M both report all three, so a token derived from
either changes. §6.8-0002 matching is byte-exact, so a version-4 token and its version-5 equivalent
**do not match**: caches keyed on version-4 tokens are stale and must be **invalidated, not
migrated**. A device reporting none of the three is unaffected.

That this bumps rather than adds is the §4 additive test run mechanically: all three are core
feature bits assigned long before the registry baseline `VK_HEADER_VERSION` 348, so a conformant
device could always have reported them. Version 5's own baseline is 348. They land together because
each would otherwise force a version of its own.

**Do not infer `i16` from `st16`.** `st16` is `storageBuffer16BitAccess`, a *storage* capability;
a conformant device may accept 16-bit data in a buffer and perform the arithmetic in f32. Rule V-15
in the namespace document forbids the inference, and the live test matches whole names for the same
reason — `contains("i16")` would be satisfied by the `st16` beside it in the same field.

Namespace document: `spec/namespaces/vulkan.md` at vocabulary version 5, rule V-15
([KISS#262](https://github.com/ThinkersJournal/KISS/pull/262)).

### Changed (breaking) — `Arith` widens to `u16`

`Arith(pub u8)` becomes `Arith(pub u16)`. Five bits were in use and the three new names fill a `u8`
exactly, so the next arithmetic name would have forced a breaking representation change on its own
account. Riding a break already being taken, for the same reason `ComponentType` became
`#[non_exhaustive]` during the version-2 cut.

### Added — the `vulkan:` namespace publishes a machine-readable vocabulary manifest

`kiss-vulkan-vocab/manifest/vulkan-vocabulary.json`, the KISS-CLASSIFY §6.8-0008 form of
`spec/namespaces/vulkan.md`, so a consumer binds against an artifact instead of hand-transcribing an
annex or hand-parsing its prose. `kind: "generated"`, emitted by
`examples/emit_vocabulary_manifest.rs` and byte-compared against the committed copy by
`tests/vocabulary_manifest.rs`.

The `vectors` array is the normative part, not `grammar`: two of the five fields choose their
encoding by a length-conditional switch, which no alphabet or regex expresses, so a consumer binding
only against the grammar can recognise every token and still produce the wrong one. Threshold
vectors sit at exactly 512 and 513 bytes for **both** length-conditional fields, which pins the
switch as `>` rather than `>=`.

New public API on `kiss-vulkan-vocab`: `VOCABULARY_VERSION`, `ComponentType::spelling`,
`OpClasses::alphabet`, `Arith::alphabet`, and `CoopMatrix`/`CoopVector::canonical_enumeration`.

### Changed (breaking) — the declared MSRV was never true; it is now measured

`vulkane` and `vulkan_gen` declared `rust-version = "1.85"` and **no consumer on 1.85 could
ever have compiled either of them**. Both now declare **1.88**, which is the floor that was
there all along. Two independent reasons, both predating the declaration:

- `vulkan_gen` uses **let-chains** (`if let ... && let ...`), stable only in 1.88. It is a
  *build-dependency* of `vulkane`, so this is on the consumer's compile path, not just ours.
- **`libloading 0.9` declares `rust-version = 1.88.0`** itself, so resolution fails before a
  single line is compiled.

The second is a property of the lockfile and the first is a property of our source, so the floor
does not move even for a consumer who resolves an older `libloading`.

`kiss-vulkan-vocab` and `vulkane_derive` **stay at 1.85** — measured, not assumed. Neither pulls
in a Vulkan loader, and a per-crate floor is worth more to a consumer than one number for the
workspace.

A new `MSRV` CI job builds every publishable crate at its own declared floor, **reading the
version out of `cargo metadata` rather than restating it in the workflow**. A hardcoded `1.88`
in CI would be a second copy of the promise, and the two drift the moment somebody edits the
manifest — which is precisely how this claim survived unexamined for as long as it did. The job
**builds and does not test**: a consumer compiles this crate, they do not run its dev-tests, and
gating on dev-dependencies would drag the floor up for reasons unrelated to the promise.

### Changed — the build toolchain is pinned, and it is not the same claim as the MSRV

`rust-toolchain.toml` pins **1.98.0** with `rustfmt` and `clippy`. This is the toolchain the
repo is *verified with*; the per-crate `rust-version` is the floor a *consumer* may compile
with. Neither number implies the other and CI now exercises both.

The pin exists because `stable` is a moving target resolved on different days on different
machines. A local `+stable` at 1.97.1 certified a branch that CI's `@stable` at 1.98.0 then
failed, on a lint that does not exist in 0.1.97 — so a green local run was not evidence about
CI at all.

`components` is not optional. Components are installed per toolchain *name*, so a bare
`channel` pin gets a toolchain without `rustfmt` or `clippy` and reds both jobs.

The MSRV jobs deliberately escape the pin: they pass an explicit `+<version>`, rustup's
highest-precedence selector. Verified rather than assumed — a pin that captured them would
have tested 1.98 under four leg names saying 1.85 and 1.88, which is a vacuous pass wearing
four green ticks. The existing toolchain-confirmation step is what would catch it.

### Fixed — `LICENSE-APACHE` was not the Apache License 2.0

All five copies diverged from the canonical text in two substantive places: §6 dropped "reasonable
and customary use in" from the trademark carve-out, and §9 substituted "Support" for "Additional
Liability" in both the section title and its operative clause — stating a narrower scope than the
licence it named. Replaced from `apache.org` and verified per-file against
`sha256 cfc7749b96f63bd3…`, including the blobs git stores rather than only the working tree.

**vulkane 0.13.0, kiss-vulkan-vocab 0.3.0 and vulkan_gen 0.5.0 shipped the wrong text and cannot be
corrected** — registry versions are immutable.

### Fixed — generated artifacts are pinned to LF

`.gitattributes` covered `LICENSE-*` only, so on a Windows checkout the vocabulary manifest and the
device fixture were stored as LF and checked out as CRLF. The committed manifest was therefore **not
byte-identical to a fresh emission**, and the emit-and-`git diff --exit-code` freshness gate
§6.8-0011 asks for could not be armed at all. The freshness test had been normalizing CRLF before
comparing, so it passed against a string neither party had on disk.

### Changed — versions

`kiss-vulkan-vocab` 0.3.0 → **0.4.0** and `vulkane` 0.13.0 → **0.14.0**, both breaking.

**0.4.0 rather than 0.3.x is the load-bearing part.** Vocabulary version 5 changes token bytes, and
under Cargo's 0.x rules `0.3.1` is *compatible* with the published `0.3.0` — so a consumer
requesting `kiss-vulkan-vocab = "0.3"` would have received version-5 tokens from what looks like a
patch upgrade. A breaking change wearing a compatible version number is exactly the silent kind:
nothing fails, the tokens simply stop matching the ones already in a cache.

## [0.13.0] — 2026-08-15  (kiss-vulkan-vocab 0.3.0, vulkan_gen 0.5.0)

**First release since 0.10.1.** Versions 0.10.2, 0.11.0 and 0.12.0 were prepared and their entries
are recorded below, but they were **never published to crates.io** — the `vulkan:` vocabulary work
they carried was deliberately held until the token set could be cross-verified against a published
KISS artifact rather than against our own tests. 0.13.0 therefore rolls all of it up, and everything
under 0.10.2 and older in this file is included here.

**Cross-verified against KISS `origin/main` commit `b27e858` (2026-08-15).** At that commit
`spec/namespaces/vulkan.md` is *Vocabulary version 4* with clause V-1 requiring exactly five
`.`-separated fields, and `conformance/corpus/structure_key_vectors.json` carries
`namespace_vocabulary_versions = {"cuda": 1, "vulkan": 4}` together with the token
`vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none`. `kiss-vulkan-vocab`'s
`the_pinned_vector_matches_the_kiss_artifact_when_reachable` parses and re-spells that token
byte-exactly, reading it via `git show origin/main:…` — the *published* ref rather than a local
working tree, which could otherwise pass against an edit nobody pushed.

**This release run set `KISS_REQUIRE_PUBLISHED_REF=1`**, which turns the working-tree fallback into
a failure. The helper does still fall back when the published ref cannot be read — a contributor
whose KISS checkout has no remote is not doing anything wrong — but it now announces the degraded
mode and reports which source it compared against, and the release path refuses it outright.

That distinction is not hypothetical, and it was live while this release was being cut: the KISS
checkout on this machine sat at `b82b50b` with a working tree still carrying the **four-field
version-3 token**, while `origin/main` had moved to the five-field version-4 one. A silent fallback
would have verified this release against a stale tree. The corpus's own `coverage_note` records the
same failure in the other direction — a `vulkan:` suffix that "once byte-matched green while it was
MALFORMED against vulkan v4".

That commit is named here because a registry version is immutable and the question a reader asks
later is not "was this correct" but **"correct against what?"**. A provenance field that names the
wrong thing is unfalsifiable by anything that reads it, so this one names a commit that can be
checked. It is the only vector in the crate whose expected value has an author other than us;
every other test compares the crate to itself.

### Changed (breaking) — `vulkan:` vocabulary version 4 adds a fifth field and names the packed types

The token grows a fifth field, `<coopvec>`, spelled `cv-…`:

```
vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none
```

**Every four-field token is now invalid.** This is not a widening — `VulkanTarget::parse` requires
exactly five fields, and a version-3 token fails to parse rather than defaulting its new field.
Under §6.8-0002 byte-exact matching there is no degraded answer available: a token with the wrong
arity names nothing, so rejecting it is the honest outcome and a silently-defaulted `cv-none` would
have been a wrong cell rather than a partial one. Caches keyed on four-field tokens are stale and
must be **invalidated, not migrated**.

The new field enumerates the device's cooperative-*vector* combinations — five component types plus
a transpose flag per combination — canonically ordered, with the same length-triggered
`fnv1a64-<hex16>` digest escape the `<coop>` field uses above 512 bytes. `PhysicalDevice` gains
`cooperative_vector_properties()`, and `DeviceCapabilities` gains `coopvec`.

`VK_COMPONENT_TYPE_SINT8_PACKED_NV` and `VK_COMPONENT_TYPE_UINT8_PACKED_NV` are named `i8packed`
and `u8packed`.
`ComponentType` gains `S8Packed` and `U8Packed`, and `kiss::component()` maps both — a device
reporting them previously derived `x1000491000` / `x1000491001`.

Both changes ride one version bump deliberately. The packed enumerants are assigned at
`VK_HEADER_VERSION` 348, the registry baseline recorded for version 3, so under the namespace's §4
additive test naming them requires a bump on its own; folding them in with the fifth field costs one
version instead of two. **Version 4's own baseline is 348** — that is the number the next bump diffs
against.

These types are reachable *only* through the cooperative-vector query. Cooperative-matrix properties
never report them, which is why the query had to exist before the names could be derived at all,
and why this was verified against hardware rather than argued: an RTX 4070 reports both values in
`inputInterpretation` while its cooperative-matrix properties contain neither.

### Changed (breaking) — `vulkan:` vocabulary version 3 names FP8

`VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT` and `VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT` now derive
`f8e4m3fn` and `f8e5m2` instead of falling through to the `x<n>` escape. `ComponentType`
gains `F8E4M3FN` and `F8E5M2`, and `vulkane::kiss::component` maps both.

**This changes the bytes of tokens already in the wild.** A device reporting FP8
cooperative-matrix shapes previously derived `x1000491002` / `x1000491003`; it now derives
the named spellings. §6.8-0002 matching is byte-exact, so an old token and its version-3
equivalent **do not match**: caches keyed on version-2 tokens for such a device are stale
and must be **invalidated, not migrated**.

That this is a version bump rather than an additive change is not a judgement call. The
registered namespace's §4 additive test says to compare against the registry baseline
recorded for the previous version — `VK_HEADER_VERSION` 348 — and both enumerants are
already assigned at 348 (value `1000491002`, extension 492's block). Assigned at the
baseline means a conformant device could already have reported it, so a derivable token
could already have been affected by the absence of the name. **Bump.** Devices with no FP8
support are unaffected: their tokens never contained these values.

The `fn` suffix is mandatory and load-bearing. `f8e4m3fn` is the OCP OFP8 finite variant;
`f8e4m3fnuz` is a different format that KISS **reserves with no computation semantics**, so
a deriver must never emit it. `kiss-vulkan-vocab` now asserts both directions — no named
component spells a reserved dtype, and a reserved spelling does not parse as a named
component — because a `fnuz` slip is four characters that every round-trip test in the
crate would otherwise sail past.

`VK_COMPONENT_TYPE_FLOAT_E4M3_NV` / `..._E5M2_NV` are registry *aliases* of the EXT
enumerants, so one mapping covers both; a test pins that, since a future registry splitting
them would silently return NV-only drivers to `x<n>`.

Namespace document: `spec/namespaces/vulkan.md` at vocabulary version 3, rules V-10 and V-11.

### Changed (breaking) — `vulkan:` tokens spell signed integers with an `i` prefix

Part of the KISS `sk4` coordinated schema event. The registered `vulkan:` namespace was amended to
**vocabulary version 2** ([KISS#130](https://github.com/ThinkersJournal/KISS/pull/130)), taking
signed-integer component types from `s8`/`s16`/`s32`/`s64` to `i8`/`i16`/`i32`/`i64` to match the
`i` prefix used by KISS-Classify §6.1's `structure_key` dtype set. `kiss-vulkan-vocab` follows it
here.

**Every token naming a signed-integer component type changes bytes.** §6.8-0002 matching is
byte-exact and forbids subset or implication logic, so a version-1 token and its version-2
equivalent **do not match**. Any cache keyed on a version-1 token is stale and must be
**invalidated, not migrated** — a lookup with an old key will simply miss, and a store under an old
key will never be found again. Token length is unchanged, since each name keeps its width.

The document was amended **before** the crate, and this crate's
`tests/registered_namespace.rs` guard moved in the same commit as the `ComponentType` arms.

**Variant names are unchanged and deliberately do not match their tokens.** `ComponentType::S8`
now spells `i8`. The variant is named for its Vulkan source (`VK_COMPONENT_TYPE_SINT8_KHR`); the
token follows the KISS-facing wire vocabulary. Do not "fix" either to match the other.

### Changed (breaking) — `ComponentType` is `#[non_exhaustive]`

Enum hardening, riding this breaking cut so that future dtype additions do not force another one
(KISS `sk4` RFC §10). Downstream `match` expressions over `ComponentType` now require a wildcard
arm.

**Migration note, required by RFC §10 — this trades a build break for a silent behaviour change.**
Because new variants can now land without a major version, a value that previously arrived as
`ComponentType::Other(n)` and matched a caller's catch-all will, once it is named, match the new
variant instead — with **no compile error to announce it**. If you branch on `Other(n)` for a
specific `n`, that branch can stop being taken by a dependency upgrade alone. This is the same
reclassification hazard as the `bf16` fix below, made permanent as a property of the type.

### Changed — versions

`kiss-vulkan-vocab` 0.1.0 → **0.3.0**, `vulkan_gen` 0.3.0 → **0.5.0**, and `vulkane` 0.10.1 →
**0.13.0**. All breaking, and all published together: the intermediate bumps recorded during this
work (`kiss-vulkan-vocab` 0.2.0, `vulkane` 0.11.0 / 0.12.0) were never released, so the
crates.io jump is straight from the last published versions.

The FP8 variants noted as outstanding here during development are **shipped** — see the version-3
entry above. They were **resolved by exclusion**, not by the authored §6.1 layout⇒variant table
that was originally expected to unblock them: KISS reserves the `fnuz` spellings with no
computation semantics, which leaves `fn` as the only spelling either Vulkan enumerant can denote.
Adding them was *additive* to the Rust type rather than breaking, thanks to `#[non_exhaustive]`
above — which is the payoff that change was made for. The *token* bytes still changed, which is
what drove the vocabulary version bump.

### Fixed (soundness) — driver-written enum fields could hold an invalid discriminant

Reading a Rust `enum` whose memory holds a discriminant outside its declared set is **undefined
behaviour**. Every struct that `vk.xml` marks `returnedonly="true"` is filled by the implementation, and
an implementation may report a value this `vk.xml` has never heard of — a component type or driver
ID from an extension newer than the pinned spec. The UB was therefore reachable by **upgrading a
graphics driver**, with no application change and no error path to observe.

The generator already emitted a checked `from_raw(i32) -> Option<Self>` on every enum. Typing the
field as the enum guaranteed it could never be called: the check existed and enforced nothing.

**Driver-written enum fields are now emitted as raw `i32`** — 81 fields across the `returnedonly`
structs — so the conversion becomes an explicit decision. Two deliberate exclusions: `sType` keeps
its enum type (the application writes it even in a `returnedonly` struct, which is how a `pNext`
query chain is assembled), and pointer/array fields are untouched (reading through a raw pointer is
already `unsafe` and carries its own contract).

Application-written structs are unaffected. Roughly 1,500 enum-typed fields keep ergonomic enums
and exhaustive matching, because an application only ever writes values it obtained from the enum.

**Breaking.** Safe-layer accessors that read these fields now return `Option<T>`, each paired with a
`_raw` accessor:

- `CooperativeMatrixProperties::{a,b,c,result}_type()` and `scope()` → `Option<_>`, plus
  `*_raw() -> i32`
- `PhysicalDeviceProperties::device_type()` → `Option<PhysicalDeviceType>`, plus `device_type_raw()`
- `SurfaceFormat::format()` / `color_space()` → `Option<_>`, plus `format_raw()` / `color_space_raw()`
- `DriverProperties::driver_id` → `Option<VkDriverId>`, and the new field `driver_id_raw: i32`

The pairing is load-bearing, not convenience. The checked form is **lossy by construction**: every
unrecognized value collapses to `None`, so two different unknown driver IDs become indistinguishable
through it, and a shader cache keyed on that would reuse one driver's binaries for another. Key
caches on the raw value; gate workarounds on the named one.

`Swapchain::pick_surface_format` now skips formats this build cannot name, and returns
`ERROR_FORMAT_NOT_SUPPORTED` if none remain. Use the `_raw` accessors and build the swapchain
directly if you need one of those formats.

### Fixed — `bfloat16` was spellable but not derivable from a device

`kiss::component()` maps a raw `VkComponentTypeKHR` to the KISS vocabulary's `ComponentType`.
It covered the base set (values 0–10) but had no arm for `VK_COMPONENT_TYPE_BFLOAT16_KHR`, so a
driver reporting bfloat16 cooperative matrices yielded `ComponentType::Other(1000141000)`.

`ComponentType::BF16` and its `bf16` token have always existed and round-trip correctly, which is
why the vocabulary crate's tests passed throughout — they construct the variant directly. Nothing
exercised the derivation from a live device, so the gap sat between two things that were each
individually fine.

**This is an `Other(n)` → named reclassification.** Downstream code matching `ComponentType` with
a wildcard arm compiles unchanged and behaves differently: a bfloat16 cooperative-matrix shape
that previously fell into a catch-all now matches `BF16`. Tokens derived from such a device change
accordingly — which is the point, but it does mean a cache keyed on the old token is stale.

When that fix landed, four further values were left unmapped deliberately. **All four are now
named**, by the version-3 and version-4 entries above — this paragraph records why they were held,
because the reasoning is what the two bumps had to answer:

- `FLOAT8_E4M3_EXT` / `FLOAT8_E5M2_EXT` were blocked on the KISS `sk4` schema event: the names
  denote a *layout*, KISS §3.1.5 makes the `fn`/`fnuz` suffix mandatory, and the Vulkan registry
  never says which variant is meant. **Resolved by exclusion in version 3** — KISS reserves the
  `fnuz` spellings with no computation semantics, so `fn` is the only spelling either enumerant
  can denote.
- `SINT8_PACKED_NV` / `UINT8_PACKED_NV` were held because they carry `s8`/`u8` data in a packed
  layout — a different shader-side contract from the unpacked types — and folding them onto
  `S8`/`U8` would have collapsed two distinct Vulkan values onto one token, letting a packed-only
  device satisfy a target asking for plain `s8`. **Resolved in version 4 by giving them their own
  spellings** (`i8packed` / `u8packed`) rather than an alias, which keeps the two contracts
  distinct while making the values derivable.

`ComponentType` is now `#[non_exhaustive]` (see above), so naming a value no longer requires a
coordinated major on its own account — the version bump is driven by the token bytes changing, not
by the Rust type.

## [0.10.2] — 2026-08-07

Found by a sweep for unfinished work prompted by Fuel and Baracuda each turning up
undocumented TODOs, one of which had caused several bugs before anyone spotted it. Grepping
`TODO` in this repo is a false all-clear — the deferred work here is written as prose
(`// … for now`), and that is where the vertex-format bug below was hiding.

### Fixed — `#[derive(Vertex)]` gave signed-integer fields a float format

`[i32; 3]` mapped to `Format::R32G32B32_SFLOAT` because no `R32G32B32_SINT` constant existed
when the derive was written. Both formats are 12 bytes with identical vertex-buffer layout, so
this compiled, passed the validation layers, and produced no error at any point — the GPU simply
reinterpreted each integer's bit pattern as IEEE-754. A vertex attribute of `[-1, 0, 1]` reaches
the shader as roughly `[-1e-45, 0.0, 1.4e-45]`.

- `[i32; 3]` now maps to `R32G32B32_SINT`.
- `[i32; 4]` is now supported, mapping to `R32G32B32A32_SINT`. It previously failed to compile.
- `Format::R32G32B32_SINT` and `Format::R32G32B32A32_SINT` are new public constants, and both
  are present in `Format::bytes_per_pixel` (12 and 16). The size lookup had the same gap, so
  adding the constants alone would have moved the defect rather than closed it.

Anyone deriving `Vertex` on a struct with an `[i32; 3]` field was reading garbage in the shader;
after upgrading, that attribute delivers the integers it always should have. Shaders written to
compensate for the old behaviour by declaring the input as `vec3` will need to change to `ivec3`.

### Fixed — the bindings generator emitted its fragments in a nondeterministic order

`CodeAssembler::resolve_dependencies` seeded its topological sort by iterating a `HashSet`, whose
order varies per process. Consecutive builds of the same `vk.xml` therefore produced different
`vulkan_bindings.rs` files.

Item order carries no meaning in Rust, and the emitted API was verified identical across three
differently-ordered builds — same 3,353 type definitions, same 337 deduplication decisions — so
no incorrect bindings were ever generated. The reason it still mattered: the deduplication pass
keeps whichever definition it encounters *first* and strips later ones, so the surviving
definition of any colliding pair was a function of that nondeterministic order. Generation is now
reproducible, verified byte-identical across builds.

### Changed — generator internals (`vulkan_gen` 0.4.0, breaking)

- Deduplication now determines what a fragment defines by scanning the emitted code rather than
  trusting a module's self-reported `defined_types`. Those claims are partly hand-maintained; a
  name listed but not emitted marked the type as seen and would have stripped a later fragment's
  real definition. One such phantom entry was live, and its only visible effect was a spurious
  "skipped a definition here" comment in the generated file.
- **Removed `CodeAssembler::validate_generated_code` and `AssemblerError::DuplicateType`.** The
  function was never called and could not be: fragments legitimately redefine names that earlier
  fragments emitted, which is what the deduplication pass exists to absorb — 337 times in a full
  run — so enabling it as an error would have failed the build on the normal case. Duplicate and
  undefined types are already checked by `type_integration::check_data_consistency`, which runs
  earlier over the intermediate JSON and does fail the build.
- `GeneratorMetadata` is now documented as informational. Nothing consults `priority` (ordering
  comes from the dependency graph) and nothing validates against `defined_types` / `used_types`.
- **`SafeHandlesStats` gained an `unclassified` field.** The auto-RAII generator skips handles
  that don't fit its create/destroy shape, and a skip was invisible: the build succeeded and the
  handle simply had no safe wrapper. Every handle is now sorted into hand-written, auto-wrapped,
  or explicitly excluded with a recorded reason (`KNOWN_UNWRAPPABLE`); anything else is reported
  here and logged as a build warning. Against the pinned `vk.xml` the 59 non-alias handle types
  are 30 + 25 + 4 with nothing left over, and a test enforces that rather than asserting it in a
  comment. The build stays permissive so a newer `vk.xml` still works.
- Unsuffixed hexadecimal constants are typed by magnitude instead of assumed `u32`. A literal
  wider than 32 bits previously emitted a `u32` constant that the compiler rejected as out of
  range.

### Fixed — documentation

- `CONTRIBUTING.md` was stale from the `infra-vulkan` → Vulkane rename: wrong crate name, wrong
  clone URL, Rust 1.75 against an edition-2024 crate, a `--features validation-layers` command
  for a feature that does not exist, and a doc example calling `device.allocate_memory(…)`, which
  is not a method. CMake is required by the `shaderc` and `slang` features, not by the examples.
- Removed a `VulkanTag::is_deprecated` stub from the generated tag registry that always returned
  `false`. `vk.xml` `<tag>` elements carry only `name`, `author` and `contact` — deprecation is a
  property of extensions, so the question has no answer rather than an unimplemented one.

## [0.10.1] — 2026-08-01

### Fixed — allocator memory-type selection and failure-path cleanup

Both issues surfaced from [Fuel](https://github.com/ciresnave/fuel)'s audit of a machine-level
`VIDEO_MEMORY_MANAGEMENT_INTERNAL` bugcheck (subcode `0x2D`, `DdiMapCpuHostAperture failed`) on
2026-07-31 — CPU host aperture / PCIe BAR exhaustion with ReBAR enabled. Neither is proven to
have caused that crash, but both make host-mapping pressure worse rather than better.

- **`AllocationUsage::HostVisible` no longer resolves to BAR memory when ordinary host-visible
  memory exists.** `pick_memory_type` passed identical `required` and `preferred` flag masks for
  this usage, which made `find_type`'s preference pass identical to its fallback pass — so
  selection degenerated to "first matching type wins" and the documented distinction between
  `HostVisible` and `HostVisibleDeviceLocal` was never enforced. Staging buffers could therefore
  land in the PCIe BAR window, whose exhaustion surfaces as a driver-level failure rather than a
  catchable `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

  Selection now *avoids* `DEVICE_LOCAL` for `HostVisible` as a preference, falling back when no
  other host-visible type exists (UMA and fully-host-visible-VRAM layouts still work). Note this
  is narrower than it may sound: the Vulkan spec already requires a memory type whose
  `propertyFlags` are a strict subset of another's to be enumerated first, so on most layouts the
  plain host type was already winning. The exposure is layouts where the flag sets are
  *incomparable* — `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` versus
  `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, neither a subset of the other — because there the
  spec imposes no ordering and a conforming driver may enumerate the BAR type first.

- **`vkDeviceMemory` is no longer leaked when `vkMapMemory` or sub-allocation fails.** Three
  allocation paths (default pool growth, custom pool growth, dedicated allocation) allocated a
  block and then propagated a later failure with `?`, stranding the handle: nothing owned it, so
  neither `Drop` nor `destroy_pool` could reclaim it. Blocks are 64 MiB, or 256 MiB on heaps
  ≥ 4 GiB. The reachable trigger is the damaging one — `VK_ERROR_MEMORY_MAP_FAILED` is what a
  driver reports when it cannot satisfy a host mapping, so the leak sat on the recovery path and
  made each retry likelier to fail than the last. Two of those paths could also strand a *live
  mapping*, holding host aperture rather than merely heap space.

Selection is covered by unit tests over synthetic memory layouts (`pick_memory_type_in`), including
the incomparable-flag-set case that no single machine exhibits. The leak fix is structural and
verified by inspection — forcing `vkMapMemory` to fail would require a mock ICD.

## [0.10.0] — 2026-07-31

Additive: a new optional feature and one new query. Nothing existing changes behaviour.

The `vulkan:` namespace of KISS-Classify §6.8 was registered to this crate on 2026-07-31
([KISS #120](https://github.com/ThinkersJournal/KISS/pull/120)), with the vocabulary pinned in
`spec/namespaces/vulkan.md` there. This release ships the two halves that implement it.

### Added — KISS `vulkan:` target_capability derivation (`kiss-target` feature)

- **New sibling crate [`kiss-vulkan-vocab`](kiss-vulkan-vocab/)** — the `vulkan:` capability-set vocabulary for KISS-Classify §6.8 `target_capability` tokens: canonical spelling, parsing, and byte-exact comparison. **Zero dependencies and no Vulkan linkage**, because KISS-CLASSIFY-6.9-0003 forbids producing or parsing a token from loading a compute driver — a conformance implementation must manage with its standard library alone. That constraint is enforced structurally by [`tests/zero_dependency.rs`](kiss-vulkan-vocab/tests/zero_dependency.rs), which reads the manifest and fails the build if a dependency table is ever non-empty, rather than by a comment nobody checks.

  Token grammar: `vulkan:<subgroup>.<ops>.<arith>.<coop>` — four fixed-position fields separated by `.`, parts within a field by `-`. Every set is canonically sorted and every field always present, so two independent implementations spelling the same target produce identical bytes, which §6.8-0002 requires (it permits no matching tolerance whatsoever). Legal-but-non-canonical spellings are **rejected rather than normalized**: under byte-exact matching, accepting two spellings of one target would let them silently fail to match each other. Long cooperative-matrix shape lists fall back to an `fnv1a64-<hex>` digest, triggered strictly by the encoded length of the canonical enumeration (`COOP_DIGEST_THRESHOLD`, 512 bytes — the same string the digest hashes, so there is nothing to define twice) and never by implementation preference — a preference-driven switch would let two honest derivers emit different tokens for the same target.

- **`PhysicalDevice::shader_arithmetic_features() -> Option<ShaderArithmeticFeatures>`** — `shaderFloat16` / `shaderInt8` (Vulkan 1.2 core) plus the 16-/8-bit storage-buffer access features, via `vkGetPhysicalDeviceFeatures2`. These gate whether a half-precision or quantized kernel can exist on a device at all, making them a specialization axis rather than a tuning knob. Compute precision and *storage* precision are reported separately — a device may accept 16-bit data in a storage buffer while doing the arithmetic in f32. Gated on `effective_api_version()` with the same honest-`None` discipline as the other property queries. New public type `safe::ShaderArithmeticFeatures`.

- **`vulkane::kiss` module** (behind the new optional `kiss-target` feature, off by default) — derives `vulkan:` tokens from a live `VkPhysicalDevice`. The API is deliberately **not** `device -> token`: a `target_capability` names the specialization a kernel was *built for*, not the device's capability envelope, so on a device with a 32..=64 pinnable range a wave32-pinned and a wave64-pinned kernel are different cells and an envelope-shaped token would collide them. Instead `DeviceCapabilities::of()` reads the envelope, `admissible_subgroups()` enumerates the choice axis, `target_for()` spells one concrete token per choice, and `admits()` answers the capability question — kept separate from token matching, since §6.8-0002 forbids a consumer from applying subset or implication logic when matching.

  Verified on live hardware: an AMD Radeon 610M yields three distinct tokens (`sgdyn`, `sg32`, `sg64`) from one device — the "a device admits a *set* of tokens" property demonstrated rather than asserted — with derivation proven deterministic across repeated reads, since driver-reported cooperative-matrix order is not guaranteed stable and an unsorted list would produce a token that differs run to run.

## [0.9.0] — 2026-07-30

Minor bump rather than a patch because `cooperative_matrix_properties` loses its `unsafe`
qualifier. The change only *loosens* a caller's obligations and is semver-compatible, but it
alters a public function signature, and callers who wrapped the call in `unsafe { .. }` will
newly see `unused_unsafe` — which fails a build using `-D warnings`. See the migration note
below.

### Added — compilation-target capability queries

These three fill the inputs a caller needs to describe *what a Vulkan device specializes a compute kernel for*. They are the Vulkan-side raw material for the `vulkan:` namespace of the KISS-Classify §6.8 `target_capability` descriptor, whose per-namespace vocabulary is owned by the namespace maintainer.

- `PhysicalDevice::subgroup_properties() -> Option<SubgroupProperties>` — subgroup ("wave"/"warp") width, supported stages, supported operation classes, and quad-in-all-stages, from `VkPhysicalDeviceSubgroupProperties` (Vulkan 1.1 core); plus `size_control: Option<SubgroupSizeControl>` (min/max pinnable subgroup size, max compute-workgroup subgroups, and which stages accept a pinned size) from `VkPhysicalDeviceSubgroupSizeControlProperties` (Vulkan 1.3 core / `VK_EXT_subgroup_size_control`). **This closes a hole Vulkane documented against itself**: [`ComputePipelineOptions::required_subgroup_size`](vulkane/src/safe/pipeline.rs) has always let a caller *pin* a subgroup size, and its own doc comment says the value must lie within `minSubgroupSize..=maxSubgroupSize` — bounds Vulkane offered no way to read. `SubgroupSizeControl::permits` / `permits_in_compute` validate a candidate size (power-of-two, in range, stage accepts a pinned size) before pipeline creation. **Note the instance-version prerequisite:** size control is Vulkan 1.3 core (or `VK_EXT_subgroup_size_control`), and the gate is on `effective_api_version()` — so `size_control` is `None` on an `Instance` created below 1.3 *regardless of what the device supports*, and `InstanceCreateInfo::api_version` defaults to `V1_0`. A caller at 1.2 sees an honest `None` and must raise the instance version to pin a subgroup size. The base `subgroup_properties()` query needs only 1.1. Subgroup width is the single most important Vulkan kernel-specialization axis: 32 on NVIDIA, 64 on AMD GCN/CDNA, *either* on RDNA, 8/16/32 on Intel. New public types `safe::SubgroupProperties`, `safe::SubgroupSizeControl`, `safe::SubgroupFeatureFlags`.
- `PhysicalDevice::driver_properties() -> Option<DriverProperties>` — `driver_id` (the `VkDriverId` naming the ICD), `driver_name`, `driver_info`, and the claimed CTS `conformance_version`, from `VkPhysicalDeviceDriverProperties` (Vulkan 1.2 core / `VK_KHR_driver_properties`). `PhysicalDeviceProperties::driver_version` is a bare `u32` whose bit-packing is *vendor-defined* (NVIDIA packs it (22,14,6,10), AMD (22,10,10,10), Intel-on-Windows (18,14)), so it cannot be decoded portably and is good only for equality. This gives a portable, legible driver identity for shader-cache keys and a proper enum for driver-quirk gating — RADV and AMDVLK drive the same hardware and do not make the same codegen choices. New public types `safe::DriverProperties`, `safe::ConformanceVersion`.
- `PhysicalDevice::effective_api_version() -> ApiVersion` — `min(instance apiVersion, device apiVersion)`, the version that actually governs property queries. A Vulkan implementation must behave as the version the *instance* requested, so an instance created at 1.0 leaves 1.1+ `pNext` property structs untouched **even on a 1.3 device**, and a caller gating on the device version alone reads a zeroed struct back as though it were an answer. Both queries above gate on this and return an honest `None` instead. Note `InstanceCreateInfo::api_version` defaults to `V1_0`, so raise it if a query declines unexpectedly.

### Changed — `cooperative_matrix_properties` is now safe

- `PhysicalDevice::cooperative_matrix_properties()` is no longer an `unsafe fn`. It previously placed the burden on the caller to have enabled `VK_KHR_cooperative_matrix`, because the Vulkan loader hands back a non-null stub for any KHR entry point whose *name* it knows and that stub can crash on a device that doesn't implement the extension (notably Mesa Lavapipe). The method now asks the device itself via `enumerate_extension_properties()` and returns an empty `Vec` when the extension isn't advertised, so the stub is never reached — the same honest-gating discipline `device_identity()` applies to `VK_EXT_pci_bus_info`. An empty `Vec` is therefore unambiguous rather than possibly-undefined.

  **Migration:** remove the `unsafe { .. }` wrapper — leaving it compiles but raises `unused_unsafe`, which bites under `-D warnings`. The redundant extension check *immediately guarding this call* can go too, but **keep the flag if it also gates anything else**: a caller who checked `VK_KHR_cooperative_matrix` once and reused that boolean for `DeviceFeatures::with_cooperative_matrix()`, `DeviceExtensions::khr_cooperative_matrix()`, or pipeline construction must retain it. Deleting the binding outright would leave the extension never enabled at device creation while this self-gating query still reports shapes from the *supported* list — a silently disabled cooperative-matrix path rather than a build error. This change also made the API runtime-testable for the first time — `safe_wrapper_test.rs` previously carried a standing note that no test could exist because calling the function on CI's Lavapipe was undefined behaviour; that note is replaced by a real test.

## [0.8.3] — 2026-06-28

### Added — physical-device identity

- `PhysicalDevice::device_identity() -> Option<DeviceIdentity>` exposes the device's stable identity for out-of-band correlation: `device_uuid` / `driver_uuid` (always, from `VkPhysicalDeviceIDProperties`, Vulkan 1.1 core), `device_luid` (`Some` only when the platform marks it valid — Windows) plus its `device_node_mask`, and `pci: Option<PciBusInfo>` (`Some` only when the device advertises `VK_EXT_pci_bus_info`). One `vkGetPhysicalDeviceProperties2` call, gated honestly: `None` when props2 is unavailable, and each sub-field is `Some` only when its source is actually present. This is the *join key* a caller needs to match a `VkPhysicalDevice` against an out-of-band GPU source — NVML by UUID, DXGI/D3DKMT by LUID, Linux sysfs (`gpu_busy_percent`) by PCI address — or against the same device seen through CUDA/D3D/OpenGL. Added because Vulkan exposes **no** cross-process GPU load / utilization / queue-depth query beyond the VRAM `memory_budget`; identity is the most Vulkane can (and should) provide toward that, with the load lookup itself living in a separate, API-agnostic layer. New public types `safe::DeviceIdentity` and `safe::PciBusInfo`.

### Added — Profile v1 conformance lock-in

- Vulkane is confirmed conformant to Fuel's **Kernel-Seam Interop Contract — Profile v1** (ratified 2026-06-20) in its **FDX-only, BDA-subset** role. No API change was required — the contract pins Vulkane to a *named surface*, all of which shipped in 0.8.2: `AllocatorOptions::buffer_device_address` / `Allocator::new_with_options`, `BufferUsage::SHADER_DEVICE_ADDRESS`, `DeviceFeatures::with_buffer_device_address`, and `Buffer::device_address`. Added [`tests/profile_v1_conformance.rs`](vulkane/tests/profile_v1_conformance.rs), a compile-time lock-in (mirroring the `Send + Sync` lock-ins on `Queue` / `CommandBuffer`) so a future rename, removal, or signature change of any named-surface item fails Vulkane's CI rather than `fuel-vulkan-backend`'s build. This operationalizes the contract's §7.2 rule that *a Vulkane major bump triggers a re-check of the named surface* — the surface is pinned by behavior, not by a `>= 0.8.2` version floor.

## [0.8.2] — 2026-06-19

### Added — device-address-capable allocator

- `Allocator::new_with_options(device, physical, AllocatorOptions { buffer_device_address: true })` makes every `VkDeviceMemory` block the allocator allocates carry `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. Buffers sub-allocated from such an allocator (created with `BufferUsage::SHADER_DEVICE_ADDRESS`) now return a valid GPU virtual address from `Buffer::device_address()`. Previously the flag was only set on the manual `DeviceMemory::allocate_with` path, so addresses read from pooled or `Buffer::new_bound` buffers were invalid on strict drivers. The flag lives on the block (not the buffer) because one block backs many sub-allocations, mirroring VMA's `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT`. Requires the `bufferDeviceAddress` device feature. `Allocator::new` is unchanged (defaults to `buffer_device_address: false`). Unblocks downstream consumers (Fuel) that address tensors via `buffer_reference` in shaders, where the tensor's data pointer is a `VkDeviceAddress`.

## [0.8.1] — 2026-05-21

### Added — thread-safety markers

- `safe::Queue` is now `Send + Sync` via `unsafe impl`. Vulkan queues have no thread affinity at the API level, but the application owns external synchronization per `VkQueue` for `vkQueueSubmit` / `vkQueueWaitIdle` / `vkQueueBindSparse` / `vkQueuePresentKHR`. Callers can share `&Queue` across threads (e.g. via `Arc<Mutex<Queue>>` or a scheduler) so long as concurrent submissions on the same queue handle are prevented. Unblocks downstream consumers (Fuel) that build work on worker threads and submit through a serializer.
- `safe::CommandBuffer` is now `Send + Sync` via `unsafe impl`, for the same reason. Recording APIs take `&mut self`, so the Rust borrow checker already prevents concurrent recording on the same buffer; sharing `&CommandBuffer` across threads is sound. The per-pool external-sync contract for `vkFreeCommandBuffers` (called from `Drop`) remains the caller's responsibility.
- Compile-time `Send + Sync` lock-in assertions added for both types so future field additions cannot silently regress this guarantee.

## [0.8.0] — 2026-04-21

Coverage release: full ray-tracing surface, external-memory / external-semaphore interop, synchronization-2 barriers, push descriptors, dynamic rendering, descriptor-buffer binding, timeline semaphores, subgroup-size control, memory priority, and generator-emitted ergonomic safe signatures for ~545 Vulkan commands. Two latent codegen correctness bugs fixed along the way.

### Added — curated extension wrappers

- **Ray tracing** — `safe::AccelerationStructure` (BLAS/TLAS/Generic, AABB + Triangles + Instances geometry), `safe::RayTracingPipeline` with `ShaderGroup` enum (General/TrianglesHit/ProceduralHit) and `ShaderBindingRegion`, `PhysicalDevice::ray_tracing_pipeline_properties`, `CommandBufferRecording::build_acceleration_structure` / `bind_ray_tracing_pipeline` / `trace_rays`, `Device::acceleration_structure_build_sizes`. Examples: [`ray_tracing_as_build`](vulkane/examples/ray_tracing_as_build.rs) builds a live BLAS + TLAS on the local GPU end-to-end.
- **External memory / semaphore interop** — `DeviceMemory::get_win32_handle` / `get_fd`, `Semaphore::get_win32_handle` / `get_fd` / `import_win32_handle` / `import_fd`, `Win32Handle` newtype, `SemaphoreImportWin32` / `SemaphoreImportFd`. Unblocks CUDA, HIP, DX12, and DMA-BUF bridging. Example: [`external_memory_export`](vulkane/examples/external_memory_export.rs).
- **Synchronization 2** — `CommandBufferRecording::memory_barrier2` / `image_barrier2` / `buffer_barrier2` with 64-bit `PipelineStage2` / `AccessFlags2`.
- **Dynamic rendering** — `CommandBufferRecording::begin_rendering` / `end_rendering` with `RenderingInfo` + `RenderingAttachment`.
- **Push descriptors** — `CommandBufferRecording::push_descriptor_set` taking a `&[PushDescriptorWrite]` that hides the `VkWriteDescriptorSet` layout.
- **Descriptor buffer** (`VK_EXT_descriptor_buffer`) — `DescriptorSetLayout::descriptor_buffer_size` / `descriptor_buffer_binding_offset` queries, `CommandBufferRecording::bind_descriptor_buffers` / `set_descriptor_buffer_offsets`.
- **Timeline semaphores** — `Semaphore::timeline_with_pnext` composes caller-supplied chains with the mandatory `VkSemaphoreTypeCreateInfo`.
- **Compute pipeline options** — `ComputePipelineOptions` carries `required_subgroup_size` (`VK_EXT_subgroup_size_control`), `specialization`, and `cache` in one bag; `ComputePipeline::with_options` is the general constructor.
- **Memory priority** — `MemoryAllocateInfo::priority: Option<f32>` auto-chains `VkMemoryPriorityAllocateInfoEXT`.
- **Shader integer dot product** — `PhysicalDevice::shader_integer_dot_product_properties() -> ShaderIntegerDotProductProperties` with `has_any_int8_acceleration()` helper.
- **pNext extension points** on every safe create-info builder: `DeviceCreateInfo::pnext`, `InstanceCreateInfo::pnext`, `MemoryAllocateInfo::pnext`, plus new `with_pnext` constructors on `Buffer`, `Image`, `Fence`, and `Semaphore`. Any unwrapped extension struct can now be layered on without dropping to raw.

### Added — generated ergonomic traits (Phase 3)

- `DeviceSafeExt`, `InstanceSafeExt`, `PhysicalDeviceSafeExt`, `QueueSafeExt`, `CommandBufferRecordingSafeExt` — auto-generated per-command methods with idiomatic Rust signatures alongside the raw Phase-2 `DeviceExt` etc. traits. **545 ergonomic methods** emitted from `vk.xml`:
  - **Slice coalescing** — `(count: u32, const T*)` pairs collapse into `&[T]` inputs. `cmd_pipeline_barrier(..., &[MemoryBarrier], &[BufferMemoryBarrier], &[ImageMemoryBarrier])` is one signature.
  - **Enumerate** — `(*mut u32 count, *mut T data)` pairs become `Result<Vec<T>>` / `Vec<T>` return types. `enumerate_physical_devices` issues the classic two-call count-then-fill idiom automatically.
  - **Single-output** — trailing `*mut T` parameters become `Result<T>` returns (`get_memory_win32_handle_khr(info: &…) -> Result<HANDLE>`).
  - **Reference input structs** — `*const T` parameters become `&T`.
  - **Scalar return passthrough** — `VkDeviceAddress` / `VkBool32` / typed handles pass through untouched (`get_buffer_device_address(info: &…) -> VkDeviceAddress`).
  - Commands with unsupported shapes (pointer-to-pointer, parallel slices sharing one count, `len` pointing inside a struct) fall through to the raw Phase-2 traits — no method emitted.

### Fixed — generator correctness

- **Nested C-array layout** — `VkTransformMatrixKHR.matrix[3][4]` was emitted as `[f32; 3]` (12 bytes) instead of `[[f32; 4]; 3]` (48 bytes). Every multi-dimensional `float matrix[a][b]` field in `vk.xml` was silently truncated. Fixed in `struct_gen::map_type_to_rust`; any ray-tracing workload using `VkAccelerationStructureInstanceKHR` was affected.
- **Transitive extension-dep walker** — `transitive_requires` harvested per-`<require>` `depends` attributes and treated them as extension prerequisites. In `vk.xml` those attributes mark *conditional* enum inclusion (e.g. "expose these extra debug-report enums if the user also enables debug_report"), not dependencies. Enabling `VK_KHR_acceleration_structure` therefore silently tried to enable `VK_EXT_debug_report` (an *instance* extension) at device creation, causing `ERROR_EXTENSION_NOT_PRESENT` on every driver. Fixed to use only the canonical top-level `requires` attribute.

### Breaking

- `DeviceCreateInfo` gained a `pnext: Option<&PNextChain>` field (default `None`). Callers using `..Default::default()` are unaffected; anyone constructing the struct with explicit named fields must add it (or switch to update syntax).
- `InstanceCreateInfo` gained the same `pnext` field.
- `MemoryAllocateInfo` gained `pnext` and `priority` fields. Direct struct-literal callers must supply both or use `..Default::default()` — the struct now derives `Default`.
- `CommandBufferRecording::memory_barrier2` / `image_barrier2` / `buffer_barrier2` return `Result<()>` (Sync2 function pointers may be absent on pre-1.3 devices without `VK_KHR_synchronization2`). Previously no sync2 methods existed, so this is only new-code exposure.
- `ComputePipeline::with_specialization_and_cache` is retained as a shim; new callers should prefer `ComputePipeline::with_options`.

### Test + example coverage

- 249 total workspace tests pass. 10 new generator pattern-matcher unit tests. 5 new live-device tests exercising generated ergonomic traits against a real driver. 2 new example programs that **run live** against the local GPU — `external_memory_export` exports a real Win32 HANDLE, `ray_tracing_as_build` builds a real BLAS + TLAS on the RT hardware.

## [0.7.0] — 2026-04-19

Allocator-side VRAM observability: the `Allocator` can now surface the driver's per-heap budget numbers in one call, fire budget-pressure callbacks when usage crosses a configurable threshold, and predictively check whether a prospective allocation would exceed the budget — all without requiring the user to opt into `VK_EXT_memory_budget` manually.

### Added

- **`Allocator::vram_budget()` / `Allocator::vram_used()`** — scalar-byte convenience helpers that sum the driver-reported budget and usage across every `DEVICE_LOCAL` memory heap. The single-number answer ML schedulers, profilers, and UI indicators typically want.
- **`Allocator::has_memory_budget_support()`** — returns `true` iff the budget numbers are authoritative (both `vkGetPhysicalDeviceMemoryProperties2` is loaded *and* `VK_EXT_memory_budget` is enabled on the device). Use this to distinguish "heap is empty" from "no query support".
- **Budget-pressure callback registry** — `Allocator::register_pressure_callback(threshold, hysteresis, closure)` fires a `PressureEvent` when a heap's `usage / budget` fraction rises past `threshold` (`PressureKind::Crossed`), falls back below `threshold - hysteresis` (`PressureKind::Relieved`), or — via `would_fit` — is projected to rise past `threshold` on a pending allocation (`PressureKind::Predictive`). Per-heap hysteresis latching prevents flutter near the threshold. Callbacks are invoked after every internal allocator lock has been released, so they may call back into the `Allocator` without deadlocking. `unregister_pressure_callback(id)` removes a registration.
- **`Allocator::would_fit(size, memory_type_index) -> FitStatus`** — proactively computes whether a forthcoming allocation would keep usage under the driver's soft budget, fires `Predictive` events for any threshold it would cross, and returns the projected heap stats (`current_usage`, `budget`, `projected_usage`, `projected_fraction`, `fits`). Lets schedulers free resources *before* attempting an allocation rather than reacting to a `Crossed` event after the fact.
- **`Device::enabled_extensions()` / `Device::is_extension_enabled(name)`** — introspect the final extension list sent to `vkCreateDevice`. Captures both explicit user requests and any extension the safe wrapper auto-enabled.
- **New documentation**: `vulkane/docs/DEFRAG_FOR_ML.md` — a dedicated walkthrough of the existing `build_defragmentation_plan` / `apply_defragmentation_plan` API aimed at ML-framework integrators, including a full worked tensor-pool compaction example and guidance on layering defrag under budget-based eviction.

### Changed

- **Device creation now auto-enables `VK_EXT_memory_budget`** when the physical device advertises it. The extension is passive — enabling it only causes the driver to populate `VkPhysicalDeviceMemoryBudgetPropertiesEXT` on `vkGetPhysicalDeviceMemoryProperties2` calls — so this is observable in `Device::enabled_extensions()` but has no runtime cost when unused. Opt-out is not currently exposed; file an issue if you need it.
- `Allocator::query_budget()` doc comment clarified: budget numbers are meaningful when `has_memory_budget_support()` returns `true`, which the auto-enable path makes the default on supported drivers.

## [0.6.0] — 2026-04-16

Major version: every Vulkan extension and feature bit is now reachable from safe code via generated builders. Layer 1 + Layer 2 + Layer 3 of the extension-handling architecture are all landed, plus **Phase 1 of Layer 4** — RAII wrappers for every previously-unwrapped Vulkan handle type.

### Added

- **`PNextChainable` trait and `PNextChain` builder** (Layer 2) — a generic pNext-chain mechanism replaces every hand-rolled pointer-patching site in the crate.
  - `PNextChainable` is implemented by the generator for every `#[repr(C)]` struct in `vk.xml` whose first two fields are `sType: VkStructureType` and `pNext` — **1225 impls** emitted from the current spec.
  - `PNextChain` owns heap-stable boxed nodes, relinks `pNext` pointers on push, and supports typed read-back (`get::<T>()` / `get_mut::<T>()`) for output-direction queries like `vkGetPhysicalDeviceMemoryProperties2` + `VK_EXT_memory_budget`.
  - Every ad-hoc pNext site in `vulkane` has been rewritten to use the chain (device creation, queue submit, semaphore create, memory allocate, memory-budget query).
- **Generated `DeviceFeatures`** (Layer 1) — `vulkan_gen` now emits one `with_<feature>()` builder method per unique feature bit across every struct that extends `VkPhysicalDeviceFeatures2`. **541 feature-bit methods** generated from the current spec. Name collisions between core-aggregate structs (`VkPhysicalDeviceVulkan12Features`) and promoted/extension structs (`VkPhysicalDeviceTimelineSemaphoreFeaturesKHR`) are resolved by routing the method to the highest-priority struct; the other path remains reachable via `chain_extension_feature()`.
- **Generated `DeviceExtensions` / `InstanceExtensions`** (Layer 3) — one `<vendor>_<ext>()` enable-method per non-disabled extension, with transitive `requires` resolved at generation time. **416 device + 44 instance** methods emitted from the current spec. Fresh extensions not yet in your copy of `vk.xml` are reachable through `enable_raw(name)`.
- **Generated RAII handle wrappers** (Layer 4 — Phase 1) — one safe, Drop-aware wrapper for every Vulkan handle type whose Create / Destroy pair fits the standard four-/three-parameter shape and isn't already covered by a hand-written wrapper. **25 new safe types** in `vulkane::safe::auto`, including `AccelerationStructureKHR`, `AccelerationStructureNV`, `MicromapEXT`, `VideoSessionKHR`, `VideoSessionParametersKHR`, `DeferredOperationKHR`, `DescriptorUpdateTemplate`, `PrivateDataSlot`, `ValidationCacheEXT`, `BufferView`, `SamplerYcbcrConversion`, `IndirectCommandsLayoutEXT/NV`, `IndirectExecutionSetEXT`, and more. Creating or destroying any of these previously required `unsafe { dispatch().vk… }` — now it's one safe call with automatic cleanup on drop.
- **`Allocation` now implements `Drop`** — a forgotten `allocator.free()` no longer leaks the slot in the TLSF pool. `AllocationInner` carries a `Weak<AllocatorInner>` back-reference, so the slot is reclaimed when the last `Arc<AllocationInner>` clone goes out of scope. `Allocator::free(allocation)` is kept for callers who prefer the imperative style — it now just `drop`s. `vulkane::safe::MemoryRequirements` is now re-exported from the crate root so call sites that build it directly don't need to reach into the buffer module.
- **Generated safe-method ext traits for every Vulkan command** (Layer 4 — Phase 2) — **600 safe methods** across 5 ext traits (`DeviceExt` 237, `CommandBufferRecordingExt` 266, `PhysicalDeviceExt` 78, `QueueExt` 15, `InstanceExt` 4). Every Vulkan command with a recognizable dispatch target now has a safe method — no `unsafe { dispatch().vkX.unwrap()(…) }` required anywhere in user code. Methods keep the `vk_` prefix (e.g. `vk_cmd_trace_rays_khr`), take raw Vulkan parameter types, and return `Result<VkResult>` for VkResult-returning commands (with error codes in `Err`, success codes like `VK_INCOMPLETE` / `VK_SUBOPTIMAL_KHR` in `Ok`). Users opt in per trait: `use vulkane::safe::CommandBufferRecordingExt;`. Ergonomic sugar (slice collapsing, typed output params, enumerate helpers) deferred to a future polish pass.
- `camel_to_snake` helper in `vulkan_gen::codegen` for consistent Vulkan identifier → Rust method-name conversion across generators.

### Breaking

- **`DeviceCreateInfo::enabled_extensions` is now `Option<&DeviceExtensions>`** (previously `&[&str]`). Migrate:

  ```rust
  // before
  let exts = [KHR_SWAPCHAIN_EXTENSION];
  DeviceCreateInfo { enabled_extensions: &exts, .. }
  // after
  let exts = DeviceExtensions::new().khr_swapchain();
  DeviceCreateInfo { enabled_extensions: Some(&exts), .. }
  ```

- **`InstanceCreateInfo::enabled_extensions` is now `Option<&InstanceExtensions>`** (previously `&[&str]`). Same migration pattern.
- **`DeviceFeatures` fields and hand-written builder methods are gone**, replaced by the 541 generated `with_<feature>()` methods. Callers who were constructing `DeviceFeatures { features11: …, features12: …, .. }` manually should use the builder instead. The generator picks names identical to pre-existing ones (`with_timeline_semaphore`, `with_buffer_device_address`, …) so most call sites are unaffected.
- **Hand-written extension-name constants removed** (`KHR_SURFACE_EXTENSION`, `KHR_SWAPCHAIN_EXTENSION`, `DEBUG_UTILS_EXTENSION`, `EXT_METAL_SURFACE_EXTENSION`, `KHR_WIN32/WAYLAND/XLIB/XCB_SURFACE_EXTENSION`). Use the generated `crate::raw::bindings::<NAME>_EXTENSION_NAME` constants or (preferred) the `<vendor>_<ext>()` builder methods.
- **`PNextChainable` requires `Clone + Default + 'static`** (previously `Default + 'static`). All `vk.xml`-generated structs derive `Clone`, so this is only a source-level break for code that hand-implemented the trait.

## [0.5.0] — 2026-04-16

### Added

- **`ShaderRegistry` for precompiled SPIR-V shaders** — new `vulkane::safe::shaders` module providing a small, shared abstraction for applications that ship compiled `.spv` artifacts (embedded via `include_bytes!` and/or loaded from disk).
  - `ShaderSource { name: &'static str, spv: &'static [u8] }` — one entry per compiled shader.
  - `ShaderRegistry::new().with_embedded(&[...]).with_env_override("MY_APP_OVERRIDE_DIR")` — builder-style setup.
  - `registry.load(name) -> Cow<'_, [u8]>` — bytes.
  - `registry.load_words(name) -> Vec<u32>` — SPIR-V word layout.
  - `registry.load_module(&device, name) -> ShaderModule` — full device-bound module in one call.
  - Runtime disk override: if the configured env var points at a directory and `<dir>/<name>.spv` exists, it is loaded instead of the embedded default; otherwise the registry falls through to the embedded table. Ideal for shader developers iterating without rebuilding the whole binary.

### Breaking

- **`Error::ShaderLoad` payload changed from `String` to `ShaderLoadError`.** The old variant preserved only a string description; the new one carries a structured enum (`NotFound` / `Io { name, source }` / `MalformedSpirv { name, byte_len }`) so consumers can match on the failure reason. Migration for manual constructions: convert `Error::ShaderLoad(format!("..."))` into the matching `ShaderLoadError` variant. Code that already used `From<ShaderLoadError> for Error` (via `?` on a `ShaderRegistry` call) needs no changes.

## [0.4.5] — 2026-04-15

### Added

- **Optional `slang` feature** — runtime Slang → SPIR-V compilation via the `shader-slang` crate (Khronos Slang compiler). Slang adds modules, generics, interfaces, and — most relevant for ML compute on Vulkan — built-in automatic differentiation: tag a function `[Differentiable]` and request forward and backward entry points from the same compiled module.
  - `vulkane::safe::slang::SlangSession::{new, with_search_paths, load_file}` — session-based API for compiling one module into many entry-point SPIR-V blobs.
  - `vulkane::safe::slang::SlangModule::compile_entry_point(name) -> Result<Vec<u32>, SlangError>`.
  - `vulkane::safe::slang::compile_slang_file(search_dir, module, entry)` — one-shot convenience.
  - Re-exports `CompileTarget`, `OptimizationLevel`, `Stage` from `shader-slang`.
  - New `Error::SlangCompile(String)` variant bridged from `SlangError`.
  - Requires `VULKAN_SDK` (SDK ships `slangc`) or `SLANG_DIR` at build/link time; `slang.dll` / `libslang.so` must be on the runtime library search path.
  - **Current limitation**: `shader-slang` 0.1.0 does not expose inline source compilation; Slang modules must live in `.slang` files resolved through session search paths. Will be lifted when a newer `shader-slang` ships.

## [0.4.4] — 2026-04-15

### Documentation

- Sync the crate-level `vulkane/README.md` (shown on crates.io and docs.rs) with the repo-root README: add the `shaderc` feature entry, the runtime-shader-compilation section, the naga-vs-shaderc selection table, and shaderc build requirements. The 0.4.3 release updated only the repo-root copy.

## [0.4.3] — 2026-04-15

### Added

- **Optional `shaderc` feature** — runtime GLSL/HLSL → SPIR-V compilation via the Khronos reference `glslang` compiler (wrapped by `shaderc-rs`). Complements the existing `naga` feature for cases that need full GLSL extension support, HLSL input, or glslang-only optimization passes.
  - `vulkane::safe::shaderc::compile_glsl(source, kind, file_name, entry_point) -> Result<Vec<u32>, ShadercError>` — common case.
  - `vulkane::safe::shaderc::compile_with_options(..., |opts| { ... })` — HLSL input, optimization level, macro defines, include callbacks, target Vulkan version.
  - Re-exports `ShaderKind`, `SourceLanguage`, `TargetEnv` from `shaderc`.
  - New `Error::ShadercCompile(String)` variant bridged from `ShadercError`.
  - Build requires either the LunarG Vulkan SDK (`VULKAN_SDK` env var), a system libshaderc, or a C++ build toolchain (CMake + Python + C++ compiler) for the source-build fallback. See README for details.

## [0.4.0] — 2026-04-10

### Added

- **45 Format constants** (up from 11) covering 8/16/32-bit, depth, and BC compressed formats. No more reaching into `vulkane::raw::bindings::VkFormat` for vertex attribute formats.
- **`Format::bytes_per_pixel()`** — returns the byte size per pixel for common uncompressed formats.
- **`BufferCopy::full(size)`** — one-liner for the common offset-0 copy case.
- **`#[derive(Vertex)]` proc macro** (new `vulkane_derive` crate, opt-in via `derive` feature) — auto-generates `VertexInputBinding` + `VertexInputAttribute` from `#[repr(C)]` structs. Supports `f32`, `[f32; 2..4]`, `u32`, `[u32; 2..4]`, `i32`, `[i32; 2..3]`, `[u8; 4]`, `u16`, `i16`. Provides both `::binding()` (vertex rate) and `::instance_binding()` (instance rate).
- New example: `derive_vertex` — instanced triangles using the derive macro.

## [0.3.0] — 2026-04-10

### Added

- **Pipeline builder extensions:**
  - `depth_bias(constant, slope, clamp)` — shadow acne prevention.
  - `depth_compare_op(CompareOp)` with `CompareOp` enum (NEVER / LESS / EQUAL / LESS_OR_EQUAL / GREATER / NOT_EQUAL / GREATER_OR_EQUAL / ALWAYS).
  - `InputRate` (VERTEX / INSTANCE) on `VertexInputBinding` — instanced rendering.
  - `color_attachment_count(n)` — multi-attachment / G-buffer pipelines.
  - `dynamic_viewport_scissor()` — resize-friendly pipelines with `set_viewport` / `set_scissor`.
- **Depth image views** — `ImageView::new_2d_depth` for depth-aspect views.
- **Image barrier aspect mask** — `ImageBarrier` gains `aspect_mask` field + `::color()` / `::depth()` convenience constructors.
- **`ClearValue` enum** + `begin_render_pass_ext` for mixed color + depth/stencil clear values.
- **Comparison sampler** — `SamplerCreateInfo::compare_op` for shadow map sampling.
- **Allocation helpers:**
  - `Buffer::new_bound(device, physical, info, flags)` — 5-step boilerplate → 1 call.
  - `Image::new_2d_bound(device, physical, info, flags)` — same for images + auto color view.
  - `Queue::upload_buffer<T>(device, physical, qf, data, usage)` — staging upload in one call.
  - `Queue::upload_image_rgba(device, physical, qf, w, h, pixels)` — image upload with layout transitions.
- New examples: `depth_prepass`, `instanced_mesh`, `shadow_map`, `deferred_shading`.

### Breaking

- `ImageBarrier` now requires `aspect_mask: u32` field. Use `ImageBarrier::color(...)` or `ImageBarrier::depth(...)` constructors.

## [0.2.0] — 2026-04-09

### Added

- **Typed pipeline stage and access mask constants** — `PipelineStage`, `AccessFlags` (32-bit), `PipelineStage2`, `AccessFlags2` (64-bit for Sync2). All barrier, timestamp, and sync APIs now accept these types instead of raw `u32` / `u64`.
- **Convenience constructors:**
  - `QueueCreateInfo::single(family_index)` — one queue, priority 1.0.
  - `WaitSemaphore::binary(sem, stage)` / `::timeline(sem, value, stage)`.
  - `SignalSemaphore::binary(sem)` / `::timeline(sem, value)`.
  - `RenderPass::simple_color(device, format, load, store, final_layout)`.
  - `Queue::one_shot(device, qf, |rec| { ... })` — fire-and-forget command recording.
- **Raw escape hatch** — `Device::dispatch()` and `Instance::dispatch()` expose the full dispatch tables for calling any Vulkan function alongside safe wrapper types.
- New examples: `buffer_upload`, `raw_interop`, `allocator_compute`.

### Breaking

- All barrier/sync API signatures changed from raw `u32`/`u64` to typed `PipelineStage`/`AccessFlags`. Migration: `0x800` → `PipelineStage::COMPUTE_SHADER`.
- `WaitSemaphore::dst_stage_mask` changed from `u32` to `PipelineStage`.

## [0.1.0] — 2026-04-08

### Added

- Initial release: complete Vulkan bindings generated from vk.xml + safe RAII wrapper covering compute and graphics end-to-end.
- **Raw bindings** (`vulkane::raw`) — all types, enums, structs, function pointers, and three-tier dispatch tables generated from the spec.
- **Safe wrapper** (`vulkane::safe`) — RAII handles for Instance, Device, Buffer, Image, ImageView, Sampler, DeviceMemory, ShaderModule, DescriptorSetLayout/Pool/Set, PipelineLayout, ComputePipeline, GraphicsPipeline (with builder), RenderPass, Framebuffer, Surface (Win32/Wayland/Xlib/Xcb/Metal), Swapchain, CommandPool/Buffer, Fence, Semaphore (binary + timeline), QueryPool.
- **VMA-style sub-allocator** — TLSF + linear pools, custom user pools, dedicated allocations, persistent mapping, defragmentation, memory budget queries.
- **Device groups** — unified single/multi-GPU device representation with per-allocation and per-submission device masks.
- **DeviceFeatures builder** — Vulkan 1.0/1.1/1.2/1.3 feature chain construction.
- **Optional `naga` feature** — `compile_glsl` + `compile_wgsl` → SPIR-V at runtime.
- **`fetch-spec` feature** — auto-download vk.xml from Khronos GitHub.
- 7 bundled examples: device_info, fill_buffer, compute_square, compute_image_invert, compile_shader, headless_triangle, textured_quad, windowed_triangle.
- Tree-based XML parser (roxmltree), vk.xml api-attribute filtering, VKSC profile exclusion.
- CI on Linux/Windows/macOS with Mesa Lavapipe for headless GPU tests.



---

# Repository log

Changes that do **not** reach a published crate: CI, the local gate harness, and
manifest comments. Kept out of `[Unreleased]` because a consumer reading that
section is being told what arrives next, and none of this ever arrives.

Verified rather than assumed — `cargo package --list` contains
`.github/workflows/ci.yml`, `.github/local_gates.py`, `.github/msrv_matrix.py`,
`.gitignore`, `CONTRIBUTING.md` and `CHANGELOG.md` **zero** times, against one
hit for `README.md` as a control. `Cargo.toml` is packaged, but the manifest cargo
GENERATES keeps only cargo's own header comments -- an author's comment
survives solely in `Cargo.toml.orig`, which nothing compiles. Measured on a
real `cargo package`: the floor reason appears 0 times in the generated
manifest and twice in `.orig`, while `rust-version` itself appears in both.
So the floor reasons below ship as bytes and not as behaviour.

Full reasoning for each lives in its pull request and beside the code it
governs. This is a pointer, not a second copy.
### Added — a floor that is not necessary is a false promise, and nothing checked

The MSRV jobs proved every declared floor **sufficient**: the crate builds there. Nothing proved
one **necessary**. The two directions fail differently and only one is loud — a floor set too low
goes red immediately, while a floor set too high is green forever and turns away consumers who
could have compiled, none of whom ever appear to say so. Same one-directional asymmetry as the
vocabulary work in 0.14.0, where every capability the device reported was named and nothing
checked that every name was producible.

`msrv-minimality` asks the other question: each crate must **fail** to compile one minor below its
declared floor.

**The obvious form of that check is unsound, twice, and both were measured before it was written.**

```
cargo +1.87 build -p vulkan_gen
  error: rustc 1.87.0 is not supported by the following package
```

That is cargo enforcing the **declared** `rust-version`. The check would prove the floor necessary
by observing that cargo obeys the floor — **the number causing the failure that justifies it.** A
check whose subject enforces its own precondition cannot return the interesting answer; it passes
on every crate in every repository whether or not the floor is real, and it does so by producing a
red exactly where a red was predicted.

```
cargo +1.84 build -p kiss-vulkan-vocab
  error: failed to load manifest for workspace member ...
```

`edition = "2024"` requires 1.85, so any cargo below the edition floor dies reading the workspace
and never reaches the code — a failure at the wrong layer reported as a finding about the right
one.

So the job passes `--ignore-rust-version`, which reaches
`error[E0658]: let expressions in this position are unstable` — a feature gate rather than
bookkeeping — and a leg is emitted only where `floor - 1` is still at or above the edition floor.

**Both 1.88 floors were measured minimal before the job existed**, so a red is a regression rather
than a first measurement, and the repair is to lower the declared floor rather than to raise the
version the job steps to. `kiss-vulkan-vocab` and `vulkane_derive` sit at edition 2024's own floor
and are minimal by construction — there is no lower version to fail at, which is an answer rather
than a gap.

**A failure is only evidence if it is the right failure.** A missing toolchain, an unreadable
manifest and a network error all exit non-zero and would otherwise read as "the floor is
necessary", so the job requires the toolchain to identify itself first and the failure to carry a
compiler diagnostic.

**Limitation, kept separate rather than folded in.** `--ignore-rust-version` ignores every
`rust-version`, including dependencies'. The leg measures whether the CODE needs the floor.
`vulkane`'s 1.88 has two independent drivers — let-chains and `libloading 0.9` declaring
`1.88.0` — and this exercises the first. Dependency-necessity is a `cargo metadata` question.

### Changed — each floor now says why, where the floor is

All four `rust-version` fields carried a bare number. The justification was prose in
`CONTRIBUTING.md`, which is not where anyone bumping a floor is looking — they are in the manifest.

The comments state the KIND of reason and carry no counts. A comment reading "7 let-chain sites"
is a measurement pinned to a moment in a place nothing re-measures: it becomes 8 the next time
someone writes `if let ... && let`, with nothing failing and nothing noticing. What keeps the
reason honest is the minimality leg, which tests whether the floor is still *necessary* whatever
the reason turns out to be — not a grep asserting the stated reason is still present, which would
be a second copy of the justification, free to rot on its own.

`CONTRIBUTING.md` gains a floor-move policy: when a floor may move, what moving one obliges
(a breaking-kind changelog entry, an updated comment, and the MSRV-gated clippy lints that switch
on), and how to read a red minimality leg.

### Fixed — nothing enforced that the bundled vk.xml ships, and CI could not notice

0.14.0 put a promise on the docs.rs front page: a copy of vk.xml ships inside the crate, so
`cargo add vulkane` builds offline. **That promise was held by nothing except the fact that
nobody had deleted the file.**

Every cargo invocation in `ci.yml` passed `--features fetch-spec`, and `spec/registry/` is not in
this repository. So removing `vulkane/vk.xml` left route 1 unset, route 2a gone and route 2b
absent — and **every leg would have quietly downloaded a spec and stayed green** while offline
builds broke for every dependent. The failure is invisible by construction: the download is a
documented feature working exactly as designed.

**The step that should have caught it was named for a configuration it did not build.**
`Build (default features)` ran `cargo build -p vulkane --features fetch-spec`, and `fetch-spec` is
not a default feature (`default = ["build-support"]`). Its comment — *"we don't need to vendor a
copy of the spec"* — was written before the crate vendored one and stopped being true without
anything failing. A step whose name and comment both describe something else is worse than an
absent one, because it occupies the slot where the missing check would be noticed.

Three gates now, each demonstrated failing before being trusted:

- `Build (default features, resolving via the bundled vk.xml)` builds what its name says, so
  resolution must come from the bundled copy.
- **The bundled copy is asserted present in `cargo package --list`**, because surviving a checkout
  and surviving packaging are different claims and only the second is what dependents get. A
  temporary `exclude = ["vk.xml"]` reds it; the assertion carries its own positive control so an
  unrecognisable list fails rather than comparing two absences.
- **The no-source path is executed, not merely compiled.** vk.xml is moved aside and the build must
  FAIL with the dependent-facing diagnostic.
