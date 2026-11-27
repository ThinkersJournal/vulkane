# Subgroup collectives on non-NVIDIA targets (board 506 "A (narrowed)"): design and plan

**Status (2026-11-27): a plan and a record of verified facts; mostly landed.** The WGSL and MSL lowerings described below shipped in #186 and #195
(main 0.31.2: reductions, shuffles and broadcasts; MSL `F32` Sum, Max and Min with a NaN indicator), with the uniformity guard taken as unpopped's
`check_kernel` (0.59.1 and later; the workspace is on unpopped 0.61). **Still open from this plan: HLSL** (it waits on a published plain-HLSL entry in
`fishstringer-slang-emit`) and every kind in the "stays declined" list. The text below was written on 2026-11-26 on the branch
`day-2026-11-26-506-prep` and is kept as written except for this status and the withheld rows; where a later PR changed a statement, the PR's
fragment and ROADMAP rows 8s, 8t, 8u are the record.

> **Vote/ballot rows withheld pending the patent hold: see the PM's vote hold (2026-11-27); facts kept off main.** (The full original of this
> document, with those rows, is kept on the branch `day-2026-11-26-506-prep`; they return in a docs PR when the hold lifts.)
>
> Citation wording for any patent named here: findings, not advice; "outside as read" is a reading of public documents, not a clearance; no attorney.

## Authority

CireSnave, board 506, **"506: A"** (narrowed), archived at `C:/Projects/CIRESNAVE-DECIDED-ARCHIVE.md` item 506
(ruled 2026-11-26). Relayed by pm-kernels as:

* **LIFTED:** subgroup-scope reductions, shuffles and broadcasts on Metal (vote rows withheld: see the pointer at the top) (`simd_*`), HLSL (`Wave*`), WGSL (`subgroup*`).
* **STILL DECLINED BY NAME, with the board 434 C marker:** subgroup-scope **scans**, SPIR-V `OpGroupNonUniform`, GLSL subgroup ops,
  and **every block-scope collective** (the archive: whether a block-wide collective reads on claim 1 "is claim construction",
  which only a person can settle).
* The item's own consequence note: "a lifted emitter needs a uniformity guard": the IR's collective means every lane, the
  hardware acts on *active* threads. Patent basis: `docs/PATENT-SWEEP.md` (Unpopped), section "US8539204B2 claims 1,12,13 against
  non-NVIDIA subgroup and wave functions". Residual risk, in the archive's words: "outside as read" is a reading of public
  documents, not a clearance.

## The ruled set, by IR kind

Only `Stmt::Collective` with `scope: Scope::Subgroup`, value type an integer or `F32` (and `F16` where the dialect spells it):

| `CollectiveKind` | in the ruled set | decline otherwise |
|---|---|---|
| `Reduce(Sum, Prod, Max, Min, And, Or, Xor)` | yes (`And/Or/Xor`: integer only) | `Reduce(Mean)` and any other op, by name |
| `Shuffle { Idx, Up, Down, Xor }` | yes, see "Edge semantics" | |
| `Broadcast { lane }` | yes | WGSL: `lane` must be a const-expression, see below |
| `Scan { .. }` (every op, inclusive or exclusive) | **no** | by name, 434 C marker |
| `ReduceWith`, `ScanWith`, `Stmt::CollectiveTuple` | **no** | by name, 434 C marker |
| any `scope: Block` | **no** | by name, 434 C marker |

GLSL and SPIR-V: every `Collective` stays refused, as today.

## Per-dialect functions (what is verified, and where)

### MSL (Metal) : from the MSL Shading Language spec 4.1 section 6.10.2 (as already cited in `vulkane-msl-emit` docs)

`Reduce`: `simd_sum`, `simd_product`, `simd_max`, `simd_min`, `simd_and`, `simd_or`, `simd_xor`. `Shuffle`: `simd_shuffle` (Idx),
`simd_shuffle_up`, `simd_shuffle_down`, `simd_shuffle_xor`. `Broadcast`: `simd_broadcast`.
Scans (`simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum`) stay declined. Constraints already recorded in the emitter: the functions act on
the active threads ("A thread needs to only read data from another active thread"), and `T` excludes `bool`, `bfloat`, `long`, `ulong` (those dtypes stay declined by name).
**READ VERBATIM (2026-11-26) from the Metal Shading Language Specification version 4.1 (dated 2026-06-04), section 6.10.2 "SIMD-Group Functions", Tables 6.14 and 6.15** (the PDF downloaded to a scratch directory and text-extracted with pymupdf; pages 217-221):

* Active threads: "An active thread is a thread that is executing. An inactive thread is a thread that is not executing. ... A thread needs to only read data from another active thread in the SIMD-group." `T` is "a scalar or vector of any integer or floating-point type, except: bool, bfloat, long, ulong, void, size_t, ptrdiff_t"; the bitwise functions need an integer `Ti`.
* `simd_shuffle_down(data, delta)`: "This function doesn't modify the upper delta lanes of data because it doesn't wrap values around the SIMD-group." `simd_shuffle_up`: "doesn't modify the lower delta lanes of data because it doesn't wrap". **Both match the IR rule (an out-of-range source reads the lane's OWN value) and need no select.** `delta` "needs to be the same for all threads in the SIMD-group".
* `simd_shuffle_xor(value, mask)`: "the thread whose SIMD lane ID is equal to the bitwise XOR (^) of the caller's SIMD lane ID and mask. The value of mask needs to be the same for all threads". **The text states NO result when `lane ^ mask` is outside the SIMD-group**, so the IR's own-value rule for `Xor` must be spelled by an explicit in-range select (and is device-tested at the edge lanes), or declined by name.
* `simd_shuffle(data, simd_lane_id)`: "needs to be a valid SIMD lane ID but doesn't have to be the same for all threads": the IR's `Idx` (`operand mod n`) needs an explicit `% threads_per_simdgroup` before the call.
* `simd_broadcast(data, broadcast_lane_id)`: "The specification doesn't define the behavior when broadcast_lane_id isn't a valid SIMD lane ID or isn't the same for all threads in a SIMD-group." So a `Broadcast` lane must be provably valid and uniform (a `ConstInt` below the subgroup size), else decline by name.
* Reductions (Table 6.15): `simd_sum`, `simd_product`, `simd_max`, `simd_min` (`T`), `simd_and`, `simd_or`, `simd_xor` (`Ti`): "across all active threads in the SIMD-group and broadcasts the result to all active threads". The prefix functions (`simd_prefix_exclusive_sum`, ...) are scans: declined (the ruling).
* **Still UNVERIFIED:** the minimum Metal/OS availability per function as it matters for the probe device (the tables list macOS 2.1+ / iOS 2.3+ for the reductions).

### WGSL : read from the WGSL spec source, `gpuweb/gpuweb` `wgsl/index.bs` on `main`, 2026-11-26 (rendered section numbers in brackets)

* Needs the enable extension `subgroups` (4.1.1; the literal `enable subgroups;` text is **inferred** from the other `enable` examples:
  the directive grammar is an include file not read). `subgroup_invocation_id` and `subgroup_size` are `u32` built-in inputs of
  compute and fragment stages (13.3.1.1.17-18); subgroup sizes are powers of two in [4, 128] (15.5).
* `Reduce`: `subgroupAdd`, `subgroupMul`, `subgroupMax`, `subgroupMin` (numeric T), `subgroupAnd`, `subgroupOr`, `subgroupXor` (integer T) [17.12.1, .10, .8, .9, .3, .11, .13].
* `Broadcast`: `subgroupBroadcast(e, id)`: **"`id` must be a const-expression in the range [0, 128)"**, and "It is a dynamic error if `id` does not select an active invocation" [17.12.6]. So a `Broadcast` whose `lane` is not a `ConstInt` is **declined by name** (the spec's own advice, `subgroupShuffle`, has different out-of-range semantics).
* `Shuffle`: `subgroupShuffle(e, id)`, `subgroupShuffleUp(e, delta)`, `subgroupShuffleDown(e, delta)`, `subgroupShuffleXor(e, mask)` [17.12.12]: "An indeterminate value is returned if `id` does not select an active invocation"; `delta`/`mask` greater than 127 is an error; a `subgroup_uniformity` diagnostic fires when `delta`/`mask` is not uniform.
* **Inactive invocations:** every function above is defined over "active invocations" (15.5: "The invocations in a subgroup that concurrently execute a subgroup operation are active for that operation. Other invocations in the subgroup are inactive for that operation."). The call must be provably in uniform control flow or a `subgroup_uniformity` diagnostic fires (error by default, per the text read); the spec itself warns "There is considerable non-portability among underlying devices when operating in non-uniform control flow" (15.5).
* **Not read verbatim** (seen only in a tool summary): 15.6.3 "Subgroup Operations" and the final default severity of the diagnostic. Quote them before relying on them.

### HLSL : read from the DirectXShaderCompiler wiki page "Wave Intrinsics" (the learn.microsoft.com page returned HTTP 404)

* `Reduce`: `WaveActiveSum`, `WaveActiveProduct`, `WaveActiveMin`, `WaveActiveMax` (`<type>`: half, float, double, int, uint, short, ushort, uint64_t, with 1 to 4 components, where the shader model has it), `WaveActiveBitAnd`, `WaveActiveBitOr`, `WaveActiveBitXor` (integer only). "The order of operations is undefined."
* `Broadcast` / `Shuffle(Idx)`: `WaveReadLaneAt(expr, laneIndex)`: "The result is undefined on a helper lane, or if the lane referred to by laneIndex is inactive or a helper lane." `WaveReadLaneFirst` reads the lowest active lane.
* **No dedicated shuffle up/down/xor intrinsic** on that page ("NOT FOUND"); a computed lane through `WaveReadLaneAt` can express them. `QuadRead*` are 4-lane quads only (not subgroup shuffles). `WavePrefixSum`/`WavePrefixProduct` are exclusive scans: declined (the ruling).
* "These intrinsics are dependent on active lanes and therefore flow control." Per-intrinsic minimum Shader Model: **not on that page** (the page says "As of v6.0"; SM 6.5 "additional Wave Intrinsics" are only linked: UNVERIFIED); the dxc-measured acceptance is below. Drivers are not required to implement wave ops for SM 6.0 (the `WaveOps` cap: UNVERIFIED on any device here).
* **Measured with dxc 1.9.0.5347 and slangc 2026.8 (2026-11-26), on fishstringer's 14 forms at `c9051c40` (`Slang::lower_kernel_hlsl`):**
  `slangc -target hlsl` accepts 14/14; `dxc` then accepts 13/14 at `cs_6_0` (the lowest model tried; no model below 6.0 exists), so the minimum shader model **dxc accepts** is 6.0 for the forms that compile (a compiler fact, not a driver guarantee: the wiki says drivers need not implement wave ops at SM 6.0). The one rejected form is `WaveActiveBitOr` on `I32` at every model 6_0..6_9: dxc's `WaveActiveBitAnd/Or/Xor` accept **only `uint`** (int, float, half rejected; measured), against the wiki's `<int_type>`; an I32 reduction must be spelled `(int)WaveActiveBitOr((uint)x)`. `WaveActiveSum/Product/Min/Max` accept int, uint, float and half (half at `cs_6_2` with `-enable-16bit-types`); `WaveReadLaneAt` accepts int, uint, float; Controls: garbage rejected, a plain kernel accepted, the SPIR-V from `-spirv` holds the `OpGroupNonUniform` op.
* **UPDATE (fishstringer#54 `4b9f00cb`, `Slang::lower_kernel_hlsl_plain`, not yet published):** the plain HLSL text carries `[WaveSize(n)]` and requires launch `subgroup` min == max, a power of two in 4..=128 (else decline by name). Measured with dxc 1.9.0.5347: `[WaveSize(32)]` is **rejected below shader model 6.6** (`attribute wavesize requires shader model 6.6 or greater`) and accepted at `cs_6_6` and `cs_6_7`; under `-spirv` it is accepted at `vulkan1.1` and `vulkan1.3`. So the minimum shader model for that text is **6.6**, not the 6.0 measured above on the forms without the attribute.
* **The emitted text is Slang, not HLSL:** it wraps each collective in `__target_switch { case hlsl: ... break; }` and `dxc` rejects all 14 with `use of undeclared identifier '__target_switch'`; the same text with the guard removed compiles. `vulkane-hlsl-emit` feeds its text to dxc directly, so it needs guard-free HLSL from fishstringer (asked), not a string strip. With `dxc -spirv` (hlsl-emit's flags) any Wave op needs `-fspv-target-env=vulkan1.1` or newer (else "Vulkan 1.1 is required for Wave Operation"); `required_dxc_args` has no target env today.
* **Edge rule:** fishstringer already spells it in the emitted text (`Idx`: `((id % WaveGetLaneCount()) + LaneCount) % LaneCount`; `Up/Down/Xor`: a range test, own value otherwise). **Executing them on a device is UNMEASURED.**
* **HLSL goes through `fishstringer-slang-emit` today** (`vulkane-hlsl-emit::lower_kernel` delegates), and that emitter refuses every `Collective` for every device (`src/gaps.rs`, `kernel_lower.rs`). The lift for HLSL is a **FishStringer change** (pm-kernels agreed, 2026-11-26): vulkane does not edit Slang. Until it publishes on unpopped 0.59, HLSL is not started.

## The HLSL back door to SPIR-V, and the guard (pm-kernels 2026-11-26; measured with dxc 1.9.0.5347, slangc 2026.8)

HLSL with a Wave intrinsic compiles to SPIR-V `OpGroupNonUniform*` under `dxc -spirv -fspv-target-env=vulkan1.1+` (measured), which is exactly
what board 506 keeps declined ("SPIR-V OpGroupNonUniform and GLSL subgroup ops stay declined"). So the HLSL text that `vulkane-hlsl-emit` hands out for a kernel that holds a
lowered `Collective` must carry a guard prefix, prepended before any other line:

```
#ifdef __spirv__
#error "subgroup collectives are declined for SPIR-V (board 506)"
#endif
#ifdef __SLANG__
#error "subgroup collectives are declined for non-HLSL targets (board 506)"
#endif
```

Measured, one macro at a time, with controls (each probe file a distinct name: on Windows two names differing only in case are the SAME file, which silently
corrupted a first probe):

| macro | `dxc -spirv` | plain `dxc -T cs_6_x` (DXIL) | `slangc -lang hlsl -target spirv` |
|---|---|---|---|
| `__spirv__` | **defined** | not defined | not defined |
| `__SLANG__`, `__SLANG_COMPILER__`, `__HLSL__` | not defined | not defined | **defined** |

* The first guard: `dxc -spirv` (every target env tried: none, 1.1, 1.3) REJECTS with the `#error` text; plain `dxc -T cs_6_0` and `cs_6_6` ACCEPT; the **guard-removed mutant** is ACCEPTED under `-spirv`; a guard naming a macro dxc does not define (`__SPIRV_NOT_DEFINED__`) is ACCEPTED under `-spirv` (so the rejection is the `__spirv__` line and nothing else).
* **The first guard alone does not close `slangc`:** `slangc -lang hlsl -target spirv` ACCEPTS the guarded text (control: also accepts the unguarded text). Hence the second guard on `__SLANG__`: the HLSL is for `dxc` DXIL only. (Not yet re-run with the second guard in place: do it when the code exists.)
* **Other HLSL front ends are UNMEASURED:** `glslang -D -V` on this machine rejects the unguarded control too (a parse error on the Slang-style declarations), so it says nothing either way. **A text prefix is default-safe, not enforcement:** anyone can delete it, as anyone can delete the `__target_switch` guard on fishstringer's Slang text; the README note is the legal half.
* **Pipeline (proposed to fishstringer, cc pm-kernels):** `vulkane-hlsl-emit::lower_kernel` is pure Rust and runs no external tool, so it cannot run `slangc`. fishstringer returns the guard-free plain HLSL (a second text beside its `__target_switch` Slang text), and `vulkane-hlsl-emit`'s retarget step (a) **refuses by name** any output that contains a `Wave` intrinsic and does not start with the guard, and (b) prepends the guard itself when fishstringer's text lacks it, so the guarantee lives in the crate that is published as HLSL.
* **Flags:** `required_dxc_args` for a collective kernel keeps **no `-spirv` and no `-fspv-target-env`** (it has no target env today; do not add one for collectives).
* **Tests (after the take; `tests/collective_guard.rs` in `vulkane-hlsl-emit`, skipping loudly without dxc unless `VULKANE_REQUIRE_DXC`):** (1) `dxc -spirv` on the output of a collective kernel FAILS with the guard text; (2) plain `dxc -T cs_6_x` on it SUCCEEDS; (3) the guard-removed mutant PASSES under `-spirv` (proves (1) can fail); (4) the wrong-macro control passes under `-spirv`; (5) `slangc -lang hlsl -target spirv` on it FAILS (second guard); (6) `required_dxc_args` for that kernel contains neither `-spirv` nor `-fspv-target-env`; (7) output with a `Wave` intrinsic and no guard is refused by name.

## Edge semantics: the IR (oracle) is not the target functions

Measured from `unpopped` 0.58.5 `src/oracle/kernel_collective.rs` (the oracle `run_launch_with_subgroup` compares against):

* `Shuffle Idx`: lane `operand mod n` (n = subgroup size). `Up`/`Down`/`Xor`: the source lane `i - operand`, `i + operand`, `i ^ operand`, and **if it is outside `0..n` the lane reads its own value**.

The target functions differ at the edges: WGSL returns an indeterminate value for a source that is not an active invocation, and errors on an out-of-range constant; HLSL's `WaveReadLaneAt` is undefined for an inactive source lane. So a lowering must spell the IR's rule explicitly, for example `Up`: `select(e, subgroupShuffleUp(e, delta), subgroup_invocation_id >= delta)` (WGSL evaluates both operands, so the call stays in uniform flow), and `Idx`: the lane taken `% subgroup_size`. **Each such spelling is UNMEASURED on a device until its test runs against the oracle at the edge lanes**; a spelling that cannot be shown equal is declined by name, not approximated.

## What the guard (unpopped 0.59.1 `check_kernel`) gives, and what it does not

From unpopped's U63 notice: `check_kernel` refuses a `Subgroup` `Collective`/`CollectiveTuple` after a `Return` under lane-divergent control
(`KernelError::CollectiveAfterReturn`) and admits control from `threadIdx.y/.z` when the launch states one subgroup size that divides `block.x`.
`unpopped::fixtures::divergent_fixtures()` (feature `fixtures`) gives two refused kernels. **Each emitter's docs state the assumption**:
"acts on active threads; the checker guard admits only uniform placements", with a test that a divergent placement is refused
(from that fixture). The guard is a checker rule: an emitter reached without `check_kernel` is not protected, so `lower_kernel` keeps its own
`check_kernel` call first (as `vulkane-kernel-text::lower_kernel` already does) and a test removes nothing silently.

## Structure of the change (accepted by pm-kernels)

1. `vulkane-kernel-text::Dialect` gains `subgroup_collectives(&self) -> bool` (default `false`) and a spelling hook that writes the call and assigns the statement's result local; `declined_by_name` becomes kind-aware (ruled set lowered only when the dialect says yes and the scope is `Subgroup`; every other kind keeps the 434 C marker). A new default method is not breaking; a new public field on `Layout`/`Decl` is (measured on #177), so add hooks, not fields, or label the PR BREAKING after `cargo-semver-checks` (0.51.0, against the published version).
2. PRs, one at a time on pm-assist's release, each under 150k hand-written characters: (A) kernel-text capability + MSL (`vulkane-msl-emit` is on the unmerged stack: it ships behind it); (B) WGSL; (C) HLSL after FishStringer's slang-emit lowers.
3. Tests per dialect: a table test listing **every** `CollectiveKind` beside its function (ruled set lowered; scans, `ReduceWith`/`ScanWith`, `CollectiveTuple`, block scope, GLSL and SPIR-V still refused WITH the 434 C marker), the vacuity mutants (delete a decline, expect a named failure), the divergent-placement refusal, and device tests: Metal on the hosted macOS probe against the oracle (the f32/bf16 flush note applies); HLSL via `dxc` if a subgroup-capable device is present, else UNMEASURED; WGSL via the naga/wgpu path if available, else UNMEASURED.
4. Patent note in each emitter's README: basis board 506 + PATENT-SWEEP US8539204B2 claims 1/12/13, dated.

## Open questions (none blocks the plan; each blocks the code it names)

* WGSL 15.6.3 text and the default severity of `subgroup_uniformity` (quote before relying).
* Which dtypes each function accepts per dialect beyond the table above (bool, `Bf16`, 64-bit): decline by name until each is shown.
* KISS 0.24.0 added seven dtype tokens (pm-kernels): an unknown token is a decline by name; check where this crate maps dtype tokens when the take lands.
