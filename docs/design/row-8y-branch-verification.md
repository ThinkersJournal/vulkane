# Row 8y: what the 36 old remote branches held, and why deleting them lost nothing

**Status (2026-11-28): done.** Of the 36 remote branches this lane had at 2026-11-27 (all but two were never the head of a pull request: their work reached `main` through squash merges; the two exceptions, `day-2026-11-27-docs-rows` and `-doc506`, were #196 and #197), 34 are deleted from `origin`; `day-2026-11-26-506-prep` (the full 506 plan with vote and ballot rows: held by the PM's vote hold) and `day-2026-11-26-9b` (ROADMAP 9c/9b, being sliced) remain. This file is the record of the check that justified the deletions. The tips of all 36 were saved first, as a verified `git bundle` (`C:\Projects\.lane-state\vulkane-branches-20261127.bundle`) and a tips list (`C:\Projects\.lane-state\vulkane-branch-tips-20261127.txt`, main at `274fcd97e8d9f58402222ea34da9266a5d0be665`): both live outside this repository, so a deleted branch can be restored with `git fetch <bundle> <branch>`.

## Why the obvious checks cannot answer the question

Every landing of this lane's work was a **squash** merge (#181 to #195), so `git branch --merged`, `merge-base --is-ancestor`, `git merge-tree` and `gh pr list --head <branch> --state merged` all mislead: the branch commits never become ancestors of `main`, and for most of these branches no pull request was ever opened, so the forge has nothing to show. A first pass that looked up every added LINE of a branch in `origin/main` called most of them UNPROVEN, because lines are rewritten by later work (a doc comment, a rustfmt wrap) without the work being missing.

## Method that was used

For each branch: take the tip and its merge-base, list the **symbols it adds** (the names of the functions, tests, types and constants its diff introduces, excluding `Cargo.lock`), and look each name up among the symbols defined on `origin/main`. A name present on main means that work reached main under the same name. **A name missing from main is a finding, not a verdict**: each missing name below was opened and judged by hand. Controls: the method DOES discriminate: on the two branches that held unmerged work it lists their pins as missing (`day-2026-11-26-g1-review`: 4 names, `-patent-audit`: 5 names), which is how those revivals were found, while 13 other branches list none. (A docs-only branch such as `day-2026-11-26-docs-msl-simdgroup` adds 0 symbols, so it is not a control of anything.)

## Result over the 36 branches (main `274fcd9`)

* **13 branches: every added symbol is on main** (nothing to lose).
* **9 branches: only `bx` is missing.** `bx` is a helper in `vulkane-spirv-emit/tests/support/atomic_wide_kernel.rs`, a file that moved to `vulkane-test-support/src/atomic_wide_kernel.rs` (the file exists there on main) when the test support became one crate: a move, not a loss.
* **8 branches: only `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` is missing.** A test written when MSL declined every collective; the board 506 "A" ruling then lowered the ruled kinds (#194, #195), so the test no longer states what is true. `vulkane-msl-emit/tests/subgroup_collectives.rs` on main now holds the surviving statement, `a_collective_tuple_is_refused_by_name` (the tuple form is still refused); the old name is read as superseded, not as a test to restore.
* **2 branches: only `bf16_and_the_inexact_half_ops_are_declined_by_name` is missing.** Only the two old MSL branches (`day-2026-11-27-chain-msl1`, `day-2026-12-02-8k-msl1`) carry it, in `vulkane-msl-emit/tests/kernel_lowering.rs`; main has it split in two, `a_bf16_constant_not_exact_at_bfloat_is_declined` and `the_inexact_half_ops_are_declined_by_name`, in the same file: a split, not a loss.
* **4 branches carry other missing names, each handled by name:** `day-2026-11-26-g1-review` and `day-2026-11-26-patent-audit` held pins and a refusal test that never reached main; they were REVIVED, not deleted unread (g1-review as #199, patent-audit as #200, including the `Place::Raw`/`AllocRaw` refusal test that `take-unpopped-0.57.0` also carried); `day-2026-11-23-8k-g1` (`int_cast`, `float_kernels_stay_a_typed_decline`) and `take-unpopped-0.57.0` were checked name by name against the rewrites that replaced them.

**What this proves and what it does not.** It proves that every name these branches introduced is on main, or is one of the missing names above, each of which was read. It does not prove byte-identical files or equal behaviour (a name can survive while its body changed); the per-name judgements above are the human part.

## The 36 branches

| branch | tip | merge-base | symbols added | missing | missing names |
|---|---|---|---|---|---|
| `day-2026-11-23-8k` | `004cfa0` | `1ebb820` | 255 | 1 | `bx` |
| `day-2026-11-23-8k-g1` | `1b448bf` | `4dcf43f` | 143 | 3 | `int_cast`, `float_kernels_stay_a_typed_decline`, `bx` |
| `day-2026-11-23-8k-g3` | `b3ccc9a` | `4dcf43f` | 235 | 1 | `bx` |
| `day-2026-11-23-8k-gf` | `5623540` | `4dcf43f` | 305 | 1 | `bx` |
| `day-2026-11-23-8k-gf2` | `a35438e` | `4dcf43f` | 374 | 1 | `bx` |
| `day-2026-11-26-506-prep` | `edcc842` | `a224561` | 435 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `day-2026-11-26-9b` | `2a7e347` | `012489b` | 77 | 0 | none |
| `day-2026-11-26-docs-msl-simdgroup` | `c3f631a` | `aa5cfee` | 0 | 0 | none |
| `day-2026-11-26-g1-review` | `d96d899` | `012489b` | 4 | 4 | `one_cell_kernel`, `declined_naming`, `a_wide_kernel_declines_division_atomics_and_a_specialization_constant_extent`, `an_i64_kernel_is_declined_by_name` |
| `day-2026-11-26-patent-audit` | `a844b7f` | `012489b` | 9 | 5 | `declined_by_checker`, `a_parameter_dependent_grid_is_declined_by_the_walker`, `a_non_wrapping_cast_and_an_integer_unary_op_are_declined_by_the_walker`, `an_unspellable_fence_exec_id_and_cast_are_typed_declines`, `an_atomic_at_cluster_scope_is_a_typed_decline` |
| `day-2026-11-26-take-0-58` | `3a5f5db` | `0840306` | 0 | 0 | none |
| `day-2026-11-27-8k-g3` | `9bbaa7f` | `a224561` | 183 | 1 | `bx` |
| `day-2026-11-27-chain-cov` | `b5bb6c1` | `aa5cfee` | 241 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `day-2026-11-27-chain-gf2a` | `45eae33` | `aa5cfee` | 51 | 0 | none |
| `day-2026-11-27-chain-gf2b` | `ee588a6` | `aa5cfee` | 108 | 0 | none |
| `day-2026-11-27-chain-gf3` | `61a5a22` | `aa5cfee` | 124 | 0 | none |
| `day-2026-11-27-chain-mask` | `fce5396` | `aa5cfee` | 231 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `day-2026-11-27-chain-mcoll` | `00b867f` | `aa5cfee` | 218 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `day-2026-11-27-chain-msl1` | `8b213de` | `aa5cfee` | 205 | 1 | `bf16_and_the_inexact_half_ops_are_declined_by_name` |
| `day-2026-11-27-chain-row9` | `f618c22` | `aa5cfee` | 215 | 0 | none |
| `day-2026-11-27-chain-wbf16` | `7c10c8c` | `aa5cfee` | 212 | 0 | none |
| `day-2026-11-27-doc506` | `9a4602d` | `274fcd9` | 0 | 0 | none |
| `day-2026-11-27-docs-rows` | `de36405` | `274fcd9` | 0 | 0 | none |
| `day-2026-11-28-8k-gf` | `e801814` | `a224561` | 253 | 1 | `bx` |
| `day-2026-11-29-8k-gf2` | `b76a2b4` | `a224561` | 319 | 1 | `bx` |
| `day-2026-11-29-8k-gf2a` | `42a7647` | `a224561` | 277 | 1 | `bx` |
| `day-2026-11-30-8k-gf2b` | `f7d3495` | `a224561` | 327 | 1 | `bx` |
| `day-2026-12-01-8k-gf3` | `9d50706` | `a224561` | 332 | 0 | none |
| `day-2026-12-02-8k-msl1` | `6f6da68` | `a224561` | 400 | 1 | `bf16_and_the_inexact_half_ops_are_declined_by_name` |
| `day-2026-12-03-row9` | `dd7a2b0` | `a224561` | 410 | 0 | none |
| `day-2026-12-04-walker-bf16` | `2bb94d3` | `a224561` | 407 | 0 | none |
| `day-2026-12-05-msl-collectives` | `1a4443d` | `a224561` | 413 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `day-2026-12-06-masked-access` | `a8be151` | `a224561` | 426 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `day-2026-12-07-stmt-coverage` | `c5f0308` | `a224561` | 435 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `macos-metal-probe` | `0b7c139` | `a224561` | 477 | 1 | `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it` |
| `take-unpopped-0.57.0` | `a466adf` | `a224561` | 437 | 3 | `a_byte_addressed_place_is_refused_by_name_not_lowered_to_a_wrong_address`, `every_kind_of_collective_is_refused_by_name_whatever_msl_function_would_spell_it`, `raw_place_kernels` |

— lane: vulkane
