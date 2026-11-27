// SPDX-License-Identifier: MIT OR Apache-2.0
//! The WGSL spellings the shared walker (`vulkane-kernel-text`) asks for (ROADMAP rows 8a, 8b, 8g).
//!
//! WGSL is not C-family (no ternary, `select(f, t, cond)`, `var<storage>` declarations, a
//! `@compute @workgroup_size` entry), so this is its own dialect. **Gaps stated, not hidden:**
//! a `Fence` has no WGSL form (only a barrier orders memory), so it is a typed decline, as are a
//! subgroup, device or cluster barrier; `workgroupBarrier()` must sit in uniform control flow, which
//! `check_kernel` enforces before any text is written (`BarrierDivergent`); a `Write` buffer is
//! `read_write` (WGSL has no write-only storage); and every `MemoryOrder` is moot because WGSL
//! atomics are relaxed only.
//!
//! **Subgroup collectives (board 506 "A", narrowed; CireSnave 2026-11-26).** The ruled reductions, shuffles, votes and broadcasts lower to
//! WGSL `subgroup*` built-ins (WGSL spec, section 17.12, read from `gpuweb/gpuweb` `wgsl/index.bs` on 2026-11-26), with `enable subgroups;`
//! and the `subgroup_invocation_id` / `subgroup_size` entry built-ins. **Acts on active invocations; the checker guard admits only uniform
//! placements:** every call sits where `check_kernel` (unpopped 0.59.1 and later) has accepted it, so a collective after a lane-divergent
//! `Return` (`CollectiveAfterReturn`) or under lane-divergent control (`BarrierDivergent`) never reaches this dialect. Spelled exactly,
//! or declined by name: `Idx` reads lane `operand mod subgroup_size` (an explicit modulo); `Up`, `Down` and `Xor` take a CONSTANT
//! `delta` or `mask` (WGSL requires them uniform and at most 127) and read the lane's own value when the source lane is outside the subgroup
//! (an explicit `select`, as the IR's oracle does and as WGSL's indeterminate result does not); a `Broadcast` lane must be a constant below the
//! launch's stated minimum subgroup size (WGSL: `id` "must be a const-expression"); `Ballot` needs a launch that states a subgroup size of at
//! most 32 (the IR ballot is a 32-bit mask). Scans, user combines, `CollectiveTuple` and block scope stay refused by name (board 434 C).
//! Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call: "outside as read" is a reading of public documents, not a clearance; no
//! attorney; findings, not advice.

use std::fmt::Write;

use unpopped::exec::{ExecId, Scope};
use unpopped::ir::{BinaryOp, ReduceOp};
use unpopped::kernel::{AtomicOp, BufferRole, CollectiveKind, MemoryOrder, ShuffleMode, VoteKind};
use vulkane_kernel_text::{BIT_OPS, COMPARISONS, Dialect, Layout, SubgroupCollective, operator};

/// The WGSL dialect of [`vulkane_kernel_text::lower_kernel`].
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct WgslKernel;

fn ty(signed: bool) -> &'static str {
    if signed { "i32" } else { "u32" }
}

fn axis(a: u8) -> Option<char> {
    ["x", "y", "z"]
        .get(usize::from(a))
        .and_then(|s| s.chars().next())
}

/// The element type of a buffer or array: `atomic<T>` when the kernel applies an atomic to it.
fn element(t: &str, atomic: bool) -> String {
    if atomic {
        format!("atomic<{t}>")
    } else {
        t.to_string()
    }
}

impl WgslKernel {
    fn declarations(layout: &Layout) -> String {
        let t = ty(layout.signed);
        let mut s = String::new();
        for b in &layout.buffers {
            if b.spec_local.is_some() {
                let one = if layout.signed { "1" } else { "1u" };
                let _ = writeln!(s, "@id({}) override {}: {t} = {one};", b.binding, b.name);
                continue;
            }
            let access = if b.role == BufferRole::Read {
                "read"
            } else {
                "read_write"
            };
            let _ = writeln!(
                s,
                "@group(0) @binding({}) var<storage, {access}> {}: array<{}>;",
                b.binding,
                b.name,
                element(t, b.atomic)
            );
        }
        for a in &layout.shared {
            let class = if a.private { "private" } else { "workgroup" };
            let len = a.len_spec.clone().unwrap_or_else(|| a.len.to_string());
            let _ = writeln!(
                s,
                "var<{class}> {}: array<{}, {len}>;",
                a.name,
                element(t, a.atomic)
            );
        }
        s
    }
}

impl Dialect for WgslKernel {
    fn name(&self) -> &'static str {
        "wgsl"
    }

    fn narrow_refusal(&self) -> Option<&'static str> {
        Some(
            "WGSL has no 8- or 16-bit integer types (its scalar types are `i32`, `u32`, `f32`, `bool` and, with `enable f16;`, `f16`), so such a buffer or local cannot be spelled; the data would have to be packed into `u32` words, which is a rewrite of the access, not a decline",
        )
    }

    fn float_refusal(&self) -> Option<&'static str> {
        Some(
            "WGSL has no float-control or contraction control (nothing like `precise`, `NoContraction` or `SignedZeroInfNanPreserve`), and its floating point evaluation rules let an implementation fuse a multiply and an add (a body of correctly rounded primitives is `correctly_rounded`, 0 ulp, in unpopped's precision contract, which a fused pair breaks) and leave NaN and infinity results indeterminate (subnormal flush alone is outside the contract's input domain and is not the reason), so a float kernel could not be held to that tier; naga writes none of those declarations either (`tests/kernel_float.rs` pins it)",
        )
    }

    fn prologue(&self, layout: &Layout) -> String {
        self.prologue_for(layout, false)
    }

    fn subgroup_collectives(&self) -> bool {
        true
    }

    fn subgroup_collective(&self, c: &SubgroupCollective<'_>) -> Result<String, String> {
        spell_subgroup_collective(c)
    }

    fn prologue_for(&self, layout: &Layout, collectives: bool) -> String {
        let t = ty(layout.signed);
        let [x, y, z] = layout.block;
        let mut s =
            String::from("// Generated by vulkane-wgsl-emit (lower_kernel): do not edit.\n");
        if collectives {
            // The enable directive precedes every declaration (WGSL spec, 4.1.1 Enable Extensions: the extension `subgroups`).
            s.push_str("enable subgroups;\n");
        }
        s.push_str(&Self::declarations(layout));
        let _ = writeln!(s, "@compute @workgroup_size({x}, {y}, {z})");
        let extra = if collectives {
            ", @builtin(subgroup_invocation_id) sg_id: u32, @builtin(subgroup_size) sg_size: u32"
        } else {
            ""
        };
        let _ = writeln!(
            s,
            "fn {}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>{extra}) {{",
            layout.name
        );
        for i in 0..layout.locals {
            let init = layout
                .spec_init(i)
                .map_or_else(|| self.lit(0, layout.signed), str::to_string);
            let _ = writeln!(s, "    var l{i}: {t} = {init};");
        }
        s
    }

    fn epilogue(&self) -> &'static str {
        "}\n"
    }

    fn lit(&self, v: i64, signed: bool) -> String {
        match (signed, v) {
            (true, i64::MIN..=-2_147_483_648) => "(-2147483647i - 1i)".to_string(),
            (true, ..=-1) => format!("({v}i)"),
            (true, _) => format!("{v}i"),
            (false, _) => format!("{v}u"),
        }
    }

    fn index(&self, name: &str, idx: &str, _is_buffer: bool) -> String {
        format!("{name}[{idx}]")
    }

    fn exec(&self, id: ExecId, signed: bool) -> Option<String> {
        let (var, a) = match id {
            ExecId::ThreadIdx(a) => ("lid", a),
            ExecId::BlockIdx(a) => ("wid", a),
            _ => return None,
        };
        let v = format!("{var}.{}", axis(a)?);
        Some(if signed { format!("i32({v})") } else { v })
    }

    fn binary(&self, op: BinaryOp, a: &str, b: &str, signed: bool) -> Option<String> {
        if let Some(sym) = operator(COMPARISONS, op) {
            let (one, zero) = (self.lit(1, signed), self.lit(0, signed));
            return Some(format!("select({zero}, {one}, {a} {sym} {b})"));
        }
        if let Some(sym) = operator(BIT_OPS, op) {
            return Some(format!("({a} {sym} {b})"));
        }
        let sym = match op {
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            _ => return None,
        };
        // A WGSL shift count is `u32`.
        let count = if signed {
            format!("(u32({b}) & 31u)")
        } else {
            format!("({b} & 31u)")
        };
        Some(format!("({a} {sym} {count})"))
    }

    fn select(&self, cond: &str, a: &str, b: &str, signed: bool) -> String {
        format!("select({b}, {a}, {})", self.truthy(cond, signed))
    }

    fn temp(&self, name: &str, rhs: &str, signed: bool) -> String {
        format!("let {name}: {} = {rhs};", ty(signed))
    }

    fn bound(&self, name: &str, rhs: &str, signed: bool) -> String {
        self.temp(name, rhs, signed)
    }

    fn barrier(&self, scope: Scope) -> Option<String> {
        (scope == Scope::Block).then(|| "storageBarrier(); workgroupBarrier();".to_string())
    }

    fn typed_atomics(&self) -> bool {
        true
    }

    /// WGSL atomics act on workgroup and storage memory; a subgroup or cluster scope has no form.
    fn atomic_scope_ok(&self, scope: Scope) -> bool {
        matches!(scope, Scope::Block | Scope::Device)
    }

    fn atomic_load(&self, elem: &str) -> Option<String> {
        Some(format!("atomicLoad(&{elem})"))
    }

    fn atomic_store(&self, elem: &str, value: &str) -> Option<String> {
        Some(format!("atomicStore(&{elem}, {value});"))
    }

    fn atomic_rmw(&self, op: AtomicOp, elem: &str, value: &str) -> Option<String> {
        let f = match op {
            AtomicOp::Add => "atomicAdd",
            AtomicOp::Min => "atomicMin",
            AtomicOp::Max => "atomicMax",
            AtomicOp::And => "atomicAnd",
            AtomicOp::Or => "atomicOr",
            AtomicOp::Xor => "atomicXor",
            AtomicOp::Exchange => "atomicExchange",
            _ => return None,
        };
        Some(format!("{f}(&{elem}, {value})"))
    }

    /// `atomicCompareExchangeWeak` may fail with the element equal to `expected` (spuriously), which
    /// the IR's compare-exchange (the old value tells whether it swapped) cannot express; so the
    /// exchange is retried until it either swaps or sees a different value.
    fn atomic_cas_lines(
        &self,
        elem: &str,
        expected: &str,
        desired: &str,
        dest: Option<&str>,
    ) -> Option<Vec<String>> {
        let mut lines = vec![
            "{".to_string(),
            format!("    let cas_e = {expected};"),
            format!("    let cas_d = {desired};"),
            "    loop {".to_string(),
            format!("        let cas_r = atomicCompareExchangeWeak(&{elem}, cas_e, cas_d);"),
            "        if (cas_r.exchanged || cas_r.old_value != cas_e) {".to_string(),
        ];
        if let Some(l) = dest {
            lines.push(format!("            {l} = cas_r.old_value;"));
        }
        lines.extend(["            break;", "        }", "    }", "}"].map(String::from));
        Some(lines)
    }

    fn fence(&self, _scope: Scope, _order: MemoryOrder) -> Option<String> {
        None
    }
}

/// The WGSL expression of one ruled subgroup collective, or the reason it has no exact spelling here.
fn spell_subgroup_collective(c: &SubgroupCollective<'_>) -> Result<String, String> {
    let v = &c.value;
    match c.kind {
        CollectiveKind::Reduce(op) => {
            let f = match op {
                ReduceOp::Sum => "subgroupAdd",
                ReduceOp::Prod => "subgroupMul",
                ReduceOp::Max => "subgroupMax",
                ReduceOp::Min => "subgroupMin",
                ReduceOp::And => "subgroupAnd",
                ReduceOp::Or => "subgroupOr",
                ReduceOp::Xor => "subgroupXor",
                other => {
                    return Err(format!(
                        "the reduction {other:?} has no WGSL subgroup built-in"
                    ));
                }
            };
            Ok(format!("{f}({v})"))
        }
        CollectiveKind::Vote(kind) => {
            if c.signed {
                return Err(
                    "a vote's result is U32 but this kernel's locals are I32 (an I32 kernel has no U32 local)"
                        .to_string(),
                );
            }
            let pred = format!("({v} != 0u)");
            match kind {
                VoteKind::All => Ok(format!("select(0u, 1u, subgroupAll({pred}))")),
                VoteKind::Any => Ok(format!("select(0u, 1u, subgroupAny({pred}))")),
                VoteKind::Ballot => match c.subgroup {
                    // `subgroupBallot` is a `vec4<u32>` (x = invocations 0..31); the IR ballot is a 32-bit mask and a typed trap beyond 32 lanes.
                    Some((_, max)) if max <= 32 => Ok(format!("subgroupBallot({pred}).x")),
                    _ => Err(
                        "a Ballot needs a launch that states a subgroup size of at most 32 (the IR ballot is a 32-bit mask)"
                            .to_string(),
                    ),
                },
                other => Err(format!("the vote {other:?} has no WGSL subgroup built-in")),
            }
        }
        CollectiveKind::Broadcast { .. } => {
            let (Some(k), Some((min, _))) = (c.operand_const, c.subgroup) else {
                return Err(
                    "a Broadcast lane must be a constant below the launch's stated subgroup size (WGSL: `id` must be a const-expression)"
                        .to_string(),
                );
            };
            if k < 0 || k >= i64::from(min) {
                return Err(format!(
                    "the Broadcast lane {k} is outside the launch's stated minimum subgroup size {min}"
                ));
            }
            Ok(format!("subgroupBroadcast({v}, {k}u)"))
        }
        CollectiveKind::Shuffle { mode, .. } => match mode {
            ShuffleMode::Idx => {
                let op = c.operand.as_deref().unwrap_or("0u");
                // The IR reads lane `operand mod n` (Euclidean); WGSL needs an in-range lane.
                let lane = if c.signed {
                    format!("u32((({op} % i32(sg_size)) + i32(sg_size)) % i32(sg_size))")
                } else {
                    format!("({op} % sg_size)")
                };
                Ok(format!("subgroupShuffle({v}, {lane})"))
            }
            ShuffleMode::Up | ShuffleMode::Down | ShuffleMode::Xor => {
                let Some(k) = c.operand_const else {
                    return Err(
                        "a shuffle up, down or xor needs a CONSTANT delta or mask (WGSL requires it uniform; a constant is the case this emitter can prove)"
                            .to_string(),
                    );
                };
                if !(0..=127).contains(&k) {
                    return Err(format!(
                        "the delta or mask {k} is outside 0..=127 (WGSL: a constant above 127 is a shader-creation error)"
                    ));
                }
                Ok(match mode {
                    ShuffleMode::Up => {
                        format!("select({v}, subgroupShuffleUp({v}, {k}u), sg_id >= {k}u)")
                    }
                    ShuffleMode::Down => {
                        format!(
                            "select({v}, subgroupShuffleDown({v}, {k}u), {k}u < (sg_size - sg_id))"
                        )
                    }
                    _ => format!(
                        "select({v}, subgroupShuffleXor({v}, {k}u), (sg_id ^ {k}u) < sg_size)"
                    ),
                })
            }
            other => Err(format!(
                "the shuffle mode {other:?} has no WGSL subgroup built-in"
            )),
        },
        other => Err(format!(
            "the collective {other:?} has no WGSL subgroup built-in"
        )),
    }
}
