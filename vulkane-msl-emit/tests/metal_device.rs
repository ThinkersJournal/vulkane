// SPDX-License-Identifier: MIT OR Apache-2.0
//! Device-run oracle for the MSL emitter on the hosted macOS job (`.github/workflows/metal-probe.yml`).
//!
//! Two tests share one case table, driven by `VULKANE_MSL_DEVICE_DIR`:
//! * `device_cases_are_written`: lowers each case and writes `<name>.metal` and `<name>.case` (entry, grid, block, the
//!   input buffers and the zero/sentinel-filled output buffer, as hex words) into the directory;
//! * `device_results_match_the_oracles`: after `.github/metal/run.swift` dispatched each case on the Metal device and wrote
//!   `<name>.out` (`simd N`, then `words ...` or `error ...`), checks every output with the SAME oracles the Vulkan device
//!   tests use (the shared kernels' `expected`, `float_kernel::same`, `half_kernel::same`), and the mutant/control cases
//!   must DIFFER from the oracle they would have matched.
//!
//! With the variable unset both tests only check that the table lowers (so a case that stops lowering fails everywhere).

use std::fmt::Write as _;
use std::path::PathBuf;

use unpopped::backend::Backend;
use unpopped::kernel::KernelFn;
use unpopped_vocab::ElementKind;
use vulkane_msl_emit::Msl;
use vulkane_test_support::{
    atomic_kernel, bf16_kernel, div_kernel, float_kernel, flow_kernel, grid_kernel, half_kernel,
    masked_kernel, narrow_kernel, private_kernel, reverse_kernel, subgroup_kernel, wide_kernel,
};

const SENTINEL: u32 = 0xAAAA_AAAA;

/// What the device printed for one case.
struct Out {
    simd: u32,
    words: Vec<u32>,
}

type Check = Box<dyn Fn(&Out) -> Result<(), String>>;

struct Case {
    name: String,
    kernel: KernelFn,
    /// `None`: one threadgroup per `cells / block` of the 1-D output.
    grid: Option<[u32; 3]>,
    inputs: Vec<Vec<u32>>,
    out_cells: usize,
    check: Check,
}

fn eq_words(want: Vec<u32>) -> Check {
    Box::new(move |o| {
        if o.words == want {
            return Ok(());
        }
        let i = o
            .words
            .iter()
            .zip(&want)
            .position(|(g, w)| g != w)
            .unwrap_or(0);
        Err(format!(
            "first difference at cell {i}: got {:#010x} want {:#010x} ({} cells, {} wrong)",
            o.words.get(i).copied().unwrap_or(0),
            want.get(i).copied().unwrap_or(0),
            o.words.len(),
            o.words.iter().zip(&want).filter(|(g, w)| g != w).count()
        ))
    })
}

/// A mutant or control: the device output must NOT match what the correct kernel produces.
fn differs_from(c: Check) -> Check {
    Box::new(move |o| match c(o) {
        Ok(()) => Err(
            "the mutant/control matched the correct oracle: this device cannot tell them apart"
                .into(),
        ),
        Err(_) => Ok(()),
    })
}

fn float_cases(v: &mut Vec<Case>) {
    use float_kernel::Case as C;
    for (tag, case) in [
        ("muladd", C::MulAdd),
        ("max", C::Max),
        ("min", C::Min),
        ("floor", C::Floor),
    ] {
        let mut k = float_kernel::float_kernel(ElementKind::F32, case);
        k.name = "float_case".into();
        let (a, b, want) = float_kernel::data::<f32>(case);
        v.push(Case {
            name: format!("f32_{tag}"),
            kernel: k,
            grid: None,
            inputs: vec![a, b],
            out_cells: want.len(),
            check: Box::new(move |o| float_kernel::same(false, &o.words, &want)),
        });
        let mut k = float_kernel::float_kernel(ElementKind::F16, case);
        k.name = "float_case".into();
        let (a, b, want, mutant) = half_kernel::data(case);
        let (w2, cells) = (want.clone(), want.len());
        v.push(Case {
            name: format!("f16_{tag}"),
            kernel: k.clone(),
            grid: None,
            inputs: vec![a.clone(), b.clone()],
            out_cells: cells,
            check: Box::new(move |o| half_kernel::same(&o.words, &w2)),
        });
        // The case's mutant (fused, flushed, NaN-ignoring ...) must NOT be what the device computes.
        v.push(Case {
            name: format!("f16_{tag}_vs_mutant"),
            kernel: k,
            grid: None,
            inputs: vec![a, b],
            out_cells: cells,
            check: differs_from(Box::new(move |o| half_kernel::same(&o.words, &mutant))),
        });
        // Bf16 (ROADMAP 8p): the same shape through the bfloat16 oracle; compile-only until the device run.
        let mut k = float_kernel::float_kernel(ElementKind::Bf16, case);
        k.name = "float_case".into();
        let (a, b, want, mutant) = bf16_kernel::data(case);
        let (w2, cells) = (want.clone(), want.len());
        v.push(Case {
            name: format!("bf16_{tag}"),
            kernel: k.clone(),
            grid: None,
            inputs: vec![a.clone(), b.clone()],
            out_cells: cells,
            check: Box::new(move |o| bf16_kernel::same(&o.words, &w2)),
        });
        v.push(Case {
            name: format!("bf16_{tag}_vs_mutant"),
            kernel: k,
            grid: None,
            inputs: vec![a, b],
            out_cells: cells,
            check: differs_from(Box::new(move |o| bf16_kernel::same(&o.words, &mutant))),
        });
    }
}

/// Subnormal operands and results (the shared `float_kernel::PAIRS` never reach one). MSL 8.1 lets a device flush a
/// denormal to zero, so a case accepts the exact IEEE answer OR the flushed one, and says which the device gave
/// (`DEVICE note`): the measurement is the point, a wrong third value is the failure.
fn subnormal_cases(v: &mut Vec<Case>) {
    use float_kernel::Case as C;
    let mut k = float_kernel::float_kernel(ElementKind::F32, C::MulAdd);
    k.name = "float_case".into();
    let n = float_kernel::THREADS as usize;
    // `0x00000001` is the smallest f32 subnormal; times 0.5 it ties to zero, and zero plus the operand is the operand.
    v.push(Case {
        name: "f32_muladd_subnormal".into(),
        kernel: k,
        grid: None,
        inputs: vec![vec![0x0000_0001; n], vec![0.5f32.to_bits(); n]],
        out_cells: n,
        check: Box::new(move |o| {
            if o.words == vec![1; n] {
                eprintln!("DEVICE note f32 subnormals are PRESERVED");
                Ok(())
            } else if o.words == vec![0; n] {
                eprintln!("DEVICE note f32 subnormals are FLUSHED to zero (MSL 8.1 allows it)");
                Ok(())
            } else {
                Err(format!(
                    "neither exact (1) nor flushed (0): first cell {:#x}",
                    o.words[0]
                ))
            }
        }),
    });
    let mut k = float_kernel::float_kernel(ElementKind::Bf16, C::MulAdd);
    k.name = "float_case".into();
    let (a, b, exact, flushed) = bf16_kernel::data_subnormal();
    let cells = exact.len();
    v.push(Case {
        name: "bf16_muladd_subnormal".into(),
        kernel: k,
        grid: None,
        inputs: vec![a, b],
        out_cells: cells,
        check: Box::new(move |o| {
            if bf16_kernel::same(&o.words, &exact).is_ok() {
                eprintln!("DEVICE note bf16 subnormals are PRESERVED");
                Ok(())
            } else if bf16_kernel::same(&o.words, &flushed).is_ok() {
                eprintln!("DEVICE note bf16 subnormals are FLUSHED to zero (MSL 8.1 allows it)");
                Ok(())
            } else {
                Err("neither the exact nor the flushed bfloat16 result".into())
            }
        }),
    });
}

/// 64-bit integers (ROADMAP 8q): the shared wide kernel at `I64` and `U64` (every result crosses the 32-bit boundary).
fn wide_cases(v: &mut Vec<Case>) {
    use wide_kernel::expected_words;
    for (ty, tag, signed) in [
        (ElementKind::I64, "i64", true),
        (ElementKind::U64, "u64", false),
    ] {
        let mut k = wide_kernel::wide_kernel(ty);
        k.name = "wide_case".into();
        let (want, other) = (expected_words(signed), expected_words(!signed));
        let cells = want.len();
        v.push(Case {
            name: format!("wide_{tag}"),
            kernel: k.clone(),
            grid: Some([1, 1, 1]),
            inputs: vec![],
            out_cells: cells,
            check: eq_words(want),
        });
        // The other signedness' oracle (logical vs arithmetic shift, MIN / -1) must NOT be what the device computes.
        v.push(Case {
            name: format!("wide_{tag}_vs_other_signedness"),
            kernel: k,
            grid: Some([1, 1, 1]),
            inputs: vec![],
            out_cells: cells,
            check: differs_from(eq_words(other)),
        });
    }
}

/// 32-bit integer `Div` (the new total lowering): every pair of the shared dividends and divisors, with the rule's
/// corners (a zero divisor, signed `MIN / -1`), at `I32` and `U32`; the zero-divisor mutant (an unguarded `/`) must differ.
fn div_cases(v: &mut Vec<Case>) {
    for (ty, tag, signed) in [
        (ElementKind::I32, "i32", true),
        (ElementKind::U32, "u32", false),
    ] {
        let mut k = div_kernel::div_kernel(ty);
        k.name = "div_case".into();
        let (a, b) = div_kernel::inputs();
        let want: Vec<u32> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| div_kernel::expected(signed, x, y))
            .collect();
        let mutant: Vec<u32> = a
            .iter()
            .zip(&b)
            .zip(&want)
            .map(|((&x, &y), &w)| {
                if y == 0 {
                    div_kernel::unguarded_zero(x)
                } else {
                    w
                }
            })
            .collect();
        let n = want.len();
        v.push(Case {
            name: format!("div_{tag}"),
            kernel: k.clone(),
            grid: Some([1, 1, 1]),
            inputs: vec![a.clone(), b.clone()],
            out_cells: n,
            check: eq_words(want),
        });
        v.push(Case {
            name: format!("div_{tag}_vs_unguarded_zero_divisor"),
            kernel: k,
            grid: Some([1, 1, 1]),
            inputs: vec![a, b],
            out_cells: n,
            check: differs_from(eq_words(mutant)),
        });
    }
}

fn narrow_cases(v: &mut Vec<Case>) {
    use narrow_kernel::{THREADS, input_words, widened, widened_wrong, width, wrapped};
    for (elem, tag) in [
        (ElementKind::I8, "i8"),
        (ElementKind::U8, "u8"),
        (ElementKind::I16, "i16"),
        (ElementKind::U16, "u16"),
    ] {
        let mut k = narrow_kernel::widen_kernel(elem);
        k.name = "narrow_case".into();
        let want: Vec<u32> = (0..THREADS).map(|t| widened(elem, t)).collect();
        let wrong: Vec<u32> = (0..THREADS).map(|t| widened_wrong(elem, t)).collect();
        v.push(Case {
            name: format!("narrow_widen_{tag}"),
            kernel: k.clone(),
            grid: None,
            inputs: vec![input_words(elem, 256)],
            out_cells: 256,
            check: Box::new(move |o| {
                eq_words(want.clone())(&Out {
                    simd: 0,
                    words: o.words[..THREADS as usize].to_vec(),
                })
            }),
        });
        v.push(Case {
            name: format!("narrow_widen_{tag}_vs_wrong_extension"),
            kernel: k,
            grid: None,
            inputs: vec![input_words(elem, 256)],
            out_cells: 256,
            check: differs_from(Box::new(move |o| {
                eq_words(wrong.clone())(&Out {
                    simd: 0,
                    words: o.words[..THREADS as usize].to_vec(),
                })
            })),
        });
        let mut k = narrow_kernel::wrap_kernel(elem);
        k.name = "narrow_case".into();
        let w = width(elem);
        v.push(Case {
            name: format!("narrow_wrap_{tag}"),
            kernel: k,
            grid: None,
            inputs: vec![input_words(elem, 256)],
            out_cells: 256,
            check: Box::new(move |o| {
                let bytes: Vec<u8> = o.words.iter().flat_map(|x| x.to_le_bytes()).collect();
                for t in 0..THREADS as usize {
                    let mut raw = [0u8; 4];
                    raw[..w].copy_from_slice(&bytes[t * w..t * w + w]);
                    let (g, e) = (u32::from_le_bytes(raw), wrapped(elem, t as u32));
                    if g != e {
                        return Err(format!("t={t}: got {g:#x} want {e:#x}"));
                    }
                }
                Ok(())
            }),
        });
    }
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let n = 64u32;
    let input: Vec<u32> = (0..n).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    v.push(Case {
        name: "reverse".into(),
        kernel: reverse_kernel::reverse(n, true, reverse_kernel::SharedLen::Static),
        grid: Some([1, 1, 1]),
        inputs: vec![input],
        out_cells: n as usize,
        check: eq_words(want),
    });
    let pin: Vec<u32> = (0..n).map(|t| (t * 5 + 1) % 4).collect();
    let pwant: Vec<u32> = (0..n).map(|t| t + 10 * pin[t as usize]).collect();
    v.push(Case {
        name: "private_index".into(),
        kernel: private_kernel::private_index(n, unpopped::exec::MemorySpace::Private),
        grid: Some([1, 1, 1]),
        inputs: vec![pin],
        out_cells: n as usize,
        check: eq_words(pwant),
    });
    v.push(Case {
        name: "masked".into(),
        kernel: masked_kernel::masked_kernel(),
        grid: Some([1, 1, 1]),
        inputs: vec![masked_kernel::input()],
        out_cells: masked_kernel::THREADS as usize,
        check: eq_words(
            (0..masked_kernel::THREADS)
                .map(masked_kernel::expected)
                .collect(),
        ),
    });
    let nx = (grid_kernel::BLOCK[0] * grid_kernel::GRID[0]) as usize;
    let ny = (grid_kernel::BLOCK[1] * grid_kernel::GRID[1]) as usize;
    v.push(Case {
        name: "grid".into(),
        kernel: grid_kernel::grid_kernel(),
        grid: Some(grid_kernel::GRID),
        inputs: vec![],
        out_cells: grid_kernel::cells(),
        check: eq_words(
            (0..grid_kernel::cells())
                .map(|i| {
                    grid_kernel::expected(
                        (i % nx) as u32,
                        ((i / nx) % ny) as u32,
                        (i / (nx * ny)) as u32,
                    )
                })
                .collect(),
        ),
    });
    let flow_want: Vec<u32> = (0..256)
        .map(|t| flow_kernel::expected(t).unwrap_or(SENTINEL))
        .collect();
    v.push(Case {
        name: "flow".into(),
        kernel: flow_kernel::flow_kernel(),
        grid: None,
        inputs: vec![vec![0u32; 256]],
        out_cells: 256,
        check: eq_words(flow_want),
    });
    for racy in [false, true] {
        let total = SENTINEL.wrapping_add(atomic_kernel::THREADS);
        let ok: Check = Box::new(move |o| {
            if o.words[0] == total {
                Ok(())
            } else {
                Err(format!(
                    "lost updates: got {:#x} want {total:#x}",
                    o.words[0]
                ))
            }
        });
        v.push(Case {
            name: if racy {
                "atomic_racy_vs_exact_sum"
            } else {
                "atomic"
            }
            .into(),
            kernel: atomic_kernel::atomic_kernel(racy),
            grid: Some([atomic_kernel::BLOCKS, 1, 1]),
            inputs: vec![],
            out_cells: 4,
            check: if racy { differs_from(ok) } else { ok },
        });
    }
    v.push(Case {
        name: "subgroup".into(),
        kernel: subgroup_kernel::subgroup_kernel(1, 128),
        grid: Some([1, 1, 1]),
        inputs: vec![],
        out_cells: subgroup_kernel::BLOCK as usize,
        check: Box::new(|o| {
            let want: Vec<u32> = (0..subgroup_kernel::BLOCK)
                .map(|t| subgroup_kernel::expected(t, o.simd))
                .collect();
            eq_words(want)(o)
        }),
    });
    float_cases(&mut v);
    subnormal_cases(&mut v);
    narrow_cases(&mut v);
    wide_cases(&mut v);
    div_cases(&mut v);
    v
}

/// A hand-written MSL case: not emitted, so the device can answer a question about the language itself (here: what Metal
/// permits for a 64-bit atomic, per the specification) before the emitter spells it.
struct RawCase {
    name: &'static str,
    source: &'static str,
    entry: &'static str,
    block: [u32; 3],
    grid: [u32; 3],
    out_cells: usize,
    /// A pipeline the device cannot create is recorded as UNMEASURED-ON-HARDWARE (with the error), not a failure: the
    /// question is whether this device can run it at all. If it runs, `check` still decides.
    refusal_is_a_note: bool,
    check: Check,
}

/// The 64-bit atomic modify functions exactly as MSL 4.1 section 6.16.4.6 gives them: `atomic_max_explicit` and
/// `atomic_min_explicit` (no `fetch`; they return void) on `device atomic_ulong*` with `memory_order_relaxed`.
fn raw_cases() -> Vec<RawCase> {
    let fill = 0xAAAA_AAAA_AAAA_AAAAu64;
    let max_base = 0xBBBB_BBBB_0000_0000u64;
    vec![RawCase {
        name: "raw_atomic_ulong_max_min",
        source: "// Hand-written (not emitted): MSL 4.1 6.16.4.6 atomic modify functions on atomic_ulong.
#include <metal_stdlib>
using namespace metal;
// threadsPerThreadgroup: 64 x 1 x 1
kernel void au_minmax(
    device atomic_ulong* b0 [[buffer(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]])
{
    ulong g = (ulong)(gid.x * 64u + tid.x);
    atomic_max_explicit(&b0[0], 0xBBBBBBBB00000000ul + g, memory_order_relaxed);
    atomic_min_explicit(&b0[1], (g + 1ul) + 8589934592ul, memory_order_relaxed);
}
",
        entry: "au_minmax",
        block: [64, 1, 1],
        grid: [16, 1, 1],
        out_cells: 4,
        refusal_is_a_note: true,
        check: Box::new(move |o| {
            let c: Vec<u64> = o
                .words
                .chunks(2)
                .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
                .collect();
            // Unsigned max over 1024 threads of `max_base + g` (above the fill) ends at `max_base + 1023`; the min of
            // `g + 1 + 2^33` ends at `1 + 2^33` (below the fill). Either alone left at the fill is a no-op atomic.
            if c[0] != max_base + 1023 {
                return Err(format!("max cell: got {:#x} want {:#x} (fill {fill:#x})", c[0], max_base + 1023));
            }
            if c[1] != (1u64 << 33) + 1 {
                return Err(format!("min cell: got {:#x} want {:#x}", c[1], (1u64 << 33) + 1));
            }
            Ok(())
        }),
    }]
}

fn block_of(src: &str) -> [u32; 3] {
    let line = src
        .lines()
        .find_map(|l| l.strip_prefix("// threadsPerThreadgroup: "))
        .expect("the emitted source names its threadgroup size");
    let d: Vec<u32> = line
        .split(" x ")
        .map(|x| x.trim().parse().expect("a dimension"))
        .collect();
    [d[0], d[1], d[2]]
}

fn hex(words: &[u32]) -> String {
    words
        .iter()
        .map(|w| format!("{w:08x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn device_cases_are_written() {
    let dir = std::env::var_os("VULKANE_MSL_DEVICE_DIR").map(PathBuf::from);
    let mut names = Vec::new();
    for c in cases() {
        let g = Msl
            .lower_kernel(&c.kernel)
            .unwrap_or_else(|e| panic!("{}: {e:?}", c.name));
        let block = block_of(&g.source);
        let grid = c
            .grid
            .unwrap_or([(c.out_cells as u32).div_ceil(block[0]), 1, 1]);
        if let Some(d) = &dir {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join(format!("{}.metal", c.name)), &g.source).unwrap();
            let mut s = String::new();
            let _ = writeln!(s, "entry {}", g.name);
            let _ = writeln!(s, "grid {} {} {}", grid[0], grid[1], grid[2]);
            let _ = writeln!(s, "block {} {} {}", block[0], block[1], block[2]);
            for i in &c.inputs {
                let _ = writeln!(s, "buf {}", hex(i));
            }
            let _ = writeln!(s, "buf {}", hex(&vec![SENTINEL; c.out_cells]));
            std::fs::write(d.join(format!("{}.case", c.name)), s).unwrap();
        }
        names.push(c.name);
    }
    for r in raw_cases() {
        if let Some(d) = &dir {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join(format!("{}.metal", r.name)), r.source).unwrap();
            let mut s = String::new();
            let _ = writeln!(s, "entry {}", r.entry);
            let _ = writeln!(s, "grid {} {} {}", r.grid[0], r.grid[1], r.grid[2]);
            let _ = writeln!(s, "block {} {} {}", r.block[0], r.block[1], r.block[2]);
            let _ = writeln!(s, "buf {}", hex(&vec![SENTINEL; r.out_cells]));
            std::fs::write(d.join(format!("{}.case", r.name)), s).unwrap();
        }
        names.push(r.name.to_string());
    }
    assert!(names.len() >= 25, "{} cases: {names:?}", names.len());
    if let Some(d) = &dir {
        std::fs::write(d.join("CASES.txt"), names.join("\n") + "\n").unwrap();
    }
}

#[test]
fn device_results_match_the_oracles() {
    let Some(dir) = std::env::var_os("VULKANE_MSL_DEVICE_DIR").map(PathBuf::from) else {
        return;
    };
    if !dir.join("RAN").exists() {
        return; // the writing step ran, the device step has not (a plain local run)
    }
    let (mut bad, mut n) = (Vec::new(), 0);
    let all = cases()
        .into_iter()
        .map(|c| (c.name, c.out_cells, c.check, false))
        .chain(raw_cases().into_iter().map(|r| {
            (
                r.name.to_string(),
                r.out_cells,
                r.check,
                r.refusal_is_a_note,
            )
        }));
    for (name, out_cells, check, refusal_is_a_note) in all {
        n += 1;
        let text = std::fs::read_to_string(dir.join(format!("{name}.out")))
            .unwrap_or_else(|e| format!("error no .out file: {e}"));
        let mut simd = 0;
        let mut words = None;
        let mut err = None;
        for l in text.lines() {
            if let Some(r) = l.strip_prefix("simd ") {
                simd = r.trim().parse().unwrap_or(0);
            } else if let Some(r) = l.strip_prefix("words") {
                words = Some(
                    r.split_whitespace()
                        .map(|h| u32::from_str_radix(h, 16).unwrap())
                        .collect::<Vec<u32>>(),
                );
            } else if l.starts_with("error") {
                err = Some(l.to_string());
            }
        }
        let r = match (words, err) {
            (Some(words), _) if words.len() == out_cells => (check)(&Out { simd, words }),
            (_, Some(e)) if refusal_is_a_note && e.contains("makeComputePipelineState") => {
                eprintln!("DEVICE note UNMEASURED-ON-HARDWARE {name}: {e}");
                Ok(())
            }
            (_, Some(e)) => Err(e),
            _ => Err(format!("no usable output: {text}")),
        };
        match r {
            Ok(()) => eprintln!("DEVICE ok    {}", name),
            Err(e) => {
                eprintln!("DEVICE FAIL  {}: {e}", name);
                bad.push(name);
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {n} device cases failed: {bad:?}",
        bad.len()
    );
}
