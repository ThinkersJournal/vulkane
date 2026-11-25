// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for `Cast` in `Backend::lower_kernel`: every case validates under spirv-val,
//! `RoundingModeRTE` is declared only for a conversion that rounds (int to float, `f64` to `f32`) and
//! `Float64` only where an `f64` appears, and what is not spelled is declined by name.

use unpopped::backend::Backend;
use unpopped::ir::{RoundMode, Saturation};
use unpopped_vocab::ElementKind::{F16, F32, F64, I64};

use crate::tests::assert_valid;
use crate::{SpirV, required_features};

use cast_kernel::{cases, kernel};
use vulkane_test_support::cast_kernel;

#[test]
fn every_cast_validates_and_declares_rte_only_when_it_rounds() {
    for c in cases() {
        let g = SpirV.lower_kernel(&kernel(&c)).expect("lowers");
        assert_valid(&g);
        let need = required_features(g.words().expect("words")).expect("a module");
        let bits = |t: unpopped_vocab::ElementKind| match t {
            F64 => 64,
            F16 => 16,
            _ => 32,
        };
        let is_f = |t| matches!(t, F16 | F32 | F64);
        // A conversion that rounds: integer to float, or a narrowing float conversion.
        let rounds = (!is_f(c.src) && is_f(c.dst))
            || (is_f(c.src) && is_f(c.dst) && bits(c.src) > bits(c.dst));
        let want_rte = |w: u32| rounds && bits(c.dst) == w;
        assert_eq!(
            need.rounding_mode_rte_f16,
            want_rte(16),
            "{} {need:?}",
            c.name
        );
        assert_eq!(
            need.rounding_mode_rte_f32,
            want_rte(32),
            "{} {need:?}",
            c.name
        );
        assert_eq!(
            need.rounding_mode_rte_f64,
            want_rte(64),
            "{} {need:?}",
            c.name
        );
        let has_f16 = [c.src, c.dst, c.compute].contains(&F16);
        assert_eq!(need.shader_float16, has_f16, "{} {need:?}", c.name);
        assert_eq!(need.storage_buffer_16bit, has_f16, "{} {need:?}", c.name);
        let has_f64 = [c.src, c.dst, c.compute].contains(&F64);
        assert_eq!(need.shader_float64, has_f64, "{} {need:?}", c.name);
        assert_eq!(
            g.source.matches("OpCapability Float64").count(),
            usize::from(has_f64),
            "{}",
            c.name
        );
    }
}

#[test]
fn casts_that_are_not_spelled_are_declined_by_name() {
    let base = || {
        cases()
            .into_iter()
            .find(|c| c.name == "i32_to_f32")
            .unwrap()
    };
    let f2i = || {
        cases()
            .into_iter()
            .find(|c| c.name == "f32_to_i32_nearest_even")
            .unwrap()
    };
    let mut bad = Vec::new();
    let mut c = base();
    c.round = RoundMode::TowardZero;
    bad.push((c, "NearestEven"));
    let mut c = f2i();
    c.round = RoundMode::NearestAway;
    bad.push((c, "Round"));
    let mut c = f2i();
    c.sat = Saturation::Wrap;
    bad.push((c, "Clamp"));
    let mut c = cases()
        .into_iter()
        .find(|c| c.name == "f64_to_f32")
        .unwrap();
    c.sat = Saturation::ClampFinite;
    bad.push((c, "NearestEven"));
    for (c, needle) in bad {
        match SpirV.lower_kernel(&kernel(&c)) {
            Err(e) => {
                let text = format!("{e:?}");
                assert!(text.contains(needle), "{}: {text}", c.name);
            }
            Ok(_) => panic!("{}: expected a decline", c.name),
        }
    }
    // Control: the unmodified cases lower, so the declines are the cast and not the kernel shape.
    SpirV.lower_kernel(&kernel(&base())).expect("lowers");
    SpirV.lower_kernel(&kernel(&f2i())).expect("lowers");
    let _ = I64;
}
