// SPDX-License-Identifier: MIT OR Apache-2.0
//! `glslangValidator` and `spirv-val` as the GLSL live tests use them.

use std::io::Write;
use std::process::{Command, Stdio};

use unpopped::backend::GeneratedKernel;

/// `glslangValidator` to SPIR-V words; `None` when it is absent (a failure under `VULKANE_REQUIRE_GLSLANG`).
pub fn glslang_words(k: &GeneratedKernel) -> Option<Vec<u32>> {
    let dir = tempfile::tempdir().unwrap();
    let (src, spv) = (dir.path().join("k.comp"), dir.path().join("k.spv"));
    std::fs::write(&src, &k.source).unwrap();
    let out = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.1", "-S", "comp", "-o"])
        .arg(&spv)
        .arg(&src)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_GLSLANG").is_none(),
                "VULKANE_REQUIRE_GLSLANG is set but `glslangValidator` could not be run: {e}"
            );
            return None;
        }
    };
    assert!(
        out.status.success(),
        "glslangValidator failed:\n{}{}\n--- source ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
        k.source
    );
    let bytes = std::fs::read(&spv).unwrap();
    Some(
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    )
}

/// `spirv-val --target-env vulkan1.1` over `words`; `None` when it is absent and not required.
pub fn validates(words: &[u32]) -> Option<(bool, String)> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.1", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            let _ = c.stdin.take().expect("piped stdin").write_all(&bytes);
            c.wait_with_output()
        });
    match child {
        Ok(o) => Some((
            o.status.success(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
        )),
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_SPIRV_VAL").is_none(),
                "VULKANE_REQUIRE_SPIRV_VAL is set but `spirv-val` could not be run: {e}"
            );
            None
        }
    }
}
