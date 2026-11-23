// SPDX-License-Identifier: MIT OR Apache-2.0
//! The device features a generated module needs the **host** to have enabled.
//!
//! The emitter cannot see the device, so a module that declares `Int64` (any `I64`/`U64` kernel) or
//! `Int64Atomics` (a 64-bit atomic) is only valid on a device created with `shaderInt64` (and, for
//! the atomics, `shaderBufferInt64Atomics` for buffers or `shaderSharedInt64Atomics` for workgroup
//! arrays). Measured on this lane's harness GPU: a device created *without* `shaderInt64` ran an
//! `Int64` module without any error (no validation layer; the spec leaves it undefined), so the
//! driver is no guard. [`required_features`] reads the module's own capabilities so the host can
//! check before it creates the pipeline.

use rspirv::dr::{Operand, load_words};
use rspirv::spirv::Capability;

/// What a module requires of the device it runs on, read from its `OpCapability` list.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RequiredFeatures {
    /// `Int64`: the device must have been created with `shaderInt64`.
    pub shader_int64: bool,
    /// `Int64Atomics`: `shaderBufferInt64Atomics` (storage buffers) and/or
    /// `shaderSharedInt64Atomics` (workgroup arrays), per the memory the atomics touch.
    pub int64_atomics: bool,
}

/// The device features `words` (a SPIR-V module) requires, or why it could not be read.
///
/// # Errors
/// The words are not a loadable SPIR-V module.
pub fn required_features(words: &[u32]) -> Result<RequiredFeatures, String> {
    let module = load_words(words).map_err(|e| format!("not a SPIR-V module: {e}"))?;
    let mut need = RequiredFeatures::default();
    for inst in &module.capabilities {
        if let Some(Operand::Capability(c)) = inst.operands.first() {
            match c {
                Capability::Int64 => need.shader_int64 = true,
                Capability::Int64Atomics => need.int64_atomics = true,
                _ => {}
            }
        }
    }
    Ok(need)
}
