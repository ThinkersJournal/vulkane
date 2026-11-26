// SPDX-License-Identifier: MIT OR Apache-2.0
//! The device features a generated module needs the **host** to have enabled. **The driver is NOT a
//! guard**: a missing feature is not refused at pipeline creation (measured below), so this is the
//! host's check, not a nicety.
//!
//! The emitter cannot see the device, so a module that declares `Int64` (any `I64`/`U64` kernel) or
//! `Int64Atomics` (a 64-bit atomic) is only valid on a device created with `shaderInt64` (and, for
//! the atomics, `shaderBufferInt64Atomics` for buffers or `shaderSharedInt64Atomics` for workgroup
//! arrays). Measured on this lane's harness GPU: a device created *without* `shaderInt64` ran an
//! `Int64` module without any error (no validation layer; the spec leaves it undefined), so the
//! driver is no guard. [`required_features`] reads the module's own capabilities so the host can
//! check before it creates the pipeline.

use rspirv::dr::{Operand, load_words};
use rspirv::spirv::{Capability, ExecutionMode};

/// What a module requires of the device it runs on, read from its `OpCapability` list.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RequiredFeatures {
    /// `Int64`: the device must have been created with `shaderInt64`.
    pub shader_int64: bool,
    /// `Int64Atomics`: `shaderBufferInt64Atomics` (storage buffers) and/or
    /// `shaderSharedInt64Atomics` (workgroup arrays), per the memory the atomics touch.
    pub int64_atomics: bool,
    /// `Int8`: `shaderInt8` (Vulkan 1.2 `VkPhysicalDeviceVulkan12Features`).
    pub shader_int8: bool,
    /// `StorageBuffer8BitAccess`: `storageBuffer8BitAccess` (Vulkan 1.2).
    pub storage_buffer_8bit: bool,
    /// `Int16`: `shaderInt16` (core `VkPhysicalDeviceFeatures`).
    pub shader_int16: bool,
    /// `StorageBuffer16BitAccess`: `storageBuffer16BitAccess` (Vulkan 1.1).
    pub storage_buffer_16bit: bool,
    /// `Float64`: `shaderFloat64`.
    pub shader_float64: bool,
    /// `DenormPreserve 32`: `shaderDenormPreserveFloat32` (`VkPhysicalDeviceFloatControlsProperties`).
    pub denorm_preserve_f32: bool,
    /// `DenormPreserve 64`: `shaderDenormPreserveFloat64`.
    pub denorm_preserve_f64: bool,
    /// `SignedZeroInfNanPreserve 32`: `shaderSignedZeroInfNanPreserveFloat32`.
    pub signed_zero_inf_nan_preserve_f32: bool,
    /// `SignedZeroInfNanPreserve 64`: `shaderSignedZeroInfNanPreserveFloat64`.
    pub signed_zero_inf_nan_preserve_f64: bool,
    /// `Float16`: `shaderFloat16` (Vulkan 1.2 `VkPhysicalDeviceVulkan12Features`).
    pub shader_float16: bool,
    /// `DenormPreserve 16`: `shaderDenormPreserveFloat16`.
    pub denorm_preserve_f16: bool,
    /// `SignedZeroInfNanPreserve 16`: `shaderSignedZeroInfNanPreserveFloat16`.
    pub signed_zero_inf_nan_preserve_f16: bool,
    /// `RoundingModeRTE 16`: `shaderRoundingModeRTEFloat16`.
    pub rounding_mode_rte_f16: bool,
    /// `RoundingModeRTE 32`: `shaderRoundingModeRTEFloat32` (a rounding conversion at 32 bits).
    pub rounding_mode_rte_f32: bool,
    /// `RoundingModeRTE 64`: `shaderRoundingModeRTEFloat64`.
    pub rounding_mode_rte_f64: bool,
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
                Capability::Float64 => need.shader_float64 = true,
                Capability::Float16 => need.shader_float16 = true,
                Capability::Int8 => need.shader_int8 = true,
                Capability::StorageBuffer8BitAccess => need.storage_buffer_8bit = true,
                Capability::Int16 => need.shader_int16 = true,
                Capability::StorageBuffer16BitAccess => need.storage_buffer_16bit = true,
                _ => {}
            }
        }
    }
    // The float-controls modes are execution modes (the width is their operand), not capabilities.
    for inst in &module.execution_modes {
        if let [
            _,
            Operand::ExecutionMode(mode),
            Operand::LiteralBit32(width),
        ] = inst.operands.as_slice()
        {
            match (mode, width) {
                (ExecutionMode::DenormPreserve, 32) => need.denorm_preserve_f32 = true,
                (ExecutionMode::DenormPreserve, 64) => need.denorm_preserve_f64 = true,
                (ExecutionMode::DenormPreserve, 16) => need.denorm_preserve_f16 = true,
                (ExecutionMode::SignedZeroInfNanPreserve, 16) => {
                    need.signed_zero_inf_nan_preserve_f16 = true;
                }
                (ExecutionMode::RoundingModeRTE, 16) => need.rounding_mode_rte_f16 = true,
                (ExecutionMode::RoundingModeRTE, 32) => need.rounding_mode_rte_f32 = true,
                (ExecutionMode::RoundingModeRTE, 64) => need.rounding_mode_rte_f64 = true,
                (ExecutionMode::SignedZeroInfNanPreserve, 32) => {
                    need.signed_zero_inf_nan_preserve_f32 = true;
                }
                (ExecutionMode::SignedZeroInfNanPreserve, 64) => {
                    need.signed_zero_inf_nan_preserve_f64 = true;
                }
                _ => {}
            }
        }
    }
    Ok(need)
}
