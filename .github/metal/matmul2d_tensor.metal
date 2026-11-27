// SPDX-License-Identifier: MIT OR Apache-2.0
// Compile-only probe (never run): the matmul2d TensorOp example of MSL 4.1 section 7.2.1 (Metal Performance Primitives),
// copied from the spec. Needs Metal 4. Lifted by CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance;
// no attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

[[ kernel ]] void matrixMultiply(
  tensor<device half, dextents<int, 2>> a [[ buffer(0) ]],
  tensor<device half, dextents<int, 2>> b [[ buffer(1) ]],
  tensor<device half, dextents<int, 2>> c [[ buffer(2) ]],
  uint2 tgid [[threadgroup_position_in_grid]]) {
  constexpr auto matmulDescriptor = tensor_ops::matmul2d_descriptor(64, 32, 0);
  tensor_ops::matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
  auto mA = a.slice(0, tgid.y * 64);
  auto mB = b.slice(tgid.x * 32, 0);
  auto mC = c.slice(tgid.x * 32, tgid.y * 64);
  matmulOp.run(mA, mB, mC);
}
