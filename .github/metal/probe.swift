// SPDX-License-Identifier: MIT OR Apache-2.0
// Records what MTLCreateSystemDefaultDevice returns on this runner. Always exits 0: a hosted VM
// with no GPU is a measurement, not a failure.
import Metal
if let d = MTLCreateSystemDefaultDevice() {
    print("device: \(d.name)")
    print("lowPower: \(d.isLowPower) unifiedMemory: \(d.hasUnifiedMemory) registryID: \(d.registryID)")
    print("supportsFamily apple7: \(d.supportsFamily(.apple7)) apple9: \(d.supportsFamily(.apple9)) metal3: \(d.supportsFamily(.metal3))")
} else {
    print("device: none")
}
let all = MTLCopyAllDevices()
print("MTLCopyAllDevices: \(all.count)")
