// SPDX-License-Identifier: MIT OR Apache-2.0
// Dispatches each `<name>.case` in a directory on the default Metal device and writes `<name>.out`
// (`simd N` then `words <hex...>`, or `error <text>`). The oracle lives in the Rust test
// (`vulkane-msl-emit/tests/metal_device.rs`), not here: this program only executes.
// Case lines: entry NAME / grid X Y Z / block X Y Z / buf <hex words> (inputs in bind order, the output last).
import Foundation
import Metal

let dir = URL(fileURLWithPath: CommandLine.arguments[1])
guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else {
    print("no Metal device or command queue")
    exit(2)
}
print("device: \(dev.name)")

func words(_ s: Substring) -> [UInt32] { s.split(separator: " ").map { UInt32($0, radix: 16)! } }

func run(_ name: String) -> String {
    let text = try! String(contentsOf: dir.appendingPathComponent("\(name).case"), encoding: .utf8)
    var entry = "", grid = MTLSize(), block = MTLSize()
    var bufs: [[UInt32]] = []
    for l in text.split(separator: "\n") {
        let p = l.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
        switch p[0] {
        case "entry": entry = String(p[1])
        case "grid", "block":
            let d = p[1].split(separator: " ").map { Int($0)! }
            if p[0] == "grid" { grid = MTLSize(width: d[0], height: d[1], depth: d[2]) } else { block = MTLSize(width: d[0], height: d[1], depth: d[2]) }
        case "buf": bufs.append(p.count > 1 ? words(p[1]) : [])
        default: break
        }
    }
    let lib: MTLLibrary
    do { lib = try dev.makeLibrary(URL: dir.appendingPathComponent("\(name).metallib")) } catch { return "error makeLibrary: \(error)" }
    guard let fn = lib.makeFunction(name: entry) else { return "error makeFunction(\(entry)) returned nil" }
    let pso: MTLComputePipelineState
    do { pso = try dev.makeComputePipelineState(function: fn) } catch { return "error makeComputePipelineState: \(error)" }
    var mtl: [MTLBuffer] = []
    for b in bufs {
        let n = max(b.count * 4, 4)
        guard let m = dev.makeBuffer(length: n, options: .storageModeShared) else { return "error makeBuffer(\(n)) returned nil" }
        if !b.isEmpty { b.withUnsafeBytes { m.contents().copyMemory(from: $0.baseAddress!, byteCount: b.count * 4) } }
        mtl.append(m)
    }
    guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return "error command buffer/encoder nil" }
    enc.setComputePipelineState(pso)
    for (i, m) in mtl.enumerated() { enc.setBuffer(m, offset: 0, index: i) }
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: block)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    if cmd.status != .completed { return "error command status \(cmd.status.rawValue): \(String(describing: cmd.error))" }
    let out = mtl[mtl.count - 1]
    let n = bufs[bufs.count - 1].count
    let p = out.contents().bindMemory(to: UInt32.self, capacity: n)
    let hex = (0..<n).map { String(format: "%08x", p[$0]) }.joined(separator: " ")
    return "simd \(pso.threadExecutionWidth)\nwords \(hex)"
}

let names = try! String(contentsOf: dir.appendingPathComponent("CASES.txt"), encoding: .utf8).split(separator: "\n").map(String.init)
for n in names {
    let r = run(n)
    try! r.write(to: dir.appendingPathComponent("\(n).out"), atomically: true, encoding: .utf8)
    print(r.hasPrefix("error") ? "ERROR \(n): \(r)" : "ran   \(n) (\(r.split(separator: "\n")[0]))")
}
try! "done".write(to: dir.appendingPathComponent("RAN"), atomically: true, encoding: .utf8)
