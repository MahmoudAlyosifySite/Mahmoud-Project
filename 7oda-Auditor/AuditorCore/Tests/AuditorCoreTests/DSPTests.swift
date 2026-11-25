import Foundation
import XCTest
import AuditorCore

private struct DSPFixtureFile: Decodable {
    struct BiquadCase: Decodable {
        let kind: String
        let fc: Double
        let fs: Double
        let q: Double
        let coeffs: [Double]
    }
    struct FilterCase: Decodable {
        let name: String?
        let kind: String
        let fc: Double
        let fs: Double
        let q: Double
        let pad: Int
        let dcGain: Double
        let x: [Double]
        let lfilter: [Double]
        let filtfilt: [Double]
    }
    struct ResampleCase: Decodable {
        let name: String?
        let t: [Double]
        let v: [Double]
        let grid: [Double]
        let gapTolerance: Double
        let values: [Double]
        let missing: [Bool]
    }
    struct SortDedupCase: Decodable {
        let name: String?
        let input: [String: [Double]]
        let output: [String: [Double]]
    }
    let biquads: [BiquadCase]
    let filters: [FilterCase]
    let resample: [ResampleCase]
    let sortDedup: [SortDedupCase]?
}

final class DSPTests: XCTestCase {
    private let tolerance = 1e-9

    private func design(_ kind: String, _ fc: Double, _ fs: Double, _ q: Double) -> Biquad? {
        switch kind {
        case "highpass": return Biquad.highPass(cutoff: fc, fs: fs, q: q)
        case "lowpass": return Biquad.lowPass(cutoff: fc, fs: fs, q: q)
        case "bandpass": return Biquad.bandPass(center: fc, fs: fs, q: q)
        default:
            XCTFail("unknown biquad kind '\(kind)'")
            return nil
        }
    }

    /// SPEC 11.3, tolerance `1e-9 · max(1, |x|)`.
    func testDSPFixtureParity() {
        guard let file = Fixture.decode(DSPFixtureFile.self, "dsp_vectors") else { return }
        XCTAssertFalse(file.biquads.isEmpty)
        XCTAssertFalse(file.filters.isEmpty)
        XCTAssertFalse(file.resample.isEmpty)
        for (i, c) in file.biquads.enumerated() {
            guard let b = design(c.kind, c.fc, c.fs, c.q) else { continue }
            assertClose([b.b0, b.b1, b.b2, b.a1, b.a2], c.coeffs, rel: tolerance, "biquads[\(i)] \(c.kind) fc=\(c.fc)")
        }
        for (i, c) in file.filters.enumerated() {
            guard let b = design(c.kind, c.fc, c.fs, c.q) else { continue }
            let label = "filters[\(i)] \(c.name ?? c.kind) n=\(c.x.count)"
            assertClose(b.filter(c.x), c.lfilter, rel: tolerance, "\(label) lfilter")
            assertClose(b.filtfiltZ(c.x, pad: c.pad, dcGain: c.dcGain), c.filtfilt, rel: tolerance, "\(label) filtfiltZ")
        }
        for (i, c) in file.resample.enumerated() {
            let r = Resampler.resample(t: c.t, v: c.v, grid: c.grid, gapTolerance: c.gapTolerance)
            let label = "resample[\(i)] \(c.name ?? "")"
            assertClose(r.values, c.values, rel: tolerance, "\(label) values")
            XCTAssertEqual(r.missing, c.missing, "\(label) missing")
        }
        for (i, c) in (file.sortDedup ?? []).enumerated() {
            let label = "sortDedup[\(i)] \(c.name ?? "")"
            if c.input["gravityX"] != nil {
                let m = StreamCleaner.sortDedup(motionStream(c.input))
                let expected = motionStream(c.output)
                XCTAssertEqual(m.t, expected.t, "\(label) t")
                XCTAssertEqual(m.gravityZ, expected.gravityZ, "\(label) gravityZ")
                XCTAssertEqual(m.rotationX, expected.rotationX, "\(label) rotationX")
            } else {
                let input = SensorStream(t: c.input["t"] ?? [], x: c.input["x"] ?? [],
                                         y: c.input["y"] ?? [], z: c.input["z"] ?? [])
                let expected = SensorStream(t: c.output["t"] ?? [], x: c.output["x"] ?? [],
                                            y: c.output["y"] ?? [], z: c.output["z"] ?? [])
                XCTAssertEqual(StreamCleaner.sortDedup(input), expected, label)
            }
        }
    }

    private func motionStream(_ d: [String: [Double]]) -> MotionStream {
        return MotionStream(
            t: d["t"] ?? [],
            gravityX: d["gravityX"] ?? [], gravityY: d["gravityY"] ?? [], gravityZ: d["gravityZ"] ?? [],
            userAccelX: d["userAccelX"] ?? [], userAccelY: d["userAccelY"] ?? [], userAccelZ: d["userAccelZ"] ?? [],
            rotationX: d["rotationX"] ?? [], rotationY: d["rotationY"] ?? [], rotationZ: d["rotationZ"] ?? [],
            attitudeW: d["attitudeW"] ?? [], attitudeX: d["attitudeX"] ?? [],
            attitudeY: d["attitudeY"] ?? [], attitudeZ: d["attitudeZ"] ?? [])
    }

    func testFiltfiltEdgeCases() {
        let lp = Biquad.lowPass(cutoff: 3.0, fs: 100.0, q: Biquad.butterworthQ)
        let hp = Biquad.highPass(cutoff: 5.0, fs: 100.0, q: Biquad.butterworthQ)
        XCTAssertEqual(lp.filtfiltZ([], pad: 30, dcGain: 1.0), [])
        XCTAssertEqual(lp.filtfiltZ([2.5], pad: 30, dcGain: 1.0), [2.5])
        XCTAssertEqual(hp.filtfiltZ([2.5], pad: 30, dcGain: 0.0), [0.0])
        XCTAssertEqual(hp.filtfiltZ([1.0, 2.0], pad: 30, dcGain: 0.0).count, 2)
        // A constant passes a low-pass unchanged and vanishes through a high-pass.
        let constant = [Double](repeating: -0.98, count: 50)
        for v in lp.filtfiltZ(constant, pad: 30, dcGain: 1.0) { XCTAssertEqual(v, -0.98, accuracy: 1e-12) }
        for v in hp.filtfiltZ(constant, pad: 30, dcGain: 0.0) { XCTAssertEqual(v, 0.0, accuracy: 1e-12) }
        XCTAssertEqual(lp.filter([]), [])
        XCTAssertEqual(Biquad.butterworthQ, 0.7071067811865475)
    }

    func testResampleEdgeCases() {
        let empty = Resampler.resample(t: [], v: [], grid: [0.0, 0.01], gapTolerance: 0.025)
        XCTAssertEqual(empty.values, [0.0, 0.0])
        XCTAssertEqual(empty.missing, [true, true])
        let one = Resampler.resample(t: [1.0], v: [5.0], grid: [0.99, 1.0, 1.02, 1.03], gapTolerance: 0.025)
        XCTAssertEqual(one.values, [5.0, 5.0, 5.0, 5.0])
        XCTAssertEqual(one.missing, [true, false, false, true])
        let two = Resampler.resample(t: [0.0, 0.01, 0.1], v: [0.0, 1.0, 2.0], grid: [0.005, 0.05], gapTolerance: 0.025)
        XCTAssertEqual(two.values[0], 0.5, accuracy: 1e-12)
        XCTAssertEqual(two.missing, [false, true])
        XCTAssertEqual(Resampler.resample(t: [0.0], v: [1.0], grid: [], gapTolerance: 0.025).values, [])
    }

    func testSortDedup() {
        let s = SensorStream(t: [3.0, 1.0, 2.0, 2.0, 1.0, .nan], x: [30, 10, 20, 21, 11, 99],
                             y: [0, 0, 0, 0, 0, 0], z: [3, 1, 2, 2, 1, 9])
        let c = StreamCleaner.sortDedup(s)
        XCTAssertEqual(c.t, [1.0, 2.0, 3.0])
        XCTAssertEqual(c.x, [10, 20, 30])  // first of equal timestamps (stable order) is kept
        XCTAssertEqual(c.z, [1, 2, 3])
        let ragged = SensorStream(t: [1.0, 2.0, 3.0], x: [1.0], y: [1.0, 2.0], z: [])
        XCTAssertEqual(StreamCleaner.sortDedup(ragged), SensorStream.empty)
        XCTAssertEqual(StreamCleaner.sortDedup(MotionStream.empty), MotionStream.empty)
        let motion = MotionStream(t: [2.0, 1.0], gravityX: [0, 0], gravityY: [0, 0], gravityZ: [-2, -1],
                                  rotationX: [0, 0], rotationY: [0, 0], rotationZ: [0.2, 0.1])
        let m = StreamCleaner.sortDedup(motion)
        XCTAssertEqual(m.t, [1.0, 2.0])
        XCTAssertEqual(m.gravityZ, [-1, -2])
        XCTAssertEqual(m.rotationZ, [0.1, 0.2])
        XCTAssertEqual(m.attitudeW, [])
    }
}
