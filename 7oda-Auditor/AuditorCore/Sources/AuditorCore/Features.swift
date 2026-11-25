import Foundation

/// Hand-made features of SPEC 7 (`featureVersion = 2`, d = 39).
///
/// Energies are summed over samples and events before any logarithm is taken, and the
/// response is described mainly by shapes (axis shares and normalized cross products of
/// simultaneously sampled channels), so the unknown sampling phase of an aliased carrier
/// cancels. Every sum is left to right (window sums added to running totals in time
/// order); no feature compares computed values, and every quotient carries `ε`, so all
/// features are finite for any input.
public enum FeatureExtractor {
    public static let version = 2

    /// The 39 feature names in index order.
    public static let names: [String] = {
        let a = ["ax", "ay", "az"]
        let g = ["gx", "gy", "gz"]
        let x = ["axy", "axz", "ayz"]
        var n = ["tap_gain_accel", "tap_log_accel_gyro"]
        n += a.map { "tap_burst_share_" + $0 }
        n += x.map { "tap_burst_cross_" + $0 }
        n += g.map { "tap_burst_share_" + $0 }
        n += a.map { "tap_tail_share_" + $0 }
        n += x.map { "tap_tail_cross_" + $0 }
        n += g.map { "tap_tail_share_" + $0 }
        n += ["tap_tail_log_accel_gyro", "ramp_gain_accel", "ramp_log_accel_gyro"]
        n += a.map { "ramp_share_" + $0 }
        n += x.map { "ramp_cross_" + $0 }
        n += g.map { "ramp_share_" + $0 }
        n += ["quiet_tremor_accel", "quiet_tremor_gyro", "quiet_rms_accel", "quiet_rms_gyro",
              "gravity_x", "gravity_y", "gravity_z"]
        return n
    }()

    /// Indices `R` of SPEC 7.5, reported relative to the phone on the table.
    public static let calibrationRelativeIndices: [Int] = [0, 1, 20, 21, 22]

    static let epsilon = 1e-12

    /// The absolute features of SPEC 7.2-7.4.
    public static func extractAbsolute(_ p: ProcessedSignal) -> [Double] {
        return tapFeatures(p) + rampFeatures(p) + quietFeatures(p)
    }

    /// SPEC 7.5: `f[r] − calibration.tableFeatures[r]` for `r` in `R` when a calibration is given.
    public static func extract(_ p: ProcessedSignal, calibration: Calibration?) -> [Double] {
        var f = extractAbsolute(p)
        if let table = calibration?.tableFeatures {
            for r in calibrationRelativeIndices where r < f.count && r < table.count {
                f[r] = f[r] - table[r]
            }
        }
        return f
    }

    // MARK: - Helpers

    static func lnRatio(_ a: Double, _ b: Double) -> Double {
        return log((a + epsilon) / (b + epsilon))
    }

    /// Running per-channel energy (and cross-product) totals of one three-channel stream.
    struct Shape {
        static let pairs = [(0, 1), (0, 2), (1, 2)]
        var energy = [0.0, 0.0, 0.0]
        var crossSum = [0.0, 0.0, 0.0]
        let withCross: Bool

        init(cross: Bool) {
            withCross = cross
        }

        /// Adds the window `[a, b)` (empty when `b <= a`): each window sum is added to the total.
        mutating func add(_ signal: [[Double]], _ a: Int, _ b: Int) {
            guard signal.count == 3 else { return }
            let lo = max(a, 0)
            let hi = min(b, signal[0].count, signal[1].count, signal[2].count)
            if hi <= lo { return }
            for c in 0..<3 {
                var s = 0.0
                for i in lo..<hi {
                    let v = signal[c][i]
                    s = s + v * v
                }
                energy[c] = energy[c] + s
            }
            if withCross {
                for (k, pair) in Shape.pairs.enumerated() {
                    var s = 0.0
                    for i in lo..<hi {
                        s = s + signal[pair.0][i] * signal[pair.1][i]
                    }
                    crossSum[k] = crossSum[k] + s
                }
            }
        }

        var total: Double {
            return (energy[0] + energy[1]) + energy[2]
        }

        func shares() -> [Double] {
            let t = total
            return energy.map { $0 / (t + FeatureExtractor.epsilon) }
        }

        func cross() -> [Double] {
            let t = total
            return crossSum.map { $0 / (t + FeatureExtractor.epsilon) }
        }
    }

    /// Left-to-right sum of `values[a..<b]`, empty (0) when the window is empty.
    static func windowSum(_ values: [Double], _ a: Int, _ b: Int) -> Double {
        let lo = max(a, 0)
        let hi = min(b, values.count)
        if hi <= lo { return 0.0 }
        return Numerics.sum(values, lo..<hi)
    }

    // MARK: - Taps (SPEC 7.1, 7.2): indices 0-20

    static func tapFeatures(_ p: ProcessedSignal) -> [Double] {
        let events = p.challenge.events
        let n = p.n
        let endNext = p.endIndex + 25
        var eaFull = 0.0
        var egFull = 0.0
        var i2 = 0.0
        var burstA = Shape(cross: true)
        var burstG = Shape(cross: false)
        var tailA = Shape(cross: true)
        var tailG = Shape(cross: false)
        for (ei, event) in events.enumerated() {
            guard case .tap(let tap) = event else { continue }
            let o = p.index(ofPatternTime: tap.time)
            let next = ei + 1 < events.count ? p.index(ofPatternTime: events[ei + 1].time) : endNext
            let u = min(next - 1, n)
            let fullA = max(o - 1, 0)
            let fullB = min(o + 9, u)
            if fullB <= fullA { continue }  // a tap whose full window is empty is skipped entirely
            eaFull = eaFull + windowSum(p.energyAccel, fullA, fullB)
            egFull = egFull + windowSum(p.energyGyro, fullA, fullB)
            i2 = i2 + tap.intensity * tap.intensity
            let burst = (max(o - 1, 0), min(o + 2, u))
            let tail = (max(o + 2, 0), min(o + 9, u))
            burstA.add(p.accelHP, burst.0, burst.1)
            burstG.add(p.gyroHP, burst.0, burst.1)
            tailA.add(p.accelHP, tail.0, tail.1)
            tailG.add(p.gyroHP, tail.0, tail.1)
        }
        var f = [lnRatio(eaFull, i2), lnRatio(eaFull, egFull)]
        f += burstA.shares() + burstA.cross() + burstG.shares()
        f += tailA.shares() + tailA.cross() + tailG.shares()
        f.append(lnRatio(tailA.total, tailG.total))
        return f
    }

    // MARK: - Ramps (SPEC 7.3): indices 21-31

    static func rampFeatures(_ p: ProcessedSignal) -> [Double] {
        let cfg = p.config
        let lag = p.alignment.lag
        var shapeA = Shape(cross: true)
        var shapeG = Shape(cross: false)
        var i2 = 0.0
        for event in p.challenge.events {
            guard case .ramp(let ramp) = event else { continue }
            let s = p.clamp(p.index(ofPatternTime: ramp.time))
            let e = p.clamp(p.index(ofPatternTime: ramp.time + ramp.duration))
            let a = s + 2
            let b = e
            if b <= a { continue }
            let tj = ramp.time
            let dj = ramp.duration
            for i in a..<b {
                var tau = Double(i - lag) / cfg.fs - cfg.preRoll - tj
                tau = min(max(tau, 0.0), dj)
                let c = ramp.intensity(at: tau)
                i2 = i2 + c * c
            }
            shapeA.add(p.accelHP, a, b)
            shapeG.add(p.gyroHP, a, b)
        }
        var f = [lnRatio(shapeA.total, i2), lnRatio(shapeA.total, shapeG.total)]
        f += shapeA.shares() + shapeA.cross() + shapeG.shares()
        return f
    }

    // MARK: - Quiet and grip (SPEC 7.4): indices 32-38

    static func quietFeatures(_ p: ProcessedSignal) -> [Double] {
        let cfg = p.config
        let q = p.quietRange
        let bp = cfg.tremorBandPass
        let ba = p.accel.map { bp.filtfiltZ($0, pad: cfg.pad, dcGain: 0.0) }
        let bg = p.gyro.map { bp.filtfiltZ($0, pad: cfg.pad, dcGain: 0.0) }
        let tremorA = sumOfSquares(ba, count: p.n)
        let tremorG = sumOfSquares(bg, count: p.n)
        var f = [
            log(Numerics.mean(tremorA, q) + 1e-10),
            log(Numerics.mean(tremorG, q) + 1e-10),
            log(Numerics.mean(p.energyAccel, q).squareRoot() + 1e-6),
            log(Numerics.mean(p.energyGyro, q).squareRoot() + 1e-6),
        ]
        let gx = p.gravity.count > 0 ? Numerics.mean(p.gravity[0]) : 0.0
        let gy = p.gravity.count > 1 ? Numerics.mean(p.gravity[1]) : 0.0
        let gz = p.gravity.count > 2 ? Numerics.mean(p.gravity[2]) : 0.0
        let norm = (gx * gx + gy * gy + gz * gz).squareRoot()
        if norm < 1e-6 {
            f += [0.0, 0.0, -1.0]
        } else {
            f += [gx / norm, gy / norm, gz / norm]
        }
        return f
    }

    /// `x[n]² + y[n]² + z[n]²` (left to right) for a three-channel signal.
    static func sumOfSquares(_ s: [[Double]], count: Int) -> [Double] {
        var out = [Double](repeating: 0.0, count: max(count, 0))
        guard s.count == 3, s.allSatisfy({ $0.count >= count }) else { return out }
        for i in 0..<out.count {
            out[i] = s[0][i] * s[0][i] + s[1][i] * s[1][i] + s[2][i] * s[2][i]
        }
        return out
    }
}
