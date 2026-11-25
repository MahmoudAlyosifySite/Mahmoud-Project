import Foundation

// MARK: - Biquads (SPEC 5.3, 5.4)

/// A second-order IIR section with `a0` normalized to 1 (RBJ cookbook designs).
public struct Biquad: Equatable {
    public var b0: Double
    public var b1: Double
    public var b2: Double
    public var a1: Double
    public var a2: Double

    public init(b0: Double, b1: Double, b2: Double, a1: Double, a2: Double) {
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
    }

    /// Q of the Butterworth-equivalent sections, computed as `1 / sqrt(2)`
    /// (0.7071067811865475; `sqrt(0.5)` is one ulp larger).
    public static let butterworthQ: Double = 1.0 / (2.0).squareRoot()

    private static func design(_ fc: Double, _ fs: Double, _ q: Double) -> (c: Double, alpha: Double) {
        let w0 = 2.0 * Double.pi * fc / fs
        let c = cos(w0)
        let s = sin(w0)
        return (c, s / (2.0 * q))
    }

    private static func normalized(_ b0: Double, _ b1: Double, _ b2: Double,
                                   _ a0: Double, _ a1: Double, _ a2: Double) -> Biquad {
        return Biquad(b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0)
    }

    /// RBJ high-pass: `b = [(1+c)/2, -(1+c), (1+c)/2]`, `a = [1+α, -2c, 1-α]`.
    public static func highPass(cutoff: Double, fs: Double, q: Double) -> Biquad {
        let (c, alpha) = design(cutoff, fs, q)
        return normalized((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// RBJ low-pass: `b = [(1-c)/2, (1-c), (1-c)/2]`, `a = [1+α, -2c, 1-α]`.
    public static func lowPass(cutoff: Double, fs: Double, q: Double) -> Biquad {
        let (c, alpha) = design(cutoff, fs, q)
        return normalized((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// RBJ band-pass with 0 dB peak: `b = [α, 0, -α]`, `a = [1+α, -2c, 1-α]`.
    public static func bandPass(center: Double, fs: Double, q: Double) -> Biquad {
        let (c, alpha) = design(center, fs, q)
        return normalized(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// Direct form I with zero initial state (SPEC 5.3), evaluated left to right as
    /// `b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]`.
    public func filter(_ x: [Double]) -> [Double] {
        var out = [Double](repeating: 0.0, count: x.count)
        var x1 = 0.0
        var x2 = 0.0
        var y1 = 0.0
        var y2 = 0.0
        for i in 0..<x.count {
            let xi = x[i]
            let yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = yi
            x2 = x1
            x1 = xi
            y2 = y1
            y1 = yi
        }
        return out
    }

    /// Zero-phase filter of SPEC 5.4: odd extension by `P = min(pad, n - 1)` samples,
    /// forward and backward passes started from the first sample (step-free).
    /// `dcGain` is 1 for low-pass and 0 for high-pass and band-pass.
    public func filtfiltZ(_ x: [Double], pad: Int, dcGain: Double) -> [Double] {
        let n = x.count
        if n == 0 { return [] }
        if n == 1 { return [dcGain * x[0]] }
        let p = min(max(pad, 0), n - 1)
        var xp: [Double] = []
        xp.reserveCapacity(n + 2 * p)
        var k = p
        while k >= 1 {
            xp.append(2.0 * x[0] - x[k])
            k -= 1
        }
        xp.append(contentsOf: x)
        k = n - 2
        while k >= n - 1 - p {
            xp.append(2.0 * x[n - 1] - x[k])
            k -= 1
        }
        let c1 = xp[0]
        var y = filter(xp.map { $0 - c1 })
        for i in 0..<y.count { y[i] = y[i] + dcGain * c1 }
        let r = Array(y.reversed())
        let c2 = r[0]
        var z = filter(r.map { $0 - c2 })
        for i in 0..<z.count { z[i] = z[i] + dcGain * c2 }
        let zr = Array(z.reversed())
        return Array(zr[p..<(p + n)])
    }
}

// MARK: - Resampling (SPEC 5.2)

public enum Resampler {
    /// Linear interpolation of `(t, v)` onto `grid` with missing flags (SPEC 5.2).
    /// `t` must already be sorted and deduplicated. For each grid time `g`, `j` is the
    /// largest index with `t[j] <= g`; before the first sample the first value is used
    /// (missing), after the last sample the last value (missing if more than
    /// `gapTolerance` later), otherwise the interpolated value (missing if the
    /// bracketing samples are more than `gapTolerance` apart). No samples: zeros, all missing.
    public static func resample(t: [Double], v: [Double], grid: [Double],
                                gapTolerance: Double) -> (values: [Double], missing: [Bool]) {
        let m = min(t.count, v.count)
        let count = grid.count
        if m == 0 {
            return ([Double](repeating: 0.0, count: count), [Bool](repeating: true, count: count))
        }
        var values = [Double](repeating: 0.0, count: count)
        var missing = [Bool](repeating: false, count: count)
        for k in 0..<count {
            let g = grid[k]
            var lo = 0
            var hi = m
            while lo < hi {
                let mid = (lo + hi) / 2
                if t[mid] <= g {
                    lo = mid + 1
                } else {
                    hi = mid
                }
            }
            let j = lo - 1
            if j < 0 {
                values[k] = v[0]
                missing[k] = true
            } else if j == m - 1 {
                values[k] = v[m - 1]
                missing[k] = (g - t[m - 1]) > gapTolerance
            } else {
                let t0 = t[j]
                let t1 = t[j + 1]
                let w = (g - t0) / (t1 - t0)
                let v0 = v[j]
                values[k] = v0 + w * (v[j + 1] - v0)
                missing[k] = (t1 - t0) > gapTolerance
            }
        }
        return (values, missing)
    }
}

// MARK: - Sort and deduplicate (SPEC 4)

public enum StreamCleaner {
    /// Indices of the samples kept by SPEC 4: stable sort by `t`, then drop every sample
    /// whose `t` is not strictly greater than the previous kept one. Non-finite
    /// timestamps are dropped first.
    static func keptIndices(_ t: [Double], count: Int) -> [Int] {
        let n = min(max(count, 0), t.count)
        let order = (0..<n).filter { t[$0].isFinite }.sorted { a, b in
            if t[a] != t[b] { return t[a] < t[b] }
            return a < b
        }
        var keep: [Int] = []
        keep.reserveCapacity(order.count)
        var last = 0.0
        for i in order {
            if keep.isEmpty || t[i] > last {
                keep.append(i)
                last = t[i]
            }
        }
        return keep
    }

    public static func sortDedup(_ s: SensorStream) -> SensorStream {
        let n = min(s.t.count, s.x.count, s.y.count, s.z.count)
        let keep = keptIndices(s.t, count: n)
        return SensorStream(t: keep.map { s.t[$0] }, x: keep.map { s.x[$0] },
                            y: keep.map { s.y[$0] }, z: keep.map { s.z[$0] })
    }

    /// The sample count is limited by `t`, gravity and rotation (the channels used by
    /// processing); a shorter user-acceleration or attitude channel comes back empty.
    public static func sortDedup(_ m: MotionStream) -> MotionStream {
        let n = min(m.t.count, m.gravityX.count, m.gravityY.count, m.gravityZ.count,
                    m.rotationX.count, m.rotationY.count, m.rotationZ.count)
        let keep = keptIndices(m.t, count: n)
        func pick(_ values: [Double]) -> [Double] {
            if values.count < n { return [] }
            return keep.map { values[$0] }
        }
        return MotionStream(
            t: pick(m.t),
            gravityX: pick(m.gravityX), gravityY: pick(m.gravityY), gravityZ: pick(m.gravityZ),
            userAccelX: pick(m.userAccelX), userAccelY: pick(m.userAccelY), userAccelZ: pick(m.userAccelZ),
            rotationX: pick(m.rotationX), rotationY: pick(m.rotationY), rotationZ: pick(m.rotationZ),
            attitudeW: pick(m.attitudeW), attitudeX: pick(m.attitudeX),
            attitudeY: pick(m.attitudeY), attitudeZ: pick(m.attitudeZ))
    }
}
