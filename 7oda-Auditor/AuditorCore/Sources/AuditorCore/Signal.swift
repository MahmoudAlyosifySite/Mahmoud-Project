import Foundation

// MARK: - Config (SPEC 5.1)

public struct SignalConfig: Equatable {
    public var fs: Double
    public var preRoll: Double
    public var postRoll: Double
    public var hpCutoff: Double
    public var hpQ: Double
    public var motionLowPass: Double
    public var motionQ: Double
    /// Tremor band-pass center `sqrt(8 * 12)` Hz and `Q = center / 4`.
    public var tremorCenter: Double
    public var tremorQ: Double
    public var pad: Int
    public var gapTolerance: Double
    public var lagMin: Int
    public var lagMax: Int
    public var templateDecay: Double
    public var accelNoiseFloor: Double
    public var gyroNoiseFloor: Double

    public init(
        fs: Double = 100.0,
        preRoll: Double = 0.25,
        postRoll: Double = 0.25,
        hpCutoff: Double = 5.0,
        hpQ: Double = Biquad.butterworthQ,
        motionLowPass: Double = 3.0,
        motionQ: Double = Biquad.butterworthQ,
        tremorCenter: Double = (8.0 * 12.0).squareRoot(),
        tremorQ: Double = (8.0 * 12.0).squareRoot() / 4.0,
        pad: Int = 30,
        gapTolerance: Double = 0.025,
        lagMin: Int = -10,
        lagMax: Int = 25,
        templateDecay: Double = 0.03,
        accelNoiseFloor: Double = 1e-6,
        gyroNoiseFloor: Double = 1e-6
    ) {
        self.fs = fs
        self.preRoll = preRoll
        self.postRoll = postRoll
        self.hpCutoff = hpCutoff
        self.hpQ = hpQ
        self.motionLowPass = motionLowPass
        self.motionQ = motionQ
        self.tremorCenter = tremorCenter
        self.tremorQ = tremorQ
        self.pad = pad
        self.gapTolerance = gapTolerance
        self.lagMin = lagMin
        self.lagMax = lagMax
        self.templateDecay = templateDecay
        self.accelNoiseFloor = accelNoiseFloor
        self.gyroNoiseFloor = gyroNoiseFloor
    }

    public static let `default` = SignalConfig()

    var highPass: Biquad { return Biquad.highPass(cutoff: hpCutoff, fs: fs, q: hpQ) }
    var motionLowPassFilter: Biquad { return Biquad.lowPass(cutoff: motionLowPass, fs: fs, q: motionQ) }
    var tremorBandPass: Biquad { return Biquad.bandPass(center: tremorCenter, fs: fs, q: tremorQ) }
}

// MARK: - Results

public struct AlignmentResult: Equatable {
    public var lag: Int
    public var ncc: Double
    public var peakMargin: Double

    public init(lag: Int, ncc: Double, peakMargin: Double) {
        self.lag = lag
        self.ncc = ncc
        self.peakMargin = peakMargin
    }
}

/// Everything SPEC 5 derives from one recording, on the common grid of length `n`.
/// Three-channel quantities are `[x, y, z]`, each of length `n`.
public struct ProcessedSignal {
    public let config: SignalConfig
    public let challenge: ChallengeSpec
    public let n: Int
    public let gridStart: Double
    public let accel: [[Double]]
    public let gyro: [[Double]]
    public let gravity: [[Double]]
    public let rotation: [[Double]]
    public let accelHP: [[Double]]
    public let gyroHP: [[Double]]
    public let energyAccel: [Double]
    public let energyGyro: [Double]
    public let noiseAccel: Double
    public let noiseGyro: Double
    /// Normalized energy `env` (SPEC 5.5).
    public let envelope: [Double]
    /// Template amplitude `h` at lag 0 (SPEC 5.6).
    public let template: [Double]
    public let alignment: AlignmentResult
    public let gapFraction: Double
    public let sampleRate: Double

    // Internal extras (mirrors of the reference's `Processed`).
    let grid: [Double]
    let accelMissing: [Bool]
    let gyroMissing: [Bool]
    /// Amplitude envelope `x = sqrt(env)`.
    let amplitude: [Double]
    /// Pearson correlation for every lag in `lagMin...lagMax`, in that order.
    let nccByLag: [Double]

    /// `idx(t) = Int((preRoll + t) * fs + 0.5) + lag` (SPEC 5.7, unclamped).
    public func index(ofPatternTime t: Double) -> Int {
        return Numerics.truncatingInt((config.preRoll + t) * config.fs + 0.5) + alignment.lag
    }

    /// Quiet window `Q` of SPEC 5.7: `[0, onsetFirst - 2)`, or the tail window
    /// `[min(endIdx + 5, N - 10), N)` (floored at 0) if that is shorter than 10 samples.
    public var quietRange: Range<Int> {
        let end = clamp(onsetFirst - 2)
        if end < 10 {
            let start = max(0, min(endIndex + 5, n - 10))
            return start..<max(start, n)
        }
        return 0..<end
    }

    /// `[onsetFirst, endIdx)` with both ends clamped to `[0, N]`.
    public var eventRange: Range<Int> {
        let a = clamp(onsetFirst)
        let b = clamp(endIndex)
        return a..<max(a, b)
    }

    /// `endIdx = idx(duration)` (unclamped).
    var endIndex: Int {
        return index(ofPatternTime: challenge.duration)
    }

    /// `onsetFirst = idx(events[0].time)` (unclamped); `idx(0)` without events.
    var onsetFirst: Int {
        return index(ofPatternTime: challenge.events.first?.time ?? 0.0)
    }

    /// Clamps an index to `[0, N]` (for range bounds).
    func clamp(_ i: Int) -> Int {
        return min(max(i, 0), n)
    }
}

// MARK: - Processing (SPEC 5.2-5.7)

public enum SignalProcessor {
    /// Upper bound on the grid length (1000 s at 100 Hz) so that a corrupted duration
    /// cannot exhaust memory; real patterns use about 200 samples.
    static let maximumGridLength = 100_000

    public static func process(_ recording: Recording, config: SignalConfig = .default) -> ProcessedSignal {
        let cfg = config
        let challenge = recording.challenge
        let fs = cfg.fs

        // 5.2 Grid.
        let t0 = recording.hapticStart - cfg.preRoll
        let rawN = Numerics.truncatingInt((cfg.preRoll + challenge.duration + cfg.postRoll) * fs + 0.5)
        let n = min(max(rawN, 0), maximumGridLength)
        var grid = [Double](repeating: 0.0, count: n)
        for k in 0..<n {
            grid[k] = t0 + Double(k) / fs
        }

        // 5.2 Resampling.
        let acc = StreamCleaner.sortDedup(recording.accel)
        let gyr = StreamCleaner.sortDedup(recording.gyro)
        let mot = StreamCleaner.sortDedup(recording.motion)
        let tol = cfg.gapTolerance
        func resample3(_ t: [Double], _ channels: [[Double]]) -> (values: [[Double]], missing: [Bool]) {
            var values: [[Double]] = []
            var missing = [Bool](repeating: true, count: n)
            for (i, channel) in channels.enumerated() {
                let r = Resampler.resample(t: t, v: channel, grid: grid, gapTolerance: tol)
                values.append(r.values)
                if i == 0 { missing = r.missing }
            }
            return (values, missing)
        }
        let accelR = resample3(acc.t, [acc.x, acc.y, acc.z])
        let gyroR = resample3(gyr.t, [gyr.x, gyr.y, gyr.z])
        let gravityR = resample3(mot.t, [mot.gravityX, mot.gravityY, mot.gravityZ])
        let rotationR = resample3(mot.t, [mot.rotationX, mot.rotationY, mot.rotationZ])

        var missingCount = 0
        for k in 0..<n where accelR.missing[k] || gyroR.missing[k] {
            missingCount += 1
        }
        let gapFraction = n > 0 ? Double(missingCount) / Double(n) : 0.0

        var sampleRate = 0.0
        if acc.t.count >= 2 {
            var diffs = [Double](repeating: 0.0, count: acc.t.count - 1)
            for i in 1..<acc.t.count {
                diffs[i - 1] = acc.t[i] - acc.t[i - 1]
            }
            sampleRate = 1.0 / Numerics.median(diffs)
        }

        // 5.5 Derived signals.
        let hp = cfg.highPass
        let accelHP = accelR.values.map { hp.filtfiltZ($0, pad: cfg.pad, dcGain: 0.0) }
        let gyroHP = gyroR.values.map { hp.filtfiltZ($0, pad: cfg.pad, dcGain: 0.0) }
        var ea = [Double](repeating: 0.0, count: n)
        var eg = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            ea[i] = accelHP[0][i] * accelHP[0][i] + accelHP[1][i] * accelHP[1][i] + accelHP[2][i] * accelHP[2][i]
            eg[i] = gyroHP[0][i] * gyroHP[0][i] + gyroHP[1][i] * gyroHP[1][i] + gyroHP[2][i] * gyroHP[2][i]
        }
        let preEnd = Numerics.truncatingInt(cfg.preRoll * fs + 0.5) - 5
        let noiseA: Double
        let noiseG: Double
        if preEnd < 1 {
            noiseA = cfg.accelNoiseFloor
            noiseG = cfg.gyroNoiseFloor
        } else {
            let range = 0..<min(preEnd, n)
            noiseA = max(Numerics.mean(ea, range), cfg.accelNoiseFloor)
            noiseG = max(Numerics.mean(eg, range), cfg.gyroNoiseFloor)
        }
        var env = [Double](repeating: 0.0, count: n)
        var amp = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            env[i] = ea[i] / noiseA + eg[i] / noiseG
            amp[i] = env[i].squareRoot()
        }

        // 5.6 Template and alignment.
        let tmpl = template(challenge, n: n, config: cfg)
        let aligned = align(amplitude: amp, template: tmpl, lagMin: cfg.lagMin, lagMax: cfg.lagMax)

        return ProcessedSignal(
            config: cfg,
            challenge: challenge,
            n: n,
            gridStart: t0,
            accel: accelR.values,
            gyro: gyroR.values,
            gravity: gravityR.values,
            rotation: rotationR.values,
            accelHP: accelHP,
            gyroHP: gyroHP,
            energyAccel: ea,
            energyGyro: eg,
            noiseAccel: noiseA,
            noiseGyro: noiseG,
            envelope: env,
            template: tmpl,
            alignment: aligned.result,
            gapFraction: gapFraction,
            sampleRate: sampleRate,
            grid: grid,
            accelMissing: accelR.missing,
            gyroMissing: gyroR.missing,
            amplitude: amp,
            nccByLag: aligned.nccByLag)
    }

    /// Template amplitude `h[n]` at lag 0 (SPEC 5.6): tap and ramp sums are each
    /// accumulated in event order from 0, then added.
    static func template(_ challenge: ChallengeSpec, n: Int, config cfg: SignalConfig) -> [Double] {
        let taps = challenge.taps
        let ramps = challenge.ramps
        let decay = cfg.templateDecay
        var out = [Double](repeating: 0.0, count: max(n, 0))
        for k in 0..<out.count {
            let tau = Double(k) / cfg.fs - cfg.preRoll
            var tapSum = 0.0
            for tap in taps where tau >= tap.time {
                tapSum = tapSum + tap.intensity * exp(-(tau - tap.time) / decay)
            }
            var rampSum = 0.0
            for ramp in ramps {
                let t = ramp.time
                let d = ramp.duration
                if t <= tau && tau < t + d {
                    rampSum = rampSum + ramp.intensity(at: tau - t)
                } else if tau >= t + d {
                    rampSum = rampSum + ramp.intensity(at: d) * exp(-(tau - t - d) / decay)
                }
            }
            out[k] = tapSum + rampSum
        }
        return out
    }

    /// Two-pass Pearson correlation of `x[xStart..<xStart+count]` and
    /// `h[hStart..<hStart+count]`; 0 if either centered sum of squares is 0.
    static func pearson(_ x: [Double], _ xStart: Int, _ h: [Double], _ hStart: Int, count: Int) -> Double {
        if count <= 0 { return 0.0 }
        var sx = 0.0
        var sh = 0.0
        for i in 0..<count {
            sx = sx + x[xStart + i]
            sh = sh + h[hStart + i]
        }
        let mx = sx / Double(count)
        let mh = sh / Double(count)
        var sxx = 0.0
        var shh = 0.0
        var sxh = 0.0
        for i in 0..<count {
            let dx = x[xStart + i] - mx
            let dh = h[hStart + i] - mh
            sxx = sxx + dx * dx
            shh = shh + dh * dh
            sxh = sxh + dx * dh
        }
        if sxx == 0.0 || shh == 0.0 { return 0.0 }
        return sxh / (sxx * shh).squareRoot()
    }

    /// Lag search of SPEC 5.6: pairs `x[n]` with `h[n - L]`; `lag` = argmax (ties to the
    /// smaller `|L|`, then the smaller `L`); `peakMargin = ncc - max{ncc(L') : |L' - lag| >= 4}`.
    static func align(amplitude x: [Double], template h: [Double],
                      lagMin: Int, lagMax: Int) -> (result: AlignmentResult, nccByLag: [Double]) {
        guard lagMin <= lagMax else {
            return (AlignmentResult(lag: 0, ncc: 0.0, peakMargin: 0.0), [])
        }
        let n = min(x.count, h.count)
        let lags = Array(lagMin...lagMax)
        var ncc = [Double](repeating: 0.0, count: lags.count)
        for (i, lag) in lags.enumerated() {
            let lo = max(0, lag)
            let hi = min(n, n + lag)
            if hi - lo < 10 { continue }
            ncc[i] = pearson(x, lo, h, lo - lag, count: hi - lo)
        }
        let order = lags.indices.sorted { a, b in
            let la = abs(lags[a])
            let lb = abs(lags[b])
            if la != lb { return la < lb }
            return lags[a] < lags[b]
        }
        var best = order[0]
        for i in order.dropFirst() where ncc[i] > ncc[best] {
            best = i
        }
        let bestLag = lags[best]
        let bestValue = ncc[best]
        var far: Double? = nil
        for (i, lag) in lags.enumerated() where abs(lag - bestLag) >= 4 {
            if let current = far {
                if ncc[i] > current { far = ncc[i] }
            } else {
                far = ncc[i]
            }
        }
        let peakMargin = far.map { bestValue - $0 } ?? bestValue
        return (AlignmentResult(lag: bestLag, ncc: bestValue, peakMargin: peakMargin), ncc)
    }
}
