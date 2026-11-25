import Foundation

// MARK: - Types (SPEC 6)

public enum QualityMode: String, Codable {
    /// Enrollment and checks: the phone is held in one hand.
    case handheld
    /// Calibration: the phone lies flat on a table.
    case onTable
}

public enum QualityIssue: String, Codable, CaseIterable {
    case sensorGaps, lowSampleRate, resting, moving, weakAlignment, weakSignal, notResting

    /// User-facing message (SPEC 6.3).
    public var message: String {
        switch self {
        case .resting: return "Hold the phone in one hand"
        case .moving: return "Hold still"
        case .notResting: return "Put the phone flat on a table and don't touch it"
        case .sensorGaps, .weakAlignment: return "Try again"
        case .weakSignal: return "Vibration too weak — remove a thick case?"
        case .lowSampleRate: return "Motion sensors are too slow on this device"
        }
    }
}

/// Metrics of SPEC 6.1 (JSON keys are the property names).
public struct QualityMetrics: Codable, Equatable {
    public var gapFraction: Double
    public var sampleRate: Double
    public var quietGyroRMS: Double
    public var quietAccelRMS: Double
    public var gravityZ: Double
    public var motionRMS: Double
    public var gravityAngleDeg: Double
    public var ncc: Double
    public var peakMargin: Double
    public var snr: Double
    public var lag: Int

    public init(gapFraction: Double, sampleRate: Double, quietGyroRMS: Double, quietAccelRMS: Double,
                gravityZ: Double, motionRMS: Double, gravityAngleDeg: Double, ncc: Double,
                peakMargin: Double, snr: Double, lag: Int) {
        self.gapFraction = gapFraction
        self.sampleRate = sampleRate
        self.quietGyroRMS = quietGyroRMS
        self.quietAccelRMS = quietAccelRMS
        self.gravityZ = gravityZ
        self.motionRMS = motionRMS
        self.gravityAngleDeg = gravityAngleDeg
        self.ncc = ncc
        self.peakMargin = peakMargin
        self.snr = snr
        self.lag = lag
    }
}

/// Thresholds of SPEC 6.2, plus the 0.05 rad/s motion limit of the `notResting` rule (6.3).
public struct QualityConfig: Equatable {
    public var maxGapFraction: Double
    public var minSampleRate: Double
    public var restingGyroFactor: Double
    public var restingGyroAbsolute: Double
    public var flatGravityZ: Double
    public var maxMotionRMS: Double
    public var maxGravityAngleDeg: Double
    public var minNCC: Double
    public var minPeakMargin: Double
    public var minSNR: Double
    public var tableMaxMotionRMS: Double

    public init(maxGapFraction: Double = 0.10, minSampleRate: Double = 70.0, restingGyroFactor: Double = 2.5,
                restingGyroAbsolute: Double = 0.004, flatGravityZ: Double = 0.95, maxMotionRMS: Double = 0.30,
                maxGravityAngleDeg: Double = 10.0, minNCC: Double = 0.30, minPeakMargin: Double = 0.05,
                minSNR: Double = 2.0, tableMaxMotionRMS: Double = 0.05) {
        self.maxGapFraction = maxGapFraction
        self.minSampleRate = minSampleRate
        self.restingGyroFactor = restingGyroFactor
        self.restingGyroAbsolute = restingGyroAbsolute
        self.flatGravityZ = flatGravityZ
        self.maxMotionRMS = maxMotionRMS
        self.maxGravityAngleDeg = maxGravityAngleDeg
        self.minNCC = minNCC
        self.minPeakMargin = minPeakMargin
        self.minSNR = minSNR
        self.tableMaxMotionRMS = tableMaxMotionRMS
    }

    public static let `default` = QualityConfig()
}

public struct QualityReport: Codable, Equatable {
    public var metrics: QualityMetrics
    public var issues: [QualityIssue]
    public var passed: Bool

    public init(metrics: QualityMetrics, issues: [QualityIssue], passed: Bool? = nil) {
        self.metrics = metrics
        self.issues = issues
        self.passed = passed ?? issues.isEmpty
    }
}

// MARK: - Gate (SPEC 6.1, 6.3)

public enum QualityGate {
    public static func evaluate(_ p: ProcessedSignal, calibration: Calibration?, mode: QualityMode,
                                config: QualityConfig = .default) -> QualityReport {
        let m = metrics(p)
        let cfg = config
        var issues: [QualityIssue] = []
        let weakAlignment = m.ncc < cfg.minNCC || m.peakMargin < cfg.minPeakMargin
        let weakSignal = m.snr < cfg.minSNR
        if m.gapFraction > cfg.maxGapFraction { issues.append(.sensorGaps) }
        if m.sampleRate < cfg.minSampleRate { issues.append(.lowSampleRate) }
        switch mode {
        case .handheld:
            let restThreshold: Double
            if let calibration = calibration {
                restThreshold = cfg.restingGyroFactor * calibration.quietGyroRMS
            } else {
                restThreshold = cfg.restingGyroAbsolute
            }
            if m.gravityZ >= cfg.flatGravityZ && m.quietGyroRMS < restThreshold { issues.append(.resting) }
            if m.motionRMS > cfg.maxMotionRMS || m.gravityAngleDeg > cfg.maxGravityAngleDeg { issues.append(.moving) }
            if weakAlignment { issues.append(.weakAlignment) }
            if weakSignal { issues.append(.weakSignal) }
        case .onTable:
            if weakAlignment { issues.append(.weakAlignment) }
            if weakSignal { issues.append(.weakSignal) }
            if m.gravityZ < cfg.flatGravityZ || m.motionRMS > cfg.tableMaxMotionRMS { issues.append(.notResting) }
        }
        return QualityReport(metrics: m, issues: issues, passed: issues.isEmpty)
    }

    /// QualityMetrics of SPEC 6.1.
    static func metrics(_ p: ProcessedSignal) -> QualityMetrics {
        let n = p.n
        let q = p.quietRange
        let e = p.eventRange
        let lp = p.config.motionLowPassFilter
        let rl = p.rotation.map { lp.filtfiltZ($0, pad: p.config.pad, dcGain: 1.0) }
        var motionEnergy = [Double](repeating: 0.0, count: n)
        if rl.count == 3 {
            for i in 0..<n where i < rl[0].count && i < rl[1].count && i < rl[2].count {
                motionEnergy[i] = rl[0][i] * rl[0][i] + rl[1][i] * rl[1][i] + rl[2][i] * rl[2][i]
            }
        }
        let snr = Numerics.mean(p.envelope, e) / max(Numerics.mean(p.envelope, q), 1e-9)
        let gravityZ = (n > 0 && p.gravity.count == 3) ? abs(Numerics.mean(p.gravity[2])) : 0.0
        return QualityMetrics(
            gapFraction: p.gapFraction,
            sampleRate: p.sampleRate,
            quietGyroRMS: Numerics.mean(p.energyGyro, q).squareRoot(),
            quietAccelRMS: Numerics.mean(p.energyAccel, q).squareRoot(),
            gravityZ: gravityZ,
            motionRMS: Numerics.mean(motionEnergy).squareRoot(),
            gravityAngleDeg: gravityAngleDeg(p),
            ncc: p.alignment.ncc,
            peakMargin: p.alignment.peakMargin,
            snr: snr,
            lag: p.alignment.lag)
    }

    /// Angle between mean gravity over the first 20 and the last 20 grid samples (degrees).
    static func gravityAngleDeg(_ p: ProcessedSignal) -> Double {
        let n = p.n
        guard n > 0, p.gravity.count == 3, p.gravity.allSatisfy({ $0.count == n }) else { return 0.0 }
        let k = min(20, n)
        let g = p.gravity
        let a = (Numerics.mean(g[0], 0..<k), Numerics.mean(g[1], 0..<k), Numerics.mean(g[2], 0..<k))
        let b = (Numerics.mean(g[0], (n - k)..<n), Numerics.mean(g[1], (n - k)..<n), Numerics.mean(g[2], (n - k)..<n))
        let na = (a.0 * a.0 + a.1 * a.1 + a.2 * a.2).squareRoot()
        let nb = (b.0 * b.0 + b.1 * b.1 + b.2 * b.2).squareRoot()
        if na < 1e-9 || nb < 1e-9 { return 0.0 }
        let dot = a.0 * b.0 + a.1 * b.1 + a.2 * b.2
        let cosine = min(max(dot / (na * nb), -1.0), 1.0)
        return acos(cosine) * 180.0 / Double.pi
    }
}
