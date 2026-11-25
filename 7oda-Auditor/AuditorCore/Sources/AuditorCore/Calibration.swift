import Foundation

/// On-table calibration (SPEC 8): the phone's own response, used to make some features
/// relative (SPEC 7.6) and to scale the resting test of the quality gate (SPEC 6.3).
public struct Calibration: Codable, Equatable {
    public var version: Int
    public var createdAt: Date
    /// Median absolute feature vector over the passing patterns (all features stored).
    public var tableFeatures: [Double]
    public var quietGyroRMS: Double
    public var quietAccelRMS: Double
    public var sampleRate: Double
    public var ncc: Double
    public var snr: Double
    public var patternsUsed: Int
    public var device: String

    /// Patterns played during a calibration.
    public static let patternCount = 20
    /// Minimum number of passing patterns.
    public static let minimumPassing = 12
    public static let failureMessage = "Calibration failed: keep the phone still on a hard table"

    public init(version: Int = 1, createdAt: Date, tableFeatures: [Double], quietGyroRMS: Double,
                quietAccelRMS: Double, sampleRate: Double, ncc: Double, snr: Double,
                patternsUsed: Int, device: String) {
        self.version = version
        self.createdAt = createdAt
        self.tableFeatures = tableFeatures
        self.quietGyroRMS = quietGyroRMS
        self.quietAccelRMS = quietAccelRMS
        self.sampleRate = sampleRate
        self.ncc = ncc
        self.snr = snr
        self.patternsUsed = patternsUsed
        self.device = device
    }

    /// Uses only `processed[i]` whose `reports[i].passed` (pairs beyond the shorter array
    /// are ignored); nil if fewer than 12 pass. `tableFeatures[k]` is the median of the
    /// absolute feature `k`; the RMS values, sample rate, NCC and SNR are medians of the
    /// quality metrics.
    public static func build(from processed: [ProcessedSignal], reports: [QualityReport],
                             device: String, now: Date) -> Calibration? {
        var used: [(ProcessedSignal, QualityReport)] = []
        for (p, r) in zip(processed, reports) where r.passed {
            used.append((p, r))
        }
        if used.count < minimumPassing { return nil }
        let features = used.map { FeatureExtractor.extractAbsolute($0.0) }
        let d = features.map { $0.count }.min() ?? 0
        var table = [Double](repeating: 0.0, count: d)
        for k in 0..<d {
            table[k] = Numerics.median(features.map { $0[k] })
        }
        func med(_ value: (QualityMetrics) -> Double) -> Double {
            return Numerics.median(used.map { value($0.1.metrics) })
        }
        return Calibration(
            version: 1,
            createdAt: now,
            tableFeatures: table,
            quietGyroRMS: med { $0.quietGyroRMS },
            quietAccelRMS: med { $0.quietAccelRMS },
            sampleRate: med { $0.sampleRate },
            ncc: med { $0.ncc },
            snr: med { $0.snr },
            patternsUsed: used.count,
            device: device)
    }
}
