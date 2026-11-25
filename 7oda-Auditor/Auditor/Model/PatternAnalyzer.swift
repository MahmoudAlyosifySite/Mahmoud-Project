import Foundation
import AuditorCore

/// One recorded pattern after the whole pipeline: processing, quality gate, v0
/// features and (when the model is available) the v1 embedding.
struct AnalyzedPattern {
    let recording: Recording
    let processed: ProcessedSignal
    let quality: QualityReport
    /// v0 features relative to the calibration (SPEC 7).
    let features: [Double]
    /// v1 embedding (SPEC 14), nil when the encoder is unavailable or failed.
    let embedding: [Double]?
    let handLogits: [Double]?
    let encoderError: String?

    var passed: Bool { quality.passed }

    /// User message of the first quality issue (SPEC 6.3), nil when passed.
    var qualityMessage: String? { quality.issues.first?.message }
}

enum PatternAnalyzer {
    /// SignalProcessor.process → QualityGate.evaluate → FeatureExtractor.extract and the encoder.
    static func analyze(_ recording: Recording, mode: QualityMode, calibration: Calibration?,
                        qualityConfig: QualityConfig, encoder: EmbeddingEncoder?) -> AnalyzedPattern {
        let processed = SignalProcessor.process(recording)
        let quality = QualityGate.evaluate(processed, calibration: calibration, mode: mode, config: qualityConfig)
        let features = FeatureExtractor.extract(processed, calibration: calibration)
        var embedding: [Double]?
        var hand: [Double]?
        var encoderError: String?
        if let encoder = encoder {
            do {
                let out = try encoder.encode(processed)
                embedding = out.embedding
                hand = out.handLogits
            } catch {
                encoderError = error.localizedDescription
            }
        }
        return AnalyzedPattern(recording: recording, processed: processed, quality: quality, features: features,
                               embedding: embedding, handLogits: hand, encoderError: encoderError)
    }
}

/// Small numeric helpers for the Lab (plots and per-channel SNR).
enum SignalSummary {
    /// Euclidean magnitude per grid index of a 3×N matrix.
    static func magnitude(_ m: [[Double]]) -> [Double] {
        guard m.count == 3 else { return [] }
        let n = min(m[0].count, m[1].count, m[2].count)
        return (0..<n).map { i in (m[0][i] * m[0][i] + m[1][i] * m[1][i] + m[2][i] * m[2][i]).squareRoot() }
    }

    /// Mean power in the event window over mean power in the quiet window, per channel
    /// (accel HP x, y, z, then gyro HP x, y, z).
    static func channelSNR(_ p: ProcessedSignal) -> [(name: String, snr: Double)] {
        let names = ["accel x", "accel y", "accel z", "gyro x", "gyro y", "gyro z"]
        let channels = p.accelHP + p.gyroHP
        var result: [(name: String, snr: Double)] = []
        for (name, x) in zip(names, channels) {
            let snr = meanPower(x, p.eventRange) / max(meanPower(x, p.quietRange), 1e-12)
            result.append((name: name, snr: snr))
        }
        return result
    }

    private static func meanPower(_ x: [Double], _ r: Range<Int>) -> Double {
        let idx = r.clamped(to: 0..<x.count)
        guard !idx.isEmpty else { return 0 }
        return idx.reduce(0.0) { $0 + x[$1] * x[$1] } / Double(idx.count)
    }
}
