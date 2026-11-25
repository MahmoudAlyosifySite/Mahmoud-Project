import Foundation
import AuditorCore

/// User settings (Lab). Decoding tolerates missing keys so older files keep working.
struct AppSettings: Codable, Equatable {
    var recognizer: RecognizerKind = .v1
    // v0 (SPEC 9.2 / 9.4)
    var absFactor: Double = ModelConfig.default.absFactor
    var nameMargin: Double = IdentifierConfig.default.nameMargin
    var handMargin: Double = IdentifierConfig.default.handMargin
    // v1 (SPEC 14.3)
    var slack: Double = EmbeddingConfig.default.slack
    var v1NameMargin: Double = EmbeddingConfig.default.nameMargin
    // shared
    var maxPatterns: Int = IdentifierConfig.default.maxPatterns
    var minNCC: Double = QualityConfig.default.minNCC
    var minPeakMargin: Double = QualityConfig.default.minPeakMargin
    var handChallenge: Bool = false
    var learnFromConfidentChecks: Bool = false
    var anonymizeExports: Bool = false
    var calibrationSkipped: Bool = false

    init() {}

    private enum CodingKeys: String, CodingKey {
        case recognizer, absFactor, nameMargin, handMargin, slack, v1NameMargin, maxPatterns, minNCC, minPeakMargin
        case handChallenge, learnFromConfidentChecks, anonymizeExports, calibrationSkipped
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = AppSettings()
        recognizer = try c.decodeIfPresent(RecognizerKind.self, forKey: .recognizer) ?? d.recognizer
        absFactor = try c.decodeIfPresent(Double.self, forKey: .absFactor) ?? d.absFactor
        nameMargin = try c.decodeIfPresent(Double.self, forKey: .nameMargin) ?? d.nameMargin
        handMargin = try c.decodeIfPresent(Double.self, forKey: .handMargin) ?? d.handMargin
        slack = try c.decodeIfPresent(Double.self, forKey: .slack) ?? d.slack
        v1NameMargin = try c.decodeIfPresent(Double.self, forKey: .v1NameMargin) ?? d.v1NameMargin
        maxPatterns = try c.decodeIfPresent(Int.self, forKey: .maxPatterns) ?? d.maxPatterns
        minNCC = try c.decodeIfPresent(Double.self, forKey: .minNCC) ?? d.minNCC
        minPeakMargin = try c.decodeIfPresent(Double.self, forKey: .minPeakMargin) ?? d.minPeakMargin
        handChallenge = try c.decodeIfPresent(Bool.self, forKey: .handChallenge) ?? d.handChallenge
        learnFromConfidentChecks = try c.decodeIfPresent(Bool.self, forKey: .learnFromConfidentChecks)
            ?? d.learnFromConfidentChecks
        anonymizeExports = try c.decodeIfPresent(Bool.self, forKey: .anonymizeExports) ?? d.anonymizeExports
        calibrationSkipped = try c.decodeIfPresent(Bool.self, forKey: .calibrationSkipped) ?? d.calibrationSkipped
    }

    var modelConfig: ModelConfig {
        var m = ModelConfig.default
        m.absFactor = absFactor
        return m
    }

    var identifierConfig: IdentifierConfig {
        var c = IdentifierConfig.default
        c.nameMargin = nameMargin
        c.handMargin = handMargin
        c.maxPatterns = max(1, maxPatterns)
        return c
    }

    var embeddingConfig: EmbeddingConfig {
        var c = EmbeddingConfig.default
        c.slack = slack
        c.nameMargin = v1NameMargin
        c.maxPatterns = max(1, maxPatterns)
        return c
    }

    var qualityConfig: QualityConfig {
        var q = QualityConfig.default
        q.minNCC = minNCC
        q.minPeakMargin = minPeakMargin
        return q
    }
}
