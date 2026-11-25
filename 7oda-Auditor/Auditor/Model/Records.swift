import Foundation
import AuditorCore

/// Which recognizer answers a check.
enum RecognizerKind: String, Codable, CaseIterable, Identifiable {
    /// Learned Core ML encoder + cosine prototypes (SPEC 14).
    case v1
    /// Hand-made features + Gaussian model (SPEC 9).
    case v0

    var id: String { rawValue }

    var title: String {
        switch self {
        case .v1: return "v1 (learned encoder)"
        case .v0: return "v0 (hand-made features)"
        }
    }

    var other: RecognizerKind { self == .v1 ? .v0 : .v1 }
}

/// An enrolled person. `id` is the `personID` of prototypes and labels.
struct Person: Codable, Equatable, Identifiable, Hashable {
    var id: String
    var name: String
    var createdAt: Date

    init(id: String = UUID().uuidString, name: String, createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
    }
}

/// A v1 embedding sample with the app-side links: the raw recording it came from and
/// the encoder version that produced it (so it can be recomputed when the model changes).
struct StoredEmbedding: Codable, Equatable, Identifiable {
    var id: UUID
    var recordingID: UUID?
    var encoderVersion: String
    var sample: EmbeddingSample

    init(id: UUID = UUID(), recordingID: UUID?, encoderVersion: String, sample: EmbeddingSample) {
        self.id = id
        self.recordingID = recordingID
        self.encoderVersion = encoderVersion
        self.sample = sample
    }

    var prototype: PrototypeKey { sample.prototype }
    var source: SampleSource { sample.source }
}

/// Keeps at most `cap` adaptive embeddings per prototype (the newest); enrollment and
/// accessory samples are never dropped. Same rule as `Identifier.applyAdaptiveCap`.
func applyAdaptiveCap(_ items: [StoredEmbedding], cap: Int) -> [StoredEmbedding] {
    let limit = max(cap, 0)
    var adaptiveByPrototype: [PrototypeKey: [Int]] = [:]
    for (i, item) in items.enumerated() where item.source == .adaptive {
        adaptiveByPrototype[item.prototype, default: []].append(i)
    }
    var drop = Set<Int>()
    for (_, indices) in adaptiveByPrototype where indices.count > limit {
        let ordered = indices.sorted { a, b in
            let da = items[a].sample.createdAt, db = items[b].sample.createdAt
            return da != db ? da < db : a < b
        }
        for i in ordered.prefix(indices.count - limit) { drop.insert(i) }
    }
    return items.enumerated().filter { !drop.contains($0.offset) }.map { $0.element }
}

/// Compact, storable summary of one recognizer's decision.
struct DecisionSummary: Codable, Equatable {
    var outcome: String          // identified | handUnclear | needMorePatterns | unknown
    var personID: String?
    var hand: Hand?
    var reason: String?
    var patternsUsed: Int
    var bestDistance: Double?
    var marginName: Double?
    var marginHand: Double?
    var handLogOdds: Double
    var absThreshold: Double

    init(_ d: Decision) {
        switch d.outcome {
        case .identified(let personID, let hand):
            outcome = "identified"; self.personID = personID; self.hand = hand; reason = nil
        case .identifiedHandUnclear(let personID):
            outcome = "handUnclear"; self.personID = personID; hand = nil; reason = nil
        case .needMorePatterns:
            outcome = "needMorePatterns"; personID = nil; hand = nil; reason = nil
        case .unknown(let r):
            outcome = "unknown"; personID = nil; hand = nil; reason = r.rawValue
        }
        patternsUsed = d.patternsUsed
        bestDistance = d.bestDistance.isFinite ? d.bestDistance : nil
        marginName = d.marginName.isFinite ? d.marginName : nil
        marginHand = d.marginHand.isFinite ? d.marginHand : nil
        handLogOdds = d.handLogOdds
        absThreshold = d.absThreshold
    }
}

/// One finished check, logged for the checks CSV (both recognizers' outcomes).
struct CheckLogEntry: Codable, Equatable, Identifiable {
    var id: UUID
    var createdAt: Date
    var purpose: RecordingPurpose
    var recognizer: RecognizerKind
    var recordingIDs: [UUID]
    var patternsUsed: Int
    var attempts: Int
    var seconds: Double
    var v0: DecisionSummary?
    var v1: DecisionSummary?
    var truthPersonID: String?
    var truthHand: Hand?
    var requestedHand: Hand?
    var challengePassed: Bool?
    var qualityFailure: String?
}
