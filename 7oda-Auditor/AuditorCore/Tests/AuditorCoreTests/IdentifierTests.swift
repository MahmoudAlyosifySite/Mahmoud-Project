import Foundation
import XCTest
import AuditorCore

final class IdentifierTests: XCTestCase {
    private let date = Date(timeIntervalSince1970: 1_795_514_400)

    /// `count` samples around `center` (uniform noise of half-width 1).
    private func samples(_ key: PrototypeKey, center: [Double], count: Int, seed: UInt64,
                         source: SampleSource = .enrollment) -> [TemplateSample] {
        var rng = Xoshiro256StarStar(seed: seed)
        return (0..<count).map { i in
            TemplateSample(prototype: key, source: source, createdAt: date.addingTimeInterval(Double(i)),
                           features: center.map { $0 + rng.uniform(-1, 1) })
        }
    }

    func testPrototypeOrder() {
        let keys = [
            PrototypeKey(personID: "B", hand: .right, accessory: .none),
            PrototypeKey(personID: "A", hand: .left, accessory: .watch),
            PrototypeKey(personID: "A", hand: .right, accessory: .other),
            PrototypeKey(personID: "A", hand: .right, accessory: .ring),
            PrototypeKey(personID: "A", hand: .left, accessory: .none),
            PrototypeKey(personID: "A", hand: .right, accessory: .none),
        ]
        XCTAssertEqual(keys.sorted(), [
            PrototypeKey(personID: "A", hand: .right, accessory: .none),
            PrototypeKey(personID: "A", hand: .right, accessory: .ring),
            PrototypeKey(personID: "A", hand: .right, accessory: .other),
            PrototypeKey(personID: "A", hand: .left, accessory: .none),
            PrototypeKey(personID: "A", hand: .left, accessory: .watch),
            PrototypeKey(personID: "B", hand: .right, accessory: .none),
        ])
        // Unicode scalar order, like Python: "Z" (U+005A) < "a" (U+0061).
        XCTAssertLessThan(PrototypeKey(personID: "Z", hand: .left), PrototypeKey(personID: "a", hand: .right))
    }

    func testNoPeople() {
        let identifier = Identifier(samples: [])
        XCTAssertNil(identifier.model)
        XCTAssertNil(identifier.handModel)
        XCTAssertEqual(identifier.prototypes, [])
        let d = identifier.decide([[1, 2, 3]])
        XCTAssertEqual(d.outcome, .unknown(.noPeople))
        XCTAssertEqual(d.patternsUsed, 1)
        XCTAssertEqual(d.absThreshold, 0.0)
        XCTAssertNil(d.bestPrototype)
        XCTAssertEqual(d.marginName, Double.infinity)
        XCTAssertEqual(d.personScores, [])
        XCTAssertEqual(identifier.adaptiveSamples(for: d, patterns: [[1, 2, 3]], now: date), [])
    }

    func testSinglePersonSingleHand() {
        let key = PrototypeKey(personID: "A", hand: .right)
        let center = [1.0, -2.0, 3.0, 0.5]
        let train = samples(key, center: center, count: 12, seed: 1)
        let identifier = Identifier(samples: train)
        XCTAssertEqual(identifier.prototypes, [key])
        XCTAssertNotNil(identifier.model)
        XCTAssertNil(identifier.handModel, "no left-hand samples, so no hand head")

        XCTAssertEqual(identifier.decide([]).outcome, .needMorePatterns)
        XCTAssertEqual(identifier.decide([]).patternsUsed, 0)

        guard let mean = identifier.model?.mean else { return }
        let decision = identifier.decide([mean])
        XCTAssertEqual(decision.outcome, .identified(personID: "A", hand: .right))
        XCTAssertEqual(decision.bestPrototype, key)
        XCTAssertEqual(decision.marginName, Double.infinity)
        XCTAssertEqual(decision.marginHand, Double.infinity)
        XCTAssertEqual(decision.handLogOdds, 0.0)
        XCTAssertEqual(decision.bestDistance, 0.0, accuracy: 1e-9)
        XCTAssertEqual(decision.personScores.count, 1)
        XCTAssertEqual(decision.personScores.first?.probability ?? 0, 1.0, accuracy: 1e-12)
        XCTAssertEqual(decision.prototypeDistances.map { $0.prototype }, [key])

        let adaptive = identifier.adaptiveSamples(for: decision, patterns: [mean], now: date)
        XCTAssertEqual(adaptive.count, 1)
        XCTAssertEqual(adaptive.first?.source, .adaptive)
        XCTAssertEqual(adaptive.first?.prototype, key)
        XCTAssertEqual(adaptive.first?.features, mean)
        XCTAssertEqual(adaptive.first?.createdAt, date)
        XCTAssertEqual(adaptive.first?.featureVersion, FeatureExtractor.version)

        // A stranger far away: more patterns first, then Unknown (no match).
        let far = center.map { $0 + 100.0 }
        XCTAssertEqual(identifier.decide([far]).outcome, .needMorePatterns)
        let unknown = identifier.decide([far, far, far])
        XCTAssertEqual(unknown.outcome, .unknown(.noMatch))
        XCTAssertEqual(unknown.patternsUsed, 3)
        XCTAssertEqual(identifier.adaptiveSamples(for: unknown, patterns: [far, far, far], now: date), [])

        // Vectors of another dimension never crash and never match.
        XCTAssertEqual(identifier.decide([[1.0], [1.0], [1.0]]).outcome, .unknown(.noMatch))
    }

    func testTwoPeopleBothHands() {
        let a = [0.0, 0.0, 0.0, 0.0]
        let b = [8.0, -8.0, 8.0, -8.0]
        var train: [TemplateSample] = []
        train += samples(PrototypeKey(personID: "A", hand: .right), center: a, count: 10, seed: 1)
        train += samples(PrototypeKey(personID: "A", hand: .left), center: a.map { $0 + 4.0 }, count: 10, seed: 2)
        train += samples(PrototypeKey(personID: "B", hand: .right), center: b, count: 10, seed: 3)
        train += samples(PrototypeKey(personID: "B", hand: .left), center: b.map { $0 + 4.0 }, count: 10, seed: 4)
        let identifier = Identifier(samples: train)
        XCTAssertEqual(identifier.prototypes.count, 4)
        XCTAssertNotNil(identifier.handModel)
        XCTAssertEqual(identifier.handModel?.classCounts, [20, 20])

        let decision = identifier.decide([b, b])
        XCTAssertEqual(decision.patternsUsed, 2)
        XCTAssertEqual(decision.bestPrototype, PrototypeKey(personID: "B", hand: .right))
        XCTAssertTrue(decision.marginName > 0 && decision.marginName.isFinite)
        XCTAssertTrue(decision.marginHand > 0 && decision.marginHand.isFinite)
        XCTAssertEqual(decision.personScores.first?.personID, "B")
        let total = decision.personScores.reduce(0.0) { $0 + $1.probability }
        XCTAssertEqual(total, 1.0, accuracy: 1e-12)
        XCTAssertEqual(decision.prototypeDistances.count, 4)
    }

    func testAdaptiveCap() {
        let p = PrototypeKey(personID: "A", hand: .right)
        let q = PrototypeKey(personID: "A", hand: .left)
        var all: [TemplateSample] = []
        all += samples(p, center: [0, 0], count: 3, seed: 1, source: .enrollment)
        var adaptive = samples(p, center: [0, 0], count: 25, seed: 2, source: .adaptive)
        adaptive.reverse()  // newest first in the list: the cap must use createdAt, not position
        all += adaptive
        all += samples(q, center: [0, 0], count: 4, seed: 3, source: .adaptive)
        all += samples(p, center: [0, 0], count: 2, seed: 4, source: .accessory)

        let capped = Identifier.applyAdaptiveCap(all, cap: 20)
        XCTAssertEqual(capped.count, all.count - 5)
        XCTAssertEqual(capped.filter { $0.prototype == p && $0.source == .adaptive }.count, 20)
        XCTAssertEqual(capped.filter { $0.prototype == q }.count, 4)
        XCTAssertEqual(capped.filter { $0.source == .enrollment }.count, 3)
        XCTAssertEqual(capped.filter { $0.source == .accessory }.count, 2)
        // The five oldest adaptive samples of p (createdAt offsets 0...4) are gone.
        let keptOffsets = capped.filter { $0.prototype == p && $0.source == .adaptive }
            .map { Int($0.createdAt.timeIntervalSince(date)) }
        XCTAssertEqual(keptOffsets, Array((5...24).reversed()))
        // Order of the kept samples is preserved.
        let ids = capped.map { $0.id }
        XCTAssertEqual(ids, all.map { $0.id }.filter { ids.contains($0) })
        XCTAssertEqual(Identifier.applyAdaptiveCap(all, cap: 0).filter { $0.source == .adaptive }.count, 0)
    }

    func testConfigCodingDefaults() throws {
        let partial = try JSONDecoder().decode(IdentifierConfig.self, from: Data("{\"nameMargin\": 7.5}".utf8))
        XCTAssertEqual(partial.nameMargin, 7.5)
        XCTAssertEqual(partial.handMargin, 3.0)
        XCTAssertEqual(partial.maxPatterns, 3)
        let model = try JSONDecoder().decode(ModelConfig.self, from: Data("{}".utf8))
        XCTAssertEqual(model, ModelConfig.default)
        let data = try JSONEncoder().encode(IdentifierConfig.default)
        XCTAssertEqual(try JSONDecoder().decode(IdentifierConfig.self, from: data), IdentifierConfig.default)
    }

    func testTemplateSampleCoding() throws {
        let s = TemplateSample(prototype: PrototypeKey(personID: "A", hand: .left, accessory: .ring), source: .accessory,
                               createdAt: date, features: [1.5, -2.0], recordingID: UUID())
        let data = try RecordingCoding.encoder().encode([s])
        XCTAssertEqual(try RecordingCoding.decoder().decode([TemplateSample].self, from: data), [s])
    }
}

final class EvaluationTests: XCTestCase {
    func testConfusionMatrix() throws {
        var m = ConfusionMatrix()
        XCTAssertEqual(m.accuracy, 0.0)
        m.add(truth: "A", predicted: "A")
        m.add(truth: "A", predicted: "Unknown")
        m.add(truth: "B", predicted: "B")
        m.add(truth: "B", predicted: "B")
        XCTAssertEqual(m.labels, ["A", "Unknown", "B"])
        XCTAssertEqual(m.counts, [[1, 1, 0], [0, 0, 0], [0, 0, 2]])
        XCTAssertEqual(m.accuracy, 0.75)
        XCTAssertEqual(m.total, 4)
        XCTAssertEqual(m.count(truth: "A", predicted: "Unknown"), 1)
        XCTAssertEqual(m.count(truth: "C", predicted: "A"), 0)
        let data = try JSONEncoder().encode(m)
        XCTAssertEqual(try JSONDecoder().decode(ConfusionMatrix.self, from: data), m)
        var preset = ConfusionMatrix(labels: ["x", "y"])
        XCTAssertEqual(preset.counts, [[0, 0], [0, 0]])
        preset.add(truth: "y", predicted: "x")
        XCTAssertEqual(preset.counts, [[0, 0], [1, 0]])
    }
}
