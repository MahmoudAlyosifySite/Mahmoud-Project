import Foundation
import XCTest
@testable import AuditorCore

private struct V1File: Decodable {
    struct TensorCase: Decodable {
        let name: String
        let lag: Int
        let n: Int
        let recording: Recording
        let tensor: [Double]
    }
    struct TensorBlock: Decodable {
        let channels: Int
        let length: Int
        let patternStart: Int
        let cases: [TensorCase]
    }
    struct Sample: Decodable {
        let personID: String
        let hand: Hand
        let accessory: Accessory
        let source: SampleSource
        let embedding: [Double]
    }
    struct Expected: Decodable {
        let prototypes: [[String]]
        let prototypeVectors: [[Double]]
        let loo: [Double]
        let simThreshold: Double
        let absThreshold: Double
        let handWeights: [Double]?
        let handBias: Double?
    }
    struct Outcome: Decodable {
        let type: String
        let personID: String?
        let hand: Hand?
        let reason: UnknownReason?
    }
    struct Score: Decodable {
        let personID: String
        let probability: Double
        let bestDistance: Double
    }
    struct Query: Decodable {
        let name: String
        let patterns: [[Double]]
        let similarities: [[Double]]
        let handLogOdds: [Double]
        let outcome: Outcome
        let bestDistance: Double?
        let marginName: Double?
        let marginHand: Double?
        let meanDistances: [Double]
        let meanHandLogOdds: Double
        let bestPrototype: [String]
        let personScores: [Score]
        let adaptiveEligible: Bool
    }
    struct IdentifierBlock: Decodable {
        let config: EmbeddingConfig
        let dimension: Int
        let samples: [Sample]
        let expected: Expected
        let queries: [Query]
    }
    let tensor: TensorBlock
    let identifier: IdentifierBlock
}

private func prototypeKey(_ p: [String]) -> PrototypeKey? {
    guard p.count == 3, let hand = Hand(rawValue: p[1]), let accessory = Accessory(rawValue: p[2]) else { return nil }
    return PrototypeKey(personID: p[0], hand: hand, accessory: accessory)
}

private func decisionOutcome(_ o: V1File.Outcome) -> DecisionOutcome? {
    switch o.type {
    case "identified":
        guard let p = o.personID, let h = o.hand else { return nil }
        return .identified(personID: p, hand: h)
    case "identifiedHandUnclear":
        guard let p = o.personID else { return nil }
        return .identifiedHandUnclear(personID: p)
    case "needMorePatterns":
        return .needMorePatterns
    case "unknown":
        guard let r = o.reason else { return nil }
        return .unknown(r)
    default:
        return nil
    }
}

final class V1Tests: XCTestCase {
    private let tolerance = 1e-6
    private let date = Date(timeIntervalSince1970: 1_795_514_400)

    /// SPEC 14.5: tensor `|Δ| ≤ 1e-5 · max(1, |x|)`.
    func testEncoderInputFixtureParity() {
        guard let file = Fixture.decode(V1File.self, "v1_vectors") else { return }
        let block = file.tensor
        XCTAssertEqual(block.channels, EncoderInput.channels)
        XCTAssertEqual(block.length, EncoderInput.length)
        XCTAssertEqual(block.patternStart, EncoderInput.patternStart)
        XCTAssertFalse(block.cases.isEmpty)
        for c in block.cases {
            let p = SignalProcessor.process(c.recording)
            XCTAssertEqual(p.n, c.n, "\(c.name) n")
            XCTAssertEqual(p.alignment.lag, c.lag, "\(c.name) lag")
            let x = EncoderInput.build(p)
            assertClose(x.map { Double($0) }, c.tensor, rel: 1e-5, "\(c.name) tensor")
        }
    }

    /// SPEC 14.5: identifier as for the model (`1e-6 · max(1, |x|)`, outcomes exact).
    func testEmbeddingIdentifierFixtureParity() {
        guard let file = Fixture.decode(V1File.self, "v1_vectors") else { return }
        let block = file.identifier
        let samples = block.samples.map { s in
            EmbeddingSample(prototype: PrototypeKey(personID: s.personID, hand: s.hand, accessory: s.accessory),
                            embedding: s.embedding, source: s.source, createdAt: date)
        }
        let identifier = EmbeddingIdentifier(samples: samples, config: block.config)
        let e = block.expected
        XCTAssertEqual(identifier.prototypes, e.prototypes.compactMap { prototypeKey($0) }, "prototypes")
        XCTAssertEqual(identifier.prototypeVectors.count, e.prototypeVectors.count)
        for (k, pair) in zip(identifier.prototypeVectors, e.prototypeVectors).enumerated() {
            assertClose(pair.0, pair.1, rel: tolerance, "prototypeVectors[\(k)]")
        }
        assertClose(identifier.looSimilarities, e.loo, rel: tolerance, "loo")
        assertClose(identifier.simThreshold, e.simThreshold, rel: tolerance, "simThreshold")
        assertClose(identifier.absThreshold, e.absThreshold, rel: tolerance, "absThreshold")
        if let w = e.handWeights {
            if let actual = identifier.handWeights {
                assertClose(actual, w, rel: tolerance, "handWeights")
                assertClose(identifier.handBias, e.handBias ?? 0.0, rel: tolerance, "handBias")
            } else {
                XCTFail("hand model expected")
            }
        } else {
            XCTAssertNil(identifier.handWeights)
        }

        XCTAssertFalse(block.queries.isEmpty)
        for q in block.queries {
            let name = "query \(q.name)"
            for (j, pattern) in q.patterns.enumerated() {
                if j < q.similarities.count {
                    assertClose(identifier.similarities(pattern), q.similarities[j], rel: tolerance, "\(name) similarities[\(j)]")
                }
                if j < q.handLogOdds.count {
                    assertClose(identifier.handLogOdds(pattern), q.handLogOdds[j], rel: tolerance, "\(name) handLogOdds[\(j)]")
                }
            }
            let decision = identifier.decide(q.patterns)
            if let expected = decisionOutcome(q.outcome) {
                XCTAssertEqual(decision.outcome, expected, "\(name) outcome")
            } else {
                XCTFail("\(name): unreadable expected outcome")
            }
            XCTAssertEqual(decision.patternsUsed, q.patterns.count, "\(name) patternsUsed")
            assertClose(decision.bestDistance, q.bestDistance ?? Double.infinity, rel: tolerance, "\(name) bestDistance")
            assertClose(decision.marginName, q.marginName ?? Double.infinity, rel: tolerance, "\(name) marginName")
            assertClose(decision.marginHand, q.marginHand ?? Double.infinity, rel: tolerance, "\(name) marginHand")
            assertClose(decision.prototypeDistances.map { $0.distance }, q.meanDistances, rel: tolerance, "\(name) meanDistances")
            assertClose(decision.handLogOdds, q.meanHandLogOdds, rel: tolerance, "\(name) handLogOdds")
            assertClose(decision.absThreshold, e.absThreshold, rel: tolerance, "\(name) absThreshold")
            XCTAssertEqual(decision.bestPrototype, prototypeKey(q.bestPrototype), "\(name) bestPrototype")
            XCTAssertEqual(decision.personScores.map { $0.personID }, q.personScores.map { $0.personID }, "\(name) personScores order")
            for (s, pair) in zip(decision.personScores, q.personScores).enumerated() {
                assertClose(pair.0.probability, pair.1.probability, rel: tolerance, "\(name) personScores[\(s)].probability")
                assertClose(pair.0.bestDistance, pair.1.bestDistance, rel: tolerance, "\(name) personScores[\(s)].bestDistance")
            }
            let adaptive = identifier.adaptiveSamples(for: decision, patterns: q.patterns, now: date)
            XCTAssertEqual(!adaptive.isEmpty, q.adaptiveEligible, "\(name) adaptiveEligible")
        }
    }

    // MARK: - Encoder input

    func testEncoderInputLayout() {
        let p = SignalProcessor.process(Synthetic.recording(seed: 7, delay: 0.05))
        let lag = p.alignment.lag
        XCTAssertEqual(lag, 5)
        XCTAssertEqual(p.index(ofPatternTime: 0.0), EncoderInput.patternStart + lag)
        let x = EncoderInput.build(p)
        let length = EncoderInput.length
        XCTAssertEqual(x.count, EncoderInput.channels * length)
        func clipped(_ v: Double) -> Float { return Float(min(max(v, -50.0), 50.0)) }
        for u in 0..<length {
            let src = u + lag
            if src < p.n {
                XCTAssertEqual(x[u], clipped(p.accelHP[0][src] / 0.02))
                XCTAssertEqual(x[4 * length + u], clipped(p.gyroHP[1][src] / 0.05))
                XCTAssertEqual(x[8 * length + u], clipped(p.gravity[2][src]))
                XCTAssertEqual(x[9 * length + u], u < p.n ? clipped(p.template[u]) : 0)
            } else {
                for c in 0..<EncoderInput.channels { XCTAssertEqual(x[c * length + u], 0) }
            }
        }
        XCTAssertTrue(x.allSatisfy { $0.isFinite && $0 >= -50 && $0 <= 50 })

        let empty = Recording(createdAt: date, purpose: .check, challenge: ChallengeGenerator.generate(seed: 2),
                              hapticStart: 1.0, accel: .empty, gyro: .empty)
        let e = EncoderInput.build(SignalProcessor.process(empty))
        XCTAssertEqual(e.count, 2560)
        XCTAssertTrue(e.allSatisfy { $0.isFinite })
    }

    // MARK: - Embedding identifier

    private func vector(_ components: [Int: Double], dimension: Int = 8) -> [Double] {
        var v = [Double](repeating: 0.0, count: dimension)
        for (i, c) in components { v[i] = c }
        return v
    }

    private func toyEmbeddings(slack: Double = 0.0) -> (EmbeddingIdentifier, [Double]) {
        var rng = Xoshiro256StarStar(seed: 5)
        let centers: [(PrototypeKey, [Double])] = [
            (PrototypeKey(personID: "A", hand: .right), vector([0: 1.0, 2: 0.2])),
            (PrototypeKey(personID: "A", hand: .left), vector([0: 1.0, 2: -0.2])),
            (PrototypeKey(personID: "B", hand: .right), vector([1: 1.0, 3: 0.2])),
            (PrototypeKey(personID: "B", hand: .left), vector([1: 1.0, 3: -0.2])),
        ]
        var samples: [EmbeddingSample] = []
        for (key, center) in centers {
            for i in 0..<4 {
                let noisy = center.map { $0 + 0.02 * rng.uniform(-1, 1) }
                samples.append(EmbeddingSample(prototype: key, embedding: noisy, source: .enrollment,
                                               createdAt: date.addingTimeInterval(Double(i))))
            }
        }
        let config = EmbeddingConfig(slack: slack)
        return (EmbeddingIdentifier(samples: samples, config: config), centers[0].1)
    }

    func testEmbeddingIdentifierBehaviour() {
        let none = EmbeddingIdentifier(samples: [])
        XCTAssertEqual(none.prototypes, [])
        XCTAssertEqual(none.absThreshold, 2.0)
        XCTAssertEqual(none.decide([[1.0, 0.0]]).outcome, .unknown(.noPeople))

        let (identifier, aRight) = toyEmbeddings()
        XCTAssertEqual(identifier.prototypes.count, 4)
        XCTAssertNotNil(identifier.handWeights)
        XCTAssertEqual(identifier.looSimilarities.count, 16)
        XCTAssertEqual(identifier.decide([]).outcome, .needMorePatterns)

        // Scale does not matter: patterns are renormalized.
        let decision = identifier.decide([aRight.map { $0 * 3.0 }])
        XCTAssertEqual(decision.outcome, .identified(personID: "A", hand: .right))
        XCTAssertEqual(decision.bestPrototype, PrototypeKey(personID: "A", hand: .right))
        XCTAssertGreaterThan(decision.handLogOdds, 0.0)
        XCTAssertGreaterThan(decision.marginName, 0.5)
        XCTAssertEqual(decision.personScores.first?.personID, "A")
        XCTAssertEqual(decision.prototypeDistances.count, 4)
        XCTAssertLessThanOrEqual(decision.bestDistance, decision.absThreshold)
        // absThreshold is tiny here, so the 0.05 adaptive margin is not met.
        XCTAssertEqual(identifier.adaptiveSamples(for: decision, patterns: [aRight], now: date), [])

        let stranger = vector([5: 1.0])
        XCTAssertEqual(identifier.decide([stranger]).outcome, .needMorePatterns)
        XCTAssertEqual(identifier.decide([stranger, stranger, stranger]).outcome, .unknown(.noMatch))
        XCTAssertEqual(identifier.decide([[1.0], [1.0], [1.0]]).outcome, .unknown(.noMatch))

        let (lenient, _) = toyEmbeddings(slack: 0.2)
        let accepted = lenient.decide([aRight])
        XCTAssertEqual(accepted.outcome, .identified(personID: "A", hand: .right))
        let adaptive = lenient.adaptiveSamples(for: accepted, patterns: [aRight, aRight], now: date)
        XCTAssertEqual(adaptive.count, 2)
        XCTAssertTrue(adaptive.allSatisfy { $0.source == .adaptive && $0.createdAt == date })
        XCTAssertEqual(adaptive.first?.prototype, PrototypeKey(personID: "A", hand: .right))
    }

    func testEmbeddingAdaptiveCapAndCoding() throws {
        let key = PrototypeKey(personID: "A", hand: .left)
        var samples = [EmbeddingSample(prototype: key, embedding: [1, 0], source: .enrollment, createdAt: date)]
        for i in 0..<25 {
            samples.append(EmbeddingSample(prototype: key, embedding: [1, Double(i)], source: .adaptive,
                                           createdAt: date.addingTimeInterval(Double(25 - i))))
        }
        let capped = EmbeddingIdentifier.applyAdaptiveCap(samples, cap: EmbeddingConfig.default.adaptiveMaxPerPrototype)
        XCTAssertEqual(capped.count, 21)
        XCTAssertEqual(capped.first?.source, .enrollment)
        // The oldest are the last five in the list (createdAt offsets 1...5).
        XCTAssertEqual(capped.last?.embedding, [1, 19])

        let data = try RecordingCoding.encoder().encode(samples)
        XCTAssertEqual(try RecordingCoding.decoder().decode([EmbeddingSample].self, from: data), samples)
        let partial = try JSONDecoder().decode(EmbeddingConfig.self, from: Data("{\"slack\": 0.02}".utf8))
        XCTAssertEqual(partial.slack, 0.02)
        XCTAssertEqual(partial.nameMargin, 0.03)
        XCTAssertEqual(partial.maxPatterns, 3)
        XCTAssertEqual(EmbeddingConfig.default.handShrinkage, 0.1)
    }
}
