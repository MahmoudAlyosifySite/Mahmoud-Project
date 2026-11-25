import Foundation
import XCTest
@testable import AuditorCore

private struct ModelBlock: Decodable {
    struct Sample: Decodable {
        let personID: String
        let hand: Hand
        let accessory: Accessory
        let source: SampleSource
        let features: [Double]
    }
    struct Expected: Decodable {
        let prototypes: [[String]]
        let mean: [Double]
        let scale: [Double]
        let shrinkage: Double
        let absThreshold: Double
        let handShrinkage: Double?
        let handAbsThreshold: Double?
        let shrinkageLW: Double?
        let mu: Double?
        let classCounts: [Int]?
        let classMeans: [[Double]]?
        let looDistances: [Double]?
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
        let name: String?
        let patterns: [[Double]]
        let distances: [[Double]]
        let handLogOdds: [Double]
        let outcome: Outcome
        let bestDistance: Double?
        let marginName: Double?
        let marginHand: Double?
        let meanDistances: [Double]?
        let meanHandLogOdds: Double?
        let bestPrototype: [String]?
        let personScores: [Score]?
        let adaptiveEligible: Bool?
    }
    let config: ModelConfig
    let identifierConfig: IdentifierConfig
    let samples: [Sample]
    let expected: Expected
    let queries: [Query]
}

private struct ModelFileExtras: Decodable {
    let singlePerson: ModelBlock?
}

final class ModelTests: XCTestCase {
    private let tolerance = 1e-6
    private let fixedDate = Date(timeIntervalSince1970: 1_795_514_400)

    private func key(_ p: [String]) -> PrototypeKey? {
        guard p.count == 3, let hand = Hand(rawValue: p[1]), let accessory = Accessory(rawValue: p[2]) else { return nil }
        return PrototypeKey(personID: p[0], hand: hand, accessory: accessory)
    }

    private func outcome(_ o: ModelBlock.Outcome) -> DecisionOutcome? {
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

    /// SPEC 11.5, tolerance `1e-6 · max(1, |x|)`, outcomes exact.
    func testModelFixtureParity() {
        guard let block = Fixture.decode(ModelBlock.self, "model_vectors") else { return }
        check(block, label: "main")
        XCTAssertGreaterThanOrEqual(Set(block.samples.map { $0.personID }).count, 3)
    }

    func testSinglePersonFixtureParity() {
        guard let extras = Fixture.decode(ModelFileExtras.self, "model_vectors") else { return }
        guard let block = extras.singlePerson else {
            XCTFail("model_vectors.json has no singlePerson block")
            return
        }
        check(block, label: "singlePerson")
    }

    private func check(_ block: ModelBlock, label: String) {
        let samples = block.samples.map { s in
            TemplateSample(prototype: PrototypeKey(personID: s.personID, hand: s.hand, accessory: s.accessory),
                           source: s.source, createdAt: fixedDate, features: s.features)
        }
        let identifier = Identifier(samples: samples, modelConfig: block.config, config: block.identifierConfig)
        let e = block.expected
        XCTAssertEqual(identifier.prototypes, e.prototypes.compactMap { key($0) }, "\(label) prototypes")
        XCTAssertEqual(identifier.prototypes.count, e.prototypes.count, "\(label) prototype count")
        guard let model = identifier.model else {
            XCTFail("\(label): model was not trained")
            return
        }
        assertClose(model.mean, e.mean, rel: tolerance, "\(label) mean")
        assertClose(model.scale, e.scale, rel: tolerance, "\(label) scale")
        assertClose(model.shrinkage, e.shrinkage, rel: tolerance, "\(label) shrinkage")
        assertClose(model.absThreshold, e.absThreshold, rel: tolerance, "\(label) absThreshold")
        if let mu = e.mu { assertClose(model.mu, mu, rel: tolerance, "\(label) mu") }
        if let lw = e.shrinkageLW { assertClose(model.shrinkageLW, lw, rel: tolerance, "\(label) shrinkageLW") }
        if let counts = e.classCounts { XCTAssertEqual(model.classCounts, counts, "\(label) classCounts") }
        if let means = e.classMeans {
            XCTAssertEqual(model.classMeans.count, means.count, "\(label) classMeans count")
            for (k, pair) in zip(model.classMeans, means).enumerated() {
                assertClose(pair.0, pair.1, rel: tolerance, "\(label) classMeans[\(k)]")
            }
        }
        if let loo = e.looDistances { assertClose(model.looDistances, loo, rel: tolerance, "\(label) looDistances") }
        if let handShrinkage = e.handShrinkage {
            if let head = identifier.handModel {
                assertClose(head.shrinkage, handShrinkage, rel: tolerance, "\(label) handShrinkage")
                if let t = e.handAbsThreshold { assertClose(head.absThreshold, t, rel: tolerance, "\(label) handAbsThreshold") }
            } else {
                XCTFail("\(label): hand model expected")
            }
        }

        XCTAssertFalse(block.queries.isEmpty, "\(label) has no queries")
        for (qi, q) in block.queries.enumerated() {
            let name = "\(label) query \(q.name ?? String(qi))"
            XCTAssertEqual(q.distances.count, q.patterns.count, "\(name) distances per pattern")
            for (j, pattern) in q.patterns.enumerated() {
                if j < q.distances.count {
                    assertClose(model.distances(pattern), q.distances[j], rel: tolerance, "\(name) distances[\(j)]")
                }
                if j < q.handLogOdds.count {
                    var h = 0.0
                    if let head = identifier.handModel {
                        let d = head.distances(pattern)
                        h = (d[1] - d[0]) / 2.0
                    }
                    assertClose(h, q.handLogOdds[j], rel: tolerance, "\(name) handLogOdds[\(j)]")
                }
            }
            let decision = identifier.decide(q.patterns)
            if let expected = outcome(q.outcome) {
                XCTAssertEqual(decision.outcome, expected, "\(name) outcome")
            } else {
                XCTFail("\(name): unreadable expected outcome \(q.outcome)")
            }
            XCTAssertEqual(decision.patternsUsed, q.patterns.count, "\(name) patternsUsed")
            assertClose(decision.bestDistance, q.bestDistance ?? Double.infinity, rel: tolerance, "\(name) bestDistance")
            assertClose(decision.marginName, q.marginName ?? Double.infinity, rel: tolerance, "\(name) marginName")
            assertClose(decision.marginHand, q.marginHand ?? Double.infinity, rel: tolerance, "\(name) marginHand")
            if let mean = q.meanDistances {
                assertClose(decision.prototypeDistances.map { $0.distance }, mean, rel: tolerance, "\(name) meanDistances")
                XCTAssertEqual(decision.prototypeDistances.map { $0.prototype }, identifier.prototypes, "\(name) prototype order")
            }
            if let h = q.meanHandLogOdds { assertClose(decision.handLogOdds, h, rel: tolerance, "\(name) handLogOdds") }
            if let best = q.bestPrototype { XCTAssertEqual(decision.bestPrototype, key(best), "\(name) bestPrototype") }
            if let scores = q.personScores {
                XCTAssertEqual(decision.personScores.map { $0.personID }, scores.map { $0.personID }, "\(name) personScores order")
                for (s, pair) in zip(decision.personScores, scores).enumerated() {
                    assertClose(pair.0.probability, pair.1.probability, rel: tolerance, "\(name) personScores[\(s)].probability")
                    assertClose(pair.0.bestDistance, pair.1.bestDistance, rel: tolerance, "\(name) personScores[\(s)].bestDistance")
                }
            }
            if let eligible = q.adaptiveEligible {
                let adaptive = identifier.adaptiveSamples(for: decision, patterns: q.patterns, now: fixedDate)
                XCTAssertEqual(!adaptive.isEmpty, eligible, "\(name) adaptiveEligible")
                if eligible {
                    XCTAssertEqual(adaptive.count, q.patterns.count)
                    XCTAssertTrue(adaptive.allSatisfy { $0.source == .adaptive && $0.prototype == decision.bestPrototype })
                }
            }
        }
    }

    // MARK: - Edge cases

    func testFitRejectsEmptyAndMalformedInput() {
        let config = ModelConfig.default
        XCTAssertNil(GaussianModel.fit(features: [], labels: [], classCount: 1, config: config))
        XCTAssertNil(GaussianModel.fit(features: [[1, 2], [1]], labels: [0, 0], classCount: 1, config: config))
        XCTAssertNil(GaussianModel.fit(features: [[1, 2]], labels: [1], classCount: 1, config: config))
        XCTAssertNil(GaussianModel.fit(features: [[1, 2]], labels: [0, 0], classCount: 1, config: config))
    }

    func testSingleSampleIsDegenerateButUsable() {
        guard let m = GaussianModel.fit(features: [[1.0, 2.0, 3.0]], labels: [0], classCount: 1, config: .default) else {
            XCTFail("a single sample must give a model")
            return
        }
        XCTAssertTrue(m.degenerate)
        XCTAssertEqual(m.shrinkage, 1.0)
        XCTAssertEqual(m.looDistances, [])
        XCTAssertEqual(m.absThreshold, 1.5 * 2.0 * 3.0)
        XCTAssertEqual(m.scale, [1e-6, 1e-6, 1e-6])
        XCTAssertEqual(m.distances([1.0, 2.0, 3.0]), [0.0])
        XCTAssertEqual(m.distances([1.0, 2.0]), [Double.infinity])
        XCTAssertTrue(m.distances([1.0, 2.0, 3.000001])[0] > 0.0)
    }

    func testClassWithoutSamplesAndShrinkageBounds() {
        var rng = Xoshiro256StarStar(seed: 11)
        var x: [[Double]] = []
        var labels: [Int] = []
        for i in 0..<30 {
            let k = i % 2
            x.append((0..<5).map { j in Double(k * 3 + j) + rng.uniform(-1, 1) })
            labels.append(k)
        }
        guard let m = GaussianModel.fit(features: x, labels: labels, classCount: 3, config: .default) else {
            XCTFail("fit failed")
            return
        }
        XCTAssertEqual(m.classCounts, [15, 15, 0])
        XCTAssertEqual(m.classMeans[2], [0, 0, 0, 0, 0])
        XCTAssertFalse(m.degenerate)
        XCTAssertTrue(m.shrinkage >= 0.1 && m.shrinkage <= 1.0)
        XCTAssertEqual(m.looDistances.count, 30)
        XCTAssertTrue(m.absThreshold > 0.0 && m.absThreshold.isFinite)
        let d = m.distances(x[0])
        XCTAssertEqual(d.count, 3)
        XCTAssertLessThan(d[0], d[1])
    }

    func testQuantileMatchesNumpyLinear() {
        // numpy.quantile([1, 2, 3, 4], q) for q = 0, 0.5, 0.95, 1 is 1, 2.5, 3.85, 4.
        XCTAssertEqual(Numerics.quantileLinear([4, 1, 3, 2], 0.0), 1.0)
        XCTAssertEqual(Numerics.quantileLinear([4, 1, 3, 2], 0.5), 2.5)
        XCTAssertEqual(Numerics.quantileLinear([4, 1, 3, 2], 0.95) ?? 0, 3.85, accuracy: 1e-12)
        XCTAssertEqual(Numerics.quantileLinear([4, 1, 3, 2], 1.0), 4.0)
        XCTAssertEqual(Numerics.quantileLinear([7], 0.95), 7.0)
        XCTAssertNil(Numerics.quantileLinear([], 0.5))
        XCTAssertEqual(Numerics.median([3, 1, 2]), 2.0)
        XCTAssertEqual(Numerics.median([4, 1, 3, 2]), 2.5)
        XCTAssertEqual(Numerics.median([]), 0.0)
        XCTAssertEqual(Numerics.solvePartialPivot([[0, 1], [2, 0]], [3, 4]) ?? [], [2, 3])
        XCTAssertNil(Numerics.solvePartialPivot([[1, 2], [2, 4]], [1, 2]))
    }
}
