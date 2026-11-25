import Foundation

// MARK: - Encoder input (SPEC 14.1)

/// The input tensor of the v1 learned encoder, built from a processed signal.
public enum EncoderInput {
    public static let channels = 10, length = 256, patternStart = 25

    static let accelScale = 0.02
    static let gyroScale = 0.05
    static let clip = 50.0

    /// Row-major `[channel][u]`, `channels * length` values. Time index `u` reads grid
    /// index `n = u + lag`; outside `[0, N)` every channel is 0. Channels: `accelHP / 0.02`,
    /// `gyroHP / 0.05`, gravity, and the lag-0 template at `u`. As in the reference
    /// (`research/v1/data.py`, `assemble_tensor`), the template channel is written only
    /// where `u + lag` is inside the grid (and `u < N`). Computed in Double, clipped to
    /// `[-50, 50]`, then rounded to Float.
    public static func build(_ p: ProcessedSignal) -> [Float] {
        var x = [Double](repeating: 0.0, count: channels * length)
        let n = p.n
        let lag = p.alignment.lag
        let channelsOK = p.accelHP.count == 3 && p.gyroHP.count == 3 && p.gravity.count == 3
        let streams = p.accelHP + p.gyroHP + p.gravity
        let lengthsOK = streams.allSatisfy({ $0.count == n }) && p.template.count == n
        if channelsOK && lengthsOK {
            for u in 0..<length {
                let src = u + lag
                if src < 0 || src >= n { continue }
                for c in 0..<3 {
                    x[c * length + u] = p.accelHP[c][src] / accelScale
                    x[(3 + c) * length + u] = p.gyroHP[c][src] / gyroScale
                    x[(6 + c) * length + u] = p.gravity[c][src]
                }
                if u < n { x[9 * length + u] = p.template[u] }
            }
        }
        return x.map { Float(min(max($0, -clip), clip)) }
    }
}

// MARK: - Embedding samples and config (SPEC 14.3, 14.5)

/// A stored encoder embedding of one prototype.
public struct EmbeddingSample: Codable, Equatable {
    public var prototype: PrototypeKey
    public var embedding: [Double]
    public var source: SampleSource
    public var createdAt: Date

    public init(prototype: PrototypeKey, embedding: [Double], source: SampleSource = .enrollment,
                createdAt: Date = Date()) {
        self.prototype = prototype
        self.embedding = embedding
        self.source = source
        self.createdAt = createdAt
    }
}

public struct EmbeddingConfig: Equatable, Codable {
    public var looQuantile: Double
    public var slack: Double
    public var nameMargin: Double
    public var handMargin: Double
    public var handShrinkage: Double
    public var maxPatterns: Int
    public var adaptiveMaxPerPrototype: Int

    public init(looQuantile: Double = 0.95, slack: Double = 0.0, nameMargin: Double = 0.03, handMargin: Double = 0.0,
                handShrinkage: Double = 0.1, maxPatterns: Int = 3, adaptiveMaxPerPrototype: Int = 20) {
        self.looQuantile = looQuantile
        self.slack = slack
        self.nameMargin = nameMargin
        self.handMargin = handMargin
        self.handShrinkage = handShrinkage
        self.maxPatterns = maxPatterns
        self.adaptiveMaxPerPrototype = adaptiveMaxPerPrototype
    }

    public static let `default` = EmbeddingConfig()

    private enum CodingKeys: String, CodingKey {
        case looQuantile, slack, nameMargin, handMargin, handShrinkage, maxPatterns, adaptiveMaxPerPrototype
    }

    /// Missing keys take their defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = EmbeddingConfig.default
        looQuantile = try c.decodeIfPresent(Double.self, forKey: .looQuantile) ?? d.looQuantile
        slack = try c.decodeIfPresent(Double.self, forKey: .slack) ?? d.slack
        nameMargin = try c.decodeIfPresent(Double.self, forKey: .nameMargin) ?? d.nameMargin
        handMargin = try c.decodeIfPresent(Double.self, forKey: .handMargin) ?? d.handMargin
        handShrinkage = try c.decodeIfPresent(Double.self, forKey: .handShrinkage) ?? d.handShrinkage
        maxPatterns = try c.decodeIfPresent(Int.self, forKey: .maxPatterns) ?? d.maxPatterns
        adaptiveMaxPerPrototype = try c.decodeIfPresent(Int.self, forKey: .adaptiveMaxPerPrototype)
            ?? d.adaptiveMaxPerPrototype
    }
}

// MARK: - Embedding identifier (SPEC 14.3)

/// v1 decision on encoder embeddings: cosine prototypes, a leave-one-out similarity
/// threshold and a pooled LDA right/left model. Decisions are expressed in cosine
/// distance `D = 1 − S` so that they reuse `Decision` (margins equal similarity differences).
///
/// All stored samples (any source; adaptive samples join their prototype) build the
/// prototypes, the leave-one-out threshold and the hand model. Samples whose dimension
/// differs from the first sample's are ignored; a pattern of another dimension is
/// infinitely far from every prototype.
public struct EmbeddingIdentifier {
    public let prototypes: [PrototypeKey]
    /// `1 − simThreshold`, in cosine-distance units.
    public let absThreshold: Double
    public let config: EmbeddingConfig

    // Internal details.
    let dimension: Int
    /// L2-normalized prototype vectors, in prototype order.
    let prototypeVectors: [[Double]]
    /// Leave-one-out similarities in sample order.
    let looSimilarities: [Double]
    let simThreshold: Double
    /// LDA direction `w` and bias `b` (`w·e + b > 0` means right); nil without a hand model.
    let handWeights: [Double]?
    let handBias: Double

    static let adaptiveNameFactor = 2.0
    static let adaptiveDistanceMargin = 0.05
    static let scoreTemperature = 0.05

    public init(samples: [EmbeddingSample], config: EmbeddingConfig = .default) {
        self.config = config
        let dim = samples.first?.embedding.count ?? 0
        let used: [EmbeddingSample] = dim > 0 ? samples.filter({ $0.embedding.count == dim }) : []
        dimension = dim
        let protos = Array(Set(used.map { $0.prototype })).sorted()
        prototypes = protos
        var index: [PrototypeKey: Int] = [:]
        for (i, p) in protos.enumerated() { index[p] = i }

        let unit = used.map { EmbeddingIdentifier.normalize($0.embedding) }
        var sums = [[Double]](repeating: [Double](repeating: 0.0, count: dim), count: protos.count)
        var counts = [Int](repeating: 0, count: protos.count)
        var labels: [Int] = []
        for (i, s) in used.enumerated() {
            let k = index[s.prototype] ?? 0
            labels.append(k)
            counts[k] += 1
            for j in 0..<dim { sums[k][j] = sums[k][j] + unit[i][j] }
        }
        prototypeVectors = sums.map { EmbeddingIdentifier.normalize($0) }

        var loo: [Double] = []
        for i in 0..<used.count {
            let k = labels[i]
            if counts[k] < 2 { continue }
            var rest = sums[k]
            for j in 0..<dim { rest[j] = rest[j] - unit[i][j] }
            loo.append(EmbeddingIdentifier.dot(EmbeddingIdentifier.normalize(rest), unit[i]))
        }
        looSimilarities = loo
        let threshold: Double
        if let q = Numerics.quantileLinear(loo, 1.0 - config.looQuantile) {
            threshold = q - config.slack
        } else {
            threshold = -1.0 - config.slack
        }
        simThreshold = threshold
        absThreshold = 1.0 - threshold

        let hands = used.map { $0.prototype.hand }
        let model = EmbeddingIdentifier.handModel(unit: unit, hands: hands, dimension: dim,
                                                  shrinkage: config.handShrinkage)
        handWeights = model?.w
        handBias = model?.b ?? 0.0
    }

    /// Pooled LDA right/left model of SPEC 14.3: shrunk pooled covariance of the residuals
    /// (right residuals first, then left), `w = C⁻¹(μ_r − μ_l)` by Cholesky, `b = −½ w·(μ_r + μ_l)`.
    /// nil unless both hands have at least 2 samples (or if `C` is not positive definite).
    static func handModel(unit: [[Double]], hands: [Hand], dimension d: Int,
                          shrinkage: Double) -> (w: [Double], b: Double)? {
        let right = (0..<unit.count).filter { hands[$0] == .right }
        let left = (0..<unit.count).filter { hands[$0] == .left }
        guard right.count >= 2, left.count >= 2, d > 0 else { return nil }
        func meanOf(_ rows: [Int]) -> [Double] {
            var m = [Double](repeating: 0.0, count: d)
            for i in rows {
                for j in 0..<d { m[j] = m[j] + unit[i][j] }
            }
            return m.map { $0 / Double(rows.count) }
        }
        let muR = meanOf(right)
        let muL = meanOf(left)
        var residuals: [[Double]] = []
        for i in right { residuals.append((0..<d).map { unit[i][$0] - muR[$0] }) }
        for i in left { residuals.append((0..<d).map { unit[i][$0] - muL[$0] }) }
        let n = Double(residuals.count)
        var cov = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: d)
        for r in residuals {
            for j in 0..<d {
                let rj = r[j]
                for k in j..<d { cov[j][k] = cov[j][k] + rj * r[k] }
            }
        }
        var trace = 0.0
        for j in 0..<d {
            for k in j..<d {
                cov[j][k] = cov[j][k] / n
                cov[k][j] = cov[j][k]
            }
            trace = trace + cov[j][j]
        }
        let diagonal = shrinkage * trace / Double(d)
        for j in 0..<d {
            for k in 0..<d {
                cov[j][k] = (1.0 - shrinkage) * cov[j][k] + (j == k ? diagonal : 0.0)
            }
        }
        guard let low = LinearAlgebra.cholesky(cov) else { return nil }
        let delta = (0..<d).map { muR[$0] - muL[$0] }
        let w = LinearAlgebra.choleskySolve(low, delta)
        let sum = (0..<d).map { muR[$0] + muL[$0] }
        return (w, -0.5 * dot(w, sum))
    }

    /// Cosine similarity of the (renormalized) pattern with every prototype;
    /// `-infinity` for a pattern of the wrong dimension.
    func similarities(_ pattern: [Double]) -> [Double] {
        guard pattern.count == dimension, dimension > 0 else {
            return [Double](repeating: -Double.infinity, count: prototypes.count)
        }
        let e = EmbeddingIdentifier.normalize(pattern)
        return prototypeVectors.map { EmbeddingIdentifier.dot(e, $0) }
    }

    /// Hand log-odds `h = w·e + b` for "right" (0 without a hand model or for a wrong dimension).
    func handLogOdds(_ pattern: [Double]) -> Double {
        guard let w = handWeights, pattern.count == w.count else { return 0.0 }
        return EmbeddingIdentifier.dot(w, EmbeddingIdentifier.normalize(pattern)) + handBias
    }

    /// SPEC 14.3 on the given pattern embeddings, in cosine-distance units.
    public func decide(_ patterns: [[Double]]) -> Decision {
        let m = patterns.count
        if prototypes.isEmpty {
            return Decision(outcome: .unknown(.noPeople), patternsUsed: m, absThreshold: absThreshold)
        }
        if m == 0 {
            return Decision(outcome: .needMorePatterns, patternsUsed: 0, absThreshold: absThreshold)
        }
        let kCount = prototypes.count
        let perPattern = patterns.map { similarities($0) }
        var sbar = [Double](repeating: 0.0, count: kCount)
        for k in 0..<kCount {
            var s = 0.0
            for j in 0..<m { s = s + perPattern[j][k] }
            sbar[k] = s / Double(m)
        }
        let dbar = sbar.map { 1.0 - $0 }
        let hasHead = handWeights != nil
        let hbar = hasHead ? Numerics.mean(patterns.map { handLogOdds($0) }) : 0.0

        var kstar = 0
        for k in 1..<max(kCount, 1) where dbar[k] < dbar[kstar] {
            kstar = k
        }
        let best = dbar[kstar]
        let person = prototypes[kstar].personID
        let hand = prototypes[kstar].hand
        var otherPerson = Double.infinity
        var otherHand = Double.infinity
        for k in 0..<kCount {
            let p = prototypes[k]
            if p.personID != person {
                if dbar[k] < otherPerson { otherPerson = dbar[k] }
            } else if p.hand != hand {
                if dbar[k] < otherHand { otherHand = dbar[k] }
            }
        }
        let marginName = otherPerson - best
        let marginHand = otherHand - best
        let absOK = best <= absThreshold
        let nameOK = marginName >= config.nameMargin
        let headOK = !hasHead || (hand == .right ? hbar > 0.0 : hbar < 0.0)
        let handOK = marginHand >= config.handMargin && headOK

        let outcome: DecisionOutcome
        if absOK && nameOK && handOK {
            outcome = .identified(personID: person, hand: hand)
        } else if m < config.maxPatterns {
            outcome = .needMorePatterns
        } else if absOK && nameOK {
            outcome = .identifiedHandUnclear(personID: person)
        } else if !absOK {
            outcome = .unknown(.noMatch)
        } else {
            outcome = .unknown(.ambiguous)
        }
        var distances: [PrototypeDistance] = []
        for (k, p) in prototypes.enumerated() {
            distances.append(PrototypeDistance(prototype: p, distance: dbar[k]))
        }
        return Decision(
            outcome: outcome, patternsUsed: m, bestPrototype: prototypes[kstar], bestDistance: best,
            marginName: marginName, marginHand: marginHand, handLogOdds: hbar, absThreshold: absThreshold,
            personScores: personScores(sbar), prototypeDistances: distances)
    }

    /// Softmax over persons of `S̄_max(person) / 0.05`, sorted by probability (descending),
    /// ties in person (prototype) order; `bestDistance = 1 − S̄_max(person)`.
    func personScores(_ sbar: [Double]) -> [PersonScore] {
        var persons: [String] = []
        var best: [Double] = []
        for (k, p) in prototypes.enumerated() {
            if let i = persons.firstIndex(of: p.personID) {
                if sbar[k] > best[i] { best[i] = sbar[k] }
            } else {
                persons.append(p.personID)
                best.append(sbar[k])
            }
        }
        if persons.isEmpty { return [] }
        let logits = best.map { $0 / EmbeddingIdentifier.scoreTemperature }
        var top = logits[0]
        for v in logits.dropFirst() where v > top { top = v }
        let probabilities: [Double]
        if top.isFinite {
            let ex = logits.map { exp($0 - top) }
            let total = Numerics.sum(ex)
            probabilities = ex.map { $0 / total }
        } else {
            probabilities = [Double](repeating: 1.0 / Double(persons.count), count: persons.count)
        }
        var scores: [PersonScore] = []
        for i in 0..<persons.count {
            scores.append(PersonScore(personID: persons[i], probability: probabilities[i], bestDistance: 1.0 - best[i]))
        }
        let order = scores.indices.sorted { a, b in
            let pa = scores[a].probability
            let pb = scores[b].probability
            if pa != pb && !(pa.isNaN || pb.isNaN) { return pa > pb }
            if pa.isNaN != pb.isNaN { return pb.isNaN }
            return a < b
        }
        return order.map { scores[$0] }
    }

    /// Template updates (9.5 analog): after `identified` with `marginName ≥ 2 · nameMargin`
    /// and `bestDistance ≤ absThreshold − 0.05`, every pattern embedding becomes an
    /// adaptive sample of prototype k*; otherwise `[]`.
    public func adaptiveSamples(for decision: Decision, patterns: [[Double]], now: Date) -> [EmbeddingSample] {
        guard !prototypes.isEmpty, let proto = decision.bestPrototype, case .identified = decision.outcome else {
            return []
        }
        guard decision.marginName >= EmbeddingIdentifier.adaptiveNameFactor * config.nameMargin,
              decision.bestDistance <= decision.absThreshold - EmbeddingIdentifier.adaptiveDistanceMargin else {
            return []
        }
        return patterns.map { EmbeddingSample(prototype: proto, embedding: $0, source: .adaptive, createdAt: now) }
    }

    /// Keeps at most `cap` adaptive samples per prototype (`config.adaptiveMaxPerPrototype`
    /// in the app), dropping the oldest by `createdAt`, then list position. Other samples are
    /// never dropped; the order of the kept samples is preserved.
    public static func applyAdaptiveCap(_ samples: [EmbeddingSample], cap: Int) -> [EmbeddingSample] {
        let limit = max(cap, 0)
        var groups: [PrototypeKey: [Int]] = [:]
        for (i, s) in samples.enumerated() where s.source == .adaptive {
            groups[s.prototype, default: []].append(i)
        }
        var drop = Set<Int>()
        for (_, indices) in groups where indices.count > limit {
            let ordered = indices.sorted { a, b in
                let da = samples[a].createdAt
                let db = samples[b].createdAt
                if da != db { return da < db }
                return a < b
            }
            for i in ordered.prefix(indices.count - limit) { drop.insert(i) }
        }
        var kept: [EmbeddingSample] = []
        for (i, s) in samples.enumerated() where !drop.contains(i) {
            kept.append(s)
        }
        return kept
    }

    // MARK: - Vector helpers

    static func dot(_ a: [Double], _ b: [Double]) -> Double {
        var s = 0.0
        for i in 0..<min(a.count, b.count) { s = s + a[i] * b[i] }
        return s
    }

    /// `v / max(‖v‖, 1e-12)`.
    static func normalize(_ v: [Double]) -> [Double] {
        let norm = dot(v, v).squareRoot()
        let scale = max(norm, 1e-12)
        return v.map { $0 / scale }
    }
}
