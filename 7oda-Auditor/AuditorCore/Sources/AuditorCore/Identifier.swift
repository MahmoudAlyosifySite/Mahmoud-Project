import Foundation

// MARK: - Config and results (SPEC 9.4, 9.5)

public struct IdentifierConfig: Equatable, Codable {
    public var nameMargin: Double
    public var handMargin: Double
    public var maxPatterns: Int
    public var adaptiveMaxPerPrototype: Int
    public var adaptiveNameFactor: Double
    public var adaptiveAbsFactor: Double

    public init(nameMargin: Double = 6.0, handMargin: Double = 3.0, maxPatterns: Int = 3,
                adaptiveMaxPerPrototype: Int = 20, adaptiveNameFactor: Double = 2.0, adaptiveAbsFactor: Double = 0.75) {
        self.nameMargin = nameMargin
        self.handMargin = handMargin
        self.maxPatterns = maxPatterns
        self.adaptiveMaxPerPrototype = adaptiveMaxPerPrototype
        self.adaptiveNameFactor = adaptiveNameFactor
        self.adaptiveAbsFactor = adaptiveAbsFactor
    }

    public static let `default` = IdentifierConfig()

    private enum CodingKeys: String, CodingKey {
        case nameMargin, handMargin, maxPatterns, adaptiveMaxPerPrototype, adaptiveNameFactor, adaptiveAbsFactor
    }

    /// Missing keys take their defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = IdentifierConfig.default
        nameMargin = try c.decodeIfPresent(Double.self, forKey: .nameMargin) ?? d.nameMargin
        handMargin = try c.decodeIfPresent(Double.self, forKey: .handMargin) ?? d.handMargin
        maxPatterns = try c.decodeIfPresent(Int.self, forKey: .maxPatterns) ?? d.maxPatterns
        adaptiveMaxPerPrototype = try c.decodeIfPresent(Int.self, forKey: .adaptiveMaxPerPrototype)
            ?? d.adaptiveMaxPerPrototype
        adaptiveNameFactor = try c.decodeIfPresent(Double.self, forKey: .adaptiveNameFactor) ?? d.adaptiveNameFactor
        adaptiveAbsFactor = try c.decodeIfPresent(Double.self, forKey: .adaptiveAbsFactor) ?? d.adaptiveAbsFactor
    }
}

public enum UnknownReason: String, Codable {
    case noPeople, noMatch, ambiguous
}

public enum DecisionOutcome: Equatable {
    case identified(personID: String, hand: Hand)
    case identifiedHandUnclear(personID: String)
    case needMorePatterns
    case unknown(UnknownReason)
}

public struct PersonScore: Equatable {
    public var personID: String
    public var probability: Double
    public var bestDistance: Double

    public init(personID: String, probability: Double, bestDistance: Double) {
        self.personID = personID
        self.probability = probability
        self.bestDistance = bestDistance
    }
}

public struct PrototypeDistance: Equatable {
    public var prototype: PrototypeKey
    public var distance: Double

    public init(prototype: PrototypeKey, distance: Double) {
        self.prototype = prototype
        self.distance = distance
    }
}

/// Result of SPEC 9.4. Margins are +infinity when there is no competing person / other hand.
public struct Decision: Equatable {
    public var outcome: DecisionOutcome
    public var patternsUsed: Int
    public var bestPrototype: PrototypeKey?
    public var bestDistance: Double
    public var marginName: Double
    public var marginHand: Double
    /// Mean hand log-odds for "right" (0 without a hand head).
    public var handLogOdds: Double
    public var absThreshold: Double
    /// Sorted by probability, descending (ties in prototype order).
    public var personScores: [PersonScore]
    /// Mean distance per prototype, in prototype order.
    public var prototypeDistances: [PrototypeDistance]

    public init(outcome: DecisionOutcome, patternsUsed: Int, bestPrototype: PrototypeKey? = nil,
                bestDistance: Double = Double.infinity, marginName: Double = Double.infinity,
                marginHand: Double = Double.infinity, handLogOdds: Double = 0.0, absThreshold: Double = 0.0,
                personScores: [PersonScore] = [], prototypeDistances: [PrototypeDistance] = []) {
        self.outcome = outcome
        self.patternsUsed = patternsUsed
        self.bestPrototype = bestPrototype
        self.bestDistance = bestDistance
        self.marginName = marginName
        self.marginHand = marginHand
        self.handLogOdds = handLogOdds
        self.absThreshold = absThreshold
        self.personScores = personScores
        self.prototypeDistances = prototypeDistances
    }
}

// MARK: - Identifier (SPEC 9.1, 9.3-9.5)

public struct Identifier {
    /// Prototypes in the order of SPEC 9.1 (= class order of `model`).
    public let prototypes: [PrototypeKey]
    public let model: GaussianModel?
    /// Two classes: right (0) and left (1); nil unless both hands have at least 2 samples.
    public let handModel: GaussianModel?
    public let modelConfig: ModelConfig
    public let config: IdentifierConfig

    /// Trains the model and the hand head. All samples must have the same feature
    /// dimension; samples whose dimension differs from the first sample's are ignored.
    public init(samples: [TemplateSample], modelConfig: ModelConfig = .default, config: IdentifierConfig = .default) {
        self.modelConfig = modelConfig
        self.config = config
        let dimension = samples.first?.features.count ?? 0
        let used = samples.filter { $0.features.count == dimension }
        let protos = Array(Set(used.map { $0.prototype })).sorted()
        prototypes = protos
        var index: [PrototypeKey: Int] = [:]
        for (i, p) in protos.enumerated() { index[p] = i }
        if used.isEmpty {
            model = nil
            handModel = nil
            return
        }
        let x = used.map { $0.features }
        let labels = used.map { index[$0.prototype] ?? 0 }
        model = GaussianModel.fit(features: x, labels: labels, classCount: protos.count, config: modelConfig)
        let hands = used.map { $0.prototype.hand }
        let rightCount = hands.filter { $0 == .right }.count
        let leftCount = hands.count - rightCount
        if rightCount >= 2 && leftCount >= 2 {
            handModel = GaussianModel.fit(features: x, labels: hands.map { $0.order }, classCount: 2,
                                          config: modelConfig)
        } else {
            handModel = nil
        }
    }

    /// `h(x) = (D²_left(x) − D²_right(x)) / 2`; 0 without a hand head.
    func handLogOdds(_ x: [Double]) -> Double {
        guard let head = handModel else { return 0.0 }
        let d = head.distances(x)
        guard d.count == 2 else { return 0.0 }
        return (d[1] - d[0]) / 2.0
    }

    /// Sequential decision of SPEC 9.4 after the given patterns (feature vectors that passed quality).
    public func decide(_ patterns: [[Double]]) -> Decision {
        let m = patterns.count
        guard let model = model, !prototypes.isEmpty else {
            return Decision(outcome: .unknown(.noPeople), patternsUsed: m, absThreshold: self.model?.absThreshold ?? 0.0)
        }
        if m == 0 {
            return Decision(outcome: .needMorePatterns, patternsUsed: 0, absThreshold: model.absThreshold)
        }
        let kCount = prototypes.count
        let perPattern = patterns.map { model.distances($0) }
        var dbar = [Double](repeating: 0.0, count: kCount)
        for k in 0..<kCount {
            var s = 0.0
            for j in 0..<m {
                let row = perPattern[j]
                s = s + (k < row.count ? row[k] : Double.infinity)
            }
            dbar[k] = s / Double(m)
        }
        var hbar = 0.0
        if handModel != nil {
            hbar = Numerics.mean(patterns.map { handLogOdds($0) })
        }
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
        let absThreshold = model.absThreshold
        let absOK = best <= absThreshold
        let nameOK = marginName >= config.nameMargin
        let headOK: Bool
        if handModel == nil {
            headOK = true
        } else {
            headOK = hand == .right ? hbar > 0.0 : hbar < 0.0
        }
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
            personScores: personScores(dbar), prototypeDistances: distances)
    }

    /// `softmax_p(−min_{k∈p} D̄_k / 2)`, sorted by probability (descending), ties in person order.
    func personScores(_ dbar: [Double]) -> [PersonScore] {
        var persons: [String] = []
        var best: [Double] = []
        for (k, p) in prototypes.enumerated() {
            if let i = persons.firstIndex(of: p.personID) {
                if dbar[k] < best[i] { best[i] = dbar[k] }
            } else {
                persons.append(p.personID)
                best.append(dbar[k])
            }
        }
        if persons.isEmpty { return [] }
        let logits = best.map { -$0 / 2.0 }
        var top = logits[0]
        for v in logits.dropFirst() where v > top { top = v }
        let ex = logits.map { exp($0 - top) }
        let total = Numerics.sum(ex)
        var scores: [PersonScore] = []
        for i in 0..<persons.count {
            scores.append(PersonScore(personID: persons[i], probability: ex[i] / total, bestDistance: best[i]))
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

    /// SPEC 9.5: one `adaptive` sample per pattern for prototype k* when the decision is
    /// `identified` with `marginName ≥ adaptiveNameFactor × nameMargin` and
    /// `bestDistance ≤ adaptiveAbsFactor × absThreshold`; otherwise `[]`.
    /// (The app also requires the "Learn from confident checks" setting.)
    public func adaptiveSamples(for decision: Decision, patterns: [[Double]], now: Date) -> [TemplateSample] {
        guard model != nil, let proto = decision.bestPrototype else { return [] }
        guard case .identified = decision.outcome else { return [] }
        guard decision.marginName >= config.adaptiveNameFactor * config.nameMargin,
              decision.bestDistance <= config.adaptiveAbsFactor * decision.absThreshold else { return [] }
        return patterns.map { x in
            TemplateSample(prototype: proto, source: .adaptive, createdAt: now, features: x,
                           featureVersion: FeatureExtractor.version)
        }
    }

    /// Keeps at most `cap` adaptive samples per prototype, dropping the oldest (by
    /// `createdAt`, then list position). Enrollment and accessory samples are never
    /// dropped; the order of the kept samples is preserved.
    public static func applyAdaptiveCap(_ samples: [TemplateSample], cap: Int) -> [TemplateSample] {
        let limit = max(cap, 0)
        var groups: [PrototypeKey: [Int]] = [:]
        var groupOrder: [PrototypeKey] = []
        for (i, s) in samples.enumerated() where s.source == .adaptive {
            if groups[s.prototype] == nil { groupOrder.append(s.prototype) }
            groups[s.prototype, default: []].append(i)
        }
        var drop = Set<Int>()
        for key in groupOrder {
            guard let indices = groups[key], indices.count > limit else { continue }
            let ordered = indices.sorted { a, b in
                let da = samples[a].createdAt
                let db = samples[b].createdAt
                if da != db { return da < db }
                return a < b
            }
            for i in ordered.prefix(indices.count - limit) { drop.insert(i) }
        }
        var kept: [TemplateSample] = []
        for (i, s) in samples.enumerated() where !drop.contains(i) {
            kept.append(s)
        }
        return kept
    }
}
