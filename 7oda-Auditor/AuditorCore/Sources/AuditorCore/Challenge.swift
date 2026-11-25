import Foundation

// MARK: - Events (SPEC 3.3, 3.4)

/// One control point of a ramp's intensity curve; `time` is seconds from the ramp start.
public struct CurvePoint: Codable, Equatable {
    public var time: Double
    public var intensity: Double

    public init(time: Double, intensity: Double) {
        self.time = time
        self.intensity = intensity
    }
}

/// A transient haptic event (SPEC 3.2).
public struct TapEvent: Codable, Equatable {
    public var time: Double
    public var intensity: Double
    public var sharpness: Double

    public init(time: Double, intensity: Double, sharpness: Double) {
        self.time = time
        self.intensity = intensity
        self.sharpness = sharpness
    }
}

/// A continuous haptic event whose intensity follows a piecewise-linear curve (SPEC 3.2, 3.3).
public struct RampEvent: Codable, Equatable {
    public var time: Double
    public var duration: Double
    public var sharpness: Double
    public var curve: [CurvePoint]

    public init(time: Double, duration: Double, sharpness: Double, curve: [CurvePoint]) {
        self.time = time
        self.duration = duration
        self.sharpness = sharpness
        self.curve = curve
    }

    /// Commanded intensity `tau` seconds after the ramp start (SPEC 3.3).
    ///
    /// `k` = largest index with `curve[k].time <= tau`; none → first value;
    /// `k` last → last value; otherwise `c_k + w * (c_{k+1} - c_k)` with
    /// `w = (tau - t_k) / (t_{k+1} - t_k)`.
    public func intensity(at tau: Double) -> Double {
        guard let first = curve.first, let last = curve.last else { return 0 }
        var k = -1
        for (index, point) in curve.enumerated() where point.time <= tau {
            k = index
        }
        if k < 0 { return first.intensity }
        if k == curve.count - 1 { return last.intensity }
        let p0 = curve[k]
        let p1 = curve[k + 1]
        let w = (tau - p0.time) / (p1.time - p0.time)
        return p0.intensity + w * (p1.intensity - p0.intensity)
    }
}

/// A challenge event. Encoded as a JSON object tagged by `"kind"` (`"tap"` or `"ramp"`), SPEC 3.4.
public enum ChallengeEvent: Codable, Equatable {
    case tap(TapEvent)
    case ramp(RampEvent)

    /// Onset time in seconds from the pattern start.
    public var time: Double {
        switch self {
        case .tap(let event): return event.time
        case .ramp(let event): return event.time
        }
    }

    private enum CodingKeys: String, CodingKey {
        case kind, time, intensity, sharpness, duration, curve
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .kind)
        switch kind {
        case "tap":
            let time = try container.decode(Double.self, forKey: .time)
            let intensity = try container.decode(Double.self, forKey: .intensity)
            let sharpness = try container.decode(Double.self, forKey: .sharpness)
            self = .tap(TapEvent(time: time, intensity: intensity, sharpness: sharpness))
        case "ramp":
            let time = try container.decode(Double.self, forKey: .time)
            let duration = try container.decode(Double.self, forKey: .duration)
            let sharpness = try container.decode(Double.self, forKey: .sharpness)
            let curve = try container.decode([CurvePoint].self, forKey: .curve)
            self = .ramp(RampEvent(time: time, duration: duration, sharpness: sharpness, curve: curve))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .kind, in: container, debugDescription: "Unknown challenge event kind '\(kind)'")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .tap(let event):
            try container.encode("tap", forKey: .kind)
            try container.encode(event.time, forKey: .time)
            try container.encode(event.intensity, forKey: .intensity)
            try container.encode(event.sharpness, forKey: .sharpness)
        case .ramp(let event):
            try container.encode("ramp", forKey: .kind)
            try container.encode(event.time, forKey: .time)
            try container.encode(event.duration, forKey: .duration)
            try container.encode(event.sharpness, forKey: .sharpness)
            try container.encode(event.curve, forKey: .curve)
        }
    }
}

// MARK: - ChallengeSpec (SPEC 3.4)

/// A complete random haptic pattern. `seed` is serialized as the hex string of SPEC 2.2.
public struct ChallengeSpec: Codable, Equatable {
    public var version: Int
    public var seed: UInt64
    /// Pattern duration in seconds (the end of the last event plus its trailing gap).
    public var duration: Double
    public var events: [ChallengeEvent]

    public init(version: Int = 1, seed: UInt64, duration: Double, events: [ChallengeEvent]) {
        self.version = version
        self.seed = seed
        self.duration = duration
        self.events = events
    }

    private enum CodingKeys: String, CodingKey {
        case version, seed, duration, events
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decodeIfPresent(Int.self, forKey: .version) ?? 1
        if let text = try? container.decode(String.self, forKey: .seed) {
            guard let parsed = SeedFormat.parse(text) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .seed, in: container, debugDescription: "Invalid seed '\(text)'")
            }
            seed = parsed
        } else {
            seed = try container.decode(UInt64.self, forKey: .seed)
        }
        duration = try container.decode(Double.self, forKey: .duration)
        events = try container.decode([ChallengeEvent].self, forKey: .events)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(version, forKey: .version)
        try container.encode(SeedFormat.string(seed), forKey: .seed)
        try container.encode(duration, forKey: .duration)
        try container.encode(events, forKey: .events)
    }

    /// Tap events in list (time) order.
    public var taps: [TapEvent] {
        return events.compactMap { event -> TapEvent? in
            if case .tap(let tap) = event { return tap }
            return nil
        }
    }

    /// Ramp events in list (time) order.
    public var ramps: [RampEvent] {
        return events.compactMap { event -> RampEvent? in
            if case .ramp(let ramp) = event { return ramp }
            return nil
        }
    }
}

// MARK: - Config (SPEC 3.1)

/// Parameter ranges of SPEC 3.1. Uniform draws use `uniform(lowerBound, upperBound)`.
public struct ChallengeConfig: Equatable {
    public var leadSilence: ClosedRange<Double>
    public var tapCount: ClosedRange<Int>
    public var tapIntensity: ClosedRange<Double>
    public var tapSharpness: ClosedRange<Double>
    /// Onset to next onset.
    public var tapGap: ClosedRange<Double>
    /// Fixed at 2: the draw order of SPEC 3.2 is defined for exactly two ramps.
    public let rampCount: Int
    public var rampDuration: ClosedRange<Double>
    public var rampSharpness: ClosedRange<Double>
    public var rampLow: ClosedRange<Double>
    public var rampMid: ClosedRange<Double>
    public var rampHigh: ClosedRange<Double>
    /// End of ramp to next onset.
    public var rampGap: ClosedRange<Double>

    public init(
        leadSilence: ClosedRange<Double> = 0.05...0.10,
        tapCount: ClosedRange<Int> = 6...8,
        tapIntensity: ClosedRange<Double> = 0.4...1.0,
        tapSharpness: ClosedRange<Double> = 0.0...1.0,
        tapGap: ClosedRange<Double> = 0.08...0.15,
        rampDuration: ClosedRange<Double> = 0.15...0.25,
        rampSharpness: ClosedRange<Double> = 0.2...0.8,
        rampLow: ClosedRange<Double> = 0.25...0.5,
        rampMid: ClosedRange<Double> = 0.25...1.0,
        rampHigh: ClosedRange<Double> = 0.75...1.0,
        rampGap: ClosedRange<Double> = 0.08...0.12
    ) {
        self.leadSilence = leadSilence
        self.tapCount = tapCount
        self.tapIntensity = tapIntensity
        self.tapSharpness = tapSharpness
        self.tapGap = tapGap
        self.rampCount = 2
        self.rampDuration = rampDuration
        self.rampSharpness = rampSharpness
        self.rampLow = rampLow
        self.rampMid = rampMid
        self.rampHigh = rampHigh
        self.rampGap = rampGap
    }

    public static let `default` = ChallengeConfig()
}

// MARK: - Generator (SPEC 3.2)

/// Turns a seed into a pattern. The draw order is normative (SPEC 3.2) so that the
/// Python reference reproduces every pattern bit for bit.
public enum ChallengeGenerator {
    public static func generate(seed: UInt64, config: ChallengeConfig = .default) -> ChallengeSpec {
        var rng = Xoshiro256StarStar(seed: seed)
        let lead = draw(&rng, config.leadSilence)
        // At least 2 taps are needed to place the two ramps in distinct slots.
        let n = max(2, rng.int(config.tapCount.lowerBound, config.tapCount.upperBound))

        let s1 = 1 + min(floorIndex(rng.uniform() * Double(n)), n - 1)
        var s2 = 1 + min(floorIndex(rng.uniform() * Double(n - 1)), n - 2)
        if s2 >= s1 { s2 += 1 }
        let slots = [s1, s2].sorted()

        var tapIntensities: [Double] = []
        var tapSharpnesses: [Double] = []
        var tapGaps: [Double] = []
        for _ in 0..<n {
            let intensity = draw(&rng, config.tapIntensity)
            let sharpness = draw(&rng, config.tapSharpness)
            let gap = draw(&rng, config.tapGap)
            tapIntensities.append(intensity)
            tapSharpnesses.append(sharpness)
            tapGaps.append(gap)
        }

        struct RampDraw {
            var duration: Double
            var sharpness: Double
            var ascending: Bool
            var low: Double
            var mid: Double
            var high: Double
            var gap: Double
        }
        var ramps: [RampDraw] = []
        for _ in 0..<2 {
            let duration = draw(&rng, config.rampDuration)
            let sharpness = draw(&rng, config.rampSharpness)
            let ascending = rng.uniform() < 0.5
            let low = draw(&rng, config.rampLow)
            let mid = draw(&rng, config.rampMid)
            let high = draw(&rng, config.rampHigh)
            let gap = draw(&rng, config.rampGap)
            ramps.append(RampDraw(duration: duration, sharpness: sharpness, ascending: ascending,
                                  low: low, mid: mid, high: high, gap: gap))
        }

        var t = lead
        var events: [ChallengeEvent] = []
        for i in 0..<n {
            events.append(.tap(TapEvent(time: t, intensity: tapIntensities[i], sharpness: tapSharpnesses[i])))
            t += tapGaps[i]
            for j in 0..<2 where slots[j] == i + 1 {
                let ramp = ramps[j]
                let d = ramp.duration
                let values = ramp.ascending ? [ramp.low, ramp.mid, ramp.high] : [ramp.high, ramp.mid, ramp.low]
                let curve = [
                    CurvePoint(time: 0.0, intensity: values[0]),
                    CurvePoint(time: d / 2.0, intensity: values[1]),
                    CurvePoint(time: d, intensity: values[2]),
                ]
                events.append(.ramp(RampEvent(time: t, duration: d, sharpness: ramp.sharpness, curve: curve)))
                t += d + ramp.gap
            }
        }
        return ChallengeSpec(version: 1, seed: seed, duration: t, events: events)
    }

    private static func draw(_ rng: inout Xoshiro256StarStar, _ range: ClosedRange<Double>) -> Double {
        return rng.uniform(range.lowerBound, range.upperBound)
    }

    /// `Int(floor(x))` for the non-negative values of SPEC 3.2.
    private static func floorIndex(_ x: Double) -> Int {
        let f = x.rounded(.down)
        guard f.isFinite, f > -9.0e18, f < 9.0e18 else { return 0 }
        return Int(f)
    }
}
