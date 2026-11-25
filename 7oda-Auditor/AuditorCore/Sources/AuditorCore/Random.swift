import Foundation

// MARK: - SPEC 2.1 SplitMix64

/// SplitMix64, used only to seed `Xoshiro256StarStar` (SPEC 2.1).
public struct SplitMix64 {
    /// Current generator state.
    public private(set) var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    /// Next 64-bit output (SPEC 2.1).
    public mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

// MARK: - SPEC 2.2 xoshiro256**

/// xoshiro256** seeded through SplitMix64 (SPEC 2.2). Every random choice in a
/// challenge comes from this generator, so a seed fully determines a pattern.
public struct Xoshiro256StarStar: RandomNumberGenerator {
    private var s0: UInt64
    private var s1: UInt64
    private var s2: UInt64
    private var s3: UInt64

    /// `s0...s3` are four consecutive outputs of `SplitMix64(seed)`.
    public init(seed: UInt64) {
        var sm = SplitMix64(seed: seed)
        let a = sm.next()
        let b = sm.next()
        let c = sm.next()
        let d = sm.next()
        s0 = a
        s1 = b
        s2 = c
        s3 = d
    }

    @inline(__always)
    private static func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 {
        return (x << k) | (x >> (64 &- k))
    }

    /// Next 64-bit output (SPEC 2.2).
    public mutating func next() -> UInt64 {
        let result = Xoshiro256StarStar.rotl(s1 &* 5, 7) &* 9
        let t = s1 << 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = Xoshiro256StarStar.rotl(s3, 45)
        return result
    }

    /// Uniform double in [0, 1): `Double(next() >> 11) * 2^-53`.
    public mutating func uniform() -> Double {
        return Double(next() >> 11) * 0x1.0p-53
    }

    /// `a + (b - a) * uniform()`.
    public mutating func uniform(_ a: Double, _ b: Double) -> Double {
        return a + (b - a) * uniform()
    }

    /// Integer in `lo...hi` (inclusive):
    /// `lo + min(Int(floor(uniform() * Double(hi - lo + 1))), hi - lo)`.
    public mutating func int(_ lo: Int, _ hi: Int) -> Int {
        let span = Double(hi - lo + 1)
        let scaled = (uniform() * span).rounded(.down)
        let offset: Int
        if scaled.isFinite && scaled < 9.0e18 && scaled > -9.0e18 {
            offset = Int(scaled)
        } else {
            offset = hi - lo
        }
        return lo + min(offset, hi - lo)
    }
}

// MARK: - Seed serialization (SPEC 2.2)

/// Seeds are written as lowercase hex with a `0x` prefix and exactly 16 digits,
/// e.g. `"0x00000000000004d2"`.
public enum SeedFormat {
    /// `"0x"` followed by 16 lowercase hex digits.
    public static func string(_ seed: UInt64) -> String {
        let hex = String(seed, radix: 16, uppercase: false)
        let zeros = String(repeating: "0", count: max(0, 16 - hex.count))
        return "0x" + zeros + hex
    }

    /// Parses `0x` followed by 1 to 16 hex digits (either case); nil otherwise.
    public static func parse(_ s: String) -> UInt64? {
        guard s.hasPrefix("0x") else { return nil }
        let digits = s.dropFirst(2)
        guard !digits.isEmpty, digits.count <= 16 else { return nil }
        for scalar in digits.unicodeScalars {
            switch scalar {
            case "0"..."9", "a"..."f", "A"..."F":
                continue
            default:
                return nil
            }
        }
        return UInt64(String(digits), radix: 16)
    }
}
