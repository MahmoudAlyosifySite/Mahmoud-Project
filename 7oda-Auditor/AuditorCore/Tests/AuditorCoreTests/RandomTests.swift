import Foundation
import XCTest
import AuditorCore

final class RandomTests: XCTestCase {
    /// SPEC 11.1: every list drawn from a fresh generator; exact equality.
    func testRNGFixtureParity() throws {
        guard let root = Fixture.json("rng_vectors") else { return }
        let cases = try root.at("cases").array()
        XCTAssertFalse(cases.isEmpty, "rng_vectors.json has no cases")
        for c in cases {
            let seedText = try c.at("seed").string()
            let seed = try c.at("seed").hex()
            XCTAssertEqual(SeedFormat.string(seed), seedText, "seed formatting")

            var sm = SplitMix64(seed: seed)
            for (i, v) in try c.at("splitmix").array().enumerated() {
                XCTAssertEqual(sm.next(), try v.hex(), "splitmix[\(i)] for seed \(seedText)")
            }
            var x = Xoshiro256StarStar(seed: seed)
            for (i, v) in try c.at("xoshiro").array().enumerated() {
                XCTAssertEqual(x.next(), try v.hex(), "xoshiro[\(i)] for seed \(seedText)")
            }
            var u = Xoshiro256StarStar(seed: seed)
            for (i, v) in try c.at("uniform").doubles().enumerated() {
                XCTAssertEqual(u.uniform(), v, "uniform[\(i)] for seed \(seedText)")
            }
            for entry in try c.at("ints").array() {
                let lo = try entry.at("lo").int()
                let hi = try entry.at("hi").int()
                var g = Xoshiro256StarStar(seed: seed)
                for (i, v) in try entry.at("values").ints().enumerated() {
                    XCTAssertEqual(g.int(lo, hi), v, "int(\(lo), \(hi))[\(i)] for seed \(seedText)")
                }
            }
            if c.has("uniformRange") {
                for entry in try c.at("uniformRange").array() {
                    let a = try entry.at("a").double()
                    let b = try entry.at("b").double()
                    var g = Xoshiro256StarStar(seed: seed)
                    for (i, v) in try entry.at("values").doubles().enumerated() {
                        XCTAssertEqual(g.uniform(a, b), v, "uniform(\(a), \(b))[\(i)] for seed \(seedText)")
                    }
                }
            }
        }
    }

    func testSplitMixKnownValue() {
        // Reference value of SplitMix64 seeded with 0.
        var sm = SplitMix64(seed: 0)
        XCTAssertEqual(sm.next(), 0xE220_A839_7B1D_CDAF)
    }

    func testSeedFormat() {
        XCTAssertEqual(SeedFormat.string(1234), "0x00000000000004d2")
        XCTAssertEqual(SeedFormat.string(0), "0x0000000000000000")
        XCTAssertEqual(SeedFormat.string(UInt64.max), "0xffffffffffffffff")
        XCTAssertEqual(SeedFormat.parse("0x00000000000004d2"), 1234)
        XCTAssertEqual(SeedFormat.parse("0xFFFFFFFFFFFFFFFF"), UInt64.max)
        XCTAssertNil(SeedFormat.parse("1234"))
        XCTAssertNil(SeedFormat.parse("0x"))
        XCTAssertNil(SeedFormat.parse("0x1g"))
        XCTAssertNil(SeedFormat.parse("0x00000000000000001"))
        for seed: UInt64 in [0, 1, 42, 0x8000_0000_0000_0000, UInt64.max] {
            XCTAssertEqual(SeedFormat.parse(SeedFormat.string(seed)), seed)
        }
    }

    func testRangesAndDeterminism() {
        var a = Xoshiro256StarStar(seed: 7)
        var b = Xoshiro256StarStar(seed: 7)
        for _ in 0..<1000 {
            let u = a.uniform()
            XCTAssertTrue(u >= 0.0 && u < 1.0)
            XCTAssertEqual(u, b.uniform())
            let r = a.uniform(0.25, 0.5)
            XCTAssertTrue(r >= 0.25 && r <= 0.5)
            _ = b.uniform(0.25, 0.5)
            let i = a.int(6, 8)
            XCTAssertTrue((6...8).contains(i))
            XCTAssertEqual(i, b.int(6, 8))
        }
        var single = Xoshiro256StarStar(seed: 9)
        XCTAssertEqual(single.int(3, 3), 3)
    }

    func testRandomNumberGeneratorConformance() {
        var g = Xoshiro256StarStar(seed: 99)
        let v = Int.random(in: 0..<10, using: &g)
        XCTAssertTrue((0..<10).contains(v))
        let shuffled = [1, 2, 3, 4].shuffled(using: &g)
        XCTAssertEqual(shuffled.sorted(), [1, 2, 3, 4])
    }
}
