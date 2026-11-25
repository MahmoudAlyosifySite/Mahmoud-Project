import Foundation
import XCTest
import AuditorCore

/// Error with a readable message (XCTest prints it when a throwing test fails).
struct FixtureError: Error, CustomStringConvertible {
    let message: String
    init(_ message: String) { self.message = message }
    var description: String { return message }
}

/// Fixture files written by `research/scripts/make_fixtures.py` (SPEC 11), bundled
/// with `.copy("Fixtures")`. A missing or unreadable file is a test failure, never a skip.
enum Fixture {
    static func url(_ name: String) -> URL? {
        if let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures") {
            return url
        }
        if let base = Bundle.module.resourceURL {
            let candidate = base.appendingPathComponent("Fixtures").appendingPathComponent(name + ".json")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    static func data(_ name: String, file: StaticString = #filePath, line: UInt = #line) -> Data? {
        guard let url = url(name) else {
            XCTFail("Fixture \(name).json is missing from the test bundle "
                    + "(expected in Tests/AuditorCoreTests/Fixtures; run research/scripts/make_fixtures.py)",
                    file: file, line: line)
            return nil
        }
        do {
            return try Data(contentsOf: url)
        } catch {
            XCTFail("Fixture \(name).json could not be read: \(error)", file: file, line: line)
            return nil
        }
    }

    /// Decodes a fixture into typed structs (dates in ISO-8601, as everywhere in the app).
    static func decode<T: Decodable>(_ type: T.Type, _ name: String,
                                     file: StaticString = #filePath, line: UInt = #line) -> T? {
        guard let data = data(name, file: file, line: line) else { return nil }
        do {
            return try RecordingCoding.decoder().decode(T.self, from: data)
        } catch {
            XCTFail("Fixture \(name).json does not match the expected schema: \(describe(error))",
                    file: file, line: line)
            return nil
        }
    }

    /// Decodes a fixture as a generic JSON tree.
    static func json(_ name: String, file: StaticString = #filePath, line: UInt = #line) -> JSON? {
        return decode(JSON.self, name, file: file, line: line)
    }

    static func describe(_ error: Error) -> String {
        guard let decoding = error as? DecodingError else { return String(describing: error) }
        func path(_ keys: [CodingKey]) -> String {
            return keys.map { (key: CodingKey) -> String in
                if let index = key.intValue { return "[\(index)]" }
                return "." + key.stringValue
            }.joined()
        }
        switch decoding {
        case .keyNotFound(let key, let ctx):
            return "missing key '\(key.stringValue)' at \(path(ctx.codingPath))"
        case .typeMismatch(let type, let ctx):
            return "type mismatch (\(type)) at \(path(ctx.codingPath)): \(ctx.debugDescription)"
        case .valueNotFound(let type, let ctx):
            return "null where \(type) was expected at \(path(ctx.codingPath))"
        case .dataCorrupted(let ctx):
            return "corrupted data at \(path(ctx.codingPath)): \(ctx.debugDescription)"
        @unknown default:
            return String(describing: error)
        }
    }
}

/// Minimal JSON tree, read without platform-dependent `NSNumber` bridging.
enum JSON: Decodable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSON])
    case object([String: JSON])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let d = try? c.decode(Double.self) {
            self = .number(d)
        } else if let s = try? c.decode(String.self) {
            self = .string(s)
        } else if let a = try? c.decode([JSON].self) {
            self = .array(a)
        } else if let o = try? c.decode([String: JSON].self) {
            self = .object(o)
        } else {
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "unsupported JSON value")
        }
    }

    var kind: String {
        switch self {
        case .null: return "null"
        case .bool: return "bool"
        case .number: return "number"
        case .string: return "string"
        case .array: return "array"
        case .object: return "object"
        }
    }

    func at(_ key: String) throws -> JSON {
        guard case .object(let o) = self else { throw FixtureError("expected an object holding '\(key)', found \(kind)") }
        guard let v = o[key] else { throw FixtureError("missing key '\(key)' (keys: \(o.keys.sorted().joined(separator: ", ")))") }
        return v
    }

    func has(_ key: String) -> Bool {
        if case .object(let o) = self { return o[key] != nil }
        return false
    }

    func array() throws -> [JSON] {
        guard case .array(let a) = self else { throw FixtureError("expected an array, found \(kind)") }
        return a
    }

    /// Numbers; `null` reads as +infinity (SPEC 11).
    func double() throws -> Double {
        switch self {
        case .number(let d): return d
        case .null: return Double.infinity
        default: throw FixtureError("expected a number, found \(kind)")
        }
    }

    func int() throws -> Int {
        let d = try double()
        guard d.isFinite, d == d.rounded(), abs(d) < 9.0e15 else { throw FixtureError("expected an integer, found \(d)") }
        return Int(d)
    }

    func string() throws -> String {
        guard case .string(let s) = self else { throw FixtureError("expected a string, found \(kind)") }
        return s
    }

    func doubles() throws -> [Double] {
        return try array().map { try $0.double() }
    }

    func ints() throws -> [Int] {
        return try array().map { try $0.int() }
    }

    /// Parses a `0x…` hex string (SPEC 2.2).
    func hex() throws -> UInt64 {
        let s = try string()
        guard let v = SeedFormat.parse(s) else { throw FixtureError("expected a 0x-prefixed hex string, found '\(s)'") }
        return v
    }
}

/// `|actual − expected| ≤ rel · max(1, |expected|)` (SPEC 11); equal infinities match.
@discardableResult
func assertClose(_ actual: Double, _ expected: Double, rel: Double, _ what: @autoclosure () -> String,
                 file: StaticString = #filePath, line: UInt = #line) -> Bool {
    if actual == expected { return true }
    if actual.isNaN || expected.isNaN || actual.isInfinite || expected.isInfinite {
        XCTFail("\(what()): got \(actual), expected \(expected)", file: file, line: line)
        return false
    }
    let tolerance = rel * max(1.0, abs(expected))
    let delta = abs(actual - expected)
    if delta > tolerance {
        XCTFail("\(what()): got \(actual), expected \(expected) (|Δ| = \(delta) > \(tolerance))", file: file, line: line)
        return false
    }
    return true
}

/// Element-wise `assertClose`; reports the length mismatch or at most 5 differing elements.
@discardableResult
func assertClose(_ actual: [Double], _ expected: [Double], rel: Double, _ what: @autoclosure () -> String,
                 file: StaticString = #filePath, line: UInt = #line) -> Bool {
    guard actual.count == expected.count else {
        XCTFail("\(what()): length \(actual.count), expected \(expected.count)", file: file, line: line)
        return false
    }
    var failures = 0
    for i in 0..<actual.count {
        if !assertClose(actual[i], expected[i], rel: rel, "\(what())[\(i)]", file: file, line: line) {
            failures += 1
            if failures >= 5 { return false }
        }
    }
    return failures == 0
}
