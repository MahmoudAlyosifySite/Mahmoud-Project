import Foundation

// MARK: - Streams (SPEC 4)

/// One accelerometer or gyroscope stream: timestamps (seconds, `systemUptime` clock)
/// and three axes. Arrays have equal length in a valid recording.
public struct SensorStream: Codable, Equatable {
    public var t: [Double]
    public var x: [Double]
    public var y: [Double]
    public var z: [Double]

    public init(t: [Double] = [], x: [Double] = [], y: [Double] = [], z: [Double] = []) {
        self.t = t
        self.x = x
        self.y = y
        self.z = z
    }

    public static let empty = SensorStream()

    private enum CodingKeys: String, CodingKey {
        case t, x, y, z
    }

    /// Missing arrays decode as empty.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        t = try c.decodeIfPresent([Double].self, forKey: .t) ?? []
        x = try c.decodeIfPresent([Double].self, forKey: .x) ?? []
        y = try c.decodeIfPresent([Double].self, forKey: .y) ?? []
        z = try c.decodeIfPresent([Double].self, forKey: .z) ?? []
    }
}

/// The Core Motion device-motion stream (gravity, user acceleration, rotation rate, attitude).
public struct MotionStream: Codable, Equatable {
    public var t: [Double]
    public var gravityX: [Double]
    public var gravityY: [Double]
    public var gravityZ: [Double]
    public var userAccelX: [Double]
    public var userAccelY: [Double]
    public var userAccelZ: [Double]
    public var rotationX: [Double]
    public var rotationY: [Double]
    public var rotationZ: [Double]
    public var attitudeW: [Double]
    public var attitudeX: [Double]
    public var attitudeY: [Double]
    public var attitudeZ: [Double]

    public init(
        t: [Double] = [],
        gravityX: [Double] = [], gravityY: [Double] = [], gravityZ: [Double] = [],
        userAccelX: [Double] = [], userAccelY: [Double] = [], userAccelZ: [Double] = [],
        rotationX: [Double] = [], rotationY: [Double] = [], rotationZ: [Double] = [],
        attitudeW: [Double] = [], attitudeX: [Double] = [], attitudeY: [Double] = [], attitudeZ: [Double] = []
    ) {
        self.t = t
        self.gravityX = gravityX
        self.gravityY = gravityY
        self.gravityZ = gravityZ
        self.userAccelX = userAccelX
        self.userAccelY = userAccelY
        self.userAccelZ = userAccelZ
        self.rotationX = rotationX
        self.rotationY = rotationY
        self.rotationZ = rotationZ
        self.attitudeW = attitudeW
        self.attitudeX = attitudeX
        self.attitudeY = attitudeY
        self.attitudeZ = attitudeZ
    }

    public static let empty = MotionStream()

    private enum CodingKeys: String, CodingKey {
        case t, gravityX, gravityY, gravityZ, userAccelX, userAccelY, userAccelZ
        case rotationX, rotationY, rotationZ, attitudeW, attitudeX, attitudeY, attitudeZ
    }

    /// Missing arrays decode as empty.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func arr(_ key: CodingKeys) throws -> [Double] {
            return try c.decodeIfPresent([Double].self, forKey: key) ?? []
        }
        t = try arr(.t)
        gravityX = try arr(.gravityX)
        gravityY = try arr(.gravityY)
        gravityZ = try arr(.gravityZ)
        userAccelX = try arr(.userAccelX)
        userAccelY = try arr(.userAccelY)
        userAccelZ = try arr(.userAccelZ)
        rotationX = try arr(.rotationX)
        rotationY = try arr(.rotationY)
        rotationZ = try arr(.rotationZ)
        attitudeW = try arr(.attitudeW)
        attitudeX = try arr(.attitudeX)
        attitudeY = try arr(.attitudeY)
        attitudeZ = try arr(.attitudeZ)
    }
}

// MARK: - Labels (SPEC 4)

public enum Hand: String, Codable, CaseIterable {
    case right, left
}

public enum Accessory: String, Codable, CaseIterable {
    case none, watch, ring, bracelet, other
}

public enum Posture: String, Codable, CaseIterable {
    case sitting, standing, freshGrip, unspecified
}

public enum RecordingPurpose: String, Codable {
    case calibration, enrollment, accessory, check, test, signalCheck
}

extension Hand {
    /// Sort and class order of SPEC 9.1/9.3: right (0) < left (1).
    var order: Int {
        switch self {
        case .right: return 0
        case .left: return 1
        }
    }
}

extension Accessory {
    /// Sort order of SPEC 9.1: none < watch < ring < bracelet < other.
    var order: Int {
        switch self {
        case .none: return 0
        case .watch: return 1
        case .ring: return 2
        case .bracelet: return 3
        case .other: return 4
        }
    }
}

/// Who held the phone, with which hand and accessory (enrollment and test recordings).
public struct RecordingLabel: Codable, Equatable {
    public var personID: String
    public var personName: String
    public var hand: Hand
    public var accessory: Accessory
    public var posture: Posture
    public var sessionID: String

    public init(personID: String, personName: String, hand: Hand, accessory: Accessory = Accessory.none,
                posture: Posture = .unspecified, sessionID: String) {
        self.personID = personID
        self.personName = personName
        self.hand = hand
        self.accessory = accessory
        self.posture = posture
        self.sessionID = sessionID
    }
}

public struct DeviceInfo: Codable, Equatable {
    public var model: String
    public var system: String
    public var app: String

    public init(model: String, system: String, app: String) {
        self.model = model
        self.system = system
        self.app = app
    }
}

// MARK: - Recording (SPEC 4)

/// Raw data of one played pattern; also the export format (one JSON object per line).
public struct Recording: Codable, Equatable {
    public static let currentSchema = "7oda-auditor.recording.v1"

    public var schema: String
    public var id: UUID
    public var createdAt: Date
    public var purpose: RecordingPurpose
    public var label: RecordingLabel?
    public var challenge: ChallengeSpec
    /// Seconds on the sensor clock (`systemUptime`) at which the haptic player started.
    public var hapticStart: Double
    public var accel: SensorStream
    public var gyro: SensorStream
    public var motion: MotionStream
    public var device: DeviceInfo?

    public init(schema: String = Recording.currentSchema, id: UUID = UUID(), createdAt: Date = Date(),
                purpose: RecordingPurpose, label: RecordingLabel? = nil, challenge: ChallengeSpec,
                hapticStart: Double, accel: SensorStream, gyro: SensorStream,
                motion: MotionStream = .empty, device: DeviceInfo? = nil) {
        self.schema = schema
        self.id = id
        self.createdAt = createdAt
        self.purpose = purpose
        self.label = label
        self.challenge = challenge
        self.hapticStart = hapticStart
        self.accel = accel
        self.gyro = gyro
        self.motion = motion
        self.device = device
    }

    private enum CodingKeys: String, CodingKey {
        case schema, id, createdAt, purpose, label, challenge, hapticStart, accel, gyro, motion, device
    }

    /// `label` and `device` may be absent or null; missing streams decode as empty.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schema = try c.decodeIfPresent(String.self, forKey: .schema) ?? Recording.currentSchema
        id = try c.decode(UUID.self, forKey: .id)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        purpose = try c.decode(RecordingPurpose.self, forKey: .purpose)
        label = try c.decodeIfPresent(RecordingLabel.self, forKey: .label)
        challenge = try c.decode(ChallengeSpec.self, forKey: .challenge)
        hapticStart = try c.decode(Double.self, forKey: .hapticStart)
        accel = try c.decodeIfPresent(SensorStream.self, forKey: .accel) ?? .empty
        gyro = try c.decodeIfPresent(SensorStream.self, forKey: .gyro) ?? .empty
        motion = try c.decodeIfPresent(MotionStream.self, forKey: .motion) ?? .empty
        device = try c.decodeIfPresent(DeviceInfo.self, forKey: .device)
    }
}

// MARK: - Coding (SPEC 4)

/// JSON coding shared by recordings, calibrations, samples and settings:
/// ISO-8601 dates in UTC without fractional seconds, sorted keys, no pretty printing.
/// Non-finite doubles (never produced by valid data) round-trip as the strings
/// "Infinity", "-Infinity" and "NaN" instead of failing the whole file.
public enum RecordingCoding {
    public static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
        return encoder
    }

    /// Reads ISO-8601 dates with or without fractional seconds.
    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { dateDecoder in
            let container = try dateDecoder.singleValueContainer()
            let text = try container.decode(String.self)
            if let date = RecordingCoding.parseDate(text) { return date }
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "Expected an ISO-8601 date, found '\(text)'")
        }
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
        return decoder
    }

    /// One compact JSON object per line, each line terminated by `\n`.
    public static func jsonLines(_ recordings: [Recording]) throws -> Data {
        let encoder = self.encoder()
        var out = Data()
        for recording in recordings {
            out.append(try encoder.encode(recording))
            out.append(0x0A)
        }
        return out
    }

    /// Parses JSON Lines; blank lines, `\r\n` line ends and a leading UTF-8 BOM are accepted.
    public static func parseJSONLines(_ data: Data) throws -> [Recording] {
        let decoder = self.decoder()
        var bytes = [UInt8](data)
        if bytes.count >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF {
            bytes.removeFirst(3)
        }
        var recordings: [Recording] = []
        var start = 0
        while start < bytes.count {
            var end = start
            while end < bytes.count && bytes[end] != 0x0A { end += 1 }
            var lo = start
            var hi = end
            while lo < hi && isSpace(bytes[lo]) { lo += 1 }
            while hi > lo && isSpace(bytes[hi - 1]) { hi -= 1 }
            if lo < hi {
                recordings.append(try decoder.decode(Recording.self, from: Data(bytes[lo..<hi])))
            }
            start = end + 1
        }
        return recordings
    }

    private static func isSpace(_ b: UInt8) -> Bool {
        return b == 0x20 || b == 0x09 || b == 0x0D || b == 0x0A
    }

    /// ISO-8601 (`2026-11-24T10:00:00Z`), optionally with fractional seconds.
    static func parseDate(_ text: String) -> Date? {
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        if let date = plain.date(from: text) { return date }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fractional.date(from: text)
    }
}
