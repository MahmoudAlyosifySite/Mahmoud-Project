import Foundation
import Security
import AuditorCore

enum RecordingError: Error, LocalizedError {
    case sensorsUnavailable
    case hapticsUnavailable
    case randomFailed

    var errorDescription: String? {
        switch self {
        case .sensorsUnavailable:
            return "Motion sensors are not available here. Recording needs a real iPhone (the Simulator has no sensors)."
        case .hapticsUnavailable:
            return "This device cannot play Core Haptics patterns. Recording needs an iPhone with a Taptic Engine."
        case .randomFailed:
            return "Could not create a random pattern."
        }
    }
}

/// What this device can do, and how it identifies itself in recordings.
enum DeviceSupport {
    static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    /// Hardware identifier such as "iPhone16,1" (the Simulator reports the simulated model).
    static var modelIdentifier: String {
        if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return simulated
        }
        var info = utsname()
        uname(&info)
        let mirror = Mirror(reflecting: info.machine)
        let bytes = mirror.children.compactMap { $0.value as? Int8 }.filter { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    static var systemDescription: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "iOS \(v.majorVersion).\(v.minorVersion)" + (v.patchVersion > 0 ? ".\(v.patchVersion)" : "")
    }

    static var appVersion: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "0"
        let build = info?["CFBundleVersion"] as? String ?? "0"
        return "\(short) (\(build))"
    }

    static var deviceInfo: DeviceInfo {
        DeviceInfo(model: modelIdentifier, system: systemDescription, app: appVersion)
    }

    /// 8 random bytes from the system CSPRNG.
    static func randomSeed() throws -> UInt64 {
        var seed: UInt64 = 0
        let status = withUnsafeMutableBytes(of: &seed) { buffer -> Int32 in
            guard let base = buffer.baseAddress else { return errSecParam }
            return SecRandomCopyBytes(kSecRandomDefault, 8, base)
        }
        guard status == errSecSuccess else { throw RecordingError.randomFailed }
        return seed
    }
}

/// Anything that can produce one recording (the real device runner, or a test double).
@MainActor
protocol PatternRecording: AnyObject {
    var canRecord: Bool { get }
    func record(purpose: RecordingPurpose, label: RecordingLabel?) async throws -> Recording
}

/// Generates a random challenge, plays it and cuts the matching sensor window (SPEC 12).
@MainActor
final class CheckRunner: PatternRecording {
    let recorder: MotionRecorder
    let haptics: HapticPlayer
    let signalConfig: SignalConfig

    init(recorder: MotionRecorder, haptics: HapticPlayer, signalConfig: SignalConfig = .default) {
        self.recorder = recorder
        self.haptics = haptics
        self.signalConfig = signalConfig
    }

    var canRecord: Bool { recorder.isAvailable && HapticPlayer.isSupported }

    /// Why recording is impossible here, or nil.
    var unavailableReason: String? {
        if !recorder.isAvailable { return RecordingError.sensorsUnavailable.errorDescription }
        if !HapticPlayer.isSupported { return RecordingError.hapticsUnavailable.errorDescription }
        return nil
    }

    func record(purpose: RecordingPurpose, label: RecordingLabel?) async throws -> Recording {
        guard recorder.isAvailable else { throw RecordingError.sensorsUnavailable }
        guard HapticPlayer.isSupported else { throw RecordingError.hapticsUnavailable }
        let startedHere = !recorder.isRunning
        if startedHere { recorder.start() }
        defer { if startedHere { recorder.stop() } }
        let warm = recorder.secondsRunning
        if warm < 0.5 {
            try await Task.sleep(nanoseconds: UInt64((0.5 - warm) * 1_000_000_000))
        }
        let seed = try DeviceSupport.randomSeed()
        let spec = ChallengeGenerator.generate(seed: seed)
        let start = try haptics.play(spec)
        let wait = spec.duration + signalConfig.postRoll + 0.1
        try await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
        let from = start - signalConfig.preRoll - 0.1
        let to = start + spec.duration + signalConfig.postRoll + 0.1
        let snap = recorder.snapshot(from: from, to: to)
        return Recording(purpose: purpose, label: label, challenge: spec, hapticStart: start,
                         accel: snap.accel, gyro: snap.gyro, motion: snap.motion,
                         device: DeviceSupport.deviceInfo)
    }
}
