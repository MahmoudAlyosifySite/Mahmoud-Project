import Foundation
@testable import Auditor
import AuditorCore

/// Deterministic synthetic recordings: damped bursts at the challenge events (the
/// template shape of SPEC 5.6) carried by a 37 Hz vibration, plus gravity and noise.
/// Each (person, hand) gets its own gains, axis mix, damping and delay.
enum SyntheticHands {
    struct Profile {
        var accelScale: Double
        var gyroScale: Double
        var delay: Double
        var decay: Double
        var tilt: Double
    }

    static func profile(person: Int, hand: Hand) -> Profile {
        let p = Double(person)
        let h = hand == .right ? 0.0 : 1.0
        return Profile(accelScale: 0.02 * (1.0 + 1.2 * p + 0.25 * h),
                       gyroScale: 0.05 * (1.0 + 0.3 * p + 0.6 * h),
                       delay: 0.04 + 0.02 * h,
                       decay: 0.03 + 0.008 * p,
                       tilt: 0.3 + 0.7 * h + 0.2 * p)
    }

    static func amplitude(_ spec: ChallengeSpec, _ tau: Double, decay: Double) -> Double {
        var s = 0.0
        for event in spec.events {
            switch event {
            case .tap(let tap):
                if tau >= tap.time { s += tap.intensity * exp(-(tau - tap.time) / decay) }
            case .ramp(let ramp):
                let t = ramp.time
                let d = ramp.duration
                if t <= tau && tau < t + d {
                    s += ramp.intensity(at: tau - t)
                } else if tau >= t + d {
                    s += ramp.intensity(at: d) * exp(-(tau - t - d) / decay)
                }
            }
        }
        return s
    }

    /// One recording. `onTable` puts the phone flat (gravity along z); `silent` plays no
    /// response at all (fails the quality gate).
    static func recording(seed: UInt64, profile: Profile, purpose: RecordingPurpose = .check,
                          label: RecordingLabel? = nil, onTable: Bool = false, silent: Bool = false,
                          noise: Double = 2e-4) -> Recording {
        let spec = ChallengeGenerator.generate(seed: seed)
        let hapticStart = 500.0
        let start = hapticStart - 0.45
        let count = Int((spec.duration + 0.9) / 0.01)
        var rng = Xoshiro256StarStar(seed: seed ^ 0xABCDEF)
        var accel = SensorStream()
        var gyro = SensorStream()
        var motion = MotionStream()
        let gravity: (Double, Double, Double) = onTable ? (0.0, 0.0, -1.0) : (0.0, -0.5, -0.866)
        let scale = silent ? 0.0 : profile.accelScale
        for k in 0..<count {
            let s = start + 0.002 + Double(k) * 0.01
            let a = amplitude(spec, s - hapticStart - profile.delay, decay: profile.decay)
            let phase = 2.0 * Double.pi * 37.0 * (s - hapticStart)
            let v = scale * a
            accel.t.append(s)
            gyro.t.append(s)
            motion.t.append(s)
            accel.x.append(gravity.0 + v * sin(phase) * cos(profile.tilt) + noise * rng.uniform(-1, 1))
            accel.y.append(gravity.1 + v * cos(phase) + noise * rng.uniform(-1, 1))
            accel.z.append(gravity.2 + v * sin(phase) * sin(profile.tilt) + noise * rng.uniform(-1, 1))
            gyro.x.append(profile.gyroScale * v * cos(phase) + noise * rng.uniform(-1, 1))
            gyro.y.append(profile.gyroScale * v * sin(phase) * cos(profile.tilt) + noise * rng.uniform(-1, 1))
            gyro.z.append(profile.gyroScale * v * sin(phase) * sin(profile.tilt) * 0.5 + noise * rng.uniform(-1, 1))
            motion.gravityX.append(gravity.0)
            motion.gravityY.append(gravity.1)
            motion.gravityZ.append(gravity.2)
            motion.userAccelX.append(0.0)
            motion.userAccelY.append(0.0)
            motion.userAccelZ.append(0.0)
            motion.rotationX.append(0.0)
            motion.rotationY.append(0.0)
            motion.rotationZ.append(0.0)
            motion.attitudeW.append(1.0)
            motion.attitudeX.append(0.0)
            motion.attitudeY.append(0.0)
            motion.attitudeZ.append(0.0)
        }
        return Recording(createdAt: Date(timeIntervalSince1970: 1_795_514_400 + Double(seed % 100_000)),
                         purpose: purpose, label: label, challenge: spec, hapticStart: hapticStart,
                         accel: accel, gyro: gyro, motion: motion,
                         device: DeviceInfo(model: "Synthetic1,1", system: "iOS 26.0", app: "test"))
    }
}

/// A `PatternRecording` that returns synthetic recordings instantly.
@MainActor
final class FakeRunner: PatternRecording {
    var canRecord: Bool { true }
    /// Who holds the phone for unlabeled recordings (checks).
    var holder: (person: Int, hand: Hand) = (0, .right)
    /// personID → synthetic person index, for labeled recordings.
    var personIndex: [String: Int] = [:]
    /// Number of upcoming recordings that carry no vibration response.
    var silentCount = 0
    private var seed: UInt64 = 100
    private(set) var recorded = 0

    func record(purpose: RecordingPurpose, label: RecordingLabel?) async throws -> Recording {
        seed += 1
        recorded += 1
        var person = holder.person
        var hand = holder.hand
        if let label = label {
            person = personIndex[label.personID] ?? 0
            hand = label.hand
        }
        let silent = silentCount > 0
        if silent { silentCount -= 1 }
        return SyntheticHands.recording(seed: seed, profile: SyntheticHands.profile(person: person, hand: hand),
                                        purpose: purpose, label: label, onTable: purpose == .calibration,
                                        silent: silent)
    }
}

enum TestSupport {
    static func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("AuditorTests-\(UUID().uuidString)",
                                                                       isDirectory: true)
    }
}
