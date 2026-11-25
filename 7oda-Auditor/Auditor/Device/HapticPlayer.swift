import CoreHaptics
import Foundation
import AuditorCore

enum HapticError: Error, LocalizedError {
    case unsupported

    var errorDescription: String? {
        switch self {
        case .unsupported: return "This device has no Taptic Engine haptics (the Simulator has none)."
        }
    }
}

/// Plays a `ChallengeSpec` on the Taptic Engine (SPEC 3.5).
final class HapticPlayer {
    private var engine: CHHapticEngine?
    private var player: CHHapticAdvancedPatternPlayer?
    private let lock = NSRecursiveLock()

    static var isSupported: Bool {
        CHHapticEngine.capabilitiesForHardware().supportsHaptics
    }

    // MARK: Pattern construction (pure)

    /// Events and parameter curves of SPEC 3.5, in event order.
    static func components(for spec: ChallengeSpec) -> (events: [CHHapticEvent], curves: [CHHapticParameterCurve]) {
        var events: [CHHapticEvent] = []
        var curves: [CHHapticParameterCurve] = []
        for event in spec.events {
            switch event {
            case .tap(let tap):
                events.append(CHHapticEvent(
                    eventType: .hapticTransient,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: Float(tap.intensity)),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: Float(tap.sharpness)),
                    ],
                    relativeTime: tap.time))
            case .ramp(let ramp):
                let d = ramp.duration
                events.append(CHHapticEvent(
                    eventType: .hapticContinuous,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: Float(ramp.sharpness)),
                    ],
                    relativeTime: ramp.time,
                    duration: d))
                let points = [
                    CHHapticParameterCurve.ControlPoint(relativeTime: 0, value: Float(ramp.intensity(at: 0))),
                    CHHapticParameterCurve.ControlPoint(relativeTime: d / 2, value: Float(ramp.intensity(at: d / 2))),
                    CHHapticParameterCurve.ControlPoint(relativeTime: d, value: Float(ramp.intensity(at: d))),
                    CHHapticParameterCurve.ControlPoint(relativeTime: d + 0.01, value: 1.0),
                ]
                curves.append(CHHapticParameterCurve(parameterID: .hapticIntensityControl,
                                                     controlPoints: points,
                                                     relativeTime: ramp.time))
            }
        }
        return (events, curves)
    }

    static func pattern(for spec: ChallengeSpec) throws -> CHHapticPattern {
        let parts = components(for: spec)
        return try CHHapticPattern(events: parts.events, parameterCurves: parts.curves)
    }

    // MARK: Engine

    /// Creates and starts the engine if needed.
    func prepare() throws {
        lock.lock(); defer { lock.unlock() }
        guard Self.isSupported else { throw HapticError.unsupported }
        if engine == nil {
            let e = try CHHapticEngine()
            e.playsHapticsOnly = true
            e.isAutoShutdownEnabled = false
            e.resetHandler = { [weak self] in
                self?.restart()
            }
            e.stoppedHandler = { [weak self] _ in
                self?.restart()
            }
            engine = e
        }
        try engine?.start()
    }

    private func restart() {
        lock.lock(); defer { lock.unlock() }
        player = nil
        try? engine?.start()
    }

    /// Plays the pattern and returns `systemUptime` read right before the start call.
    func play(_ spec: ChallengeSpec) throws -> TimeInterval {
        try prepare()
        let pattern = try Self.pattern(for: spec)
        lock.lock(); defer { lock.unlock() }
        guard let engine = engine else { throw HapticError.unsupported }
        let newPlayer = try engine.makeAdvancedPlayer(with: pattern)
        player = newPlayer
        let start = ProcessInfo.processInfo.systemUptime
        try newPlayer.start(atTime: CHHapticTimeImmediate)
        return start
    }

    func stop() {
        lock.lock(); defer { lock.unlock() }
        try? player?.stop(atTime: CHHapticTimeImmediate)
        player = nil
    }
}
