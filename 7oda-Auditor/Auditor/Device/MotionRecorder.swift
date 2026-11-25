import CoreMotion
import Foundation
import AuditorCore

/// One shared `CMMotionManager` streaming accelerometer, gyroscope and device motion at
/// 100 Hz into a thread-safe ring buffer of the last 12 s. Start/stop are reference
/// counted so overlapping screens can share it.
final class MotionRecorder {
    static let shared = MotionRecorder()

    static let interval: TimeInterval = 0.01
    static let bufferSeconds: TimeInterval = 12.0

    private struct Vec3 { var t, x, y, z: Double }
    private struct MotionSample {
        var t, gx, gy, gz, ux, uy, uz, rx, ry, rz, qw, qx, qy, qz: Double
    }

    private let manager = CMMotionManager()
    private let queue: OperationQueue
    private let lock = NSLock()
    private var accel: [Vec3] = []
    private var gyro: [Vec3] = []
    private var motion: [MotionSample] = []
    private var users = 0
    private var startedAt: TimeInterval?

    init() {
        queue = OperationQueue()
        queue.name = "auditor.motion"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
    }

    var isAccelerometerAvailable: Bool { manager.isAccelerometerAvailable }
    var isGyroAvailable: Bool { manager.isGyroAvailable }
    var isDeviceMotionAvailable: Bool { manager.isDeviceMotionAvailable }
    /// All three streams exist (false in the Simulator).
    var isAvailable: Bool { isAccelerometerAvailable && isGyroAvailable && isDeviceMotionAvailable }

    /// Seconds since the sensors were started (0 when stopped).
    var secondsRunning: TimeInterval {
        lock.lock(); defer { lock.unlock() }
        guard let s = startedAt else { return 0 }
        return ProcessInfo.processInfo.systemUptime - s
    }

    var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return users > 0
    }

    func start() {
        lock.lock()
        users += 1
        let first = users == 1
        if first {
            accel.removeAll(keepingCapacity: true)
            gyro.removeAll(keepingCapacity: true)
            motion.removeAll(keepingCapacity: true)
            startedAt = ProcessInfo.processInfo.systemUptime
        }
        lock.unlock()
        guard first else { return }

        if manager.isAccelerometerAvailable {
            manager.accelerometerUpdateInterval = Self.interval
            manager.startAccelerometerUpdates(to: queue) { [weak self] data, _ in
                guard let self = self, let d = data else { return }
                let a = d.acceleration
                self.append(accel: Vec3(t: d.timestamp, x: a.x, y: a.y, z: a.z))
            }
        }
        if manager.isGyroAvailable {
            manager.gyroUpdateInterval = Self.interval
            manager.startGyroUpdates(to: queue) { [weak self] data, _ in
                guard let self = self, let d = data else { return }
                let r = d.rotationRate
                self.append(gyro: Vec3(t: d.timestamp, x: r.x, y: r.y, z: r.z))
            }
        }
        if manager.isDeviceMotionAvailable {
            manager.deviceMotionUpdateInterval = Self.interval
            manager.startDeviceMotionUpdates(to: queue) { [weak self] data, _ in
                guard let self = self, let m = data else { return }
                let q = m.attitude.quaternion
                self.append(motion: MotionSample(
                    t: m.timestamp,
                    gx: m.gravity.x, gy: m.gravity.y, gz: m.gravity.z,
                    ux: m.userAcceleration.x, uy: m.userAcceleration.y, uz: m.userAcceleration.z,
                    rx: m.rotationRate.x, ry: m.rotationRate.y, rz: m.rotationRate.z,
                    qw: q.w, qx: q.x, qy: q.y, qz: q.z))
            }
        }
    }

    func stop() {
        lock.lock()
        users = max(0, users - 1)
        let last = users == 0
        if last { startedAt = nil }
        lock.unlock()
        guard last else { return }
        manager.stopAccelerometerUpdates()
        manager.stopGyroUpdates()
        manager.stopDeviceMotionUpdates()
    }

    // MARK: Buffer

    private func append(accel s: Vec3) {
        lock.lock(); defer { lock.unlock() }
        accel.append(s)
        Self.trim(&accel, latest: s.t)
    }

    private func append(gyro s: Vec3) {
        lock.lock(); defer { lock.unlock() }
        gyro.append(s)
        Self.trim(&gyro, latest: s.t)
    }

    private func append(motion s: MotionSample) {
        lock.lock(); defer { lock.unlock() }
        motion.append(s)
        if motion.count > 1_400, let first = motion.first, first.t < s.t - Self.bufferSeconds {
            let cut = motion.firstIndex { $0.t >= s.t - Self.bufferSeconds } ?? motion.count
            motion.removeFirst(cut)
        }
    }

    private static func trim(_ buffer: inout [Vec3], latest: Double) {
        // Trim in chunks (about every 2 s of data) to keep appends cheap.
        guard buffer.count > 1_400, let first = buffer.first, first.t < latest - bufferSeconds else { return }
        let cut = buffer.firstIndex { $0.t >= latest - bufferSeconds } ?? buffer.count
        buffer.removeFirst(cut)
    }

    /// Samples with `from <= t <= to` from all three streams.
    func snapshot(from: Double, to: Double) -> (accel: SensorStream, gyro: SensorStream, motion: MotionStream) {
        lock.lock()
        let a = accel.filter { $0.t >= from && $0.t <= to }
        let g = gyro.filter { $0.t >= from && $0.t <= to }
        let m = motion.filter { $0.t >= from && $0.t <= to }
        lock.unlock()
        let accelStream = SensorStream(t: a.map { $0.t }, x: a.map { $0.x }, y: a.map { $0.y }, z: a.map { $0.z })
        let gyroStream = SensorStream(t: g.map { $0.t }, x: g.map { $0.x }, y: g.map { $0.y }, z: g.map { $0.z })
        let motionStream = MotionStream(
            t: m.map { $0.t },
            gravityX: m.map { $0.gx }, gravityY: m.map { $0.gy }, gravityZ: m.map { $0.gz },
            userAccelX: m.map { $0.ux }, userAccelY: m.map { $0.uy }, userAccelZ: m.map { $0.uz },
            rotationX: m.map { $0.rx }, rotationY: m.map { $0.ry }, rotationZ: m.map { $0.rz },
            attitudeW: m.map { $0.qw }, attitudeX: m.map { $0.qx }, attitudeY: m.map { $0.qy },
            attitudeZ: m.map { $0.qz })
        return (accelStream, gyroStream, motionStream)
    }

    /// Measured rates (Hz) over the buffered samples: accelerometer, gyroscope, device motion.
    func measuredRates() -> (accel: Double, gyro: Double, motion: Double) {
        lock.lock(); defer { lock.unlock() }
        func rate(_ ts: [Double]) -> Double {
            guard ts.count >= 2, let first = ts.first, let last = ts.last, last > first else { return 0 }
            return Double(ts.count - 1) / (last - first)
        }
        return (rate(accel.map { $0.t }), rate(gyro.map { $0.t }), rate(motion.map { $0.t }))
    }
}
