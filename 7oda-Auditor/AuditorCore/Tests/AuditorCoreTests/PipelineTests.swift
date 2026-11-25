import Foundation
import XCTest
import AuditorCore

private struct PipelineFile: Decodable {
    struct Expected: Decodable {
        let n: Int
        let lag: Int
        let ncc: Double
        let peakMargin: Double
        let noiseAccel: Double
        let noiseGyro: Double
        let gapFraction: Double
        let sampleRate: Double
        let quietRange: [Int]
        let eventRange: [Int]
        let metrics: [String: Double?]
        let issues: [String]
        let passed: Bool?
        let featuresAbsolute: [Double]
        let features: [Double]
    }
    struct Case: Decodable {
        let name: String
        let recording: Recording
        let calibration: Calibration?
        let mode: String
        let expected: Expected
    }
    let cases: [Case]
}

final class PipelineTests: XCTestCase {
    private let tolerance = 1e-6

    private func metric(_ m: QualityMetrics, _ key: String) -> Double? {
        switch key {
        case "gapFraction": return m.gapFraction
        case "sampleRate": return m.sampleRate
        case "quietGyroRMS": return m.quietGyroRMS
        case "quietAccelRMS": return m.quietAccelRMS
        case "gravityZ": return m.gravityZ
        case "motionRMS": return m.motionRMS
        case "gravityAngleDeg": return m.gravityAngleDeg
        case "ncc": return m.ncc
        case "peakMargin": return m.peakMargin
        case "snr": return m.snr
        case "lag": return Double(m.lag)
        default: return nil
        }
    }

    /// SPEC 11.4: indices, ranges and issues exact; metrics and features `1e-6 · max(1, |x|)`.
    func testPipelineFixtureParity() {
        guard let file = Fixture.decode(PipelineFile.self, "pipeline_vectors") else { return }
        XCTAssertGreaterThanOrEqual(file.cases.count, 8)
        for c in file.cases {
            let name = c.name
            let e = c.expected
            guard let mode = QualityMode(rawValue: c.mode) else {
                XCTFail("\(name): unknown mode \(c.mode)")
                continue
            }
            let p = SignalProcessor.process(c.recording)
            XCTAssertEqual(p.n, e.n, "\(name) n")
            XCTAssertEqual(p.alignment.lag, e.lag, "\(name) lag")
            assertClose(p.alignment.ncc, e.ncc, rel: tolerance, "\(name) ncc")
            assertClose(p.alignment.peakMargin, e.peakMargin, rel: tolerance, "\(name) peakMargin")
            assertClose(p.noiseAccel, e.noiseAccel, rel: tolerance, "\(name) noiseAccel")
            assertClose(p.noiseGyro, e.noiseGyro, rel: tolerance, "\(name) noiseGyro")
            assertClose(p.gapFraction, e.gapFraction, rel: tolerance, "\(name) gapFraction")
            assertClose(p.sampleRate, e.sampleRate, rel: tolerance, "\(name) sampleRate")
            XCTAssertEqual([p.quietRange.lowerBound, p.quietRange.upperBound], e.quietRange, "\(name) quietRange")
            XCTAssertEqual([p.eventRange.lowerBound, p.eventRange.upperBound], e.eventRange, "\(name) eventRange")

            let report = QualityGate.evaluate(p, calibration: c.calibration, mode: mode)
            for key in e.metrics.keys.sorted() {
                guard let actual = metric(report.metrics, key) else {
                    XCTFail("\(name): unknown metric \(key)")
                    continue
                }
                let expected = (e.metrics[key] ?? nil) ?? Double.infinity
                if key == "lag" {
                    XCTAssertEqual(actual, expected, "\(name) metrics.lag")
                } else {
                    assertClose(actual, expected, rel: tolerance, "\(name) metrics.\(key)")
                }
            }
            XCTAssertEqual(report.issues.map { $0.rawValue }, e.issues, "\(name) issues")
            XCTAssertEqual(report.passed, e.issues.isEmpty, "\(name) passed")
            if let passed = e.passed { XCTAssertEqual(report.passed, passed, "\(name) passed flag") }

            let absolute = FeatureExtractor.extractAbsolute(p)
            assertClose(absolute, e.featuresAbsolute, rel: tolerance, "\(name) featuresAbsolute")
            let features = FeatureExtractor.extract(p, calibration: c.calibration)
            assertClose(features, e.features, rel: tolerance, "\(name) features")
            XCTAssertTrue(absolute.allSatisfy { $0.isFinite }, "\(name) features must be finite")
        }
    }

    func testCalibrationBuild() {
        guard let file = Fixture.decode(PipelineFile.self, "pipeline_vectors") else { return }
        guard let table = file.cases.first(where: { $0.expected.issues.isEmpty && $0.mode == "onTable" }) else {
            XCTFail("pipeline_vectors.json has no passing onTable case")
            return
        }
        let p = SignalProcessor.process(table.recording)
        let report = QualityGate.evaluate(p, calibration: nil, mode: .onTable)
        XCTAssertTrue(report.passed)
        var failed = report
        failed.issues = [.notResting]
        failed.passed = false
        let now = Date(timeIntervalSince1970: 1_795_514_400)
        let eleven = Array(repeating: p, count: 11)
        XCTAssertNil(Calibration.build(from: eleven, reports: Array(repeating: report, count: 11), device: "X", now: now))
        let processed = Array(repeating: p, count: 20)
        let reports = Array(repeating: report, count: 12) + Array(repeating: failed, count: 8)
        guard let calibration = Calibration.build(from: processed, reports: reports, device: "iPhone16,1", now: now) else {
            XCTFail("12 passing patterns must give a calibration")
            return
        }
        XCTAssertEqual(calibration.patternsUsed, 12)
        XCTAssertEqual(calibration.tableFeatures.count, FeatureExtractor.names.count)
        XCTAssertEqual(calibration.tableFeatures, FeatureExtractor.extractAbsolute(p))
        XCTAssertEqual(calibration.quietGyroRMS, report.metrics.quietGyroRMS)
        XCTAssertEqual(calibration.snr, report.metrics.snr)
        XCTAssertEqual(calibration.device, "iPhone16,1")
        XCTAssertEqual(calibration.createdAt, now)
        // Relative features vanish on the calibration's own pattern.
        let relative = FeatureExtractor.extract(p, calibration: calibration)
        for r in FeatureExtractor.calibrationRelativeIndices {
            XCTAssertEqual(relative[r], 0.0, accuracy: 1e-12)
        }
        // JSON round trip.
        if let data = try? RecordingCoding.encoder().encode(calibration),
           let back = try? RecordingCoding.decoder().decode(Calibration.self, from: data) {
            XCTAssertEqual(back, calibration)
        } else {
            XCTFail("calibration JSON round trip failed")
        }
    }

    func testFeatureNamesAndEdgeCases() {
        XCTAssertEqual(FeatureExtractor.version, 2)
        XCTAssertEqual(FeatureExtractor.names.count, 39)
        XCTAssertEqual(Set(FeatureExtractor.names).count, 39)
        XCTAssertEqual(FeatureExtractor.names[20], "tap_tail_log_accel_gyro")
        XCTAssertEqual(FeatureExtractor.names[38], "gravity_z")
        XCTAssertEqual(FeatureExtractor.calibrationRelativeIndices, [0, 1, 20, 21, 22])

        // Empty streams and degenerate durations: finite features, no crash.
        var spec = ChallengeGenerator.generate(seed: 8)
        for duration in [spec.duration, 0.0, -1.0, Double.nan] {
            spec.duration = duration
            let r = Recording(createdAt: Date(timeIntervalSince1970: 0), purpose: .check, challenge: spec,
                              hapticStart: 10.0, accel: .empty, gyro: .empty)
            let p = SignalProcessor.process(r)
            let f = FeatureExtractor.extractAbsolute(p)
            XCTAssertEqual(f.count, 39)
            XCTAssertTrue(f.allSatisfy { $0.isFinite }, "duration \(duration)")
            XCTAssertEqual(Array(f[36...38]), [0.0, 0.0, -1.0])
            let report = QualityGate.evaluate(p, calibration: nil, mode: .handheld)
            XCTAssertFalse(report.passed)
            XCTAssertTrue(report.issues.contains(.lowSampleRate))
            if p.n > 0 { XCTAssertTrue(report.issues.contains(.sensorGaps)) }
        }
    }

    func testQualityRulesOnSyntheticRecordings() {
        let p = SignalProcessor.process(Synthetic.recording(seed: 7, delay: 0.05))
        let report = QualityGate.evaluate(p, calibration: nil, mode: .handheld)
        XCTAssertEqual(report.issues, [], "synthetic handheld pattern should pass: \(report.metrics)")
        XCTAssertEqual(report.metrics.lag, 5)
        XCTAssertGreaterThan(report.metrics.snr, 2.0)
        // The same pattern evaluated on the table: gravity is tilted, so it is not resting.
        XCTAssertEqual(QualityGate.evaluate(p, calibration: nil, mode: .onTable).issues, [.notResting])
        // Stricter thresholds produce the matching issues.
        var strict = QualityConfig.default
        strict.minPeakMargin = 0.99
        strict.minSNR = 1e9
        XCTAssertEqual(QualityGate.evaluate(p, calibration: nil, mode: .handheld, config: strict).issues,
                       [.weakAlignment, .weakSignal])
        XCTAssertEqual(QualityIssue.weakSignal.message, "Vibration too weak — remove a thick case?")
        XCTAssertEqual(QualityIssue.allCases.count, 7)
        if let data = try? JSONEncoder().encode(report),
           let back = try? JSONDecoder().decode(QualityReport.self, from: data) {
            XCTAssertEqual(back, report)
        } else {
            XCTFail("quality report JSON round trip failed")
        }
    }
}
