import SwiftUI
import AuditorCore

/// 20 patterns with the phone flat on a table (SPEC 8). First run and Lab → Recalibrate.
struct CalibrationView: View {
    @Environment(AppModel.self) private var model
    let firstRun: Bool

    @State private var running = false
    @State private var done = 0
    @State private var passed = 0
    @State private var lastMessage: String?
    @State private var result: Calibration?
    @State private var failure: String?
    @State private var task: Task<Void, Never>?

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Put the phone flat on a hard table and don't touch it.")
                        .font(.headline)
                    Text("The phone plays \(Calibration.patternCount) short vibration patterns (about a minute) and measures how it responds on its own. Later checks are compared against this, so do it once per phone, and again if you change the case.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            if let reason = model.recordingUnavailableReason {
                Section {
                    NoteBanner(text: reason)
                }
            }

            Section("Progress") {
                ProgressView(value: Double(done), total: Double(Calibration.patternCount))
                Text("\(done) of \(Calibration.patternCount) patterns · \(passed) passed (at least \(Calibration.minimumPassing) needed)")
                    .font(.footnote.monospacedDigit())
                if let message = lastMessage {
                    Text(message).font(.footnote).foregroundStyle(.orange)
                }
            }

            if let cal = result {
                Section("Result") {
                    LabeledContent("Patterns used", value: "\(cal.patternsUsed)")
                    LabeledContent("Sample rate", value: "\(Fmt.num(cal.sampleRate, 1)) Hz")
                    LabeledContent("Quiet gyro RMS", value: Fmt.num(cal.quietGyroRMS, 5))
                    LabeledContent("Median NCC", value: Fmt.num(cal.ncc, 2))
                    LabeledContent("Median SNR", value: Fmt.num(cal.snr, 1))
                }
            }

            if let failure = failure {
                Section {
                    Text(failure).foregroundStyle(.red)
                }
            }

            Section {
                Button(running ? "Calibrating…" : (result == nil ? "Start calibration" : "Calibrate again")) {
                    start()
                }
                .disabled(running || !model.canRecord)
                if running {
                    Button("Cancel", role: .destructive) { task?.cancel() }
                }
                if firstRun {
                    Button("Skip for now") { model.skipCalibration() }
                        .disabled(running)
                }
            } footer: {
                if firstRun {
                    Text("Without a calibration the app uses absolute thresholds; you can calibrate later in Lab.")
                }
            }
        }
        .navigationTitle("Calibrate")
        .onAppear { model.beginSensorSession() }
        .onDisappear {
            task?.cancel()
            model.endSensorSession()
        }
    }

    private func start() {
        running = true
        done = 0
        passed = 0
        lastMessage = nil
        failure = nil
        result = nil
        task = Task {
            do {
                let cal = try await model.runCalibration { d, p, last in
                    done = d
                    passed = p
                    lastMessage = last.qualityMessage
                }
                if let cal = cal {
                    result = cal
                } else {
                    failure = Calibration.failureMessage
                }
            } catch is CancellationError {
                failure = "Calibration cancelled."
            } catch {
                failure = error.localizedDescription
            }
            running = false
        }
    }
}
