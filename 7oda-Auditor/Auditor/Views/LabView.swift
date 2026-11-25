import SwiftUI
import AuditorCore

/// Lab: signal plot, both recognizers' last decisions, tools, settings, device, export.
struct LabView: View {
    @Environment(AppModel.self) private var model
    @State private var exportURLs: [URL] = []
    @State private var confirmDeleteAll = false

    var body: some View {
        NavigationStack {
            List {
                Section("Last recording") {
                    if let a = model.lastAnalysis {
                        SignalPlot(analysis: a)
                            .frame(height: 230)
                        Text("\(a.recording.purpose.rawValue) · \(a.recording.challenge.events.count) events · seed \(SeedFormat.string(a.recording.challenge.seed))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        MetricsList(report: a.quality)
                    } else {
                        Text("No recording yet.").foregroundStyle(.secondary)
                    }
                }

                Section("Last check") {
                    if let check = model.lastCheck {
                        DecisionDetail(title: "v1 · learned encoder", decision: check.v1, answered: check.answeredBy == .v1)
                        DecisionDetail(title: "v0 · hand-made features", decision: check.v0, answered: check.answeredBy == .v0)
                    } else {
                        Text("No check yet.").foregroundStyle(.secondary)
                    }
                }

                Section("Tools") {
                    NavigationLink("Test run") { TestRunView() }
                    NavigationLink("Signal check") { SignalCheckView() }
                    NavigationLink("Settings") { LabSettingsView() }
                    NavigationLink("Recalibrate") { CalibrationView(firstRun: false) }
                    Button("Recompute features from recordings") {
                        Task { await model.recomputeAll() }
                    }
                    .disabled(model.busyStatus != nil)
                }

                Section("Device") {
                    DeviceInfoView()
                }

                Section {
                    Toggle("Anonymize names (P01, P02, …)", isOn: model.binding(\.anonymizeExports))
                    Button("Prepare export files") {
                        do {
                            exportURLs = try model.makeExport()
                        } catch {
                            model.errorMessage = "Export failed: \(error.localizedDescription)"
                        }
                    }
                    if !exportURLs.isEmpty {
                        ShareLink(items: exportURLs) {
                            Label("Share \(exportURLs.count) files", systemImage: "square.and.arrow.up")
                        }
                    }
                } header: {
                    Text("Export")
                } footer: {
                    Text("recordings.jsonl (raw recordings), features.csv (v0 samples) and checks.csv (both recognizers' outcomes). Exported files are not encrypted.")
                }

                Section {
                    Button("Delete all data", role: .destructive) { confirmDeleteAll = true }
                }
            }
            .navigationTitle("Lab")
            .confirmationDialog("Delete all people, recordings, calibration and settings?",
                                isPresented: $confirmDeleteAll, titleVisibility: .visible) {
                Button("Delete all data", role: .destructive) {
                    exportURLs = []
                    model.deleteAllData()
                }
            }
        }
        .onAppear { model.beginSensorSession() }
        .onDisappear { model.endSensorSession() }
    }
}

/// Model, system, sensor rates, haptics and model status.
struct DeviceInfoView: View {
    @Environment(AppModel.self) private var model
    @State private var accelRate = 0.0
    @State private var gyroRate = 0.0
    @State private var motionRate = 0.0

    var body: some View {
        LabeledContent("Model", value: DeviceSupport.modelIdentifier)
            .task {
                while !Task.isCancelled {
                    if let rates = model.motion?.measuredRates() {
                        accelRate = rates.accel
                        gyroRate = rates.gyro
                        motionRate = rates.motion
                    }
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }
        LabeledContent("System", value: DeviceSupport.systemDescription)
        LabeledContent("App", value: DeviceSupport.appVersion)
        LabeledContent("Haptics", value: HapticPlayer.isSupported ? "supported" : "not supported")
        LabeledContent("Motion sensors", value: (model.motion?.isAvailable ?? false) ? "available" : "not available")
        LabeledContent("Rates (accel / gyro / motion)",
                       value: "\(Fmt.num(accelRate, 0)) / \(Fmt.num(gyroRate, 0)) / \(Fmt.num(motionRate, 0)) Hz")
        LabeledContent("v1 model", value: model.encoder?.version ?? "unavailable")
        LabeledContent("Feature version", value: "\(FeatureExtractor.version)")
        LabeledContent("Calibration", value: calibrationText)
        if DeviceSupport.isSimulator {
            Text("Simulator: no motion sensors and no haptics. Recording screens explain this instead of recording.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var calibrationText: String {
        guard let c = model.calibration else { return "none" }
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return "\(f.string(from: c.createdAt)) · \(c.patternsUsed) patterns"
    }
}

/// Recognizer choice, thresholds of both recognizers, hand challenge, adaptive learning.
struct LabSettingsView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Form {
            Section {
                Picker("Answering recognizer", selection: model.binding(\.recognizer)) {
                    ForEach(RecognizerKind.allCases) { kind in
                        Text(kind.title).tag(kind)
                    }
                }
                if let note = model.recognizerNote {
                    Text(note).font(.footnote).foregroundStyle(.orange)
                }
            } header: {
                Text("Recognizer")
            } footer: {
                Text("Both recognizers run on every check; the other one is logged for comparison.")
            }

            Section("v0 thresholds") {
                Stepper(value: model.binding(\.absFactor), in: 0.5...5.0, step: 0.1) {
                    LabeledContent("absFactor", value: Fmt.num(model.settings.absFactor, 1))
                }
                Stepper(value: model.binding(\.nameMargin), in: 0.0...30.0, step: 0.5) {
                    LabeledContent("nameMargin", value: Fmt.num(model.settings.nameMargin, 1))
                }
                Stepper(value: model.binding(\.handMargin), in: 0.0...20.0, step: 0.5) {
                    LabeledContent("handMargin", value: Fmt.num(model.settings.handMargin, 1))
                }
            }

            Section("v1 thresholds") {
                Stepper(value: model.binding(\.slack), in: -0.2...0.2, step: 0.01) {
                    LabeledContent("slack", value: Fmt.num(model.settings.slack, 2))
                }
                Stepper(value: model.binding(\.v1NameMargin), in: 0.0...0.3, step: 0.005) {
                    LabeledContent("nameMargin", value: Fmt.num(model.settings.v1NameMargin, 3))
                }
            }

            Section("Checks and quality") {
                Stepper(value: model.binding(\.maxPatterns), in: 1...6) {
                    LabeledContent("maxPatterns", value: "\(model.settings.maxPatterns)")
                }
                Stepper(value: model.binding(\.minNCC), in: 0.0...1.0, step: 0.05) {
                    LabeledContent("minNCC", value: Fmt.num(model.settings.minNCC, 2))
                }
                Stepper(value: model.binding(\.minPeakMargin), in: 0.0...0.5, step: 0.01) {
                    LabeledContent("minPeakMargin", value: Fmt.num(model.settings.minPeakMargin, 2))
                }
            }

            Section {
                Toggle("Hand challenge", isOn: model.binding(\.handChallenge))
                Toggle("Learn from confident checks", isOn: model.binding(\.learnFromConfidentChecks))
            } footer: {
                Text("Hand challenge: the app asks for a random hand and passes only if it names you with that hand. Learning adds the patterns of very confident checks to your templates (at most 20 per hand).")
            }

            Section {
                Button("Restore defaults") {
                    var s = AppSettings()
                    s.calibrationSkipped = model.settings.calibrationSkipped
                    s.anonymizeExports = model.settings.anonymizeExports
                    model.updateSettings(s)
                }
            }
        }
        .navigationTitle("Settings")
    }
}
