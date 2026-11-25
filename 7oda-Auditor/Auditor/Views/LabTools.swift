import Charts
import SwiftUI
import AuditorCore

/// Accel-HP magnitude, gyro-HP magnitude and the aligned template of one pattern, each
/// scaled to its own maximum, with dashed markers at the aligned event onsets.
struct SignalPlot: View {
    let analysis: AnalyzedPattern

    struct Point: Identifiable {
        let id: Int
        let t: Double
        let v: Double
        let series: String
    }

    var body: some View {
        let p = analysis.processed
        let points = Self.points(p)
        let markers = Self.markers(p)
        Chart {
            ForEach(markers.indices, id: \.self) { i in
                RuleMark(x: .value("Event", markers[i]))
                    .foregroundStyle(Color.gray.opacity(0.35))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
            ForEach(points) { pt in
                LineMark(x: .value("Time (s)", pt.t), y: .value("Scaled", pt.v))
                    .foregroundStyle(by: .value("Series", pt.series))
                    .lineStyle(StrokeStyle(lineWidth: 1.2))
            }
        }
        .chartXAxisLabel("seconds from haptic start")
        .chartYAxis(.hidden)
        .chartLegend(position: .bottom)
    }

    static func points(_ p: ProcessedSignal) -> [Point] {
        let fs = p.config.fs
        let pre = p.config.preRoll
        let lag = p.alignment.lag
        let accel = SignalSummary.magnitude(p.accelHP)
        let gyro = SignalSummary.magnitude(p.gyroHP)
        let aligned: [Double] = (0..<p.n).map { n in
            let k = n - lag
            return k >= 0 && k < p.template.count ? p.template[k] : 0
        }
        var out: [Point] = []
        let series: [(String, [Double])] = [("accel HP", accel), ("gyro HP", gyro), ("template", aligned)]
        for (name, values) in series {
            let peak = max(values.max() ?? 0, 1e-12)
            for (i, x) in values.enumerated() {
                out.append(Point(id: out.count, t: Double(i) / fs - pre, v: x / peak, series: name))
            }
        }
        return out
    }

    static func markers(_ p: ProcessedSignal) -> [Double] {
        p.challenge.events.map { Double(p.index(ofPatternTime: $0.time)) / p.config.fs - p.config.preRoll }
    }
}

/// Test run: ground-truth person + hand, one check at a time, a confusion matrix per recognizer.
struct TestRunView: View {
    @Environment(AppModel.self) private var model
    @State private var personID = ""
    @State private var hand: Hand = .right
    @State private var sessionID = UUID().uuidString
    @State private var running = false
    @State private var status = ""
    @State private var last: CheckResult?
    @State private var failure: String?
    @State private var confirmClear = false

    var body: some View {
        Form {
            Section("Ground truth") {
                Picker("Person", selection: $personID) {
                    ForEach(model.people) { person in
                        Text(person.name).tag(person.id)
                    }
                }
                Picker("Hand", selection: $hand) {
                    Text("Right").tag(Hand.right)
                    Text("Left").tag(Hand.left)
                }
                .pickerStyle(.segmented)
            }

            Section {
                Button(running ? status : "Run one check") { run() }
                    .disabled(running || personID.isEmpty || !model.canRecord)
                if let r = last {
                    Text(summaryLine(r))
                        .font(.footnote)
                }
                if let failure = failure {
                    Text(failure).foregroundStyle(.red)
                }
                if let reason = model.recordingUnavailableReason {
                    Text(reason).font(.footnote).foregroundStyle(.secondary)
                }
            }

            ForEach(RecognizerKind.allCases) { kind in
                Section("Confusion matrix · \(kind.rawValue)") {
                    ConfusionMatrixView(matrix: model.confusionMatrix(for: kind))
                }
            }

            Section {
                Button("Clear test results", role: .destructive) { confirmClear = true }
            }
        }
        .navigationTitle("Test run")
        .onAppear {
            if personID.isEmpty { personID = model.people.first?.id ?? "" }
            model.beginSensorSession()
        }
        .onDisappear { model.endSensorSession() }
        .confirmationDialog("Clear all test results?", isPresented: $confirmClear, titleVisibility: .visible) {
            Button("Clear", role: .destructive) { model.clearTestResults() }
        }
    }

    private func summaryLine(_ r: CheckResult) -> String {
        let v1 = r.v1.map { model.headline(for: $0.outcome) } ?? "n/a"
        let v0 = r.v0.map { model.headline(for: $0.outcome) } ?? "n/a"
        return "Last check: v1 \(v1) · v0 \(v0)"
    }

    private func run() {
        guard !personID.isEmpty else { return }
        running = true
        failure = nil
        status = "Starting…"
        let truth = RecordingLabel(personID: personID, personName: model.personName(personID), hand: hand,
                                   accessory: Accessory.none, posture: .unspecified, sessionID: sessionID)
        Task {
            do {
                last = try await model.runCheck(purpose: .test, truth: truth) { text in status = text }
            } catch is CancellationError {
                failure = "Stopped."
            } catch {
                failure = error.localizedDescription
            }
            running = false
        }
    }
}

struct ConfusionMatrixView: View {
    let matrix: ConfusionMatrix

    var body: some View {
        if matrix.total == 0 {
            Text("No test checks yet.").foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 8) {
                Text("Accuracy \(Fmt.percent(matrix.accuracy)) over \(matrix.total) checks (rows: truth, columns: answer)")
                    .font(.footnote)
                ScrollView(.horizontal) {
                    Grid(alignment: .center, horizontalSpacing: 6, verticalSpacing: 4) {
                        GridRow {
                            Text("")
                            ForEach(matrix.labels.indices, id: \.self) { j in
                                Text(matrix.labels[j])
                                    .font(.caption2)
                                    .lineLimit(2)
                                    .frame(width: 70)
                            }
                        }
                        ForEach(matrix.labels.indices, id: \.self) { i in
                            GridRow {
                                Text(matrix.labels[i])
                                    .font(.caption2)
                                    .lineLimit(2)
                                    .frame(width: 90, alignment: .leading)
                                ForEach(matrix.labels.indices, id: \.self) { j in
                                    Text("\(cell(i, j))")
                                        .font(.footnote.monospacedDigit())
                                        .frame(width: 70, height: 26)
                                        .background(i == j ? Color.green.opacity(0.18) : Color.secondary.opacity(0.06))
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func cell(_ i: Int, _ j: Int) -> Int {
        guard i < matrix.counts.count, j < matrix.counts[i].count else { return 0 }
        return matrix.counts[i][j]
    }
}

/// Signal check: 3 patterns, per-channel SNR, "Taptic response visible: yes/no".
struct SignalCheckView: View {
    @Environment(AppModel.self) private var model
    @State private var running = false
    @State private var done = 0
    @State private var result: SignalCheckResult?
    @State private var failure: String?

    var body: some View {
        Form {
            Section {
                Text("Hold the phone in one hand and keep still. Three patterns will play.")
                Button(running ? "Playing…" : "Run signal check") { run() }
                    .disabled(running || !model.canRecord)
                if running {
                    ProgressView(value: Double(done), total: 3)
                }
                if let reason = model.recordingUnavailableReason {
                    Text(reason).font(.footnote).foregroundStyle(.secondary)
                }
                if let failure = failure {
                    Text(failure).foregroundStyle(.red)
                }
            }
            if let r = result {
                Section("Result") {
                    Label("Taptic response visible: \(r.visible ? "yes" : "no")",
                          systemImage: r.visible ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(r.visible ? Color.green : Color.orange)
                    ForEach(r.channels.indices, id: \.self) { i in
                        LabeledContent("SNR \(r.channels[i].name)", value: Fmt.num(r.channels[i].snr, 1))
                    }
                }
                Section("Patterns") {
                    ForEach(r.patterns.indices, id: \.self) { i in
                        QualityBadge(report: r.patterns[i].quality)
                    }
                }
            }
        }
        .navigationTitle("Signal check")
        .onAppear { model.beginSensorSession() }
        .onDisappear { model.endSensorSession() }
    }

    private func run() {
        running = true
        done = 0
        failure = nil
        result = nil
        Task {
            do {
                result = try await model.runSignalCheck { n in done = n }
            } catch {
                failure = error.localizedDescription
            }
            running = false
        }
    }
}
