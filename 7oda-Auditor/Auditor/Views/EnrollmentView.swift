import SwiftUI
import AuditorCore

/// Drives an `EnrollmentFlow` with the device: records each slot, repeats failed ones.
@Observable
@MainActor
final class EnrollmentController {
    var flow: EnrollmentFlow
    private(set) var accepted: [(slot: EnrollmentSlot, analysis: AnalyzedPattern)] = []
    private(set) var isRecording = false
    private(set) var lastAnalysis: AnalyzedPattern?
    private(set) var committed = false
    var errorMessage: String?

    private let model: AppModel
    @ObservationIgnored private var task: Task<Void, Never>?

    init(kind: EnrollmentKind, model: AppModel) {
        self.flow = EnrollmentFlow(kind: kind)
        self.model = model
    }

    /// Confirms the user is ready and records until the flow needs another confirmation.
    func startOrContinue() {
        guard !isRecording, flow.step == .recording else { return }
        errorMessage = nil
        flow.confirmReady()
        task = Task { await runLoop() }
    }

    private func runLoop() async {
        isRecording = true
        defer { isRecording = false }
        while flow.step == .recording && !flow.awaitingConfirmation {
            guard let slot = flow.currentSlot else { break }
            do {
                try await Task.sleep(nanoseconds: 600_000_000)
                let purpose: RecordingPurpose = flow.sampleSource == .accessory ? .accessory : .enrollment
                let recording = try await model.runner.record(purpose: purpose, label: flow.label(for: slot))
                let a = model.analyze(recording, mode: .handheld)
                lastAnalysis = a
                model.lastAnalysis = a
                apply(a, to: slot)
            } catch is CancellationError {
                return
            } catch {
                errorMessage = error.localizedDescription
                return
            }
        }
    }

    /// Feeds one analysed pattern into the flow (also used by tests).
    func apply(_ a: AnalyzedPattern, to slot: EnrollmentSlot) {
        guard flow.currentSlot == slot else { return }
        if a.passed { accepted.append((slot: slot, analysis: a)) }
        flow.submit(passed: a.passed, issues: a.quality.issues)
    }

    func cancel() {
        task?.cancel()
        task = nil
        if flow.step != .summary || !committed { flow.cancel() }
    }

    func commit() {
        guard !committed, flow.step == .summary else { return }
        model.commitEnrollment(flow: flow, patterns: accepted)
        committed = true
    }
}

/// The Enroll sheet: name → checklist → right hand rounds → left hand rounds → summary.
struct EnrollmentView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    let kind: EnrollmentKind

    @State private var controller: EnrollmentController?
    @State private var nameField = ""

    var body: some View {
        NavigationStack {
            Group {
                if let c = controller {
                    content(c)
                } else {
                    ProgressView()
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(controller?.flow.step == .summary ? "Discard" : "Cancel") {
                        controller?.cancel()
                        dismiss()
                    }
                }
            }
        }
        .interactiveDismissDisabled(true)
        .onAppear {
            if controller == nil {
                controller = EnrollmentController(kind: kind, model: model)
            }
            model.beginSensorSession()
        }
        .onDisappear {
            controller?.cancel()
            model.endSensorSession()
        }
    }

    private var title: String {
        switch kind {
        case .newPerson: return "Add person"
        case .reenroll(_, let name): return "Re-enroll \(name)"
        case .accessory(_, let name, let accessory): return "\(name) · \(accessory.rawValue)"
        }
    }

    @ViewBuilder
    private func content(_ c: EnrollmentController) -> some View {
        switch c.flow.step {
        case .name:
            nameStep(c)
        case .checklist:
            checklistStep(c)
        case .recording:
            recordingStep(c)
        case .summary:
            summaryStep(c)
        case .cancelled:
            Text("Cancelled").foregroundStyle(.secondary)
        }
    }

    private func nameStep(_ c: EnrollmentController) -> some View {
        Form {
            Section {
                TextField("Name", text: $nameField)
                    .textInputAutocapitalization(.words)
                    .submitLabel(.next)
                    .onSubmit { _ = c.flow.submitName(nameField) }
            } header: {
                Text("Who is this?")
            } footer: {
                Text("The name stays on this phone. Recordings are encrypted.")
            }
            Section {
                Button("Continue") { _ = c.flow.submitName(nameField) }
                    .disabled(nameField.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private func checklistText(_ item: EnrollmentFlow.ChecklistItem, accessory: Accessory) -> String {
        switch item {
        case .watchOff: return "Watch off"
        case .ringsOff: return "Rings off"
        case .braceletsOff: return "Bracelets off"
        case .wearingAccessory: return "I am wearing my \(accessory.rawValue)"
        }
    }

    private func checklistStep(_ c: EnrollmentController) -> some View {
        Form {
            Section {
                ForEach(c.flow.checklistItems) { item in
                    Toggle(checklistText(item, accessory: c.flow.accessory),
                           isOn: Binding(get: { c.flow.checked.contains(item) },
                                         set: { c.flow.setChecked(item, $0) }))
                }
            } header: {
                Text("Before you start")
            } footer: {
                Text("You will hold the phone in your right hand, then your left, for \(c.flow.total) short vibration patterns in \(c.flow.roundCount) rounds. Keep still while it vibrates.")
            }
            if let reason = model.recordingUnavailableReason {
                Section {
                    NoteBanner(text: reason, systemImage: "iphone.slash")
                }
            }
            Section {
                Button("Start") { _ = c.flow.confirmChecklist() }
                    .disabled(!c.flow.checklistComplete)
            }
        }
    }

    private func recordingStep(_ c: EnrollmentController) -> some View {
        ScrollView {
            VStack(spacing: 18) {
                ProgressRing(progress: c.flow.progress, label: "\(c.flow.acceptedCount)/\(c.flow.total)")
                    .frame(width: 170, height: 170)
                    .padding(.top, 12)
                Text(c.flow.roundTitle)
                    .font(.headline)
                Text(c.flow.instruction)
                    .multilineTextAlignment(.center)
                if let message = c.flow.lastMessage {
                    Label(message, systemImage: "exclamationmark.circle")
                        .foregroundStyle(.orange)
                }
                if let a = c.lastAnalysis {
                    QualityBadge(report: a.quality)
                }
                if let error = c.errorMessage {
                    NoteBanner(text: error)
                }
                if c.isRecording {
                    ProgressView("Keep still…")
                } else {
                    Button(buttonTitle(c)) { c.startOrContinue() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .disabled(!model.canRecord)
                }
            }
            .padding()
        }
    }

    private func buttonTitle(_ c: EnrollmentController) -> String {
        if c.errorMessage != nil { return "Try again" }
        if c.flow.failuresOnSlot > 0 { return "Try again" }
        if c.flow.currentSlot?.posture == .freshGrip && c.flow.currentSlot?.index != 0 { return "Picked up — next pattern" }
        return c.flow.acceptedCount == 0 ? "Start" : "Start round"
    }

    private func summaryStep(_ c: EnrollmentController) -> some View {
        Form {
            Section("Summary") {
                LabeledContent("Name", value: c.flow.name)
                LabeledContent("Right hand", value: "\(c.accepted.filter { $0.slot.hand == .right }.count) patterns")
                LabeledContent("Left hand", value: "\(c.accepted.filter { $0.slot.hand == .left }.count) patterns")
                LabeledContent("Repeated patterns", value: "\(c.flow.totalFailures)")
                if c.flow.accessory != Accessory.none {
                    LabeledContent("Accessory", value: c.flow.accessory.rawValue)
                }
            }
            Section {
                Button(c.committed ? "Saved" : "Save") {
                    c.commit()
                    dismiss()
                }
                .disabled(c.committed)
            }
        }
    }
}
