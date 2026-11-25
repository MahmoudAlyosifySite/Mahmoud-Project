import SwiftUI
import AuditorCore

/// "Who's holding?": one big button, live status, result card.
struct CheckView: View {
    @Environment(AppModel.self) private var model

    @State private var running = false
    @State private var status = ""
    @State private var requestedHand: Hand?
    @State private var result: CheckResult?
    @State private var failure: String?
    @State private var task: Task<Void, Never>?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    if let note = model.recognizerNote {
                        NoteBanner(text: note)
                    }
                    if let reason = model.recordingUnavailableReason {
                        NoteBanner(text: reason, systemImage: "iphone.slash")
                    }
                    if model.people.isEmpty {
                        NoteBanner(text: "Nobody is enrolled yet. Add a person in the People tab first.",
                                   systemImage: "person.badge.plus")
                    }
                    if let hand = requestedHand, running || result != nil {
                        Label("Hand challenge: hold the phone in your \(hand == .right ? "RIGHT" : "LEFT") hand",
                              systemImage: "hand.raised.fill")
                            .font(.headline)
                            .padding(12)
                            .frame(maxWidth: .infinity)
                            .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
                    }

                    Button {
                        start()
                    } label: {
                        VStack(spacing: 8) {
                            Image(systemName: "hand.raised.fingers.spread")
                                .font(.system(size: 54))
                            Text("Who's holding?")
                                .font(.title2.bold())
                        }
                        .foregroundStyle(.white)
                        .frame(width: 220, height: 220)
                        .background(Circle().fill(running ? Color.gray : Color.accentColor))
                    }
                    .buttonStyle(.plain)
                    .disabled(running || !model.canRecord || model.people.isEmpty)
                    .padding(.top, 8)

                    if running {
                        VStack(spacing: 8) {
                            ProgressView()
                            Text(status).font(.callout).multilineTextAlignment(.center)
                            Button("Stop", role: .cancel) { task?.cancel() }
                        }
                    } else {
                        Text("Hold the phone in one hand, the way you normally do, and keep still while it vibrates.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }

                    if let result = result {
                        ResultCard(result: result)
                    }
                    if let failure = failure {
                        Text(failure).foregroundStyle(.red).multilineTextAlignment(.center)
                    }
                }
                .padding()
            }
            .navigationTitle("Who's holding?")
        }
        .onAppear { model.beginSensorSession() }
        .onDisappear {
            task?.cancel()
            model.endSensorSession()
        }
    }

    private func start() {
        running = true
        failure = nil
        result = nil
        requestedHand = model.settings.handChallenge ? model.drawChallengeHand() : nil
        status = requestedHand == nil ? "Starting…" : "Switch hands if needed…"
        task = Task {
            do {
                if requestedHand != nil {
                    try await Task.sleep(nanoseconds: 2_500_000_000)
                }
                result = try await model.runCheck(purpose: .check, requestedHand: requestedHand) { text in
                    status = text
                }
            } catch is CancellationError {
                failure = "Check stopped."
            } catch {
                failure = error.localizedDescription
            }
            running = false
        }
    }
}

/// Result of one check: name · hand or Unknown, confidence bars, patterns and seconds.
struct ResultCard: View {
    @Environment(AppModel.self) private var model
    let result: CheckResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(headline)
                .font(.title.bold())
            if let subtitle = subtitle {
                Text(subtitle).foregroundStyle(.secondary)
            }
            if let passed = result.challengePassed {
                Label(passed ? "Hand challenge passed" : "Hand challenge failed",
                      systemImage: passed ? "checkmark.seal.fill" : "xmark.seal.fill")
                    .foregroundStyle(passed ? Color.green : Color.red)
            }
            if let d = result.decision, !d.personScores.isEmpty {
                ScoreBars(scores: d.personScores)
            }
            Text("\(result.patterns.count) pattern\(result.patterns.count == 1 ? "" : "s") used · \(result.attempts) played · \(Fmt.num(result.seconds, 1)) s · answered by \(result.answeredBy.rawValue)")
                .font(.footnote)
                .foregroundStyle(.secondary)
            ForEach(result.notes, id: \.self) { note in
                Text(note).font(.footnote).foregroundStyle(.orange)
            }
            if result.learned > 0 {
                Text("Learned from this check (\(result.learned) samples added).")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var headline: String {
        if result.qualityFailure != nil { return "No answer" }
        guard let d = result.decision else { return "No answer" }
        return model.headline(for: d.outcome)
    }

    private var subtitle: String? {
        if let q = result.qualityFailure { return "Could not record a clean pattern: \(q)" }
        guard let d = result.decision else { return nil }
        return model.reasonText(for: d.outcome)
    }
}
