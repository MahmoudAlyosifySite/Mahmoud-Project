import SwiftUI
import AuditorCore

/// Number formatting for the UI.
enum Fmt {
    static func num(_ x: Double, _ digits: Int = 3) -> String {
        if x.isNaN { return "—" }
        if x.isInfinite { return x > 0 ? "∞" : "−∞" }
        return String(format: "%.\(digits)f", x)
    }

    static func percent(_ x: Double) -> String {
        x.isFinite ? "\(Int((x * 100).rounded()))%" : "—"
    }
}

extension AppModel {
    /// A binding to one setting; every change is saved and the recognizers are rebuilt.
    func binding<T>(_ keyPath: WritableKeyPath<AppSettings, T>) -> Binding<T> {
        Binding(get: { self.settings[keyPath: keyPath] },
                set: { newValue in
                    var s = self.settings
                    s[keyPath: keyPath] = newValue
                    self.updateSettings(s)
                })
    }

    func handName(_ hand: Hand) -> String { hand == .right ? "right hand" : "left hand" }

    func prototypeName(_ key: PrototypeKey) -> String {
        var s = "\(personName(key.personID)) · \(key.hand.rawValue)"
        if key.accessory != Accessory.none { s += " · \(key.accessory.rawValue)" }
        return s
    }

    /// "Name · right hand", "Name · hand unclear", "Unknown", …
    func headline(for outcome: DecisionOutcome) -> String {
        switch outcome {
        case .identified(let person, let hand): return "\(personName(person)) · \(handName(hand))"
        case .identifiedHandUnclear(let person): return "\(personName(person)) · hand unclear"
        case .needMorePatterns: return "Undecided"
        case .unknown: return "Unknown"
        }
    }

    func reasonText(for outcome: DecisionOutcome) -> String? {
        guard case .unknown(let reason) = outcome else { return nil }
        switch reason {
        case .noPeople: return "Nobody is enrolled yet."
        case .noMatch: return "This hand does not match anyone enrolled."
        case .ambiguous: return "Too close between two people."
        }
    }
}

/// Circular progress with a label in the middle.
struct ProgressRing: View {
    var progress: Double
    var label: String

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.secondary.opacity(0.2), lineWidth: 14)
            Circle()
                .trim(from: 0, to: max(0, min(1, progress.isFinite ? progress : 0)))
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 14, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeInOut, value: progress)
            Text(label)
                .font(.title2.monospacedDigit().bold())
        }
    }
}

/// Per-person confidence bars.
struct ScoreBars: View {
    @Environment(AppModel.self) private var model
    let scores: [PersonScore]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(scores, id: \.personID) { score in
                HStack(spacing: 8) {
                    Text(model.personName(score.personID))
                        .lineLimit(1)
                        .frame(width: 110, alignment: .leading)
                    ProgressView(value: score.probability.isFinite ? max(0, min(1, score.probability)) : 0)
                    Text(Fmt.percent(score.probability))
                        .monospacedDigit()
                        .frame(width: 48, alignment: .trailing)
                }
                .font(.subheadline)
            }
        }
    }
}

/// Pass/fail and the key alignment metrics of one pattern.
struct QualityBadge: View {
    let report: QualityReport

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: report.passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundStyle(report.passed ? Color.green : Color.orange)
            Text("NCC \(Fmt.num(report.metrics.ncc, 2)) · peak \(Fmt.num(report.metrics.peakMargin, 2)) · SNR \(Fmt.num(report.metrics.snr, 1))")
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }
}

/// All quality metrics of one pattern (Lab).
struct MetricsList: View {
    let report: QualityReport

    var body: some View {
        let m = report.metrics
        VStack(alignment: .leading, spacing: 4) {
            row("Passed", report.passed ? "yes" : "no")
            if !report.issues.isEmpty {
                row("Issues", report.issues.map { $0.rawValue }.joined(separator: ", "))
            }
            row("NCC / peak margin / lag", "\(Fmt.num(m.ncc, 3)) / \(Fmt.num(m.peakMargin, 3)) / \(m.lag)")
            row("SNR", Fmt.num(m.snr, 2))
            row("Sample rate", "\(Fmt.num(m.sampleRate, 1)) Hz")
            row("Gap fraction", Fmt.num(m.gapFraction, 3))
            row("Quiet gyro / accel RMS", "\(Fmt.num(m.quietGyroRMS, 4)) / \(Fmt.num(m.quietAccelRMS, 4))")
            row("Gravity Z", Fmt.num(m.gravityZ, 3))
            row("Motion RMS", Fmt.num(m.motionRMS, 3))
            row("Gravity angle", "\(Fmt.num(m.gravityAngleDeg, 1))°")
        }
        .font(.footnote.monospacedDigit())
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack(alignment: .top) {
            Text(title).foregroundStyle(.secondary)
            Spacer()
            Text(value).multilineTextAlignment(.trailing)
        }
    }
}

/// One recognizer's decision with distances and margins (Lab).
struct DecisionDetail: View {
    @Environment(AppModel.self) private var model
    let title: String
    let decision: Decision?
    let answered: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.headline)
                if answered {
                    Text("answered")
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.2), in: Capsule())
                }
            }
            if let d = decision {
                Text(model.headline(for: d.outcome) + (model.reasonText(for: d.outcome).map { " — " + $0 } ?? ""))
                Group {
                    Text("best \(Fmt.num(d.bestDistance)) · threshold \(Fmt.num(d.absThreshold)) · patterns \(d.patternsUsed)")
                    Text("margin name \(Fmt.num(d.marginName)) · margin hand \(Fmt.num(d.marginHand)) · hand log-odds \(Fmt.num(d.handLogOdds))")
                }
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
                ForEach(Array(d.prototypeDistances.enumerated()), id: \.offset) { item in
                    HStack {
                        Text(model.prototypeName(item.element.prototype))
                        Spacer()
                        Text(Fmt.num(item.element.distance)).monospacedDigit()
                    }
                    .font(.caption)
                }
            } else {
                Text("Not evaluated").foregroundStyle(.secondary)
            }
        }
    }
}

/// A highlighted note (model fallback, simulator limits, …).
struct NoteBanner: View {
    let text: String
    var systemImage: String = "exclamationmark.triangle"

    var body: some View {
        Label(text, systemImage: systemImage)
            .font(.footnote)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
    }
}
