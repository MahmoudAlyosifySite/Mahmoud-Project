import Foundation
import AuditorCore

/// Export files of the Lab: recordings (JSON Lines), v0 features (CSV) and checks (CSV).
enum Exporter {
    /// personID → display name; with `anonymize`, P01, P02, … in enrollment order.
    static func nameMap(people: [Person], anonymize: Bool) -> [String: String] {
        var map: [String: String] = [:]
        let ordered = people.sorted { a, b in
            a.createdAt != b.createdAt ? a.createdAt < b.createdAt : a.id < b.id
        }
        for (i, p) in ordered.enumerated() {
            map[p.id] = anonymize ? String(format: "P%02d", i + 1) : p.name
        }
        return map
    }

    /// Recordings with label names taken from `names` (current or anonymized names).
    static func prepare(_ recordings: [Recording], names: [String: String]) -> [Recording] {
        recordings.map { r in
            var r = r
            if var label = r.label {
                label.personName = names[label.personID] ?? "P??"
                r.label = label
            }
            return r
        }
    }

    static func csvField(_ s: String) -> String {
        if s.contains(",") || s.contains("\"") || s.contains("\n") {
            return "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return s
    }

    static func number(_ x: Double?) -> String {
        guard let x = x else { return "" }
        guard x.isFinite else { return x > 0 ? "inf" : (x < 0 ? "-inf" : "nan") }
        return String(format: "%.6g", x)
    }

    private static func iso(_ d: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f.string(from: d)
    }

    /// One row per v0 template sample.
    static func featuresCSV(samples: [TemplateSample], names: [String: String]) -> String {
        let width = samples.map { $0.features.count }.max() ?? FeatureExtractor.names.count
        let featureNames = (0..<width).map { k in k < FeatureExtractor.names.count ? FeatureExtractor.names[k] : "f\(k)" }
        let header: [String] = ["sample_id", "recording_id", "person", "person_id", "hand", "accessory", "source",
                                "created_at", "feature_version"] + featureNames
        var lines: [String] = [header.map(csvField).joined(separator: ",")]
        for s in samples {
            var row: [String] = [s.id.uuidString, s.recordingID?.uuidString ?? "", names[s.prototype.personID] ?? "P??",
                       s.prototype.personID, s.prototype.hand.rawValue, s.prototype.accessory.rawValue,
                       s.source.rawValue, iso(s.createdAt), String(s.featureVersion)]
            row += (0..<width).map { k in k < s.features.count ? number(s.features[k]) : "" }
            lines.append(row.map(csvField).joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func decisionColumns(_ prefix: String) -> [String] {
        ["outcome", "person", "hand", "reason", "patterns", "best_distance", "margin_name", "margin_hand",
         "hand_log_odds", "abs_threshold"].map { "\(prefix)_\($0)" }
    }

    private static func decisionFields(_ d: DecisionSummary?, names: [String: String]) -> [String] {
        guard let d = d else { return Array(repeating: "", count: 10) }
        let person: String = d.personID.map { names[$0] ?? "P??" } ?? ""
        let fields: [String] = [d.outcome, person, d.hand?.rawValue ?? "", d.reason ?? "", String(d.patternsUsed),
                number(d.bestDistance), number(d.marginName), number(d.marginHand),
                number(d.handLogOdds), number(d.absThreshold)]
        return fields
    }

    /// One row per check with both recognizers' outcomes.
    static func checksCSV(entries: [CheckLogEntry], names: [String: String]) -> String {
        let header: [String] = ["check_id", "created_at", "purpose", "answered_by", "patterns_used", "attempts", "seconds",
                      "truth_person", "truth_hand", "requested_hand", "challenge_passed", "quality_failure",
                      "recording_ids"] + decisionColumns("v0") + decisionColumns("v1")
        var lines: [String] = [header.map(csvField).joined(separator: ",")]
        for e in entries {
            var truthName = ""
            if let id = e.truthPersonID { truthName = names[id] ?? "P??" }
            var challenge = ""
            if let passed = e.challengePassed { challenge = passed ? "yes" : "no" }
            let recordingList = e.recordingIDs.map { $0.uuidString }.joined(separator: " ")
            var row: [String] = [e.id.uuidString, iso(e.createdAt), e.purpose.rawValue, e.recognizer.rawValue]
            row += [String(e.patternsUsed), String(e.attempts), number(e.seconds)]
            row += [truthName, e.truthHand?.rawValue ?? "", e.requestedHand?.rawValue ?? "", challenge]
            row += [e.qualityFailure ?? "", recordingList]
            row += decisionFields(e.v0, names: names)
            row += decisionFields(e.v1, names: names)
            lines.append(row.map(csvField).joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }
}
