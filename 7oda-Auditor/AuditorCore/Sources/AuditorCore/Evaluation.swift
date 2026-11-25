import Foundation

/// Confusion matrix over string labels (used by the Lab test run and the tests).
/// `counts[t][p]` counts samples with truth `labels[t]` predicted as `labels[p]`;
/// labels are added in order of first appearance.
public struct ConfusionMatrix: Codable, Equatable {
    public var labels: [String]
    public var counts: [[Int]]

    public init(labels: [String] = []) {
        self.labels = labels
        self.counts = [[Int]](repeating: [Int](repeating: 0, count: labels.count), count: labels.count)
    }

    public mutating func add(truth: String, predicted: String) {
        let t = indexAdding(truth)
        let p = indexAdding(predicted)
        counts[t][p] += 1
    }

    /// Fraction of samples on the diagonal; 0 when empty.
    public var accuracy: Double {
        var total = 0
        var correct = 0
        for (i, row) in counts.enumerated() {
            for (j, c) in row.enumerated() {
                total += c
                if i == j { correct += c }
            }
        }
        return total > 0 ? Double(correct) / Double(total) : 0.0
    }

    /// Total number of samples added.
    public var total: Int {
        return counts.reduce(0) { $0 + $1.reduce(0, +) }
    }

    /// Count for one (truth, predicted) pair; 0 for unknown labels.
    public func count(truth: String, predicted: String) -> Int {
        guard let t = labels.firstIndex(of: truth), let p = labels.firstIndex(of: predicted),
              t < counts.count, p < counts[t].count else { return 0 }
        return counts[t][p]
    }

    /// Index of `label`, appending it (and growing the matrix) if new; also repairs a
    /// matrix whose shape does not match `labels` (e.g. decoded from a hand-edited file).
    private mutating func indexAdding(_ label: String) -> Int {
        if labels.firstIndex(of: label) == nil { labels.append(label) }
        let size = labels.count
        if counts.count > size { counts.removeLast(counts.count - size) }
        while counts.count < size { counts.append([]) }
        for i in 0..<size {
            if counts[i].count > size { counts[i].removeLast(counts[i].count - size) }
            while counts[i].count < size { counts[i].append(0) }
        }
        return labels.firstIndex(of: label) ?? 0
    }
}
