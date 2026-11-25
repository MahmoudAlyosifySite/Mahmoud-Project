import Foundation

/// Small numeric helpers whose semantics mirror `research/auditor_ref/_numerics.py`
/// exactly (left-to-right sums, numpy median and "linear" quantile).
enum Numerics {
    /// Bound used by `truncatingInt` so that later index arithmetic cannot overflow.
    static let intLimit = 1_000_000_000_000_000

    /// `Int(x)` (truncation toward zero, like Python's `int()`), but never traps:
    /// NaN gives 0 and values beyond ±1e15 are clamped.
    @inline(__always)
    static func truncatingInt(_ x: Double) -> Int {
        if x.isNaN { return 0 }
        if x >= 1.0e15 { return intLimit }
        if x <= -1.0e15 { return -intLimit }
        return Int(x)
    }

    /// Left-to-right sum starting from 0.0.
    static func sum(_ values: [Double]) -> Double {
        var s = 0.0
        for v in values { s = s + v }
        return s
    }

    /// Left-to-right sum of `values[range]` (the range must be valid).
    static func sum(_ values: [Double], _ range: Range<Int>) -> Double {
        var s = 0.0
        for i in range { s = s + values[i] }
        return s
    }

    /// `sum / count`; 0.0 for an empty input.
    static func mean(_ values: [Double]) -> Double {
        if values.isEmpty { return 0.0 }
        return sum(values) / Double(values.count)
    }

    /// Mean of `values[range]`; 0.0 for an empty range (the range must be valid).
    static func mean(_ values: [Double], _ range: Range<Int>) -> Double {
        if range.isEmpty { return 0.0 }
        return sum(values, range) / Double(range.count)
    }

    /// Median with numpy semantics (mean of the two middle values for even counts);
    /// 0.0 for an empty input.
    static func median(_ values: [Double]) -> Double {
        let n = values.count
        if n == 0 { return 0.0 }
        let sorted = values.sorted(by: totalOrderLess)
        let mid = n / 2
        if n % 2 == 1 { return sorted[mid] }
        return (sorted[mid - 1] + sorted[mid]) / 2.0
    }

    /// `numpy.quantile(values, q)` with the default "linear" method, including numpy's
    /// two-sided interpolation formula; nil for an empty input.
    static func quantileLinear(_ values: [Double], _ q: Double) -> Double? {
        let n = values.count
        if n == 0 { return nil }
        let sorted = values.sorted(by: totalOrderLess)
        let h = Double(n - 1) * q
        if h >= Double(n - 1) { return sorted[n - 1] }
        if !(h >= 0.0) { return sorted[0] }
        let i = min(max(truncatingInt(h.rounded(.down)), 0), n - 2)
        let g = h - Double(i)
        let a = sorted[i]
        let b = sorted[i + 1]
        let diff = b - a
        if g >= 0.5 { return b - diff * (1.0 - g) }
        return a + diff * g
    }

    /// Ascending order that places NaN last, so sorting never sees an inconsistent predicate.
    @inline(__always)
    static func totalOrderLess(_ a: Double, _ b: Double) -> Bool {
        if a.isNaN { return false }
        if b.isNaN { return true }
        return a < b
    }

    /// Solves `A x = b` by Gaussian elimination with partial pivoting (first row with the
    /// largest `|A[r][col]|` on ties), as `solve_partial_pivot` of the reference.
    /// Returns nil for a singular (zero-pivot) or malformed system.
    static func solvePartialPivot(_ matrix: [[Double]], _ rhs: [Double]) -> [Double]? {
        let k = rhs.count
        guard matrix.count == k, matrix.allSatisfy({ $0.count == k }) else { return nil }
        var a = matrix
        var b = rhs
        for col in 0..<k {
            var piv = col
            var best = abs(a[col][col])
            var r = col + 1
            while r < k {
                let val = abs(a[r][col])
                if val > best {
                    best = val
                    piv = r
                }
                r += 1
            }
            if piv != col {
                a.swapAt(col, piv)
                b.swapAt(col, piv)
            }
            let pivot = a[col][col]
            if pivot == 0.0 || pivot.isNaN { return nil }
            r = col + 1
            while r < k {
                let f = a[r][col] / pivot
                for c in col..<k {
                    a[r][c] = a[r][c] - f * a[col][c]
                }
                b[r] = b[r] - f * b[col]
                r += 1
            }
        }
        var x = [Double](repeating: 0.0, count: k)
        var r = k - 1
        while r >= 0 {
            var s = b[r]
            var c = r + 1
            while c < k {
                s = s - a[r][c] * x[c]
                c += 1
            }
            x[r] = s / a[r][r]
            r -= 1
        }
        return x
    }
}
