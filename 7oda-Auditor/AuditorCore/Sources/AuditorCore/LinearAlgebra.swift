import Foundation

/// Dense helpers for the Gaussian model (SPEC 9.2), written with left-to-right sums
/// in the same order as `research/auditor_ref/model.py`.
enum LinearAlgebra {
    /// `d x d` identity.
    static func identity(_ d: Int) -> [[Double]] {
        var m = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: d)
        for i in 0..<d { m[i][i] = 1.0 }
        return m
    }

    /// `s + jitter * I`.
    static func addingDiagonal(_ s: [[Double]], _ jitter: Double) -> [[Double]] {
        var m = s
        for i in 0..<m.count where i < m[i].count {
            m[i][i] = m[i][i] + jitter
        }
        return m
    }

    /// Lower-triangular `L` with `S = L Lᵀ` (Cholesky–Banachiewicz, row by row);
    /// nil on a non-positive (or NaN) pivot or a non-square input.
    static func cholesky(_ s: [[Double]]) -> [[Double]]? {
        let d = s.count
        guard s.allSatisfy({ $0.count == d }) else { return nil }
        var low = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: d)
        for i in 0..<d {
            for j in 0...i {
                var acc = s[i][j]
                for k in 0..<j {
                    acc = acc - low[i][k] * low[j][k]
                }
                if i == j {
                    if !(acc > 0.0) { return nil }
                    low[i][i] = acc.squareRoot()
                } else {
                    low[i][j] = acc / low[j][j]
                }
            }
        }
        return low
    }

    /// `‖L⁻¹ v‖²` by forward substitution (`v.count` must equal the size of `low`).
    static func forwardSquaredNorm(_ low: [[Double]], _ v: [Double]) -> Double {
        let d = low.count
        var y = [Double](repeating: 0.0, count: d)
        for i in 0..<d {
            var acc = v[i]
            for k in 0..<i {
                acc = acc - low[i][k] * y[k]
            }
            y[i] = acc / low[i][i]
        }
        var s = 0.0
        for i in 0..<d {
            s = s + y[i] * y[i]
        }
        return s
    }
}
