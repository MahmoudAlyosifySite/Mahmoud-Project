import Foundation

// MARK: - Prototypes and samples (SPEC 9.1)

/// One `(personID, hand, accessory)` combination; a class of the Gaussian model.
/// Order of SPEC 9.1: personID (Unicode scalar order, like Python string comparison),
/// then right < left, then none < watch < ring < bracelet < other.
public struct PrototypeKey: Hashable, Codable, Comparable {
    public var personID: String
    public var hand: Hand
    public var accessory: Accessory

    public init(personID: String, hand: Hand, accessory: Accessory = Accessory.none) {
        self.personID = personID
        self.hand = hand
        self.accessory = accessory
    }

    public static func < (lhs: PrototypeKey, rhs: PrototypeKey) -> Bool {
        if lhs.personID != rhs.personID {
            return lhs.personID.unicodeScalars.lexicographicallyPrecedes(rhs.personID.unicodeScalars)
        }
        if lhs.hand != rhs.hand { return lhs.hand.order < rhs.hand.order }
        return lhs.accessory.order < rhs.accessory.order
    }
}

public enum SampleSource: String, Codable {
    case enrollment, accessory, adaptive
}

/// A stored feature vector of one prototype.
public struct TemplateSample: Codable, Equatable, Identifiable {
    public var id: UUID
    public var prototype: PrototypeKey
    public var source: SampleSource
    public var createdAt: Date
    public var features: [Double]
    public var featureVersion: Int
    public var recordingID: UUID?

    public init(id: UUID = UUID(), prototype: PrototypeKey, source: SampleSource, createdAt: Date = Date(),
                features: [Double], featureVersion: Int = FeatureExtractor.version, recordingID: UUID? = nil) {
        self.id = id
        self.prototype = prototype
        self.source = source
        self.createdAt = createdAt
        self.features = features
        self.featureVersion = featureVersion
        self.recordingID = recordingID
    }
}

// MARK: - Config (SPEC 9.2)

public struct ModelConfig: Equatable, Codable {
    public var shrinkageMin: Double
    public var looQuantile: Double
    public var absFactor: Double

    public init(shrinkageMin: Double = 0.1, looQuantile: Double = 0.95, absFactor: Double = 1.5) {
        self.shrinkageMin = shrinkageMin
        self.looQuantile = looQuantile
        self.absFactor = absFactor
    }

    public static let `default` = ModelConfig()

    private enum CodingKeys: String, CodingKey {
        case shrinkageMin, looQuantile, absFactor
    }

    /// Missing keys take their defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = ModelConfig.default
        shrinkageMin = try c.decodeIfPresent(Double.self, forKey: .shrinkageMin) ?? d.shrinkageMin
        looQuantile = try c.decodeIfPresent(Double.self, forKey: .looQuantile) ?? d.looQuantile
        absFactor = try c.decodeIfPresent(Double.self, forKey: .absFactor) ?? d.absFactor
    }
}

// MARK: - Gaussian model (SPEC 9.2)

/// Shared-covariance Gaussian model: standardized features, class means, Ledoit–Wolf
/// shrunk pooled residual covariance, squared Mahalanobis distances and a
/// leave-one-out absolute threshold.
public struct GaussianModel {
    public let mean: [Double]
    /// Per-feature standard deviation after the `1e-6` floor.
    public let scale: [Double]
    /// Class means in standardized units (zero vectors for classes without samples).
    public let classMeans: [[Double]]
    public let classCounts: [Int]
    /// Shrinkage `λ` actually used (1 in the degenerate case).
    public let shrinkage: Double
    public let absThreshold: Double
    /// Leave-one-out distances, in sample order, of the samples whose class has at least 2 samples.
    public let looDistances: [Double]

    // Internal details (mirrors of the reference's attributes).
    /// Lower Cholesky factor `L` of the shrunk covariance `S`.
    let cholesky: [[Double]]
    /// `mu = trace(S_emp) / d`.
    let mu: Double
    /// `λ_LW` before the floor and clamp; NaN in the degenerate case.
    let shrinkageLW: Double
    let degenerate: Bool
    let jitter: Double
    let looIndices: [Int]

    static let scaleFloor = 1e-6
    static let jitterStart = 1e-6
    static let jitterRetries = 5

    /// Fits on `features` (n × d) with class indices `labels` in `0..<classCount`.
    /// Returns nil without samples, and for malformed input (ragged rows, label count
    /// mismatch or labels out of range).
    public static func fit(features: [[Double]], labels: [Int], classCount: Int, config: ModelConfig) -> GaussianModel? {
        let n = features.count
        guard n > 0, labels.count == n, classCount > 0 else { return nil }
        let d = features[0].count
        guard features.allSatisfy({ $0.count == d }) else { return nil }
        guard labels.allSatisfy({ $0 >= 0 && $0 < classCount }) else { return nil }
        let nD = Double(n)

        // 1. Standardize.
        var mu = [Double](repeating: 0.0, count: d)
        var sig = [Double](repeating: 0.0, count: d)
        for j in 0..<d {
            var s = 0.0
            for i in 0..<n { s = s + features[i][j] }
            let m = s / nD
            var ss = 0.0
            for i in 0..<n {
                let dv = features[i][j] - m
                ss = ss + dv * dv
            }
            mu[j] = m
            sig[j] = max((ss / nD).squareRoot(), scaleFloor)
        }
        var z = features
        for i in 0..<n {
            for j in 0..<d {
                z[i][j] = (features[i][j] - mu[j]) / sig[j]
            }
        }

        // 2. Class means and counts.
        var counts = [Int](repeating: 0, count: classCount)
        for k in labels { counts[k] += 1 }
        var cmeans = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: classCount)
        for k in 0..<classCount where counts[k] > 0 {
            var sums = [Double](repeating: 0.0, count: d)
            for i in 0..<n where labels[i] == k {
                for j in 0..<d { sums[j] = sums[j] + z[i][j] }
            }
            for j in 0..<d { cmeans[k][j] = sums[j] / Double(counts[k]) }
        }

        // 3. Residuals.
        var r = z
        for i in 0..<n {
            let m = cmeans[labels[i]]
            for j in 0..<d { r[i][j] = z[i][j] - m[j] }
        }
        let kPresent = counts.filter { $0 > 0 }.count

        // 4. Ledoit–Wolf shrinkage.
        let lw = ledoitWolf(r)
        var cov: [[Double]]
        var lambda: Double
        var degenerate = false
        var shrinkageLW = lw.lambda
        if n - kPresent < 2 || !(lw.mu > 1e-12) {
            cov = LinearAlgebra.identity(d)
            lambda = 1.0
            degenerate = true
            shrinkageLW = Double.nan
        } else {
            lambda = min(max(max(lw.lambda, config.shrinkageMin), 0.0), 1.0)
            cov = lw.empirical
            let diagonal = lambda * lw.mu
            for j in 0..<d {
                for k in 0..<d {
                    cov[j][k] = (1.0 - lambda) * lw.empirical[j][k] + (j == k ? diagonal : 0.0)
                }
            }
        }

        // 5. Cholesky with jitter.
        var jitterUsed = 0.0
        var low = LinearAlgebra.cholesky(cov)
        if low == nil {
            var jitter = jitterStart * (lw.mu > 0.0 ? lw.mu : 1.0)
            for _ in 0..<jitterRetries {
                low = LinearAlgebra.cholesky(LinearAlgebra.addingDiagonal(cov, jitter))
                jitterUsed = jitter
                if low != nil { break }
                jitter *= 10.0
            }
        }
        let factor: [[Double]]
        if let found = low {
            factor = found
        } else {
            factor = LinearAlgebra.identity(d)
            lambda = 1.0
            degenerate = true
        }

        // 7. Leave-one-out threshold.
        var loo: [Double] = []
        var looIndices: [Int] = []
        for i in 0..<n {
            let k = labels[i]
            let nk = counts[k]
            if nk < 2 { continue }
            var diff = [Double](repeating: 0.0, count: d)
            for j in 0..<d {
                let mPrime = (Double(nk) * cmeans[k][j] - z[i][j]) / Double(nk - 1)
                diff[j] = z[i][j] - mPrime
            }
            loo.append(LinearAlgebra.forwardSquaredNorm(factor, diff))
            looIndices.append(i)
        }
        let absThreshold: Double
        if let q = Numerics.quantileLinear(loo, config.looQuantile) {
            absThreshold = config.absFactor * q
        } else {
            absThreshold = config.absFactor * 2.0 * Double(d)
        }

        return GaussianModel(
            mean: mu, scale: sig, classMeans: cmeans, classCounts: counts,
            shrinkage: lambda, absThreshold: absThreshold, looDistances: loo,
            cholesky: factor, mu: lw.mu, shrinkageLW: shrinkageLW, degenerate: degenerate,
            jitter: jitterUsed, looIndices: looIndices)
    }

    /// Squared Mahalanobis distance `D²_k(x)` for every class (SPEC 9.2 step 6).
    /// A vector of the wrong dimension is infinitely far from every class.
    public func distances(_ x: [Double]) -> [Double] {
        let d = mean.count
        guard x.count == d else {
            return [Double](repeating: Double.infinity, count: classMeans.count)
        }
        var z = [Double](repeating: 0.0, count: d)
        for j in 0..<d {
            z[j] = (x[j] - mean[j]) / scale[j]
        }
        return classMeans.map { m -> Double in
            var v = [Double](repeating: 0.0, count: d)
            for j in 0..<d { v[j] = z[j] - m[j] }
            return LinearAlgebra.forwardSquaredNorm(cholesky, v)
        }
    }

    /// Ledoit–Wolf statistics of the residual matrix `R` (n × d, assumed centered),
    /// exactly as SPEC 9.2 step 4, before the floor and the clamp.
    static func ledoitWolf(_ r: [[Double]]) -> (empirical: [[Double]], mu: Double, lambda: Double) {
        let n = r.count
        let d = r.first?.count ?? 0
        if n == 0 || d == 0 {
            return (LinearAlgebra.identity(d), 0.0, 0.0)
        }
        let nD = Double(n)
        let dD = Double(d)
        // Gram matrix RᵀR and the Gram matrix of the squared entries.
        var gram = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: d)
        var gram2 = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: d)
        for i in 0..<n {
            let row = r[i]
            for j in 0..<d {
                let rj = row[j]
                let rj2 = rj * rj
                for k in j..<d {
                    let rk = row[k]
                    gram[j][k] = gram[j][k] + rj * rk
                    gram2[j][k] = gram2[j][k] + rj2 * (rk * rk)
                }
            }
        }
        for j in 0..<d {
            for k in 0..<j {
                gram[j][k] = gram[k][j]
                gram2[j][k] = gram2[k][j]
            }
        }
        var empirical = gram
        var traceSum = 0.0
        for j in 0..<d {
            for k in 0..<d { empirical[j][k] = gram[j][k] / nD }
            traceSum = traceSum + gram[j][j] / nD
        }
        let mu = traceSum / dD
        var betaRaw = 0.0
        var deltaRaw = 0.0
        for j in 0..<d {
            for k in 0..<d {
                betaRaw = betaRaw + gram2[j][k]
                deltaRaw = deltaRaw + gram[j][k] * gram[j][k]
            }
        }
        deltaRaw = deltaRaw / (nD * nD)
        var beta = (betaRaw / nD - deltaRaw) / (dD * nD)
        let delta = (deltaRaw - 2.0 * mu * traceSum + dD * (mu * mu)) / dD
        beta = min(beta, delta)
        let lambda = delta == 0.0 ? 0.0 : beta / delta
        return (empirical, mu, lambda)
    }
}
