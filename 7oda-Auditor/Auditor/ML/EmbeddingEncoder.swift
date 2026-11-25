import CoreML
import Foundation
import AuditorCore

enum EncoderError: Error, LocalizedError {
    case modelMissing
    case badOutput(String)
    case badInput(Int)

    var errorDescription: String? {
        switch self {
        case .modelMissing: return "The v1 model (HandEncoder) is missing from the app bundle."
        case .badOutput(let name): return "The v1 model returned no usable '\(name)' output."
        case .badInput(let count): return "The v1 model input has \(count) values instead of 2560."
        }
    }
}

/// Output of the v1 encoder for one pattern.
struct EncoderOutput {
    /// L2-normalized embedding (128 values), renormalized in Double.
    var embedding: [Double]
    /// Hand logits: right, left, twoHanded (logged only; the decision uses its own hand model).
    var handLogits: [Double]
}

/// Wraps the bundled `HandEncoder` Core ML model through the generic `MLModel` API
/// (input `x` float32 [1, 10, 256]; outputs `embedding` [1, 128] and `hand` [1, 3]).
final class EmbeddingEncoder {
    static let resourceName = "HandEncoder"
    static let embeddingSize = 128

    let model: MLModel
    /// Identifies the model that produced stored embeddings; a change triggers recomputation.
    let version: String

    init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: Self.resourceName, withExtension: "mlmodelc") else {
            throw EncoderError.modelMissing
        }
        let configuration = MLModelConfiguration()
        #if targetEnvironment(simulator)
        configuration.computeUnits = .cpuOnly
        #else
        configuration.computeUnits = .all
        #endif
        model = try MLModel(contentsOf: url, configuration: configuration)
        let metadata = model.modelDescription.metadata
        let declared = metadata[MLModelMetadataKey.versionString] as? String ?? ""
        version = "HandEncoder/1" + (declared.isEmpty ? "" : "/" + declared)
    }

    /// Encodes one processed pattern (input built by `EncoderInput.build`, SPEC 14.1).
    func encode(_ processed: ProcessedSignal) throws -> EncoderOutput {
        try encode(input: EncoderInput.build(processed))
    }

    /// Encodes a raw input tensor, row-major [channel][u], 2560 values.
    func encode(input: [Float]) throws -> EncoderOutput {
        let channels = EncoderInput.channels
        let length = EncoderInput.length
        guard input.count == channels * length else { throw EncoderError.badInput(input.count) }
        let array = try MLMultiArray(shape: [1, NSNumber(value: channels), NSNumber(value: length)],
                                     dataType: .float32)
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
            let sc = strides[1], su = strides[2]
            for c in 0..<channels {
                for u in 0..<length {
                    buffer[c * sc + u * su] = input[c * length + u]
                }
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: array)])
        let result = try model.prediction(from: provider)
        guard let embeddingArray = result.featureValue(for: "embedding")?.multiArrayValue else {
            throw EncoderError.badOutput("embedding")
        }
        var embedding = Self.values(embeddingArray)
        guard embedding.count == Self.embeddingSize, embedding.allSatisfy({ $0.isFinite }) else {
            throw EncoderError.badOutput("embedding")
        }
        let norm = embedding.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
        if norm > 1e-12 { embedding = embedding.map { $0 / norm } }
        let hand = result.featureValue(for: "hand")?.multiArrayValue.map { Self.values($0) } ?? []
        return EncoderOutput(embedding: embedding, handLogits: hand)
    }

    /// All values of a multi-array in linear (row-major) order, whatever its data type.
    private static func values(_ array: MLMultiArray) -> [Double] {
        (0..<array.count).map { array[$0].doubleValue }
    }
}
