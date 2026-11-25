import XCTest
@testable import Auditor
import AuditorCore

final class EncoderTests: XCTestCase {
    func testBundledModelLoadsAndReturnsAUnitEmbedding() throws {
        let encoder = try EmbeddingEncoder(bundle: .main)
        XCTAssertTrue(encoder.version.hasPrefix("HandEncoder/"))

        let recording = SyntheticHands.recording(seed: 21, profile: SyntheticHands.profile(person: 0, hand: .right))
        let processed = SignalProcessor.process(recording)
        let input = EncoderInput.build(processed)
        XCTAssertEqual(input.count, EncoderInput.channels * EncoderInput.length)

        let out = try encoder.encode(processed)
        XCTAssertEqual(out.embedding.count, 128)
        XCTAssertTrue(out.embedding.allSatisfy { $0.isFinite })
        let norm = out.embedding.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1.0, accuracy: 1e-6)
        XCTAssertEqual(out.handLogits.count, 3)

        // Deterministic for the same input.
        let again = try encoder.encode(input: input)
        for (a, b) in zip(out.embedding, again.embedding) {
            XCTAssertEqual(a, b, accuracy: 1e-6)
        }
    }

    func testDifferentHandsGiveDifferentEmbeddings() throws {
        let encoder = try EmbeddingEncoder(bundle: .main)
        let a = try encoder.encode(SignalProcessor.process(
            SyntheticHands.recording(seed: 31, profile: SyntheticHands.profile(person: 0, hand: .right))))
        let b = try encoder.encode(SignalProcessor.process(
            SyntheticHands.recording(seed: 32, profile: SyntheticHands.profile(person: 1, hand: .left))))
        let cosine = zip(a.embedding, b.embedding).reduce(0.0) { $0 + $1.0 * $1.1 }
        XCTAssertLessThan(cosine, 0.999999, "embeddings should depend on the input")
    }

    func testRejectsWrongInputSize() throws {
        let encoder = try EmbeddingEncoder(bundle: .main)
        XCTAssertThrowsError(try encoder.encode(input: [Float](repeating: 0, count: 10)))
    }
}
