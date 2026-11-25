import CryptoKit
import XCTest
@testable import Auditor
import AuditorCore

final class EncryptedStoreTests: XCTestCase {
    private var directory: URL!

    override func setUp() {
        super.setUp()
        directory = TestSupport.temporaryDirectory()
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
        super.tearDown()
    }

    func testRoundTripIsEncryptedOnDisk() throws {
        let key = SymmetricKey(size: .bits256)
        let store = try EncryptedStore(directory: directory, keyProvider: StaticKeyProvider(symmetricKey: key))
        let people = [Person(name: "Zainab Unique-Name"), Person(name: "Omar")]
        try store.write(people, to: EncryptedStore.peopleFile)

        let back = try store.read([Person].self, from: EncryptedStore.peopleFile)
        XCTAssertEqual(back?.map { $0.name }, people.map { $0.name })
        XCTAssertEqual(back?.map { $0.id }, people.map { $0.id })

        let raw = try Data(contentsOf: directory.appendingPathComponent(EncryptedStore.peopleFile))
        XCTAssertNil(raw.range(of: Data("Zainab Unique-Name".utf8)), "file must not contain plaintext")
        XCTAssertGreaterThan(raw.count, 28, "nonce + tag + ciphertext")

        XCTAssertNil(try store.read([Person].self, from: "missing.json"))
    }

    func testWrongKeyCannotRead() throws {
        let store = try EncryptedStore(directory: directory,
                                       keyProvider: StaticKeyProvider(symmetricKey: SymmetricKey(size: .bits256)))
        try store.write(AppSettings(), to: EncryptedStore.settingsFile)
        let other = try EncryptedStore(directory: directory,
                                       keyProvider: StaticKeyProvider(symmetricKey: SymmetricKey(size: .bits256)))
        XCTAssertThrowsError(try other.read(AppSettings.self, from: EncryptedStore.settingsFile))
    }

    func testRecordingsAndSettingsRoundTrip() throws {
        let store = try EncryptedStore(directory: directory,
                                       keyProvider: StaticKeyProvider(symmetricKey: SymmetricKey(size: .bits256)))
        let recording = SyntheticHands.recording(seed: 7, profile: SyntheticHands.profile(person: 0, hand: .right))
        try store.writeRecording(recording)
        XCTAssertEqual(store.recordingIDs(), [recording.id])
        let back = try XCTUnwrap(store.readRecording(id: recording.id))
        XCTAssertEqual(back.id, recording.id)
        XCTAssertEqual(back.challenge, recording.challenge)
        XCTAssertEqual(back.accel, recording.accel)
        XCTAssertEqual(back.motion, recording.motion)
        XCTAssertEqual(back.hapticStart, recording.hapticStart)

        var settings = AppSettings()
        settings.recognizer = .v0
        settings.maxPatterns = 5
        settings.v1NameMargin = 0.05
        try store.write(settings, to: EncryptedStore.settingsFile)
        XCTAssertEqual(try store.read(AppSettings.self, from: EncryptedStore.settingsFile), settings)

        try store.deleteAll()
        XCTAssertTrue(store.recordingIDs().isEmpty)
        XCTAssertNil(try store.read(AppSettings.self, from: EncryptedStore.settingsFile))
    }

    func testSettingsDecodeWithMissingKeys() throws {
        let data = Data(#"{"recognizer":"v0","maxPatterns":4}"#.utf8)
        let s = try RecordingCoding.decoder().decode(AppSettings.self, from: data)
        XCTAssertEqual(s.recognizer, .v0)
        XCTAssertEqual(s.maxPatterns, 4)
        XCTAssertEqual(s.nameMargin, IdentifierConfig.default.nameMargin)
        XCTAssertEqual(s.v1NameMargin, EmbeddingConfig.default.nameMargin)
    }
}
