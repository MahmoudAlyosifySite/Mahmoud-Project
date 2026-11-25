import CryptoKit
import Foundation
import Security
import AuditorCore

/// Supplies the 256-bit key that encrypts every stored file.
protocol KeyProvider {
    func key() throws -> SymmetricKey
}

enum StoreError: Error, LocalizedError {
    case keychain(OSStatus)
    case badKey
    case sealFailed

    var errorDescription: String? {
        switch self {
        case .keychain(let status): return "Keychain error \(status)"
        case .badKey: return "The stored encryption key is invalid"
        case .sealFailed: return "Encryption failed"
        }
    }
}

/// Keeps one random 256-bit key in the Keychain, readable only while the device is
/// unlocked and never migrated to another device or backup.
final class KeychainKeyProvider: KeyProvider {
    private let service: String
    private let account: String
    private var cached: SymmetricKey?
    private let lock = NSLock()

    init(service: String = "com.mahmoudalyosify.auditor.storage", account: String = "data-key") {
        self.service = service
        self.account = account
    }

    func key() throws -> SymmetricKey {
        lock.lock()
        defer { lock.unlock() }
        if let cached = cached { return cached }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecSuccess {
            guard let data = item as? Data, data.count == 32 else { throw StoreError.badKey }
            let key = SymmetricKey(data: data)
            cached = key
            return key
        }
        guard status == errSecItemNotFound else { throw StoreError.keychain(status) }
        let key = SymmetricKey(size: .bits256)
        let data = key.withUnsafeBytes { Data($0) }
        let add: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            kSecValueData as String: data,
        ]
        let addStatus = SecItemAdd(add as CFDictionary, nil)
        guard addStatus == errSecSuccess else { throw StoreError.keychain(addStatus) }
        cached = key
        return key
    }
}

/// A fixed key (tests).
struct StaticKeyProvider: KeyProvider {
    let symmetricKey: SymmetricKey
    func key() throws -> SymmetricKey { symmetricKey }
}

/// Files under one directory, each encrypted with AES-GCM and written with complete
/// file protection. JSON uses `RecordingCoding` (ISO-8601 dates, sorted keys).
final class EncryptedStore {
    static let peopleFile = "people.json"
    static let samplesFile = "samples.json"
    static let embeddingsFile = "embeddings.json"
    static let calibrationFile = "calibration.json"
    static let settingsFile = "settings.json"
    static let checksFile = "checks.json"
    static let recordingsFolder = "recordings"

    let directory: URL
    private let keyProvider: KeyProvider
    private let fileManager = FileManager.default

    init(directory: URL, keyProvider: KeyProvider) throws {
        self.directory = directory
        self.keyProvider = keyProvider
        let attributes: [FileAttributeKey: Any] = [.protectionKey: FileProtectionType.complete]
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: attributes)
        try fileManager.createDirectory(at: directory.appendingPathComponent(Self.recordingsFolder, isDirectory: true),
                                        withIntermediateDirectories: true, attributes: attributes)
    }

    /// Application Support/Auditor.
    static func defaultDirectory() throws -> URL {
        let base = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
        return base.appendingPathComponent("Auditor", isDirectory: true)
    }

    // MARK: Encryption

    func seal(_ plain: Data) throws -> Data {
        let box = try AES.GCM.seal(plain, using: try keyProvider.key())
        guard let combined = box.combined else { throw StoreError.sealFailed }
        return combined
    }

    func open(_ sealed: Data) throws -> Data {
        let box = try AES.GCM.SealedBox(combined: sealed)
        return try AES.GCM.open(box, using: try keyProvider.key())
    }

    // MARK: Generic values

    func url(for name: String) -> URL {
        directory.appendingPathComponent(name, isDirectory: false)
    }

    func writeData(_ data: Data, to name: String) throws {
        try seal(data).write(to: url(for: name), options: [.atomic, .completeFileProtection])
    }

    func readData(_ name: String) throws -> Data? {
        let fileURL = url(for: name)
        guard fileManager.fileExists(atPath: fileURL.path) else { return nil }
        return try open(Data(contentsOf: fileURL))
    }

    func write<T: Encodable>(_ value: T, to name: String) throws {
        try writeData(try RecordingCoding.encoder().encode(value), to: name)
    }

    func read<T: Decodable>(_ type: T.Type, from name: String) throws -> T? {
        guard let data = try readData(name) else { return nil }
        return try RecordingCoding.decoder().decode(type, from: data)
    }

    func remove(_ name: String) throws {
        let fileURL = url(for: name)
        if fileManager.fileExists(atPath: fileURL.path) {
            try fileManager.removeItem(at: fileURL)
        }
    }

    // MARK: Recordings

    private func recordingName(_ id: UUID) -> String {
        "\(Self.recordingsFolder)/\(id.uuidString).json"
    }

    func writeRecording(_ recording: Recording) throws {
        try write(recording, to: recordingName(recording.id))
    }

    func readRecording(id: UUID) throws -> Recording? {
        try read(Recording.self, from: recordingName(id))
    }

    func deleteRecording(id: UUID) throws {
        try remove(recordingName(id))
    }

    func recordingIDs() -> [UUID] {
        let folder = directory.appendingPathComponent(Self.recordingsFolder, isDirectory: true)
        let names = (try? fileManager.contentsOfDirectory(atPath: folder.path)) ?? []
        return names.compactMap { name -> UUID? in
            guard name.hasSuffix(".json") else { return nil }
            return UUID(uuidString: String(name.dropLast(5)))
        }
    }

    func allRecordings() -> [Recording] {
        recordingIDs().compactMap { try? readRecording(id: $0) }.sorted { $0.createdAt < $1.createdAt }
    }

    /// Removes every stored file (the key stays in the Keychain).
    func deleteAll() throws {
        if fileManager.fileExists(atPath: directory.path) {
            try fileManager.removeItem(at: directory)
        }
        let attributes: [FileAttributeKey: Any] = [.protectionKey: FileProtectionType.complete]
        try fileManager.createDirectory(at: directory.appendingPathComponent(Self.recordingsFolder, isDirectory: true),
                                        withIntermediateDirectories: true, attributes: attributes)
    }
}
