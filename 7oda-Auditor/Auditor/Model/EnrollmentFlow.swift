import Foundation
import AuditorCore

/// What an enrollment session records.
enum EnrollmentKind: Equatable {
    /// New person: right hand then left hand, 3 rounds (sitting, standing, fresh grip) × 5 patterns.
    case newPerson
    /// Replace the enrollment samples of an existing person (same rounds as a new person).
    case reenroll(personID: String, name: String)
    /// Accessory prototype for an existing person: 5 patterns per hand.
    case accessory(personID: String, name: String, accessory: Accessory)
}

/// One pattern slot of an enrollment session.
struct EnrollmentSlot: Equatable {
    var hand: Hand
    var posture: Posture
    var round: Int
    var index: Int
}

/// Pure state machine of the Enroll sheet (SPEC 12):
/// name → checklist → recording (slots, failed patterns repeat) → summary.
struct EnrollmentFlow {
    enum Step: Equatable { case name, checklist, recording, summary, cancelled }

    enum ChecklistItem: String, CaseIterable, Identifiable {
        case watchOff, ringsOff, braceletsOff, wearingAccessory
        var id: String { rawValue }
    }

    static let patternsPerRound = 5
    static let fullPostures: [Posture] = [.sitting, .standing, .freshGrip]
    static let hands: [Hand] = [.right, .left]
    /// After this many consecutive failures on one slot, wait for the user before retrying.
    static let pauseAfterFailures = 3

    let kind: EnrollmentKind
    let personID: String
    let sessionID: String
    let slots: [EnrollmentSlot]

    private(set) var step: Step
    private(set) var name: String
    private(set) var checked: Set<ChecklistItem> = []
    private(set) var slotIndex = 0
    private(set) var failuresOnSlot = 0
    private(set) var totalFailures = 0
    private(set) var lastMessage: String?
    private(set) var lastIssues: [QualityIssue] = []
    /// True when the next pattern needs a tap first (round start, fresh grip, repeated failures).
    private(set) var awaitingConfirmation = true

    init(kind: EnrollmentKind, personID: String = UUID().uuidString, sessionID: String = UUID().uuidString) {
        self.kind = kind
        self.sessionID = sessionID
        switch kind {
        case .newPerson:
            self.personID = personID
            name = ""
            step = .name
            slots = EnrollmentFlow.fullSlots()
        case .reenroll(let id, let existingName):
            self.personID = id
            name = existingName
            step = .checklist
            slots = EnrollmentFlow.fullSlots()
        case .accessory(let id, let existingName, _):
            self.personID = id
            name = existingName
            step = .checklist
            slots = EnrollmentFlow.accessorySlots()
        }
    }

    static func fullSlots() -> [EnrollmentSlot] {
        var out: [EnrollmentSlot] = []
        var round = 0
        for hand in hands {
            for posture in fullPostures {
                for i in 0..<patternsPerRound {
                    out.append(EnrollmentSlot(hand: hand, posture: posture, round: round, index: i))
                }
                round += 1
            }
        }
        return out
    }

    static func accessorySlots() -> [EnrollmentSlot] {
        var out: [EnrollmentSlot] = []
        for (round, hand) in hands.enumerated() {
            for i in 0..<patternsPerRound {
                out.append(EnrollmentSlot(hand: hand, posture: .unspecified, round: round, index: i))
            }
        }
        return out
    }

    // MARK: Derived

    var accessory: Accessory {
        if case .accessory(_, _, let a) = kind { return a }
        return .none
    }

    var sampleSource: SampleSource {
        if case .accessory = kind { return .accessory }
        return .enrollment
    }

    var checklistItems: [ChecklistItem] {
        if case .accessory = kind { return [.wearingAccessory] }
        return [.watchOff, .ringsOff, .braceletsOff]
    }

    var checklistComplete: Bool { checklistItems.allSatisfy { checked.contains($0) } }

    var currentSlot: EnrollmentSlot? { slotIndex < slots.count ? slots[slotIndex] : nil }
    var acceptedCount: Int { slotIndex }
    var total: Int { slots.count }
    var progress: Double { slots.isEmpty ? 1 : Double(slotIndex) / Double(slots.count) }
    var roundCount: Int { (slots.map { $0.round }.max() ?? -1) + 1 }

    /// The label recorded with the current slot's pattern.
    func label(for slot: EnrollmentSlot) -> RecordingLabel {
        RecordingLabel(personID: personID, personName: name, hand: slot.hand, accessory: accessory,
                       posture: slot.posture, sessionID: sessionID)
    }

    /// Instruction for the current round.
    var instruction: String {
        guard let slot = currentSlot else { return "Done." }
        let hand = slot.hand == .right ? "right" : "left"
        switch slot.posture {
        case .sitting:
            return "Sit down and hold the phone in your \(hand) hand the way you usually do. Keep still while it vibrates."
        case .standing:
            return "Stand up and hold the phone in your \(hand) hand. Keep still while it vibrates."
        case .freshGrip:
            return "Before each pattern, put the phone down and pick it up again with your \(hand) hand."
        case .unspecified:
            return "Wear your \(accessory.rawValue) and hold the phone in your \(hand) hand. Keep still while it vibrates."
        }
    }

    var roundTitle: String {
        guard let slot = currentSlot else { return "Finished" }
        let hand = slot.hand == .right ? "Right hand" : "Left hand"
        let posture: String
        switch slot.posture {
        case .sitting: posture = "sitting"
        case .standing: posture = "standing"
        case .freshGrip: posture = "fresh grip"
        case .unspecified: posture = accessory.rawValue
        }
        return "\(hand) · \(posture) · round \(slot.round + 1) of \(roundCount)"
    }

    // MARK: Transitions

    /// Accepts a non-empty name and moves to the checklist.
    @discardableResult
    mutating func submitName(_ raw: String) -> Bool {
        guard step == .name else { return false }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        name = trimmed
        step = .checklist
        return true
    }

    mutating func toggle(_ item: ChecklistItem) {
        if checked.contains(item) { checked.remove(item) } else { checked.insert(item) }
    }

    mutating func setChecked(_ item: ChecklistItem, _ on: Bool) {
        if on { checked.insert(item) } else { checked.remove(item) }
    }

    /// Moves to recording once every checklist item is ticked.
    @discardableResult
    mutating func confirmChecklist() -> Bool {
        guard step == .checklist, checklistComplete else { return false }
        step = slots.isEmpty ? .summary : .recording
        awaitingConfirmation = true
        return true
    }

    /// The user is ready for the next pattern.
    mutating func confirmReady() {
        guard step == .recording else { return }
        awaitingConfirmation = false
    }

    /// Result of one pattern for the current slot: a pass advances, a failure repeats it.
    mutating func submit(passed: Bool, issues: [QualityIssue]) {
        guard step == .recording, let slot = currentSlot else { return }
        if passed {
            lastMessage = nil
            lastIssues = []
            failuresOnSlot = 0
            slotIndex += 1
            if let next = currentSlot {
                awaitingConfirmation = next.round != slot.round || next.posture == .freshGrip
            } else {
                step = .summary
                awaitingConfirmation = false
            }
        } else {
            failuresOnSlot += 1
            totalFailures += 1
            lastIssues = issues
            lastMessage = issues.first?.message ?? "Try again"
            awaitingConfirmation = slot.posture == .freshGrip || failuresOnSlot % Self.pauseAfterFailures == 0
        }
    }

    mutating func cancel() {
        step = .cancelled
        awaitingConfirmation = false
    }
}
