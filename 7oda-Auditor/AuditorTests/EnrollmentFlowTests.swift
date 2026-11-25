import XCTest
@testable import Auditor
import AuditorCore

final class EnrollmentFlowTests: XCTestCase {
    func testNewPersonFlowWithAFailedPattern() {
        var flow = EnrollmentFlow(kind: .newPerson)
        XCTAssertEqual(flow.step, .name)
        XCTAssertEqual(flow.total, 30)
        XCTAssertEqual(flow.roundCount, 6)

        XCTAssertFalse(flow.submitName("   "))
        XCTAssertEqual(flow.step, .name)
        XCTAssertTrue(flow.submitName("  Mona "))
        XCTAssertEqual(flow.name, "Mona")
        XCTAssertEqual(flow.step, .checklist)

        XCTAssertFalse(flow.confirmChecklist(), "all accessories must be off first")
        flow.toggle(.watchOff)
        flow.toggle(.ringsOff)
        XCTAssertFalse(flow.confirmChecklist())
        flow.setChecked(.braceletsOff, true)
        XCTAssertTrue(flow.confirmChecklist())
        XCTAssertEqual(flow.step, .recording)
        XCTAssertTrue(flow.awaitingConfirmation)
        XCTAssertEqual(flow.currentSlot, EnrollmentSlot(hand: .right, posture: .sitting, round: 0, index: 0))

        // Confirmation gates the runner (the next recording), not the state machine.
        flow.confirmReady()
        XCTAssertFalse(flow.awaitingConfirmation)
        for _ in 0..<4 { flow.submit(passed: true, issues: []) }
        XCTAssertEqual(flow.acceptedCount, 4)
        XCTAssertFalse(flow.awaitingConfirmation)

        // A failed pattern repeats the same slot with the SPEC 6.3 message.
        flow.submit(passed: false, issues: [.moving, .weakSignal])
        XCTAssertEqual(flow.acceptedCount, 4)
        XCTAssertEqual(flow.currentSlot?.index, 4)
        XCTAssertEqual(flow.lastMessage, "Hold still")
        XCTAssertEqual(flow.failuresOnSlot, 1)
        XCTAssertEqual(flow.totalFailures, 1)
        XCTAssertFalse(flow.awaitingConfirmation)

        // Three failures in a row pause for the user.
        flow.submit(passed: false, issues: [.resting])
        flow.submit(passed: false, issues: [.resting])
        XCTAssertTrue(flow.awaitingConfirmation)
        XCTAssertEqual(flow.lastMessage, "Hold the phone in one hand")
        flow.confirmReady()

        // The repeat passes; the next round (standing) needs a confirmation.
        flow.submit(passed: true, issues: [])
        XCTAssertNil(flow.lastMessage)
        XCTAssertEqual(flow.acceptedCount, 5)
        XCTAssertEqual(flow.currentSlot?.posture, .standing)
        XCTAssertTrue(flow.awaitingConfirmation)
        flow.confirmReady()
        for _ in 0..<5 { flow.submit(passed: true, issues: []) }

        // Fresh grip: every pattern needs a confirmation.
        XCTAssertEqual(flow.currentSlot?.posture, .freshGrip)
        for _ in 0..<5 {
            XCTAssertTrue(flow.awaitingConfirmation)
            flow.confirmReady()
            flow.submit(passed: true, issues: [])
        }

        // Left hand, then the summary.
        XCTAssertEqual(flow.currentSlot?.hand, .left)
        XCTAssertEqual(flow.currentSlot?.posture, .sitting)
        while flow.step == .recording {
            flow.confirmReady()
            flow.submit(passed: true, issues: [])
        }
        XCTAssertEqual(flow.step, .summary)
        XCTAssertEqual(flow.acceptedCount, 30)
        XCTAssertEqual(flow.progress, 1.0, accuracy: 1e-12)
        XCTAssertNil(flow.currentSlot)
        XCTAssertEqual(flow.totalFailures, 3)

        // Results after the end are ignored.
        flow.submit(passed: true, issues: [])
        XCTAssertEqual(flow.acceptedCount, 30)
    }

    func testLabelsCarryPersonHandPostureAndSession() {
        var flow = EnrollmentFlow(kind: .newPerson, personID: "P-1", sessionID: "S-1")
        flow.submitName("Ali")
        let slot = EnrollmentSlot(hand: .left, posture: .standing, round: 4, index: 2)
        let label = flow.label(for: slot)
        XCTAssertEqual(label.personID, "P-1")
        XCTAssertEqual(label.personName, "Ali")
        XCTAssertEqual(label.hand, .left)
        XCTAssertEqual(label.posture, .standing)
        XCTAssertEqual(label.accessory, Accessory.none)
        XCTAssertEqual(label.sessionID, "S-1")
    }

    func testAccessoryFlow() {
        var flow = EnrollmentFlow(kind: .accessory(personID: "X", name: "Sara", accessory: .watch))
        XCTAssertEqual(flow.step, .checklist)
        XCTAssertEqual(flow.total, 10)
        XCTAssertEqual(flow.checklistItems, [.wearingAccessory])
        XCTAssertEqual(flow.sampleSource, .accessory)
        XCTAssertEqual(flow.accessory, .watch)
        flow.toggle(.wearingAccessory)
        XCTAssertTrue(flow.confirmChecklist())
        XCTAssertEqual(flow.currentSlot?.posture, .unspecified)
        XCTAssertEqual(flow.label(for: flow.currentSlot!).accessory, .watch)
        var hands: [Hand] = []
        while let slot = flow.currentSlot {
            hands.append(slot.hand)
            flow.confirmReady()
            flow.submit(passed: true, issues: [])
        }
        XCTAssertEqual(hands, Array(repeating: .right, count: 5) + Array(repeating: .left, count: 5))
        XCTAssertEqual(flow.step, .summary)
    }

    func testReenrollSkipsNameAndCancel() {
        var flow = EnrollmentFlow(kind: .reenroll(personID: "R", name: "Huda"))
        XCTAssertEqual(flow.step, .checklist)
        XCTAssertEqual(flow.personID, "R")
        XCTAssertEqual(flow.name, "Huda")
        XCTAssertEqual(flow.sampleSource, .enrollment)
        flow.cancel()
        XCTAssertEqual(flow.step, .cancelled)
        flow.submit(passed: true, issues: [])
        XCTAssertEqual(flow.acceptedCount, 0)
    }
}
