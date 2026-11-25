package com.mahmoudalyosify.auditor.model

import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.Posture
import com.mahmoudalyosify.auditor.core.QualityIssue
import com.mahmoudalyosify.auditor.core.RecordingLabel
import com.mahmoudalyosify.auditor.core.SampleSource
import java.util.UUID

/** What an enrollment session records. */
sealed class EnrollmentKind {
    /** New person: right hand then left hand, 3 rounds (sitting, standing, fresh grip) × 5 patterns. */
    data object NewPerson : EnrollmentKind()

    /** Replace the enrollment samples of an existing person (same rounds as a new person). */
    data class Reenroll(val personID: String, val name: String) : EnrollmentKind()

    /** Accessory prototype for an existing person: 5 patterns per hand. */
    data class AccessoryEnroll(val personID: String, val name: String, val accessory: Accessory) : EnrollmentKind()
}

/** One pattern slot of an enrollment session. */
data class EnrollmentSlot(val hand: Hand, val posture: Posture, val round: Int, val index: Int)

/**
 * Pure state machine of the enrollment screen (SPEC 12):
 * name → checklist → recording (slots, failed patterns repeat) → summary.
 */
class EnrollmentFlow(
    val kind: EnrollmentKind,
    personID: String = UUID.randomUUID().toString().uppercase(),
    val sessionID: String = UUID.randomUUID().toString().uppercase(),
) {
    enum class Step { name, checklist, recording, summary, cancelled }

    enum class ChecklistItem { watchOff, ringsOff, braceletsOff, wearingAccessory }

    companion object {
        const val patternsPerRound = 5
        val fullPostures = listOf(Posture.sitting, Posture.standing, Posture.freshGrip)
        val hands = listOf(Hand.right, Hand.left)

        /** After this many consecutive failures on one slot, wait for the user before retrying. */
        const val pauseAfterFailures = 3

        fun fullSlots(): List<EnrollmentSlot> {
            val out = mutableListOf<EnrollmentSlot>()
            var round = 0
            for (hand in hands) {
                for (posture in fullPostures) {
                    for (i in 0 until patternsPerRound) out.add(EnrollmentSlot(hand, posture, round, i))
                    round += 1
                }
            }
            return out
        }

        fun accessorySlots(): List<EnrollmentSlot> {
            val out = mutableListOf<EnrollmentSlot>()
            hands.forEachIndexed { round, hand ->
                for (i in 0 until patternsPerRound) out.add(EnrollmentSlot(hand, Posture.unspecified, round, i))
            }
            return out
        }
    }

    val personID: String
    val slots: List<EnrollmentSlot>

    var step: Step
        private set
    var name: String
        private set
    var checked: Set<ChecklistItem> = emptySet()
        private set
    var slotIndex = 0
        private set
    var failuresOnSlot = 0
        private set
    var totalFailures = 0
        private set
    var lastMessage: String? = null
        private set
    var lastIssues: List<QualityIssue> = emptyList()
        private set

    /** True when the next pattern needs a tap first (round start, fresh grip, repeated failures). */
    var awaitingConfirmation = true
        private set

    init {
        when (kind) {
            EnrollmentKind.NewPerson -> {
                this.personID = personID
                name = ""
                step = Step.name
                slots = fullSlots()
            }
            is EnrollmentKind.Reenroll -> {
                this.personID = kind.personID
                name = kind.name
                step = Step.checklist
                slots = fullSlots()
            }
            is EnrollmentKind.AccessoryEnroll -> {
                this.personID = kind.personID
                name = kind.name
                step = Step.checklist
                slots = accessorySlots()
            }
        }
    }

    // Derived

    val accessory: Accessory
        get() = (kind as? EnrollmentKind.AccessoryEnroll)?.accessory ?: Accessory.none

    val sampleSource: SampleSource
        get() = if (kind is EnrollmentKind.AccessoryEnroll) SampleSource.accessory else SampleSource.enrollment

    val checklistItems: List<ChecklistItem>
        get() = if (kind is EnrollmentKind.AccessoryEnroll) listOf(ChecklistItem.wearingAccessory)
        else listOf(ChecklistItem.watchOff, ChecklistItem.ringsOff, ChecklistItem.braceletsOff)

    val checklistComplete: Boolean get() = checklistItems.all { it in checked }

    val currentSlot: EnrollmentSlot? get() = slots.getOrNull(slotIndex)
    val acceptedCount: Int get() = slotIndex
    val total: Int get() = slots.size
    val progress: Double get() = if (slots.isEmpty()) 1.0 else slotIndex.toDouble() / slots.size
    val roundCount: Int get() = (slots.maxOfOrNull { it.round } ?: -1) + 1

    /** The label recorded with the current slot's pattern. */
    fun label(slot: EnrollmentSlot): RecordingLabel =
        RecordingLabel(personID = personID, personName = name, hand = slot.hand, accessory = accessory,
            posture = slot.posture, sessionID = sessionID)

    /** Instruction for the current round. */
    val instruction: String
        get() {
            val slot = currentSlot ?: return "Done."
            val hand = if (slot.hand == Hand.right) "right" else "left"
            return when (slot.posture) {
                Posture.sitting ->
                    "Sit down and hold the phone in your $hand hand the way you usually do. Keep still while it vibrates."
                Posture.standing -> "Stand up and hold the phone in your $hand hand. Keep still while it vibrates."
                Posture.freshGrip ->
                    "Before each pattern, put the phone down and pick it up again with your $hand hand."
                Posture.unspecified ->
                    "Wear your ${accessory.rawValue} and hold the phone in your $hand hand. Keep still while it vibrates."
            }
        }

    val roundTitle: String
        get() {
            val slot = currentSlot ?: return "Finished"
            val hand = if (slot.hand == Hand.right) "Right hand" else "Left hand"
            val posture = when (slot.posture) {
                Posture.sitting -> "sitting"
                Posture.standing -> "standing"
                Posture.freshGrip -> "fresh grip"
                Posture.unspecified -> accessory.rawValue
            }
            return "$hand · $posture · round ${slot.round + 1} of $roundCount"
        }

    // Transitions

    /** Accepts a non-empty name and moves to the checklist. */
    fun submitName(raw: String): Boolean {
        if (step != Step.name) return false
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return false
        name = trimmed
        step = Step.checklist
        return true
    }

    fun toggle(item: ChecklistItem) {
        checked = if (item in checked) checked - item else checked + item
    }

    fun setChecked(item: ChecklistItem, on: Boolean) {
        checked = if (on) checked + item else checked - item
    }

    /** Moves to recording once every checklist item is ticked. */
    fun confirmChecklist(): Boolean {
        if (step != Step.checklist || !checklistComplete) return false
        step = if (slots.isEmpty()) Step.summary else Step.recording
        awaitingConfirmation = true
        return true
    }

    /** The user is ready for the next pattern. */
    fun confirmReady() {
        if (step != Step.recording) return
        awaitingConfirmation = false
    }

    /** Result of one pattern for the current slot: a pass advances, a failure repeats it. */
    fun submit(passed: Boolean, issues: List<QualityIssue>) {
        if (step != Step.recording) return
        val slot = currentSlot ?: return
        if (passed) {
            lastMessage = null
            lastIssues = emptyList()
            failuresOnSlot = 0
            slotIndex += 1
            val next = currentSlot
            if (next != null) {
                awaitingConfirmation = next.round != slot.round || next.posture == Posture.freshGrip
            } else {
                step = Step.summary
                awaitingConfirmation = false
            }
        } else {
            failuresOnSlot += 1
            totalFailures += 1
            lastIssues = issues
            lastMessage = issues.firstOrNull()?.message ?: "Try again"
            awaitingConfirmation = slot.posture == Posture.freshGrip || failuresOnSlot % pauseAfterFailures == 0
        }
    }

    fun cancel() {
        step = Step.cancelled
        awaitingConfirmation = false
    }
}
