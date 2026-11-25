package com.mahmoudalyosify.auditor

import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.Posture
import com.mahmoudalyosify.auditor.core.QualityIssue
import com.mahmoudalyosify.auditor.core.SampleSource
import com.mahmoudalyosify.auditor.model.EnrollmentFlow
import com.mahmoudalyosify.auditor.model.EnrollmentKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class EnrollmentFlowTest {
    private fun readyFlow(): EnrollmentFlow {
        val flow = EnrollmentFlow(EnrollmentKind.NewPerson)
        assertTrue(flow.submitName("  Mona "))
        flow.checklistItems.forEach { flow.setChecked(it, true) }
        assertTrue(flow.confirmChecklist())
        return flow
    }

    @Test
    fun newPersonHasThirtySlotsInSixRounds() {
        val flow = EnrollmentFlow(EnrollmentKind.NewPerson)
        assertEquals(EnrollmentFlow.Step.enterName, flow.step)
        assertEquals(30, flow.total)
        assertEquals(6, flow.roundCount)
        assertEquals(Hand.right, flow.slots.first().hand)
        assertEquals(Hand.left, flow.slots.last().hand)
        assertEquals(Posture.freshGrip, flow.slots[14].posture)
        assertEquals(SampleSource.enrollment, flow.sampleSource)
    }

    @Test
    fun nameAndChecklistGateTheRecording() {
        val flow = EnrollmentFlow(EnrollmentKind.NewPerson)
        assertFalse(flow.submitName("   "))
        assertTrue(flow.submitName("  Mona "))
        assertEquals("Mona", flow.name)
        assertEquals(EnrollmentFlow.Step.checklist, flow.step)
        assertFalse(flow.confirmChecklist())
        flow.toggle(EnrollmentFlow.ChecklistItem.watchOff)
        flow.toggle(EnrollmentFlow.ChecklistItem.ringsOff)
        assertFalse(flow.checklistComplete)
        flow.toggle(EnrollmentFlow.ChecklistItem.braceletsOff)
        assertTrue(flow.confirmChecklist())
        assertEquals(EnrollmentFlow.Step.recording, flow.step)
        assertTrue(flow.awaitingConfirmation)
    }

    @Test
    fun failedPatternsRepeatTheSlotWithTheQualityMessage() {
        val flow = readyFlow()
        flow.confirmReady()
        flow.submit(passed = false, issues = listOf(QualityIssue.moving, QualityIssue.weakSignal))
        assertEquals(0, flow.slotIndex)
        assertEquals("Hold still", flow.lastMessage)
        assertFalse(flow.awaitingConfirmation)
        flow.submit(false, listOf(QualityIssue.weakSignal))
        flow.submit(false, listOf(QualityIssue.weakSignal))
        assertTrue("pause after three failures", flow.awaitingConfirmation)
        assertEquals(3, flow.totalFailures)
        flow.confirmReady()
        flow.submit(true, emptyList())
        assertEquals(1, flow.slotIndex)
        assertNull(flow.lastMessage)
        assertEquals(0, flow.failuresOnSlot)
        assertFalse(flow.awaitingConfirmation)
    }

    @Test
    fun roundsAndFreshGripWaitForTheUserAndTheFlowEndsInTheSummary() {
        val flow = readyFlow()
        flow.confirmReady()
        repeat(4) { flow.submit(true, emptyList()) }
        assertFalse(flow.awaitingConfirmation)
        flow.submit(true, emptyList())
        assertTrue("new round", flow.awaitingConfirmation)
        flow.confirmReady()
        repeat(5) { flow.submit(true, emptyList()) }
        assertEquals(Posture.freshGrip, flow.currentSlot?.posture)
        flow.confirmReady()
        flow.submit(true, emptyList())
        assertTrue("fresh grip waits before every pattern", flow.awaitingConfirmation)
        while (flow.step == EnrollmentFlow.Step.recording) {
            flow.confirmReady()
            flow.submit(true, emptyList())
        }
        assertEquals(EnrollmentFlow.Step.summary, flow.step)
        assertEquals(30, flow.acceptedCount)
        assertEquals(1.0, flow.progress, 0.0)
    }

    @Test
    fun accessoryEnrollmentRecordsFivePerHand() {
        val flow = EnrollmentFlow(EnrollmentKind.AccessoryEnroll("P", "Mona", Accessory.ring))
        assertEquals(EnrollmentFlow.Step.checklist, flow.step)
        assertEquals(10, flow.total)
        assertEquals(listOf(EnrollmentFlow.ChecklistItem.wearingAccessory), flow.checklistItems)
        assertEquals(SampleSource.accessory, flow.sampleSource)
        assertEquals(Accessory.ring, flow.label(flow.slots[0]).accessory)
        assertEquals("P", flow.personID)
        flow.cancel()
        assertEquals(EnrollmentFlow.Step.cancelled, flow.step)
    }
}
