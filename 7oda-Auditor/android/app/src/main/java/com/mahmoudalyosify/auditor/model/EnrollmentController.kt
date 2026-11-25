package com.mahmoudalyosify.auditor.model

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.mahmoudalyosify.auditor.core.QualityMode
import com.mahmoudalyosify.auditor.core.RecordingPurpose
import com.mahmoudalyosify.auditor.core.SampleSource
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** Drives one enrollment session: records the slots of an [EnrollmentFlow] and commits them. */
class EnrollmentController(kind: EnrollmentKind, private val model: AppModel, private val scope: CoroutineScope) {
    private val flowValue = EnrollmentFlow(kind)

    /** Bumped after every change of the flow, so Compose re-reads it. */
    var version by mutableIntStateOf(0)
        private set

    /** The flow; reading it subscribes the caller (a composable) to the flow's changes. */
    val flow: EnrollmentFlow
        get() {
            check(version >= 0)
            return flowValue
        }

    var accepted: List<Pair<EnrollmentSlot, AnalyzedPattern>> by mutableStateOf(emptyList())
        private set
    var isRecording by mutableStateOf(false)
        private set
    var lastAnalysis: AnalyzedPattern? by mutableStateOf(null)
        private set
    var committed by mutableStateOf(false)
        private set
    var errorMessage: String? by mutableStateOf(null)

    private var job: Job? = null

    /** Applies a change to the flow and notifies Compose. */
    fun update(change: EnrollmentFlow.() -> Unit) {
        flowValue.change()
        version += 1
    }

    fun startOrContinue() {
        if (isRecording || flow.step != EnrollmentFlow.Step.recording) return
        errorMessage = null
        update { confirmReady() }
        job = scope.launch { runLoop() }
    }

    private suspend fun runLoop() {
        isRecording = true
        try {
            while (flow.step == EnrollmentFlow.Step.recording && !flow.awaitingConfirmation) {
                val slot = flow.currentSlot ?: break
                delay(600)
                val purpose = if (flow.sampleSource == SampleSource.accessory) RecordingPurpose.accessory
                else RecordingPurpose.enrollment
                val a = model.recordAndAnalyze(purpose, flow.label(slot), QualityMode.handheld)
                lastAnalysis = a
                apply(a, slot)
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            errorMessage = e.message ?: e.javaClass.simpleName
        } finally {
            isRecording = false
        }
    }

    fun apply(a: AnalyzedPattern, slot: EnrollmentSlot) {
        if (flow.currentSlot != slot) return
        if (a.passed) accepted = accepted + (slot to a)
        update { submit(a.passed, a.quality.issues) }
    }

    fun cancel() {
        job?.cancel()
        job = null
        if (flow.step != EnrollmentFlow.Step.summary || !committed) update { cancel() }
    }

    fun commit() {
        if (committed || flow.step != EnrollmentFlow.Step.summary) return
        model.commitEnrollment(flow, accepted)
        committed = true
    }
}
