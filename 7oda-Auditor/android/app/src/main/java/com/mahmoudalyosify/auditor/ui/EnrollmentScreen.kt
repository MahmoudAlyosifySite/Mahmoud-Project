package com.mahmoudalyosify.auditor.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.MobileOff
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.Posture
import com.mahmoudalyosify.auditor.model.AppModel
import com.mahmoudalyosify.auditor.model.EnrollmentController
import com.mahmoudalyosify.auditor.model.EnrollmentFlow
import com.mahmoudalyosify.auditor.model.EnrollmentKind

@Composable
fun EnrollmentScreen(model: AppModel, kind: EnrollmentKind, onDone: () -> Unit) {
    SensorSession(model)
    val scope = rememberCoroutineScope()
    val controller = remember(kind) { EnrollmentController(kind, model, scope) }
    DisposableEffect(controller) { onDispose { controller.cancel() } }
    BackHandler {
        controller.cancel()
        onDone()
    }
    // Reading `flow` subscribes each composable to the flow's changes.
    val flow = controller.flow

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        when (flow.step) {
            EnrollmentFlow.Step.name -> NameStep(controller)
            EnrollmentFlow.Step.checklist -> ChecklistStep(model, controller)
            EnrollmentFlow.Step.recording -> RecordingStep(model, controller)
            EnrollmentFlow.Step.summary -> SummaryStep(controller, onDone)
            EnrollmentFlow.Step.cancelled -> FootnoteText("Cancelled")
        }
        if (flow.step != EnrollmentFlow.Step.summary && flow.step != EnrollmentFlow.Step.cancelled) {
            OutlinedButton(onClick = {
                controller.cancel()
                onDone()
            }, modifier = Modifier.fillMaxWidth()) { Text("Cancel") }
        }
    }
}

@Composable
private fun NameStep(controller: EnrollmentController) {
    var nameField by remember { mutableStateOf("") }
    SectionCard("Who is this?") {
        OutlinedTextField(
            value = nameField, onValueChange = { nameField = it }, singleLine = true, label = { Text("Name") },
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words, imeAction = ImeAction.Next),
            keyboardActions = KeyboardActions(onNext = { controller.update { submitName(nameField) } }),
            modifier = Modifier.fillMaxWidth(),
        )
        FootnoteText("The name stays on this phone. Recordings are encrypted.")
    }
    Button(
        onClick = { controller.update { submitName(nameField) } },
        enabled = nameField.isNotBlank(), modifier = Modifier.fillMaxWidth(),
    ) { Text("Continue") }
}

private fun checklistText(item: EnrollmentFlow.ChecklistItem, accessory: Accessory): String = when (item) {
    EnrollmentFlow.ChecklistItem.watchOff -> "Watch off"
    EnrollmentFlow.ChecklistItem.ringsOff -> "Rings off"
    EnrollmentFlow.ChecklistItem.braceletsOff -> "Bracelets off"
    EnrollmentFlow.ChecklistItem.wearingAccessory -> "I am wearing my ${accessory.rawValue}"
}

@Composable
private fun ChecklistStep(model: AppModel, controller: EnrollmentController) {
    val flow = controller.flow
    SectionCard("Before you start") {
        for (item in flow.checklistItems) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = item in flow.checked, onCheckedChange = { on -> controller.update { setChecked(item, on) } })
                Text(checklistText(item, flow.accessory))
            }
        }
        FootnoteText(
            "You will hold the phone in your right hand, then your left, for ${flow.total} short vibration patterns " +
                "in ${flow.roundCount} rounds. Keep still while it vibrates; failed patterns are repeated.",
        )
    }
    model.recordingUnavailableReason?.let { NoteBanner(it, Icons.Filled.MobileOff) }
    Button(
        onClick = { controller.update { confirmChecklist() } },
        enabled = flow.checklistComplete, modifier = Modifier.fillMaxWidth(),
    ) { Text("Start") }
}

private fun buttonTitle(controller: EnrollmentController): String {
    val flow = controller.flow
    if (controller.errorMessage != null) return "Try again"
    if (flow.failuresOnSlot > 0) return "Try again"
    val slot = flow.currentSlot
    if (slot?.posture == Posture.freshGrip && slot.index != 0) return "Picked up — next pattern"
    return if (flow.acceptedCount == 0) "Start" else "Start round"
}

@Composable
private fun RecordingStep(model: AppModel, controller: EnrollmentController) {
    val flow = controller.flow
    Column(
        Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        ProgressRing(flow.progress, "${flow.acceptedCount}/${flow.total}")
        Text(flow.roundTitle, style = MaterialTheme.typography.titleMedium)
        Text(flow.instruction, style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Center)
        flow.lastMessage?.let { message ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.ErrorOutline, contentDescription = null, tint = WarningColor)
                Spacer(Modifier.width(6.dp))
                Text(message, color = WarningColor)
            }
        }
        controller.lastAnalysis?.let { QualityBadge(it.quality) }
        controller.errorMessage?.let { NoteBanner(it) }
        if (controller.isRecording) {
            CircularProgressIndicator()
            Text("Keep still…")
        } else {
            Button(onClick = { controller.startOrContinue() }, enabled = model.canRecord) {
                Text(buttonTitle(controller))
            }
        }
    }
}

@Composable
private fun SummaryStep(controller: EnrollmentController, onDone: () -> Unit) {
    val flow = controller.flow
    SectionCard("Summary") {
        LabeledRow("Name", flow.name)
        LabeledRow("Right hand", "${controller.accepted.count { it.first.hand == Hand.right }} patterns")
        LabeledRow("Left hand", "${controller.accepted.count { it.first.hand == Hand.left }} patterns")
        LabeledRow("Repeated patterns", "${flow.totalFailures}")
        if (flow.accessory != Accessory.none) LabeledRow("Accessory", flow.accessory.rawValue)
    }
    Button(
        onClick = {
            controller.commit()
            onDone()
        },
        enabled = !controller.committed, modifier = Modifier.fillMaxWidth(),
    ) { Text(if (controller.committed) "Saved" else "Save") }
    OutlinedButton(onClick = onDone, modifier = Modifier.fillMaxWidth()) { Text("Discard") }
}
