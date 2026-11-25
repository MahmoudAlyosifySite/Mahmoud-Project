@file:OptIn(ExperimentalMaterial3Api::class)

package com.mahmoudalyosify.auditor.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.core.ConfusionMatrix
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.Posture
import com.mahmoudalyosify.auditor.core.RecordingLabel
import com.mahmoudalyosify.auditor.core.RecordingPurpose
import com.mahmoudalyosify.auditor.model.AppModel
import com.mahmoudalyosify.auditor.model.AppSettings
import com.mahmoudalyosify.auditor.model.CheckResult
import com.mahmoudalyosify.auditor.model.RecognizerKind
import com.mahmoudalyosify.auditor.model.SignalCheckResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.UUID
import kotlin.math.roundToInt

// Test run

@Composable
fun TestRunScreen(model: AppModel) {
    SensorSession(model)
    val scope = rememberCoroutineScope()
    var personID by remember { mutableStateOf(model.people.firstOrNull()?.id ?: "") }
    var hand by remember { mutableStateOf(Hand.right) }
    val sessionID = remember { UUID.randomUUID().toString().uppercase() }
    var running by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("") }
    var last by remember { mutableStateOf<CheckResult?>(null) }
    var failure by remember { mutableStateOf<String?>(null) }
    var confirmClear by remember { mutableStateOf(false) }
    var menuOpen by remember { mutableStateOf(false) }
    var job by remember { mutableStateOf<Job?>(null) }
    DisposableEffect(Unit) { onDispose { job?.cancel() } }

    fun run() {
        if (personID.isEmpty()) return
        running = true
        failure = null
        status = "Starting…"
        val truth = RecordingLabel(personID = personID, personName = model.personName(personID), hand = hand,
            accessory = Accessory.none, posture = Posture.unspecified, sessionID = sessionID)
        job = scope.launch {
            try {
                last = model.runCheck(RecordingPurpose.test, truth = truth) { status = it }
            } catch (e: CancellationException) {
                failure = "Stopped."
            } catch (e: Exception) {
                failure = e.message ?: e.javaClass.simpleName
            } finally {
                running = false
            }
        }
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SectionCard("Ground truth") {
            Box {
                OutlinedButton(onClick = { menuOpen = true }, enabled = model.people.isNotEmpty()) {
                    Text(if (personID.isEmpty()) "Pick a person" else model.personName(personID))
                }
                DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                    for (person in model.people) {
                        DropdownMenuItem(text = { Text(person.name) }, onClick = {
                            personID = person.id
                            menuOpen = false
                        })
                    }
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = hand == Hand.right, onClick = { hand = Hand.right }, label = { Text("Right") })
                FilterChip(selected = hand == Hand.left, onClick = { hand = Hand.left }, label = { Text("Left") })
            }
        }
        SectionCard {
            Button(
                onClick = { run() }, enabled = !running && personID.isNotEmpty() && model.canRecord,
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (running) status else "Run one check") }
            if (running) OutlinedButton(onClick = { job?.cancel() }, modifier = Modifier.fillMaxWidth()) { Text("Stop") }
            last?.let { r ->
                val v1 = r.v1?.let { model.headline(it.outcome) } ?: "n/a"
                val v0 = r.v0?.let { model.headline(it.outcome) } ?: "n/a"
                Text("Last check: v1 $v1 · v0 $v0")
            }
            failure?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            model.recordingUnavailableReason?.let { FootnoteText(it) }
        }
        for (kind in RecognizerKind.entries) {
            SectionCard("Confusion matrix · ${kind.name}") { ConfusionMatrixView(model.confusionMatrix(kind)) }
        }
        OutlinedButton(onClick = { confirmClear = true }, modifier = Modifier.fillMaxWidth()) {
            Text("Clear test results", color = MaterialTheme.colorScheme.error)
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear all test results?") },
            confirmButton = {
                TextButton(onClick = {
                    confirmClear = false
                    model.clearTestResults()
                }) { Text("Clear", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } },
        )
    }
}

@Composable
fun ConfusionMatrixView(matrix: ConfusionMatrix) {
    if (matrix.total == 0) {
        FootnoteText("No test checks yet.")
        return
    }
    val labels = matrix.labels
    val counts = matrix.counts
    val cellStyle = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace)
    val diagonal = SuccessColor.copy(alpha = 0.18f)
    val other = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.06f)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        FootnoteText("Accuracy ${Fmt.percent(matrix.accuracy)} over ${matrix.total} checks (rows: truth, columns: answer)")
        Column(
            Modifier.horizontalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("", style = cellStyle, modifier = Modifier.width(110.dp))
                for (label in labels) {
                    Text(label, style = cellStyle, maxLines = 2, textAlign = TextAlign.Center, modifier = Modifier.width(72.dp))
                }
            }
            labels.forEachIndexed { i, rowLabel ->
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(rowLabel, style = cellStyle, maxLines = 2, modifier = Modifier.width(110.dp))
                    labels.indices.forEach { j ->
                        val n = counts.getOrNull(i)?.getOrNull(j) ?: 0
                        Text(
                            "$n", style = cellStyle, textAlign = TextAlign.Center,
                            modifier = Modifier
                                .width(72.dp)
                                .background(if (i == j) diagonal else other)
                                .padding(vertical = 4.dp),
                        )
                    }
                }
            }
        }
    }
}

// Signal check

@Composable
fun SignalCheckScreen(model: AppModel) {
    SensorSession(model)
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var done by remember { mutableIntStateOf(0) }
    var result by remember { mutableStateOf<SignalCheckResult?>(null) }
    var failure by remember { mutableStateOf<String?>(null) }
    var job by remember { mutableStateOf<Job?>(null) }
    DisposableEffect(Unit) { onDispose { job?.cancel() } }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SectionCard {
            Text("Hold the phone in one hand and keep still. Three patterns will play.")
            Button(
                onClick = {
                    running = true
                    done = 0
                    failure = null
                    result = null
                    job = scope.launch {
                        try {
                            result = model.runSignalCheck { done = it }
                        } catch (e: CancellationException) {
                            failure = "Stopped."
                        } catch (e: Exception) {
                            failure = e.message ?: e.javaClass.simpleName
                        } finally {
                            running = false
                        }
                    }
                },
                enabled = !running && model.canRecord, modifier = Modifier.fillMaxWidth(),
            ) { Text(if (running) "Playing…" else "Run signal check") }
            if (running) LinearProgressIndicator(progress = { done / 3f }, modifier = Modifier.fillMaxWidth())
            model.recordingUnavailableReason?.let { FootnoteText(it) }
            failure?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        }
        result?.let { r ->
            SectionCard("Result") {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        if (r.visible) Icons.Filled.CheckCircle else Icons.Filled.Cancel, contentDescription = null,
                        tint = if (r.visible) SuccessColor else MaterialTheme.colorScheme.error,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Vibration response visible: ${if (r.visible) "yes" else "no"}")
                }
                for ((name, snr) in r.channels) LabeledRow("SNR $name", Fmt.num(snr, 1))
            }
            SectionCard("Patterns") {
                for (p in r.patterns) QualityBadge(p.quality)
            }
        }
    }
}

// Settings

@Composable
private fun Stepper(title: String, value: String, onMinus: () -> Unit, onPlus: () -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(title, modifier = Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace))
        Spacer(Modifier.width(8.dp))
        OutlinedButton(onClick = onMinus) { Text("−") }
        Spacer(Modifier.width(4.dp))
        OutlinedButton(onClick = onPlus) { Text("+") }
    }
}

/** Steps a value by `step` within `range`, rounded to the step grid to avoid drift. */
private fun stepped(value: Double, step: Double, range: ClosedFloatingPointRange<Double>, up: Boolean): Double {
    val next = value + if (up) step else -step
    val snapped = (next / step).roundToInt() * step
    return snapped.coerceIn(range.start, range.endInclusive)
}

@Composable
private fun DoubleStepper(
    model: AppModel,
    title: String,
    value: Double,
    digits: Int,
    step: Double,
    range: ClosedFloatingPointRange<Double>,
    change: AppSettings.(Double) -> AppSettings,
) {
    Stepper(
        title, Fmt.num(value, digits),
        onMinus = { model.updateSettings(model.settings.change(stepped(value, step, range, up = false))) },
        onPlus = { model.updateSettings(model.settings.change(stepped(value, step, range, up = true))) },
    )
}

@Composable
private fun SwitchRow(title: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(title, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
fun LabSettingsScreen(model: AppModel) {
    val s = model.settings
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SectionCard("Recognizer") {
            for (kind in RecognizerKind.entries) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = s.recognizer == kind, onClick = { model.updateSettings(s.copy(recognizer = kind)) })
                    Text(kind.title)
                }
            }
            model.recognizerNote?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = WarningColor) }
            FootnoteText("Both recognizers run on every check; the other one is logged for comparison.")
        }
        SectionCard("v0 thresholds") {
            DoubleStepper(model, "absFactor", s.absFactor, 1, 0.1, 0.5..5.0) { copy(absFactor = it) }
            DoubleStepper(model, "nameMargin", s.nameMargin, 1, 0.5, 0.0..30.0) { copy(nameMargin = it) }
            DoubleStepper(model, "handMargin", s.handMargin, 1, 0.5, 0.0..20.0) { copy(handMargin = it) }
        }
        SectionCard("v1 thresholds") {
            DoubleStepper(model, "slack", s.slack, 2, 0.01, -0.2..0.2) { copy(slack = it) }
            DoubleStepper(model, "nameMargin", s.v1NameMargin, 3, 0.005, 0.0..0.3) { copy(v1NameMargin = it) }
        }
        SectionCard("Checks and quality") {
            Stepper(
                "maxPatterns", "${s.maxPatterns}",
                onMinus = { model.updateSettings(s.copy(maxPatterns = (s.maxPatterns - 1).coerceIn(1, 6))) },
                onPlus = { model.updateSettings(s.copy(maxPatterns = (s.maxPatterns + 1).coerceIn(1, 6))) },
            )
            DoubleStepper(model, "minNCC", s.minNCC, 2, 0.05, 0.0..1.0) { copy(minNCC = it) }
            DoubleStepper(model, "minPeakMargin", s.minPeakMargin, 2, 0.01, 0.0..0.5) { copy(minPeakMargin = it) }
        }
        SectionCard {
            SwitchRow("Hand challenge", s.handChallenge) { model.updateSettings(s.copy(handChallenge = it)) }
            SwitchRow("Learn from confident checks", s.learnFromConfidentChecks) {
                model.updateSettings(s.copy(learnFromConfidentChecks = it))
            }
            FootnoteText(
                "Hand challenge: the app asks for a random hand and passes only if it names you with that hand. " +
                    "Learning adds the patterns of very confident checks to your samples (at most 20 per hand).",
            )
        }
        Button(
            onClick = {
                model.updateSettings(AppSettings(calibrationSkipped = s.calibrationSkipped, anonymizeExports = s.anonymizeExports))
            },
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Restore defaults") }
    }
}
