package com.mahmoudalyosify.auditor.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MobileOff
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.mahmoudalyosify.auditor.core.Calibration
import com.mahmoudalyosify.auditor.model.AppModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

@Composable
fun CalibrationScreen(model: AppModel, firstRun: Boolean, modifier: Modifier = Modifier) {
    SensorSession(model)
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var done by remember { mutableIntStateOf(0) }
    var passed by remember { mutableIntStateOf(0) }
    var lastMessage by remember { mutableStateOf<String?>(null) }
    var result by remember { mutableStateOf<Calibration?>(null) }
    var failure by remember { mutableStateOf<String?>(null) }
    var job by remember { mutableStateOf<Job?>(null) }
    DisposableEffect(Unit) { onDispose { job?.cancel() } }

    fun start() {
        running = true
        done = 0
        passed = 0
        lastMessage = null
        failure = null
        result = null
        job = scope.launch {
            try {
                val cal = model.runCalibration { d, p, last ->
                    done = d
                    passed = p
                    lastMessage = last.qualityMessage
                }
                if (cal != null) result = cal else failure = Calibration.failureMessage
            } catch (e: CancellationException) {
                failure = "Calibration cancelled."
            } catch (e: Exception) {
                failure = e.message ?: e.javaClass.simpleName
            } finally {
                running = false
            }
        }
    }

    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SectionCard {
            Text("Put the phone flat on a hard table and don't touch it.", style = MaterialTheme.typography.titleMedium)
            FootnoteText(
                "The phone plays ${Calibration.patternCount} short vibration patterns (about a minute) and measures " +
                    "how it responds on its own. Later checks are compared with this baseline.",
            )
        }
        model.recordingUnavailableReason?.let { NoteBanner(it, Icons.Filled.MobileOff) }
        SectionCard("Progress") {
            LinearProgressIndicator(
                progress = { done.toFloat() / Calibration.patternCount }, modifier = Modifier.fillMaxWidth(),
            )
            Text("$done of ${Calibration.patternCount} patterns · $passed passed (at least ${Calibration.minimumPassing} needed)",
                style = MaterialTheme.typography.bodySmall)
            lastMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = WarningColor) }
        }
        result?.let { cal ->
            SectionCard("Result") {
                LabeledRow("Patterns used", "${cal.patternsUsed}")
                LabeledRow("Sample rate", "${Fmt.num(cal.sampleRate, 1)} Hz")
                LabeledRow("Quiet gyro RMS", Fmt.num(cal.quietGyroRMS, 5))
                LabeledRow("Median NCC", Fmt.num(cal.ncc, 2))
                LabeledRow("Median SNR", Fmt.num(cal.snr, 1))
            }
        }
        failure?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        Button(onClick = { start() }, enabled = !running && model.canRecord, modifier = Modifier.fillMaxWidth()) {
            Text(if (running) "Calibrating…" else if (result == null) "Start calibration" else "Calibrate again")
        }
        if (running) {
            OutlinedButton(onClick = { job?.cancel() }, modifier = Modifier.fillMaxWidth()) {
                Text("Cancel", color = MaterialTheme.colorScheme.error)
            }
        }
        if (firstRun) {
            TextButton(onClick = { model.skipCalibration() }, enabled = !running, modifier = Modifier.fillMaxWidth()) {
                Text("Skip for now")
            }
            FootnoteText("Without a calibration the app uses absolute thresholds; you can calibrate later in Lab.")
        }
    }
}
