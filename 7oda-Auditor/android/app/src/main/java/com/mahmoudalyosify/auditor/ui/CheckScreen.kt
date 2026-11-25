package com.mahmoudalyosify.auditor.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.MobileOff
import androidx.compose.material.icons.filled.PanTool
import androidx.compose.material.icons.filled.PersonAdd
import androidx.compose.material.icons.filled.Verified
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.RecordingPurpose
import com.mahmoudalyosify.auditor.model.AppModel
import com.mahmoudalyosify.auditor.model.CheckResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Composable
fun CheckScreen(model: AppModel) {
    SensorSession(model)
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("") }
    var requestedHand by remember { mutableStateOf<Hand?>(null) }
    var result by remember { mutableStateOf<CheckResult?>(null) }
    var failure by remember { mutableStateOf<String?>(null) }
    var job by remember { mutableStateOf<Job?>(null) }
    DisposableEffect(Unit) { onDispose { job?.cancel() } }

    fun start() {
        running = true
        failure = null
        result = null
        requestedHand = if (model.settings.handChallenge) model.drawChallengeHand() else null
        status = if (requestedHand == null) "Starting…" else "Switch hands if needed…"
        job = scope.launch {
            try {
                if (requestedHand != null) delay(2500)
                result = model.runCheck(RecordingPurpose.check, requestedHand = requestedHand) { status = it }
            } catch (e: CancellationException) {
                failure = "Check stopped."
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
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        model.recognizerNote?.let { NoteBanner(it) }
        model.recordingUnavailableReason?.let { NoteBanner(it, Icons.Filled.MobileOff) }
        if (model.people.isEmpty()) {
            NoteBanner("Nobody is enrolled yet. Add a person in the People tab first.", Icons.Filled.PersonAdd)
        }
        val hand = requestedHand
        if (hand != null && (running || result != null)) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.15f), RoundedCornerShape(12.dp))
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
            ) {
                Icon(Icons.Filled.PanTool, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("Hand challenge: hold the phone in your ${if (hand == Hand.right) "RIGHT" else "LEFT"} hand",
                    style = MaterialTheme.typography.titleSmall)
            }
        }
        Button(
            onClick = { start() },
            enabled = !running && model.canRecord && model.people.isNotEmpty(),
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(),
            modifier = Modifier
                .padding(top = 8.dp)
                .size(220.dp),
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Filled.PanTool, contentDescription = null, modifier = Modifier.size(54.dp))
                Spacer(Modifier.size(8.dp))
                Text("Who's holding?", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            }
        }
        if (running) {
            CircularProgressIndicator()
            Text(status, style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Center)
            OutlinedButton(onClick = { job?.cancel() }) { Text("Stop") }
        } else {
            FootnoteText("Hold the phone in one hand, the way you normally do, and keep still while it vibrates.")
        }
        result?.let { ResultCard(model, it) }
        failure?.let { Text(it, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center) }
    }
}

@Composable
fun ResultCard(model: AppModel, result: CheckResult) {
    val decision = result.decision
    val headline = if (result.qualityFailure != null || decision == null) "No answer" else model.headline(decision.outcome)
    val subtitle = result.qualityFailure?.let { "Could not record a clean pattern: $it" }
        ?: decision?.let { model.reasonText(it.outcome) }
    SectionCard {
        Text(headline, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        subtitle?.let { FootnoteText(it) }
        result.challengePassed?.let { passed ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                val color: Color = if (passed) SuccessColor else MaterialTheme.colorScheme.error
                Icon(if (passed) Icons.Filled.Verified else Icons.Filled.Cancel, contentDescription = null, tint = color)
                Spacer(Modifier.width(6.dp))
                Text(if (passed) "Hand challenge passed" else "Hand challenge failed", color = color)
            }
        }
        if (decision != null && decision.personScores.isNotEmpty()) ScoreBars(model, decision.personScores)
        val n = result.patterns.size
        FootnoteText("$n pattern${if (n == 1) "" else "s"} used · ${result.attempts} played · " +
            "${Fmt.num(result.seconds, 1)} s · answered by ${result.answeredBy.name}")
        for (note in result.notes) Text(note, style = MaterialTheme.typography.bodySmall, color = WarningColor)
        if (result.learned > 0) FootnoteText("Learned from this check (${result.learned} samples added).")
    }
}
