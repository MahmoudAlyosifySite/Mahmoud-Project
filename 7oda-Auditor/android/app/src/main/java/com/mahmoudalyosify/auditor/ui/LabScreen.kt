package com.mahmoudalyosify.auditor.ui

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.navigation.NavHostController
import com.mahmoudalyosify.auditor.core.FeatureExtractor
import com.mahmoudalyosify.auditor.core.SeedFormat
import com.mahmoudalyosify.auditor.device.DeviceSupport
import com.mahmoudalyosify.auditor.device.SensorRates
import com.mahmoudalyosify.auditor.model.AppModel
import com.mahmoudalyosify.auditor.model.RecognizerKind
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

@Composable
fun LabScreen(model: AppModel, nav: NavHostController) {
    SensorSession(model)
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var exportFiles by remember { mutableStateOf<List<File>>(emptyList()) }
    var preparing by remember { mutableStateOf(false) }
    var confirmDeleteAll by remember { mutableStateOf(false) }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SectionCard("Last recording") {
            val a = model.lastAnalysis
            if (a == null) {
                FootnoteText("No recording yet.")
            } else {
                SignalPlot(a)
                FootnoteText("${a.recording.purpose.rawValue} · ${a.recording.challenge.events.size} events · " +
                    "seed ${SeedFormat.string(a.recording.challenge.seed)}")
                a.encoderError?.let { Text("v1 encoder: $it", style = MaterialTheme.typography.bodySmall, color = WarningColor) }
                MetricsList(a.quality)
            }
        }
        SectionCard("Last check") {
            val check = model.lastCheck
            if (check == null) {
                FootnoteText("No check yet.")
            } else {
                DecisionDetail(model, "v1 · learned encoder", check.v1, check.answeredBy == RecognizerKind.v1)
                DecisionDetail(model, "v0 · hand-made features", check.v0, check.answeredBy == RecognizerKind.v0)
            }
        }
        SectionCard("Tools") {
            Button(onClick = { nav.navigate(Routes.TEST_RUN) }, modifier = Modifier.fillMaxWidth()) { Text("Test run") }
            Button(onClick = { nav.navigate(Routes.SIGNAL_CHECK) }, modifier = Modifier.fillMaxWidth()) { Text("Signal check") }
            Button(onClick = { nav.navigate(Routes.SETTINGS) }, modifier = Modifier.fillMaxWidth()) { Text("Settings") }
            Button(onClick = { nav.navigate(Routes.CALIBRATE) }, modifier = Modifier.fillMaxWidth()) { Text("Recalibrate") }
            OutlinedButton(
                onClick = { scope.launch { model.recomputeAll() } },
                enabled = model.busyStatus == null, modifier = Modifier.fillMaxWidth(),
            ) { Text("Recompute features from recordings") }
        }
        SectionCard("Device") { DeviceInfo(model) }
        SectionCard("Export") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Anonymize names (P01, P02, …)", modifier = Modifier.weight(1f))
                Switch(
                    checked = model.settings.anonymizeExports,
                    onCheckedChange = { model.updateSettings(model.settings.copy(anonymizeExports = it)) },
                )
            }
            Button(
                onClick = {
                    preparing = true
                    scope.launch {
                        try {
                            exportFiles = model.makeExport()
                        } catch (e: Exception) {
                            model.errorMessage = "Export failed: ${e.message ?: e.javaClass.simpleName}"
                        } finally {
                            preparing = false
                        }
                    }
                },
                enabled = !preparing, modifier = Modifier.fillMaxWidth(),
            ) { Text(if (preparing) "Preparing…" else "Prepare export files") }
            if (exportFiles.isNotEmpty()) {
                Button(onClick = { share(context, exportFiles, model) }, modifier = Modifier.fillMaxWidth()) {
                    Text("Share ${exportFiles.size} files")
                }
                for (file in exportFiles) {
                    OutlinedButton(onClick = { share(context, listOf(file), model) }, modifier = Modifier.fillMaxWidth()) {
                        Text("Share ${file.name}")
                    }
                }
            }
            FootnoteText("recordings.jsonl (raw recordings), features.csv (v0 samples) and checks.csv (both " +
                "recognizers' outcomes). Exported files are not encrypted.")
        }
        OutlinedButton(onClick = { confirmDeleteAll = true }, modifier = Modifier.fillMaxWidth()) {
            Text("Delete all data", color = MaterialTheme.colorScheme.error)
        }
    }

    if (confirmDeleteAll) {
        AlertDialog(
            onDismissRequest = { confirmDeleteAll = false },
            title = { Text("Delete all people, recordings, calibration and settings?") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDeleteAll = false
                    exportFiles = emptyList()
                    model.deleteAllData()
                }) { Text("Delete all data", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDeleteAll = false }) { Text("Cancel") } },
        )
    }
}

/** Shares export files through the FileProvider declared in the manifest. */
private fun share(context: Context, files: List<File>, model: AppModel) {
    try {
        val authority = "${context.packageName}.exports"
        val uris: ArrayList<Uri> = ArrayList(files.map { FileProvider.getUriForFile(context, authority, it) })
        val intent = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).apply {
                type = if (files[0].name.endsWith(".csv")) "text/csv" else "application/x-ndjson"
                putExtra(Intent.EXTRA_STREAM, uris[0])
            }
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                type = "*/*"
                putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
            }
        }
        intent.putExtra(Intent.EXTRA_SUBJECT, "7oda Auditor export")
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        context.startActivity(Intent.createChooser(intent, "Share export files"))
    } catch (e: ActivityNotFoundException) {
        model.errorMessage = "No app on this phone can receive the files."
    } catch (e: Exception) {
        model.errorMessage = "Could not share the files: ${e.message ?: e.javaClass.simpleName}"
    }
}

@Composable
private fun DeviceInfo(model: AppModel) {
    var rates by remember { mutableStateOf(SensorRates(0.0, 0.0, 0.0)) }
    LaunchedEffect(Unit) {
        while (true) {
            model.motion?.let { rates = it.measuredRates() }
            delay(1000)
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        LabeledRow("Model", DeviceSupport.modelIdentifier)
        LabeledRow("System", DeviceSupport.systemDescription)
        LabeledRow("App", DeviceSupport.appVersion)
        LabeledRow("Motion sensors", if (model.motion?.isAvailable == true) "available" else "not available")
        LabeledRow("Rates (accel / gyro / motion)",
            "${Fmt.num(rates.accel, 0)} / ${Fmt.num(rates.gyro, 0)} / ${Fmt.num(rates.motion, 0)} Hz")
        model.motion?.sensorDescriptions()?.forEach { (name, value) -> LabeledRow(name, value) }
        model.haptics?.capabilities()?.forEach { (name, value) -> LabeledRow(name, value) }
        LabeledRow("v1 model", model.encoder?.version ?: "unavailable")
        model.encoderError?.let { LabeledRow("v1 model error", it) }
        LabeledRow("Feature version", "${FeatureExtractor.version}")
        val cal = model.calibration
        val calText = if (cal == null) "none" else {
            val f = DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM, FormatStyle.SHORT).withZone(ZoneId.systemDefault())
            "${f.format(cal.createdAt)} · ${cal.patternsUsed} patterns"
        }
        LabeledRow("Calibration", calText)
    }
}
