package com.mahmoudalyosify.auditor.ui

import android.os.Build
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleStartEffect
import com.mahmoudalyosify.auditor.core.Decision
import com.mahmoudalyosify.auditor.core.PersonScore
import com.mahmoudalyosify.auditor.core.QualityReport
import com.mahmoudalyosify.auditor.model.AppModel
import java.util.Locale
import kotlin.math.roundToInt

/** Number formatting shared by the screens. */
object Fmt {
    fun num(x: Double, digits: Int = 3): String {
        if (x.isNaN()) return "—"
        if (x.isInfinite()) return if (x > 0) "∞" else "−∞"
        return String.format(Locale.US, "%.${digits}f", x)
    }

    fun percent(x: Double): String = if (x.isFinite()) "${(x * 100).roundToInt()}%" else "—"
}

val WarningColor = Color(0xFFE08A00)
val SuccessColor = Color(0xFF2E9D4F)

@Composable
fun AuditorTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val scheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> darkColorScheme(primary = Color(0xFF7FD1FF), secondary = Color(0xFF9CC9E8))
        else -> lightColorScheme(primary = Color(0xFF1E5A8F), secondary = Color(0xFF3A6F96))
    }
    MaterialTheme(colorScheme = scheme, content = content)
}

/** Runs the motion sensors while the calling screen is shown and the app is started (reference counted). */
@Composable
fun SensorSession(model: AppModel) {
    LifecycleStartEffect(Unit) {
        model.beginSensorSession()
        onStopOrDispose { model.endSensorSession() }
    }
}

@Composable
fun NoteBanner(text: String, icon: ImageVector = Icons.Filled.Warning, modifier: Modifier = Modifier) {
    Row(
        modifier
            .fillMaxWidth()
            .background(WarningColor.copy(alpha = 0.15f), RoundedCornerShape(10.dp))
            .padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = WarningColor, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(8.dp))
        Text(text, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
fun SectionCard(title: String? = null, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(modifier.fillMaxWidth(), colors = CardDefaults.cardColors()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (title != null) {
                Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
            }
            content()
        }
    }
}

@Composable
fun FootnoteText(text: String, color: Color = MaterialTheme.colorScheme.onSurfaceVariant) {
    Text(text, style = MaterialTheme.typography.bodySmall, color = color)
}

@Composable
fun LabeledRow(title: String, value: String) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
        Text(title, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f))
        Spacer(Modifier.width(8.dp))
        Text(value, style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.End,
            modifier = Modifier.weight(1f))
    }
}

@Composable
fun ProgressRing(progress: Double, label: String, size: Dp = 180.dp) {
    val track = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f)
    val color = MaterialTheme.colorScheme.primary
    val fraction = if (progress.isFinite()) progress.coerceIn(0.0, 1.0).toFloat() else 0f
    Box(Modifier.size(size), contentAlignment = Alignment.Center) {
        Canvas(Modifier.size(size)) {
            val stroke = 14.dp.toPx()
            val inset = stroke / 2
            val arcSize = androidx.compose.ui.geometry.Size(this.size.width - stroke, this.size.height - stroke)
            val topLeft = androidx.compose.ui.geometry.Offset(inset, inset)
            drawArc(track, 0f, 360f, false, topLeft, arcSize, style = Stroke(stroke))
            drawArc(color, -90f, 360f * fraction, false, topLeft, arcSize, style = Stroke(stroke, cap = StrokeCap.Round))
        }
        Text(label, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun ScoreBars(model: AppModel, scores: List<PersonScore>) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        for (score in scores) {
            val p = if (score.probability.isFinite()) score.probability.coerceIn(0.0, 1.0).toFloat() else 0f
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(model.personName(score.personID), maxLines = 1, overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.bodyMedium, modifier = Modifier.width(110.dp))
                LinearProgressIndicator(progress = { p }, modifier = Modifier.weight(1f))
                Text(Fmt.percent(score.probability), style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.End, modifier = Modifier.width(52.dp))
            }
        }
    }
}

@Composable
fun QualityBadge(report: QualityReport) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(
            if (report.passed) Icons.Filled.CheckCircle else Icons.Filled.Cancel, contentDescription = null,
            tint = if (report.passed) SuccessColor else WarningColor, modifier = Modifier.size(18.dp),
        )
        Spacer(Modifier.width(8.dp))
        val m = report.metrics
        FootnoteText("NCC ${Fmt.num(m.ncc, 2)} · peak ${Fmt.num(m.peakMargin, 2)} · SNR ${Fmt.num(m.snr, 1)}")
    }
}

@Composable
fun MetricsList(report: QualityReport) {
    val m = report.metrics
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        LabeledRow("Passed", if (report.passed) "yes" else "no")
        if (report.issues.isNotEmpty()) LabeledRow("Issues", report.issues.joinToString(", ") { it.rawValue })
        LabeledRow("NCC / peak margin / lag", "${Fmt.num(m.ncc, 3)} / ${Fmt.num(m.peakMargin, 3)} / ${m.lag}")
        LabeledRow("SNR", Fmt.num(m.snr, 2))
        LabeledRow("Sample rate", "${Fmt.num(m.sampleRate, 1)} Hz")
        LabeledRow("Gap fraction", Fmt.num(m.gapFraction, 3))
        LabeledRow("Quiet gyro / accel RMS", "${Fmt.num(m.quietGyroRMS, 4)} / ${Fmt.num(m.quietAccelRMS, 4)}")
        LabeledRow("Gravity Z", Fmt.num(m.gravityZ, 3))
        LabeledRow("Motion RMS", Fmt.num(m.motionRMS, 3))
        LabeledRow("Gravity angle", "${Fmt.num(m.gravityAngleDeg, 1)}°")
    }
}

@Composable
fun DecisionDetail(model: AppModel, title: String, decision: Decision?, answered: Boolean) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            if (answered) {
                Spacer(Modifier.width(8.dp))
                Text(
                    "answered", style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier
                        .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.2f), RoundedCornerShape(50))
                        .padding(horizontal = 6.dp, vertical = 2.dp),
                )
            }
        }
        if (decision == null) {
            FootnoteText("Not evaluated")
            return@Column
        }
        Text(model.headline(decision.outcome) + (model.reasonText(decision.outcome)?.let { " — $it" } ?: ""))
        val mono = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
        Text("best ${Fmt.num(decision.bestDistance)} · threshold ${Fmt.num(decision.absThreshold)} · patterns ${decision.patternsUsed}",
            style = mono, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("margin name ${Fmt.num(decision.marginName)} · margin hand ${Fmt.num(decision.marginHand)} · " +
            "hand log-odds ${Fmt.num(decision.handLogOdds)}", style = mono, color = MaterialTheme.colorScheme.onSurfaceVariant)
        for (item in decision.prototypeDistances) {
            Row(Modifier.fillMaxWidth()) {
                Text(model.prototypeName(item.prototype), style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.weight(1f))
                Text(Fmt.num(item.distance), style = MaterialTheme.typography.labelSmall)
            }
        }
    }
}
