package com.mahmoudalyosify.auditor.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.mahmoudalyosify.auditor.core.ProcessedSignal
import com.mahmoudalyosify.auditor.model.AnalyzedPattern
import com.mahmoudalyosify.auditor.model.SignalSummary

/** One scaled series of the plot: values divided by their peak, on the grid's time axis. */
class PlotSeries(val name: String, val times: DoubleArray, val values: DoubleArray)

object SignalPlotData {
    /** accel-HP magnitude, gyro-HP magnitude and the aligned template, each scaled to its peak. */
    fun series(p: ProcessedSignal): List<PlotSeries> {
        val fs = p.config.fs
        val pre = p.config.preRoll
        val lag = p.alignment.lag
        val accel = SignalSummary.magnitude(p.accelHP)
        val gyro = SignalSummary.magnitude(p.gyroHP)
        val aligned = DoubleArray(p.n) { n ->
            val k = n - lag
            if (k >= 0 && k < p.template.size) p.template[k] else 0.0
        }
        return listOf("accel HP" to accel, "gyro HP" to gyro, "template" to aligned).map { (name, values) ->
            val peak = maxOf(values.maxOrNull() ?: 0.0, 1e-12)
            PlotSeries(name, DoubleArray(values.size) { i -> i / fs - pre }, DoubleArray(values.size) { i -> values[i] / peak })
        }
    }

    /** Event times (seconds from haptic start) after alignment. */
    fun markers(p: ProcessedSignal): List<Double> =
        p.challenge.events.map { p.index(it.time) / p.config.fs - p.config.preRoll }
}

/** Last-recording plot drawn on a Compose Canvas, with a legend. */
@Composable
fun SignalPlot(analysis: AnalyzedPattern) {
    val series = remember(analysis) { SignalPlotData.series(analysis.processed) }
    val markers = remember(analysis) { SignalPlotData.markers(analysis.processed) }
    val colors = listOf(MaterialTheme.colorScheme.primary, Color(0xFFE0663A), MaterialTheme.colorScheme.outline)
    val markerColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
    val axisColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
    val tMin = series.minOfOrNull { it.times.firstOrNull() ?: 0.0 } ?: 0.0
    val tMax = series.maxOfOrNull { it.times.lastOrNull() ?: 1.0 } ?: 1.0
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Canvas(
            Modifier
                .fillMaxWidth()
                .height(180.dp),
        ) {
            val w = size.width
            val h = size.height
            val span = if (tMax > tMin) tMax - tMin else 1.0
            fun x(t: Double): Float = ((t - tMin) / span * w).toFloat()
            fun y(v: Double): Float = (h - v.coerceIn(0.0, 1.05) / 1.05 * h).toFloat()
            drawLine(axisColor, Offset(0f, h), Offset(w, h), strokeWidth = 1.dp.toPx())
            if (0.0 in tMin..tMax) drawLine(axisColor, Offset(x(0.0), 0f), Offset(x(0.0), h), strokeWidth = 1.dp.toPx())
            val dash = PathEffect.dashPathEffect(floatArrayOf(6f, 6f))
            for (m in markers) {
                if (m < tMin || m > tMax) continue
                drawLine(markerColor, Offset(x(m), 0f), Offset(x(m), h), strokeWidth = 1.dp.toPx(), pathEffect = dash)
            }
            series.forEachIndexed { index, s ->
                if (s.values.isEmpty()) return@forEachIndexed
                val path = Path()
                path.moveTo(x(s.times[0]), y(s.values[0]))
                for (i in 1 until s.values.size) path.lineTo(x(s.times[i]), y(s.values[i]))
                drawPath(path, colors[index % colors.size], style = Stroke(width = 1.2.dp.toPx()))
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            series.forEachIndexed { index, s ->
                Box(
                    Modifier
                        .size(10.dp)
                        .background(colors[index % colors.size]),
                )
                Spacer(Modifier.width(4.dp))
                Text(s.name, style = MaterialTheme.typography.labelSmall)
                Spacer(Modifier.width(12.dp))
            }
        }
        FootnoteText("seconds from haptic start: ${Fmt.num(tMin, 2)} … ${Fmt.num(tMax, 2)} · dashed lines: events")
    }
}
