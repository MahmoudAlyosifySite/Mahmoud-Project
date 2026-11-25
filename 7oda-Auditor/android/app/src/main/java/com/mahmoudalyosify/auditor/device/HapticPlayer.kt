package com.mahmoudalyosify.auditor.device

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.os.VibrationAttributes
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.mahmoudalyosify.auditor.core.ChallengeSpec

/** Plays a `ChallengeSpec` on the phone's vibration motor (see [HapticTimeline]). */
class HapticPlayer(context: Context) {
    private val vibrator: Vibrator? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }

    val isSupported: Boolean get() = vibrator?.hasVibrator() == true

    val hasAmplitudeControl: Boolean get() = vibrator?.hasAmplitudeControl() == true

    /** Click and tick composition primitives (API 30+). */
    val primitivesSupported: Boolean
        get() {
            val v = vibrator ?: return false
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return false
            return v.areAllPrimitivesSupported(
                VibrationEffect.Composition.PRIMITIVE_CLICK, VibrationEffect.Composition.PRIMITIVE_TICK,
            )
        }

    /** Primitive duration in ms (API 31+ reports it; older devices: about one tap). */
    private fun primitiveMillis(tick: Boolean): Int {
        val v = vibrator
        if (v != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val id = if (tick) VibrationEffect.Composition.PRIMITIVE_TICK else VibrationEffect.Composition.PRIMITIVE_CLICK
            val d = v.getPrimitiveDurations(id).firstOrNull() ?: 0
            if (d > 0) return d
        }
        return HapticTimeline.TAP_MS
    }

    /** Vibrator capabilities for the Lab. */
    fun capabilities(): List<Pair<String, String>> {
        val v = vibrator
        val out = mutableListOf(
            "Vibrator" to if (isSupported) "present" else "not present",
            "Amplitude control" to if (hasAmplitudeControl) "yes" else "no",
            "Click/tick primitives" to if (primitivesSupported) "yes" else "no",
        )
        if (v != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            out.add("Vibrator id" to v.id.toString())
            val durations = listOf(VibrationEffect.Composition.PRIMITIVE_CLICK, VibrationEffect.Composition.PRIMITIVE_TICK)
                .map { v.getPrimitiveDurations(it).firstOrNull() ?: 0 }
            out.add("Click / tick duration" to "${durations[0]} / ${durations[1]} ms")
        }
        return out
    }

    /** The single effect that plays the whole pattern. */
    fun effect(spec: ChallengeSpec): VibrationEffect {
        if (!HapticTimeline.hasRamps(spec) && primitivesSupported && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val composition = VibrationEffect.startComposition()
            for (step in HapticTimeline.primitives(spec) { primitiveMillis(it) }) {
                val id = if (step.tick) VibrationEffect.Composition.PRIMITIVE_TICK
                else VibrationEffect.Composition.PRIMITIVE_CLICK
                composition.addPrimitive(id, step.scale, step.delayMillis)
            }
            return composition.compose()
        }
        val plan = HapticTimeline.waveform(spec, hasAmplitudeControl)
        if (plan.timings.isEmpty()) {
            return VibrationEffect.createOneShot(HapticTimeline.TAP_MS.toLong(), VibrationEffect.DEFAULT_AMPLITUDE)
        }
        return VibrationEffect.createWaveform(plan.timings, plan.amplitudes, -1)
    }

    /**
     * Plays the pattern and returns `SystemClock.elapsedRealtimeNanos() / 1e9` read right
     * before the vibrate call (the clock of `SensorEvent.timestamp`).
     */
    fun play(spec: ChallengeSpec): Double {
        val v = vibrator ?: throw IllegalStateException(HAPTICS_UNAVAILABLE)
        if (!v.hasVibrator()) throw IllegalStateException(HAPTICS_UNAVAILABLE)
        val effect = effect(spec)
        val start = SystemClock.elapsedRealtimeNanos() / 1e9
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            v.vibrate(effect, VibrationAttributes.createForUsage(VibrationAttributes.USAGE_MEDIA))
        } else {
            v.vibrate(effect)
        }
        return start
    }

    fun stop() {
        vibrator?.cancel()
    }

    companion object {
        const val HAPTICS_UNAVAILABLE = "This phone has no vibration motor, so it cannot play the patterns."
    }
}
