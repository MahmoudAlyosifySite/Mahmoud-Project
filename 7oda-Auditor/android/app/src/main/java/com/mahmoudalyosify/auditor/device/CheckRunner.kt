package com.mahmoudalyosify.auditor.device

import android.os.Build
import com.mahmoudalyosify.auditor.BuildConfig
import com.mahmoudalyosify.auditor.core.ChallengeGenerator
import com.mahmoudalyosify.auditor.core.DeviceInfo
import com.mahmoudalyosify.auditor.core.Recording
import com.mahmoudalyosify.auditor.core.RecordingLabel
import com.mahmoudalyosify.auditor.core.RecordingPurpose
import com.mahmoudalyosify.auditor.core.SignalConfig
import kotlinx.coroutines.delay
import java.security.SecureRandom

/** Why recording cannot happen on this device. */
class RecordingException(message: String) : Exception(message)

/** What this device can do, and how it identifies itself in recordings. */
object DeviceSupport {
    const val SENSORS_UNAVAILABLE =
        "This phone has no accelerometer or no gyroscope, so it cannot record the vibration response. " +
            "You can still look around the app, but checks and enrollment need both sensors."

    /** Manufacturer and model, e.g. "Google Pixel 8". */
    val modelIdentifier: String
        get() {
            val maker = Build.MANUFACTURER ?: ""
            val model = Build.MODEL ?: "Android"
            return if (model.startsWith(maker, ignoreCase = true) || maker.isEmpty()) model
            else "${maker.replaceFirstChar { it.uppercase() }} $model"
        }

    val systemDescription: String get() = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})"

    val appVersion: String get() = "${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})"

    val deviceInfo: DeviceInfo get() = DeviceInfo(model = modelIdentifier, system = systemDescription, app = appVersion)

    private val random = SecureRandom()

    /** 8 random bytes from the system CSPRNG. */
    fun randomSeed(): ULong {
        val bytes = ByteArray(8)
        synchronized(random) { random.nextBytes(bytes) }
        var seed = 0UL
        for (b in bytes) seed = (seed shl 8) or (b.toULong() and 0xFFUL)
        return seed
    }
}

/** Anything that can produce one recording (the real device runner, or a test double). */
interface PatternRecording {
    val canRecord: Boolean
    val unavailableReason: String?
    suspend fun record(purpose: RecordingPurpose, label: RecordingLabel?): Recording
}

/** Generates a random challenge, plays it and cuts the matching sensor window (SPEC 12). */
class CheckRunner(
    val recorder: MotionRecorder,
    val haptics: HapticPlayer,
    private val signalConfig: SignalConfig = SignalConfig.default,
) : PatternRecording {
    override val canRecord: Boolean get() = recorder.isAvailable && haptics.isSupported

    override val unavailableReason: String?
        get() = when {
            !recorder.isAvailable -> DeviceSupport.SENSORS_UNAVAILABLE
            !haptics.isSupported -> HapticPlayer.HAPTICS_UNAVAILABLE
            else -> null
        }

    override suspend fun record(purpose: RecordingPurpose, label: RecordingLabel?): Recording {
        if (!recorder.isAvailable) throw RecordingException(DeviceSupport.SENSORS_UNAVAILABLE)
        if (!haptics.isSupported) throw RecordingException(HapticPlayer.HAPTICS_UNAVAILABLE)
        val startedHere = !recorder.isRunning
        if (startedHere) recorder.start()
        try {
            val warm = recorder.secondsRunning
            if (warm < 0.5) delay(((0.5 - warm) * 1000).toLong() + 1)
            val seed = DeviceSupport.randomSeed()
            val spec = ChallengeGenerator.generate(seed)
            val start = try {
                haptics.play(spec)
            } catch (e: IllegalStateException) {
                throw RecordingException(e.message ?: HapticPlayer.HAPTICS_UNAVAILABLE)
            }
            val wait = spec.duration + signalConfig.postRoll + 0.1
            try {
                delay((wait * 1000).toLong() + 1)
            } catch (e: kotlinx.coroutines.CancellationException) {
                haptics.stop()
                throw e
            }
            val from = start - signalConfig.preRoll - 0.1
            val to = start + spec.duration + signalConfig.postRoll + 0.1
            val snap = recorder.snapshot(from, to)
            return Recording(
                purpose = purpose, label = label, challenge = spec, hapticStart = start,
                accel = snap.accel, gyro = snap.gyro, motion = snap.motion, device = DeviceSupport.deviceInfo,
            )
        } finally {
            if (startedHere) recorder.stop()
        }
    }
}
