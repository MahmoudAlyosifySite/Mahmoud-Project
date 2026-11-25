package com.mahmoudalyosify.auditor.core

import java.time.Instant
import java.util.UUID

// MARK: - Streams (SPEC 4)

/** Three-axis sensor samples with their timestamps (seconds). Arrays have equal length. */
data class SensorStream(
    val t: List<Double> = emptyList(),
    val x: List<Double> = emptyList(),
    val y: List<Double> = emptyList(),
    val z: List<Double> = emptyList(),
) {
    companion object {
        val empty: SensorStream = SensorStream()
    }
}

/** Device-motion samples (gravity in g, user acceleration in g, rotation rate in rad/s, attitude quaternion). */
data class MotionStream(
    val t: List<Double> = emptyList(),
    val gravityX: List<Double> = emptyList(),
    val gravityY: List<Double> = emptyList(),
    val gravityZ: List<Double> = emptyList(),
    val userAccelX: List<Double> = emptyList(),
    val userAccelY: List<Double> = emptyList(),
    val userAccelZ: List<Double> = emptyList(),
    val rotationX: List<Double> = emptyList(),
    val rotationY: List<Double> = emptyList(),
    val rotationZ: List<Double> = emptyList(),
    val attitudeW: List<Double> = emptyList(),
    val attitudeX: List<Double> = emptyList(),
    val attitudeY: List<Double> = emptyList(),
    val attitudeZ: List<Double> = emptyList(),
) {
    companion object {
        val empty: MotionStream = MotionStream()
    }
}

// MARK: - Labels (SPEC 4). Constant names equal the JSON raw values.

@Suppress("EnumEntryName")
enum class Hand {
    right, left;

    val rawValue: String get() = name

    companion object {
        fun from(raw: String): Hand? = entries.firstOrNull { it.name == raw }
    }
}

@Suppress("EnumEntryName")
enum class Accessory {
    none, watch, ring, bracelet, other;

    val rawValue: String get() = name

    companion object {
        fun from(raw: String): Accessory? = entries.firstOrNull { it.name == raw }
    }
}

@Suppress("EnumEntryName")
enum class Posture {
    sitting, standing, freshGrip, unspecified;

    val rawValue: String get() = name

    companion object {
        fun from(raw: String): Posture? = entries.firstOrNull { it.name == raw }
    }
}

@Suppress("EnumEntryName")
enum class RecordingPurpose {
    calibration, enrollment, accessory, check, test, signalCheck;

    val rawValue: String get() = name

    companion object {
        fun from(raw: String): RecordingPurpose? = entries.firstOrNull { it.name == raw }
    }
}

data class RecordingLabel(
    val personID: String,
    val personName: String,
    val hand: Hand,
    val accessory: Accessory = Accessory.none,
    val posture: Posture = Posture.unspecified,
    val sessionID: String,
)

data class DeviceInfo(val model: String, val system: String, val app: String)

// MARK: - Recording (SPEC 4)

/**
 * One challenge with its raw sensor data; also the export format. `accel` is in g and includes gravity,
 * `gyro` is in rad/s, all timestamps (and `hapticStart`) use the same clock in seconds.
 */
data class Recording(
    val schema: String = currentSchema,
    val id: UUID = UUID.randomUUID(),
    val createdAt: Instant = Instant.now(),
    val purpose: RecordingPurpose,
    val label: RecordingLabel? = null,
    val challenge: ChallengeSpec,
    val hapticStart: Double,
    val accel: SensorStream,
    val gyro: SensorStream,
    val motion: MotionStream = MotionStream.empty,
    val device: DeviceInfo? = null,
) {
    companion object {
        const val currentSchema: String = "7oda-auditor.recording.v1"
    }
}
