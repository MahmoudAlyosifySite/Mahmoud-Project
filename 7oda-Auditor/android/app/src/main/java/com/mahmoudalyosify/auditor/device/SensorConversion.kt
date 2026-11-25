package com.mahmoudalyosify.auditor.device

import kotlin.math.sqrt

/**
 * Converts Android sensor values to the iOS conventions of SPEC 4, so recordings from both
 * platforms go through the same pipeline and research tools.
 *
 * Android reports specific force in m/s² (+9.81 on z when the phone lies face up); Core Motion
 * reports acceleration in g with the opposite sign (−1 on z face up). Gravity and linear
 * acceleration follow the same rule. Angular rates are rad/s with the same axes on both.
 */
object SensorConversion {
    const val STANDARD_GRAVITY = 9.80665

    /** Accelerometer, gravity or linear acceleration axis: m/s² (Android) → g (iOS sign). */
    fun toIOSG(metersPerSecondSquared: Float): Double = -metersPerSecondSquared.toDouble() / STANDARD_GRAVITY

    /** Gyroscope axis: rad/s on both platforms. */
    fun rotation(radPerSecond: Float): Double = radPerSecond.toDouble()

    /** SensorEvent.timestamp (nanoseconds, elapsedRealtimeNanos base) → seconds. */
    fun seconds(timestampNanos: Long): Double = timestampNanos / 1e9

    /**
     * Unit quaternion (w, x, y, z) from a rotation-vector event: values are x·sin(θ/2),
     * y·sin(θ/2), z·sin(θ/2) and, when present, cos(θ/2).
     */
    fun quaternion(values: FloatArray): DoubleArray {
        val x = values.getOrElse(0) { 0f }.toDouble()
        val y = values.getOrElse(1) { 0f }.toDouble()
        val z = values.getOrElse(2) { 0f }.toDouble()
        val w = if (values.size >= 4) values[3].toDouble() else sqrt(maxOf(0.0, 1.0 - x * x - y * y - z * z))
        val norm = sqrt(w * w + x * x + y * y + z * z)
        return if (norm > 1e-12) doubleArrayOf(w / norm, x / norm, y / norm, z / norm) else doubleArrayOf(1.0, 0.0, 0.0, 0.0)
    }
}
