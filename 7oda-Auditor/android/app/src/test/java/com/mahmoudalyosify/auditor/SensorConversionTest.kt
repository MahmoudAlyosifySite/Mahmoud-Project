package com.mahmoudalyosify.auditor

import com.mahmoudalyosify.auditor.device.SensorConversion
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class SensorConversionTest {
    @Test
    fun faceUpGravityBecomesMinusOneG() {
        // Android reports +9.81 m/s² on z when the phone lies face up; iOS reports −1 g.
        assertEquals(-1.0, SensorConversion.toIOSG(9.80665f), 1e-6)
        assertEquals(0.5, SensorConversion.toIOSG(-4.903325f), 1e-6)
        assertEquals(0.0, SensorConversion.toIOSG(0f), 0.0)
    }

    @Test
    fun rotationIsUnchangedAndTimestampsAreSeconds() {
        assertEquals(1.25, SensorConversion.rotation(1.25f), 1e-7)
        assertEquals(81234.5612, SensorConversion.seconds(81_234_561_200_000L), 1e-9)
    }

    @Test
    fun quaternionFromRotationVector() {
        assertArrayEquals(doubleArrayOf(1.0, 0.0, 0.0, 0.0), SensorConversion.quaternion(floatArrayOf(0f, 0f, 0f, 1f)), 1e-9)
        // Without the scalar part, w is reconstructed from the unit norm.
        val q = SensorConversion.quaternion(floatArrayOf(0.6f, 0f, 0f))
        assertEquals(0.8, q[0], 1e-6)
        assertEquals(0.6, q[1], 1e-6)
        val n = SensorConversion.quaternion(floatArrayOf(0f, 0f, 2f, 0f))
        assertEquals(1.0, n[3], 1e-9)
    }
}
