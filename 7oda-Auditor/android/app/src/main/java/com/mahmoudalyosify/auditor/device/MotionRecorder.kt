package com.mahmoudalyosify.auditor.device

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.os.SystemClock
import com.mahmoudalyosify.auditor.core.MotionStream
import com.mahmoudalyosify.auditor.core.SensorStream

/** Measured delivery rates in Hz over the last two seconds. */
data class SensorRates(val accel: Double, val gyro: Double, val motion: Double)

/** One cut of the ring buffer, already in the iOS conventions of SPEC 4. */
class SensorSnapshot(val accel: SensorStream, val gyro: SensorStream, val motion: MotionStream)

/**
 * Streams accelerometer, gyroscope, gravity, linear acceleration and game rotation vector at
 * 10 000 µs on a HandlerThread into a thread-safe ring buffer of the last 12 s. Start/stop are
 * reference counted so overlapping screens can share it.
 *
 * The motion stream (Core Motion's device motion) is assembled from the fused sensors: one
 * sample per gravity event, carrying the latest linear acceleration, gyroscope rate and
 * attitude. Without a gravity sensor, gravity is a low-pass estimate of the accelerometer.
 */
class MotionRecorder(context: Context) {
    companion object {
        const val SAMPLING_PERIOD_US = 10_000
        const val BUFFER_SECONDS = 12.0
        private const val GRAVITY_LOW_PASS = 0.1f
    }

    private class Vec3(val t: Double, val x: Double, val y: Double, val z: Double)

    private class MotionSample(
        val t: Double,
        val gx: Double, val gy: Double, val gz: Double,
        val ux: Double, val uy: Double, val uz: Double,
        val rx: Double, val ry: Double, val rz: Double,
        val qw: Double, val qx: Double, val qy: Double, val qz: Double,
    )

    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager?
    private val accelerometer: Sensor? = manager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope: Sensor? = manager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val gravitySensor: Sensor? = manager?.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val linearSensor: Sensor? = manager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val rotationSensor: Sensor? = manager?.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

    private val lock = Any()
    private val accel = ArrayDeque<Vec3>()
    private val gyro = ArrayDeque<Vec3>()
    private val motion = ArrayDeque<MotionSample>()
    private var users = 0
    private var startedAtNanos: Long? = null
    private var thread: HandlerThread? = null

    // Latest values of the fused sensors, Android units (m/s², rad/s).
    private val lastGravity = FloatArray(3)
    private var haveGravity = false
    private val lastLinear = FloatArray(3)
    private var haveLinear = false
    private val lastGyro = FloatArray(3)
    private val lastAccel = FloatArray(3)
    private var haveAccel = false
    private var lastQuaternion = doubleArrayOf(1.0, 0.0, 0.0, 0.0)

    val isAccelerometerAvailable: Boolean get() = accelerometer != null
    val isGyroAvailable: Boolean get() = gyroscope != null
    val hasGravitySensor: Boolean get() = gravitySensor != null
    val hasRotationVector: Boolean get() = rotationSensor != null

    /** Accelerometer and gyroscope both exist (the fused streams fall back if missing). */
    val isAvailable: Boolean get() = isAccelerometerAvailable && isGyroAvailable

    /** Seconds since the sensors were started (0 when stopped). */
    val secondsRunning: Double
        get() = synchronized(lock) {
            val s = startedAtNanos ?: return 0.0
            (SystemClock.elapsedRealtimeNanos() - s) / 1e9
        }

    val isRunning: Boolean get() = synchronized(lock) { users > 0 }

    /** Names and vendors of the sensors in use, for the Lab. */
    fun sensorDescriptions(): List<Pair<String, String>> {
        fun d(s: Sensor?): String = s?.let { "${it.name} (${it.vendor}), min delay ${it.minDelay} µs" } ?: "missing"
        return listOf(
            "Accelerometer" to d(accelerometer),
            "Gyroscope" to d(gyroscope),
            "Gravity" to d(gravitySensor),
            "Linear acceleration" to d(linearSensor),
            "Game rotation vector" to d(rotationSensor),
        )
    }

    private val listener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) = handle(event)
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    fun start() {
        val first = synchronized(lock) {
            users += 1
            val isFirst = users == 1
            if (isFirst) {
                accel.clear(); gyro.clear(); motion.clear()
                haveGravity = false; haveLinear = false
                lastGyro.fill(0f)
                lastQuaternion = doubleArrayOf(1.0, 0.0, 0.0, 0.0)
                haveAccel = false
                startedAtNanos = SystemClock.elapsedRealtimeNanos()
            }
            isFirst
        }
        if (!first) return
        val m = manager ?: return
        val t = HandlerThread("auditor.motion", Process.THREAD_PRIORITY_URGENT_DISPLAY).also { it.start() }
        val handler = Handler(t.looper)
        synchronized(lock) { thread = t }
        for (sensor in listOf(accelerometer, gyroscope, gravitySensor, linearSensor, rotationSensor)) {
            if (sensor != null) m.registerListener(listener, sensor, SAMPLING_PERIOD_US, handler)
        }
    }

    fun stop() {
        val t: HandlerThread? = synchronized(lock) {
            if (users == 0) return
            users -= 1
            if (users > 0) return
            startedAtNanos = null
            val current = thread
            thread = null
            current
        }
        manager?.unregisterListener(listener)
        t?.quitSafely()
    }

    private fun handle(event: SensorEvent) {
        val t = SensorConversion.seconds(event.timestamp)
        val v = event.values
        synchronized(lock) {
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    append(accel, Vec3(t, SensorConversion.toIOSG(v[0]), SensorConversion.toIOSG(v[1]),
                        SensorConversion.toIOSG(v[2])))
                    for (i in 0..2) lastAccel[i] = v[i]
                    haveAccel = true
                    if (gravitySensor == null) {
                        // Low-pass gravity estimate and its residual stand in for the fused sensors.
                        if (!haveGravity) {
                            for (i in 0..2) lastGravity[i] = v[i]
                            haveGravity = true
                        } else {
                            for (i in 0..2) lastGravity[i] += GRAVITY_LOW_PASS * (v[i] - lastGravity[i])
                        }
                        if (linearSensor == null) {
                            for (i in 0..2) lastLinear[i] = v[i] - lastGravity[i]
                            haveLinear = true
                        }
                        appendMotion(t)
                    }
                }
                Sensor.TYPE_GYROSCOPE -> {
                    append(gyro, Vec3(t, SensorConversion.rotation(v[0]), SensorConversion.rotation(v[1]),
                        SensorConversion.rotation(v[2])))
                    for (i in 0..2) lastGyro[i] = v[i]
                }
                Sensor.TYPE_GRAVITY -> {
                    for (i in 0..2) lastGravity[i] = v[i]
                    haveGravity = true
                    appendMotion(t)
                }
                Sensor.TYPE_LINEAR_ACCELERATION -> {
                    for (i in 0..2) lastLinear[i] = v[i]
                    haveLinear = true
                }
                Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                    lastQuaternion = SensorConversion.quaternion(v)
                }
            }
        }
    }

    /** Called with the lock held. */
    private fun appendMotion(t: Double) {
        if (!haveGravity) return
        val q = lastQuaternion
        val linear = when {
            haveLinear -> lastLinear
            haveAccel -> FloatArray(3) { i -> lastAccel[i] - lastGravity[i] }
            else -> FloatArray(3)
        }
        val sample = MotionSample(
            t = t,
            gx = SensorConversion.toIOSG(lastGravity[0]),
            gy = SensorConversion.toIOSG(lastGravity[1]),
            gz = SensorConversion.toIOSG(lastGravity[2]),
            ux = SensorConversion.toIOSG(linear[0]),
            uy = SensorConversion.toIOSG(linear[1]),
            uz = SensorConversion.toIOSG(linear[2]),
            rx = SensorConversion.rotation(lastGyro[0]),
            ry = SensorConversion.rotation(lastGyro[1]),
            rz = SensorConversion.rotation(lastGyro[2]),
            qw = q[0], qx = q[1], qy = q[2], qz = q[3],
        )
        motion.addLast(sample)
        while (motion.isNotEmpty() && motion.first().t < t - BUFFER_SECONDS) motion.removeFirst()
    }

    private fun append(buffer: ArrayDeque<Vec3>, s: Vec3) {
        buffer.addLast(s)
        while (buffer.isNotEmpty() && buffer.first().t < s.t - BUFFER_SECONDS) buffer.removeFirst()
    }

    /** All samples with `from <= t <= to`, in the iOS conventions of SPEC 4. */
    fun snapshot(from: Double, to: Double): SensorSnapshot = synchronized(lock) {
        fun stream(buffer: ArrayDeque<Vec3>): SensorStream {
            val items = buffer.filter { it.t in from..to }
            return SensorStream(items.map { it.t }, items.map { it.x }, items.map { it.y }, items.map { it.z })
        }
        val m = motion.filter { it.t in from..to }
        SensorSnapshot(
            accel = stream(accel),
            gyro = stream(gyro),
            motion = MotionStream(
                t = m.map { it.t },
                gravityX = m.map { it.gx }, gravityY = m.map { it.gy }, gravityZ = m.map { it.gz },
                userAccelX = m.map { it.ux }, userAccelY = m.map { it.uy }, userAccelZ = m.map { it.uz },
                rotationX = m.map { it.rx }, rotationY = m.map { it.ry }, rotationZ = m.map { it.rz },
                attitudeW = m.map { it.qw }, attitudeX = m.map { it.qx },
                attitudeY = m.map { it.qy }, attitudeZ = m.map { it.qz },
            ),
        )
    }

    /** Delivery rates over the last two seconds of each stream (0 when stopped). */
    fun measuredRates(): SensorRates = synchronized(lock) {
        fun rate(times: List<Double>): Double {
            if (times.size < 2) return 0.0
            val latest = times.last()
            val recent = times.filter { it >= latest - 2.0 }
            if (recent.size < 2) return 0.0
            val span = recent.last() - recent.first()
            return if (span > 0) (recent.size - 1) / span else 0.0
        }
        SensorRates(rate(accel.map { it.t }), rate(gyro.map { it.t }), rate(motion.map { it.t }))
    }
}
