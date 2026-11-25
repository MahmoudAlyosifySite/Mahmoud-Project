package com.mahmoudalyosify.auditor.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.io.File
import java.time.Instant

/** Fixtures shared with the Swift package (written by research/scripts/make_fixtures.py). */
object Fixtures {
    private const val RELATIVE = "AuditorCore/Tests/AuditorCoreTests/Fixtures"

    /** Walks up from the working directory (the module directory under Gradle) to find the fixture. */
    fun file(name: String): File {
        val start = File(System.getProperty("user.dir")).absoluteFile
        var dir: File? = start
        while (dir != null) {
            val candidate = File(dir, "$RELATIVE/$name.json")
            if (candidate.isFile) return candidate
            dir = dir.parentFile
        }
        throw AssertionError(
            "Fixture $name.json not found above $start (expected ../../$RELATIVE relative to android/core; " +
                "run research/scripts/make_fixtures.py)",
        )
    }

    fun json(name: String): JsonElement = Json.parseToJsonElement(file(name).readText())
}

val FIXED_DATE: Instant = Instant.ofEpochSecond(1_795_514_400L)

// MARK: - JSON access with readable errors

fun JsonElement.at(key: String): JsonElement {
    val o = this as? JsonObject ?: throw AssertionError("expected an object holding '$key'")
    return o[key] ?: throw AssertionError("missing key '$key' (keys: ${o.keys.sorted().joinToString()})")
}

fun JsonElement.has(key: String): Boolean = (this as? JsonObject)?.containsKey(key) == true

/** Value of `key` unless it is absent or null. */
fun JsonElement.opt(key: String): JsonElement? {
    val v = (this as? JsonObject)?.get(key)
    return if (v == null || v is JsonNull) null else v
}

fun JsonElement.arr(): JsonArray = this as? JsonArray ?: throw AssertionError("expected an array")

fun JsonElement.isNull(): Boolean = this is JsonNull

/** A number; JSON null reads as +infinity (SPEC 11). */
fun JsonElement.dbl(): Double {
    if (this is JsonNull) return Double.POSITIVE_INFINITY
    val p = this as? JsonPrimitive ?: throw AssertionError("expected a number")
    if (p.isString) throw AssertionError("expected a number, found string '${p.content}'")
    return p.content.toDouble()
}

fun JsonElement.int(): Int {
    val d = dbl()
    if (!d.isFinite() || d != Math.rint(d)) throw AssertionError("expected an integer, found $d")
    return d.toInt()
}

fun JsonElement.str(): String {
    val p = this as? JsonPrimitive ?: throw AssertionError("expected a string")
    if (!p.isString) throw AssertionError("expected a string, found ${p.content}")
    return p.content
}

fun JsonElement.bool(): Boolean {
    val p = this as? JsonPrimitive ?: throw AssertionError("expected a bool")
    return when (p.content) {
        "true" -> true
        "false" -> false
        else -> throw AssertionError("expected a bool, found ${p.content}")
    }
}

fun JsonElement.dbls(): List<Double> = arr().map { it.dbl() }

fun JsonElement.ints(): List<Int> = arr().map { it.int() }

fun JsonElement.matrix(): List<List<Double>> = arr().map { it.dbls() }

fun JsonElement.hex(): ULong {
    val s = str()
    return SeedFormat.parse(s) ?: throw AssertionError("expected a 0x-prefixed hex string, found '$s'")
}

fun JsonElement.prototypeKey(): PrototypeKey {
    val p = arr().map { it.str() }
    if (p.size != 3) throw AssertionError("expected [personID, hand, accessory], found $p")
    return PrototypeKey(
        p[0],
        Hand.from(p[1]) ?: throw AssertionError("bad hand ${p[1]}"),
        Accessory.from(p[2]) ?: throw AssertionError("bad accessory ${p[2]}"),
    )
}

fun JsonElement.outcome(): DecisionOutcome {
    val type = at("type").str()
    return when (type) {
        "identified" -> DecisionOutcome.Identified(
            at("personID").str(),
            Hand.from(at("hand").str()) ?: throw AssertionError("bad hand"),
        )
        "identifiedHandUnclear" -> DecisionOutcome.IdentifiedHandUnclear(at("personID").str())
        "needMorePatterns" -> DecisionOutcome.NeedMorePatterns
        "unknown" -> DecisionOutcome.Unknown(UnknownReason.from(at("reason").str()) ?: throw AssertionError("bad reason"))
        else -> throw AssertionError("unreadable outcome type '$type'")
    }
}

// MARK: - Failure collector (reports many mismatches at once, like XCTFail)

class Checks(private val limit: Int = 40) {
    private val failures = ArrayList<String>()
    private var count = 0

    fun fail(message: String) {
        count += 1
        if (failures.size < limit) failures.add(message)
    }

    fun check(condition: Boolean, message: () -> String) {
        if (!condition) fail(message())
    }

    fun <T> equal(actual: T, expected: T, what: String) {
        if (actual != expected) fail("$what: got $actual, expected $expected")
    }

    fun close(actual: Double, expected: Double, rel: Double, what: String): Boolean {
        if (actual == expected) return true
        if (actual.isNaN() || expected.isNaN() || actual.isInfinite() || expected.isInfinite()) {
            fail("$what: got $actual, expected $expected")
            return false
        }
        val tolerance = rel * maxOf(1.0, Math.abs(expected))
        val delta = Math.abs(actual - expected)
        if (delta > tolerance) {
            fail("$what: got $actual, expected $expected (|d| = $delta > $tolerance)")
            return false
        }
        return true
    }

    fun close(actual: List<Double>, expected: List<Double>, rel: Double, what: String): Boolean {
        if (actual.size != expected.size) {
            fail("$what: length ${actual.size}, expected ${expected.size}")
            return false
        }
        var bad = 0
        for (i in actual.indices) {
            if (!close(actual[i], expected[i], rel, "$what[$i]")) {
                bad += 1
                if (bad >= 5) return false
            }
        }
        return bad == 0
    }

    fun close(actual: DoubleArray, expected: List<Double>, rel: Double, what: String): Boolean =
        close(actual.toList(), expected, rel, what)

    fun done() {
        if (count > 0) {
            throw AssertionError("$count check(s) failed:\n" + failures.joinToString("\n"))
        }
    }
}

/** Runs `body` with a collector and fails the test if any check failed. */
fun checks(body: (Checks) -> Unit) {
    val c = Checks()
    body(c)
    c.done()
}
