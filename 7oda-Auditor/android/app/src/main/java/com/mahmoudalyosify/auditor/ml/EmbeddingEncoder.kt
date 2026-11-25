package com.mahmoudalyosify.auditor.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.mahmoudalyosify.auditor.core.EncoderInput
import com.mahmoudalyosify.auditor.core.ProcessedSignal
import com.mahmoudalyosify.auditor.model.EncoderOutput
import com.mahmoudalyosify.auditor.model.PatternEncoder
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlin.math.sqrt

/**
 * Wraps the bundled `hand_encoder.onnx` (input `x` float32 [1, 10, 256]; outputs `embedding`
 * [1, 128] and `hand` [1, 3]) with ONNX Runtime on the CPU.
 */
class EmbeddingEncoder private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    override val version: String,
) : PatternEncoder {
    companion object {
        const val ASSET = "hand_encoder.onnx"
        const val EMBEDDING_SIZE = 128

        /** Loads the model from the app assets; throws with a readable message when it cannot. */
        fun load(context: Context): EmbeddingEncoder {
            val bytes = try {
                context.assets.open(ASSET).use { it.readBytes() }
            } catch (e: Exception) {
                throw IllegalStateException("The v1 model ($ASSET) is missing from the app.", e)
            }
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
            }
            val session = env.createSession(bytes, options)
            val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
            val shortHash = digest.take(6).joinToString("") { "%02x".format(it) }
            return EmbeddingEncoder(env, session, "HandEncoder/1/onnx-$shortHash")
        }
    }

    private val lock = Any()

    override fun encode(processed: ProcessedSignal): EncoderOutput = encode(EncoderInput.build(processed))

    /** Encodes a raw input tensor, row-major [channel][u], 2560 values. */
    fun encode(input: FloatArray): EncoderOutput {
        val channels = EncoderInput.channels
        val length = EncoderInput.length
        require(input.size == channels * length) {
            "The v1 model input has ${input.size} values instead of ${channels * length}."
        }
        return synchronized(lock) {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, channels.toLong(), length.toLong()))
                .use { tensor ->
                    session.run(mapOf("x" to tensor)).use { result ->
                        val embeddingValue = result.get("embedding").orElse(null)
                            ?: throw IllegalStateException("The v1 model returned no 'embedding' output.")
                        var embedding = floats(embeddingValue.value).map { it.toDouble() }
                        if (embedding.size != EMBEDDING_SIZE || !embedding.all { it.isFinite() }) {
                            throw IllegalStateException("The v1 model returned no usable 'embedding' output.")
                        }
                        val norm = sqrt(embedding.sumOf { it * it })
                        if (norm > 1e-12) embedding = embedding.map { it / norm }
                        val hand = result.get("hand").orElse(null)?.let { v ->
                            floats(v.value).map { it.toDouble() }
                        } ?: emptyList()
                        EncoderOutput(embedding, hand)
                    }
                }
        }
    }

    /** Flattens a [1, k] (or [k]) float output. */
    private fun floats(value: Any?): List<Float> = when (value) {
        is FloatArray -> value.toList()
        is Array<*> -> value.flatMap { floats(it) }
        is Float -> listOf(value)
        else -> throw IllegalStateException("Unexpected model output type ${value?.javaClass?.simpleName}")
    }
}
