package eu.codefionn.llmleaf.client.model

import eu.codefionn.llmleaf.client.RawJson
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

/**
 * An embeddings request. Like [ChatRequest], `extra` keys are merged at the top level, so the
 * struct uses a custom serializer over an internal derived [EmbeddingRequestWire]. `input`
 * serialises as a bare string when it has one element, else an array.
 */
@Serializable(with = EmbeddingRequestSerializer::class)
public data class EmbeddingRequest(
    val model: String,
    val input: List<String>,
    val dimensions: Int? = null,
    val encodingFormat: String? = null, // "float" | "base64"
    val extra: RawJson? = null,
)

@Serializable
internal data class EmbeddingRequestWire(
    @SerialName("model") val model: String,
    @Serializable(with = StringOrListSerializer::class)
    @SerialName("input") val input: List<String>,
    @SerialName("dimensions") val dimensions: Int? = null,
    @SerialName("encoding_format") val encodingFormat: String? = null,
)

public object EmbeddingRequestSerializer : KSerializer<EmbeddingRequest> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("EmbeddingRequest")

    override fun serialize(encoder: Encoder, value: EmbeddingRequest) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("EmbeddingRequest requires kotlinx.serialization JSON")
        val json = jsonEncoder.json
        val wire = EmbeddingRequestWire(
            model = value.model,
            input = value.input,
            dimensions = value.dimensions,
            encodingFormat = value.encodingFormat,
        )
        val base = json.encodeToJsonElement(EmbeddingRequestWire.serializer(), wire) as JsonObject
        jsonEncoder.encodeJsonElement(mergeExtra(json, base, value.extra))
    }

    override fun deserialize(decoder: Decoder): EmbeddingRequest {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("EmbeddingRequest requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        val wire = jsonDecoder.json.decodeFromJsonElement(EmbeddingRequestWire.serializer(), obj)
        return EmbeddingRequest(wire.model, wire.input, wire.dimensions, wire.encodingFormat)
    }
}

/**
 * A single embedding. The wire carries `embedding` as either a float array (`encoding_format:
 * "float"`) or a base64 string of little-endian f32 bytes (`encoding_format: "base64"`); both
 * decode into [embedding].
 */
@Serializable(with = EmbeddingSerializer::class)
public data class Embedding(
    val `object`: String = "embedding",
    val index: Int = 0,
    val embedding: List<Float> = emptyList(),
)

public object EmbeddingSerializer : KSerializer<Embedding> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("Embedding")

    override fun serialize(encoder: Encoder, value: Embedding) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("Embedding requires kotlinx.serialization JSON")
        val obj = JsonObject(
            mapOf(
                "object" to JsonPrimitive(value.`object`),
                "index" to JsonPrimitive(value.index),
                "embedding" to JsonArray(value.embedding.map { JsonPrimitive(it) }),
            ),
        )
        jsonEncoder.encodeJsonElement(obj)
    }

    override fun deserialize(decoder: Decoder): Embedding {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("Embedding requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        val json = jsonDecoder.json
        val rawObject = (obj["object"] as? JsonPrimitive)?.content ?: "embedding"
        val index = (obj["index"] as? JsonPrimitive)?.content?.toIntOrNull() ?: 0
        val floats = when (val e = obj["embedding"]) {
            is JsonArray -> json.decodeFromJsonElement(ListSerializer(Float.serializer()), e)
            is JsonPrimitive -> decodeBase64Floats(e.content)
            null -> emptyList()
            else -> error("embedding must be an array or base64 string")
        }
        return Embedding(rawObject, index, floats)
    }
}

/** Decodes a base64 string of little-endian f32 bytes into a list of floats (SPEC.md). */
@OptIn(ExperimentalEncodingApi::class)
internal fun decodeBase64Floats(b64: String): List<Float> {
    val bytes = Base64.decode(b64)
    require(bytes.size % 4 == 0) { "base64 embedding length is not a multiple of 4" }
    val out = ArrayList<Float>(bytes.size / 4)
    var i = 0
    while (i < bytes.size) {
        // Little-endian: byte 0 is the least significant.
        val bits = (bytes[i].toInt() and 0xFF) or
            ((bytes[i + 1].toInt() and 0xFF) shl 8) or
            ((bytes[i + 2].toInt() and 0xFF) shl 16) or
            ((bytes[i + 3].toInt() and 0xFF) shl 24)
        out.add(Float.fromBits(bits))
        i += 4
    }
    return out
}

@Serializable
public data class EmbeddingResponse(
    @SerialName("object") val `object`: String = "list",
    @SerialName("data") val data: List<Embedding> = emptyList(),
    @SerialName("model") val model: String,
    @SerialName("usage") val usage: Usage? = null,
)
