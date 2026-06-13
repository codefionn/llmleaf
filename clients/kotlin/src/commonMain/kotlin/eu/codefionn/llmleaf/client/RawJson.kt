package eu.codefionn.llmleaf.client

import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonEncoder

/**
 * A free-form JSON field carried as a raw JSON string.
 *
 * Per SPEC.md, fields like `ChatRequest.extra`, `FunctionDef.parameters`,
 * `ResponseFormat.json_schema`, `EmbeddingRequest.extra`, `SpeechRequest.extra` and
 * `ModelEntry.default_parameters` are arbitrary JSON. On encode the parsed JSON value is
 * **spliced** into the body at that key (never double-encoded as a string); on decode the
 * sub-object is captured back as a raw JSON string.
 *
 * Wrap the raw text in [RawJson] and the [RawJsonSerializer] handles the splice in both
 * directions. The contained [value] is exactly what the caller passed in / what the server
 * sent, e.g. `{"foo":1}` or `[1,2,3]`.
 */
@Serializable(with = RawJsonSerializer::class)
public class RawJson(public val value: String) {
    override fun toString(): String = value

    override fun equals(other: Any?): Boolean = other is RawJson && other.value == value

    override fun hashCode(): Int = value.hashCode()

    public companion object {
        /** Builds a [RawJson] from an already-parsed [JsonElement]. */
        public fun of(element: JsonElement): RawJson = RawJson(LenientJson.encodeToString(JsonElement.serializer(), element))
    }
}

/**
 * Splices a [RawJson] into the surrounding JSON object instead of emitting it as a string
 * literal. Requires a kotlinx.serialization JSON encoder/decoder (the SDK always uses one).
 */
public object RawJsonSerializer : KSerializer<RawJson> {
    // The descriptor models the value as a generic JSON element; the real work happens
    // against the Json{Encoder,Decoder} below so the value is inlined, not stringified.
    override val descriptor: SerialDescriptor = JsonElement.serializer().descriptor

    override fun serialize(encoder: Encoder, value: RawJson) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("RawJson can only be serialised with kotlinx.serialization JSON")
        val element = jsonEncoder.json.parseToJsonElement(value.value)
        jsonEncoder.encodeJsonElement(element)
    }

    override fun deserialize(decoder: Decoder): RawJson {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("RawJson can only be deserialised with kotlinx.serialization JSON")
        val element = jsonDecoder.decodeJsonElement()
        return RawJson(jsonDecoder.json.encodeToString(JsonElement.serializer(), element))
    }
}

/**
 * The single JSON configuration the whole SDK shares. `encodeDefaults = false` keeps absent
 * optional fields off the wire (matching the proto's `*_UNSPECIFIED ⇔ field absent` rule),
 * `ignoreUnknownKeys` makes responses forward-compatible, and `explicitNulls = false` keeps
 * `null` optionals out of request bodies.
 */
public val LenientJson: Json = Json {
    encodeDefaults = false
    explicitNulls = false
    ignoreUnknownKeys = true
    isLenient = true
}
