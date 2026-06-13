package eu.codefionn.llmleaf.client.model

import kotlinx.serialization.KSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonPrimitive

/**
 * Serializer for a `List<String>` field that the wire accepts as either a bare string (one
 * element) or an array. Used for `ChatRequest.stop` and `EmbeddingRequest.input`.
 *
 * On encode: a single element emits a bare string, otherwise an array (SPEC.md: "emitting an
 * array is fine" too, but a bare string for one element matches the OpenAI surface exactly).
 * On decode: a bare string becomes a one-element list; an array becomes the list.
 */
public object StringOrListSerializer : KSerializer<List<String>> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("StringOrList")

    override fun serialize(encoder: Encoder, value: List<String>) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("StringOrList requires kotlinx.serialization JSON")
        if (value.size == 1) {
            jsonEncoder.encodeJsonElement(JsonPrimitive(value[0]))
        } else {
            jsonEncoder.encodeJsonElement(
                JsonArray(value.map { JsonPrimitive(it) }),
            )
        }
    }

    override fun deserialize(decoder: Decoder): List<String> {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("StringOrList requires kotlinx.serialization JSON")
        return when (val element = jsonDecoder.decodeJsonElement()) {
            is JsonPrimitive -> listOf(element.content)
            is JsonArray -> jsonDecoder.json.decodeFromJsonElement(
                ListSerializer(String.serializer()),
                element,
            )
            else -> error("expected a string or array of strings")
        }
    }
}
