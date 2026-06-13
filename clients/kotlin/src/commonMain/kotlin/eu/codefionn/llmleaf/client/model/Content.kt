package eu.codefionn.llmleaf.client.model

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** The `image_url` object inside a [ContentPart.ImageUrlPart]. */
@Serializable
public data class ImageUrl(
    @SerialName("url") val url: String,
    @SerialName("detail") val detail: String? = null, // "auto" | "low" | "high"
)

/**
 * One part of a multimodal message. Discriminated on the wire by `type`:
 *   {"type":"text","text":"..."}
 *   {"type":"image_url","image_url":{"url":"...","detail":"auto"}}
 */
@Serializable(with = ContentPartSerializer::class)
public sealed interface ContentPart {
    public data class Text(val text: String) : ContentPart

    public data class ImageUrlPart(val imageUrl: ImageUrl) : ContentPart
}

/** Manual discriminated-union serializer keyed on `type` (proto `ContentPart.part` oneof). */
public object ContentPartSerializer : KSerializer<ContentPart> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ContentPart")

    override fun serialize(encoder: Encoder, value: ContentPart) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ContentPart requires kotlinx.serialization JSON")
        val json = jsonEncoder.json
        val element: JsonElement = when (value) {
            is ContentPart.Text -> JsonObject(
                mapOf(
                    "type" to JsonPrimitive("text"),
                    "text" to JsonPrimitive(value.text),
                ),
            )
            is ContentPart.ImageUrlPart -> JsonObject(
                mapOf(
                    "type" to JsonPrimitive("image_url"),
                    "image_url" to json.encodeToJsonElement(ImageUrl.serializer(), value.imageUrl),
                ),
            )
        }
        jsonEncoder.encodeJsonElement(element)
    }

    override fun deserialize(decoder: Decoder): ContentPart {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ContentPart requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as? JsonObject
            ?: error("content part must be a JSON object")
        return when ((obj["type"] as? JsonPrimitive)?.content) {
            "text" -> ContentPart.Text((obj["text"] as JsonPrimitive).content)
            "image_url" -> ContentPart.ImageUrlPart(
                jsonDecoder.json.decodeFromJsonElement(ImageUrl.serializer(), obj.getValue("image_url")),
            )
            else -> error("unknown content part type: ${obj["type"]}")
        }
    }
}

/**
 * A message's `content`: either a plain [Text] string or an array of [Parts] (proto
 * `ChatMessage.content` oneof). It serialises as a bare JSON string when [Text], else as an
 * array of content-part objects.
 */
@Serializable(with = MessageContentSerializer::class)
public sealed interface MessageContent {
    public data class Text(val text: String) : MessageContent

    public data class Parts(val parts: List<ContentPart>) : MessageContent

    public companion object {
        /** Convenience constructor for the common plain-text case. */
        public fun text(value: String): MessageContent = Text(value)
    }
}

public object MessageContentSerializer : KSerializer<MessageContent> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("MessageContent")

    override fun serialize(encoder: Encoder, value: MessageContent) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("MessageContent requires kotlinx.serialization JSON")
        when (value) {
            is MessageContent.Text -> jsonEncoder.encodeJsonElement(JsonPrimitive(value.text))
            is MessageContent.Parts -> {
                val arr = jsonEncoder.json.encodeToJsonElement(
                    ListSerializer(ContentPartSerializer),
                    value.parts,
                )
                jsonEncoder.encodeJsonElement(arr)
            }
        }
    }

    override fun deserialize(decoder: Decoder): MessageContent {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("MessageContent requires kotlinx.serialization JSON")
        return when (val element = jsonDecoder.decodeJsonElement()) {
            is JsonPrimitive -> MessageContent.Text(element.content)
            is JsonArray -> MessageContent.Parts(
                jsonDecoder.json.decodeFromJsonElement(ListSerializer(ContentPartSerializer), element),
            )
            else -> error("content must be a string or array")
        }
    }
}
