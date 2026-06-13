package eu.codefionn.llmleaf.client.model

import eu.codefionn.llmleaf.client.RawJson
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** A function the model called. `arguments` is a JSON-encoded string (OpenAI shape). */
@Serializable
public data class FunctionCall(
    @SerialName("name") val name: String,
    @SerialName("arguments") val arguments: String,
)

@Serializable
public data class ToolCall(
    @SerialName("id") val id: String,
    @SerialName("type") val type: String = "function",
    @SerialName("function") val function: FunctionCall,
)

/** Incremental tool-call fragment on a streaming delta; fields arrive piecemeal. */
@Serializable
public data class FunctionCallDelta(
    @SerialName("name") val name: String? = null,
    @SerialName("arguments") val arguments: String? = null,
)

@Serializable
public data class ToolCallDelta(
    @SerialName("index") val index: Int,
    @SerialName("id") val id: String? = null,
    @SerialName("type") val type: String? = null,
    @SerialName("function") val function: FunctionCallDelta? = null,
)

/** A function the model MAY call. `parameters` is a raw JSON-Schema object (spliced inline). */
@Serializable
public data class FunctionDef(
    @SerialName("name") val name: String,
    @SerialName("description") val description: String? = null,
    @SerialName("parameters") val parameters: RawJson? = null,
)

@Serializable
public data class ToolDef(
    @SerialName("type") val type: String = "function",
    @SerialName("function") val function: FunctionDef,
)

@Serializable
public data class FunctionName(
    @SerialName("name") val name: String,
)

@Serializable
public data class NamedToolChoice(
    @SerialName("type") val type: String = "function",
    @SerialName("function") val function: FunctionName,
)

/**
 * `tool_choice`: either a bare mode string (`"auto"`/`"none"`/`"required"`) or a named-function
 * object (proto `ToolChoice.choice` oneof).
 */
@Serializable(with = ToolChoiceSerializer::class)
public sealed interface ToolChoice {
    public data class Mode(val mode: String) : ToolChoice

    public data class Named(val named: NamedToolChoice) : ToolChoice

    public companion object {
        public val AUTO: ToolChoice = Mode("auto")
        public val NONE: ToolChoice = Mode("none")
        public val REQUIRED: ToolChoice = Mode("required")

        public fun function(name: String): ToolChoice = Named(NamedToolChoice(function = FunctionName(name)))
    }
}

public object ToolChoiceSerializer : KSerializer<ToolChoice> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ToolChoice")

    override fun serialize(encoder: Encoder, value: ToolChoice) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ToolChoice requires kotlinx.serialization JSON")
        when (value) {
            is ToolChoice.Mode -> jsonEncoder.encodeJsonElement(JsonPrimitive(value.mode))
            is ToolChoice.Named -> jsonEncoder.encodeJsonElement(
                jsonEncoder.json.encodeToJsonElement(NamedToolChoice.serializer(), value.named),
            )
        }
    }

    override fun deserialize(decoder: Decoder): ToolChoice {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ToolChoice requires kotlinx.serialization JSON")
        return when (val element = jsonDecoder.decodeJsonElement()) {
            is JsonPrimitive -> ToolChoice.Mode(element.content)
            is JsonObject -> ToolChoice.Named(
                jsonDecoder.json.decodeFromJsonElement(NamedToolChoice.serializer(), element),
            )
            else -> error("tool_choice must be a string or object")
        }
    }
}

/** response_format: {"type":"text"|"json_object"|"json_schema", "json_schema": {...}}. */
@Serializable
public data class ResponseFormat(
    @SerialName("type") val type: String,
    @SerialName("json_schema") val jsonSchema: RawJson? = null,
)
