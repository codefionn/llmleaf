@file:OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)

package eu.codefionn.llmleaf.client.model

import eu.codefionn.llmleaf.client.RawJson
import kotlinx.serialization.EncodeDefault
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
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put

// ----------------------------------------------------------------------------
// Responses (POST /v1/responses) — the OpenAI Responses dialect over the same
// canonical core. See SPEC.md "POST /v1/responses" and the proto's Responses
// section. Like the chat dialect these are hand-written kotlinx.serialization
// types that own the bytes on the wire (the Wire-generated proto types under
// `eu.codefionn.llmleaf.v1` only prove the schema compiles).
// ----------------------------------------------------------------------------

/** Reads a string field, treating JSON `null` as absent. */
private fun JsonObject.str(key: String): String? = (this[key] as? JsonPrimitive)?.contentOrNull

// --- Content parts ---------------------------------------------------------

/**
 * One content part of a Responses message. The wire `"type"` token matches the set field's
 * name (SPEC.md):
 *   {"type":"input_text","text":"..."}
 *   {"type":"input_image","image_url":"<url>","detail":"auto"}   (image_url is a STRING here,
 *       not the chat dialect's nested `{url}` object)
 *   {"type":"output_text","text":"...","annotations":[]}          (annotations emitted as `[]`)
 */
@Serializable(with = ResponseContentPartSerializer::class)
public sealed interface ResponseContentPart {
    public data class InputText(val text: String) : ResponseContentPart

    public data class InputImage(val imageUrl: String, val detail: String? = null) : ResponseContentPart

    public data class OutputText(val text: String) : ResponseContentPart
}

public object ResponseContentPartSerializer : KSerializer<ResponseContentPart> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponseContentPart")

    override fun serialize(encoder: Encoder, value: ResponseContentPart) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponseContentPart requires kotlinx.serialization JSON")
        val element: JsonElement = when (value) {
            is ResponseContentPart.InputText -> buildJsonObject {
                put("type", "input_text")
                put("text", value.text)
            }
            is ResponseContentPart.InputImage -> buildJsonObject {
                put("type", "input_image")
                put("image_url", value.imageUrl) // plain string
                value.detail?.let { put("detail", it) }
            }
            is ResponseContentPart.OutputText -> buildJsonObject {
                put("type", "output_text")
                put("text", value.text)
                // Constructed output_text parts always carry an (empty) annotations array.
                put("annotations", JsonArray(emptyList()))
            }
        }
        jsonEncoder.encodeJsonElement(element)
    }

    override fun deserialize(decoder: Decoder): ResponseContentPart {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponseContentPart requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as? JsonObject
            ?: error("content part must be a JSON object")
        return when (obj.str("type")) {
            "input_text" -> ResponseContentPart.InputText(obj.str("text") ?: "")
            "input_image" -> ResponseContentPart.InputImage(obj.str("image_url") ?: "", obj.str("detail"))
            "output_text" -> ResponseContentPart.OutputText(obj.str("text") ?: "")
            else -> error("unknown response content part type: ${obj["type"]}")
        }
    }
}

/**
 * A Responses message's `content`: a bare string or an array of [ResponseContentPart]s. Serialises
 * as a JSON string when [Text], else as an array of content-part objects.
 */
@Serializable(with = ResponseContentSerializer::class)
public sealed interface ResponseContent {
    public data class Text(val text: String) : ResponseContent

    public data class Parts(val parts: List<ResponseContentPart>) : ResponseContent

    public companion object {
        public fun text(value: String): ResponseContent = Text(value)

        public fun parts(vararg parts: ResponseContentPart): ResponseContent = Parts(parts.toList())
    }
}

public object ResponseContentSerializer : KSerializer<ResponseContent> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponseContent")

    override fun serialize(encoder: Encoder, value: ResponseContent) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponseContent requires kotlinx.serialization JSON")
        when (value) {
            is ResponseContent.Text -> jsonEncoder.encodeJsonElement(JsonPrimitive(value.text))
            is ResponseContent.Parts -> jsonEncoder.encodeJsonElement(
                jsonEncoder.json.encodeToJsonElement(ListSerializer(ResponseContentPartSerializer), value.parts),
            )
        }
    }

    override fun deserialize(decoder: Decoder): ResponseContent {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponseContent requires kotlinx.serialization JSON")
        return when (val element = jsonDecoder.decodeJsonElement()) {
            is JsonPrimitive -> ResponseContent.Text(element.content)
            is JsonArray -> ResponseContent.Parts(
                jsonDecoder.json.decodeFromJsonElement(ListSerializer(ResponseContentPartSerializer), element),
            )
            else -> error("message content must be a string or array")
        }
    }
}

// --- Items -----------------------------------------------------------------

/**
 * One item of the request `input` array or the response `output` array. The wire discriminator is
 * `"type"`; a plain message is a role-keyed object with **no** `"type"` (SPEC.md), the rest carry
 * their type token. Decoding dispatches on `"type"` (absent / `"message"` → a [ResponseMessageItem]).
 */
@Serializable(with = ResponseItemSerializer::class)
public sealed interface ResponseItem

/**
 * A conversation message item. On input `role` is `"user"` / `"system"` / `"developer"` /
 * `"assistant"` and `content` is a bare string or input parts; on output `role` is `"assistant"`,
 * `content` is output_text parts and `id` / `status` are set. Serialises role-keyed, without a
 * `"type"` key.
 */
@Serializable(with = ResponseMessageItemSerializer::class)
public data class ResponseMessageItem(
    val role: String,
    val content: ResponseContent,
    val id: String? = null,
    val status: String? = null, // output only: "in_progress" | "completed"
) : ResponseItem {
    public companion object {
        public fun user(text: String): ResponseMessageItem =
            ResponseMessageItem(role = "user", content = ResponseContent.text(text))

        public fun system(text: String): ResponseMessageItem =
            ResponseMessageItem(role = "system", content = ResponseContent.text(text))

        public fun developer(text: String): ResponseMessageItem =
            ResponseMessageItem(role = "developer", content = ResponseContent.text(text))

        public fun assistant(text: String): ResponseMessageItem =
            ResponseMessageItem(role = "assistant", content = ResponseContent.text(text))
    }
}

public object ResponseMessageItemSerializer : KSerializer<ResponseMessageItem> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponseMessageItem")

    override fun serialize(encoder: Encoder, value: ResponseMessageItem) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponseMessageItem requires kotlinx.serialization JSON")
        val json = jsonEncoder.json
        jsonEncoder.encodeJsonElement(
            buildJsonObject {
                value.id?.let { put("id", it) }
                put("role", value.role)
                put("content", json.encodeToJsonElement(ResponseContentSerializer, value.content))
                value.status?.let { put("status", it) }
            },
        )
    }

    override fun deserialize(decoder: Decoder): ResponseMessageItem {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponseMessageItem requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        val content = obj["content"]
            ?.let { jsonDecoder.json.decodeFromJsonElement(ResponseContentSerializer, it) }
            ?: ResponseContent.Text("")
        return ResponseMessageItem(
            role = obj.str("role") ?: "",
            content = content,
            id = obj.str("id"),
            status = obj.str("status"),
        )
    }
}

/**
 * A function call the model made. `call_id` pairs it with its [ResponseFunctionCallOutputItem];
 * `arguments` is the raw JSON string exactly as emitted. Serialises with `"type":"function_call"`.
 */
@Serializable(with = ResponseFunctionCallItemSerializer::class)
public data class ResponseFunctionCallItem(
    val callId: String,
    val name: String,
    val arguments: String,
    val id: String? = null,
    val status: String? = null,
) : ResponseItem

public object ResponseFunctionCallItemSerializer : KSerializer<ResponseFunctionCallItem> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponseFunctionCallItem")

    override fun serialize(encoder: Encoder, value: ResponseFunctionCallItem) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponseFunctionCallItem requires kotlinx.serialization JSON")
        jsonEncoder.encodeJsonElement(
            buildJsonObject {
                put("type", "function_call")
                value.id?.let { put("id", it) }
                put("call_id", value.callId)
                put("name", value.name)
                put("arguments", value.arguments)
                value.status?.let { put("status", it) }
            },
        )
    }

    override fun deserialize(decoder: Decoder): ResponseFunctionCallItem {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponseFunctionCallItem requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        return ResponseFunctionCallItem(
            callId = obj.str("call_id") ?: "",
            name = obj.str("name") ?: "",
            arguments = obj.str("arguments") ?: "",
            id = obj.str("id"),
            status = obj.str("status"),
        )
    }
}

/** The caller's answer to a function call, replayed on the next turn (`"type":"function_call_output"`). */
@Serializable(with = ResponseFunctionCallOutputItemSerializer::class)
public data class ResponseFunctionCallOutputItem(
    val callId: String,
    val output: String,
    val id: String? = null,
) : ResponseItem

public object ResponseFunctionCallOutputItemSerializer : KSerializer<ResponseFunctionCallOutputItem> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponseFunctionCallOutputItem")

    override fun serialize(encoder: Encoder, value: ResponseFunctionCallOutputItem) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponseFunctionCallOutputItem requires kotlinx.serialization JSON")
        jsonEncoder.encodeJsonElement(
            buildJsonObject {
                put("type", "function_call_output")
                value.id?.let { put("id", it) }
                put("call_id", value.callId)
                put("output", value.output)
            },
        )
    }

    override fun deserialize(decoder: Decoder): ResponseFunctionCallOutputItem {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponseFunctionCallOutputItem requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        return ResponseFunctionCallOutputItem(
            callId = obj.str("call_id") ?: "",
            output = obj.str("output") ?: "",
            id = obj.str("id"),
        )
    }
}

/**
 * A reasoning ("thinking") item (`"type":"reasoning"`). [summary] entries serialise as
 * `{"type":"summary_text","text"}` and [content] entries as `{"type":"reasoning_text","text"}` —
 * the list an entry lives in decides its wire token (SPEC.md). [encryptedContent] is opaque and
 * MUST be echoed back verbatim in the next request's input to continue an encrypted reasoning turn.
 */
@Serializable(with = ResponseReasoningItemSerializer::class)
public data class ResponseReasoningItem(
    val id: String? = null,
    val summary: List<String> = emptyList(),
    val content: List<String> = emptyList(),
    val encryptedContent: String? = null,
) : ResponseItem

public object ResponseReasoningItemSerializer : KSerializer<ResponseReasoningItem> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponseReasoningItem")

    override fun serialize(encoder: Encoder, value: ResponseReasoningItem) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponseReasoningItem requires kotlinx.serialization JSON")
        jsonEncoder.encodeJsonElement(
            buildJsonObject {
                put("type", "reasoning")
                value.id?.let { put("id", it) }
                if (value.summary.isNotEmpty()) {
                    put("summary", reasoningTexts("summary_text", value.summary))
                }
                if (value.content.isNotEmpty()) {
                    put("content", reasoningTexts("reasoning_text", value.content))
                }
                value.encryptedContent?.let { put("encrypted_content", it) }
            },
        )
    }

    override fun deserialize(decoder: Decoder): ResponseReasoningItem {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponseReasoningItem requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        return ResponseReasoningItem(
            id = obj.str("id"),
            summary = obj.reasoningTexts("summary"),
            content = obj.reasoningTexts("content"),
            encryptedContent = obj.str("encrypted_content"),
        )
    }

    private fun reasoningTexts(type: String, texts: List<String>): JsonArray =
        JsonArray(
            texts.map {
                buildJsonObject {
                    put("type", type)
                    put("text", it)
                }
            },
        )

    private fun JsonObject.reasoningTexts(key: String): List<String> =
        (this[key] as? JsonArray)?.mapNotNull { (it as? JsonObject)?.str("text") } ?: emptyList()
}

/**
 * Discriminated-union serializer for [ResponseItem], keyed on the wire `"type"` (proto
 * `ResponseItem.item` oneof). A plain message is emitted / accepted without a `"type"` key.
 */
public object ResponseItemSerializer : KSerializer<ResponseItem> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponseItem")

    override fun serialize(encoder: Encoder, value: ResponseItem) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponseItem requires kotlinx.serialization JSON")
        val json = jsonEncoder.json
        val element = when (value) {
            is ResponseMessageItem -> json.encodeToJsonElement(ResponseMessageItemSerializer, value)
            is ResponseFunctionCallItem -> json.encodeToJsonElement(ResponseFunctionCallItemSerializer, value)
            is ResponseFunctionCallOutputItem ->
                json.encodeToJsonElement(ResponseFunctionCallOutputItemSerializer, value)
            is ResponseReasoningItem -> json.encodeToJsonElement(ResponseReasoningItemSerializer, value)
        }
        jsonEncoder.encodeJsonElement(element)
    }

    override fun deserialize(decoder: Decoder): ResponseItem {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponseItem requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as? JsonObject
            ?: error("response item must be a JSON object")
        val json = jsonDecoder.json
        return when (val type = obj.str("type")) {
            null, "message" -> json.decodeFromJsonElement(ResponseMessageItemSerializer, obj)
            "function_call" -> json.decodeFromJsonElement(ResponseFunctionCallItemSerializer, obj)
            "function_call_output" -> json.decodeFromJsonElement(ResponseFunctionCallOutputItemSerializer, obj)
            "reasoning" -> json.decodeFromJsonElement(ResponseReasoningItemSerializer, obj)
            else -> error("unknown response item type: $type")
        }
    }
}

// --- Tools / reasoning config ----------------------------------------------

/**
 * A tool the model MAY call — **flat** in this dialect (`type` / `name` / `parameters` at the top
 * level, no nested `function` object, unlike the chat dialect). `parameters` is a raw JSON-Schema
 * object (spliced inline). `type` is always emitted.
 */
@Serializable
public data class ResponsesToolDef(
    @EncodeDefault(EncodeDefault.Mode.ALWAYS)
    @SerialName("type") val type: String = "function",
    @SerialName("name") val name: String,
    @SerialName("description") val description: String? = null,
    @SerialName("parameters") val parameters: RawJson? = null,
    @SerialName("strict") val strict: Boolean? = null,
)

/** The flat named `tool_choice` object: `{"type":"function","name":"..."}`. */
@Serializable
public data class ResponsesNamedToolChoice(
    @SerialName("name") val name: String,
    @EncodeDefault(EncodeDefault.Mode.ALWAYS)
    @SerialName("type") val type: String = "function",
)

/**
 * `tool_choice`: a bare mode string (`"auto"`/`"none"`/`"required"`) or the FLAT named object
 * (proto `ResponsesToolChoice.choice` oneof).
 */
@Serializable(with = ResponsesToolChoiceSerializer::class)
public sealed interface ResponsesToolChoice {
    public data class Mode(val mode: String) : ResponsesToolChoice

    public data class Named(val named: ResponsesNamedToolChoice) : ResponsesToolChoice

    public companion object {
        public val AUTO: ResponsesToolChoice = Mode("auto")
        public val NONE: ResponsesToolChoice = Mode("none")
        public val REQUIRED: ResponsesToolChoice = Mode("required")

        public fun function(name: String): ResponsesToolChoice = Named(ResponsesNamedToolChoice(name))
    }
}

public object ResponsesToolChoiceSerializer : KSerializer<ResponsesToolChoice> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponsesToolChoice")

    override fun serialize(encoder: Encoder, value: ResponsesToolChoice) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponsesToolChoice requires kotlinx.serialization JSON")
        when (value) {
            is ResponsesToolChoice.Mode -> jsonEncoder.encodeJsonElement(JsonPrimitive(value.mode))
            is ResponsesToolChoice.Named -> jsonEncoder.encodeJsonElement(
                jsonEncoder.json.encodeToJsonElement(ResponsesNamedToolChoice.serializer(), value.named),
            )
        }
    }

    override fun deserialize(decoder: Decoder): ResponsesToolChoice {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponsesToolChoice requires kotlinx.serialization JSON")
        return when (val element = jsonDecoder.decodeJsonElement()) {
            is JsonPrimitive -> ResponsesToolChoice.Mode(element.content)
            is JsonObject -> ResponsesToolChoice.Named(
                jsonDecoder.json.decodeFromJsonElement(ResponsesNamedToolChoice.serializer(), element),
            )
            else -> error("tool_choice must be a string or object")
        }
    }
}

/** `reasoning`: `{"effort":"minimal"|"low"|"medium"|"high"|..., "summary":...}`. */
@Serializable
public data class ResponsesReasoning(
    @SerialName("effort") val effort: String? = null,
    @SerialName("summary") val summary: String? = null,
)

// --- Request ---------------------------------------------------------------

/**
 * The request `input`: a bare string (one user message) or an array of [ResponseItem]s. Serialises
 * as a JSON string when [Text], else as an array of item objects.
 */
@Serializable(with = ResponsesInputSerializer::class)
public sealed interface ResponsesInput {
    public data class Text(val text: String) : ResponsesInput

    public data class Items(val items: List<ResponseItem>) : ResponsesInput

    public companion object {
        public fun text(value: String): ResponsesInput = Text(value)

        public fun items(items: List<ResponseItem>): ResponsesInput = Items(items)

        public fun items(vararg items: ResponseItem): ResponsesInput = Items(items.toList())
    }
}

public object ResponsesInputSerializer : KSerializer<ResponsesInput> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponsesInput")

    override fun serialize(encoder: Encoder, value: ResponsesInput) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponsesInput requires kotlinx.serialization JSON")
        when (value) {
            is ResponsesInput.Text -> jsonEncoder.encodeJsonElement(JsonPrimitive(value.text))
            is ResponsesInput.Items -> jsonEncoder.encodeJsonElement(
                jsonEncoder.json.encodeToJsonElement(ListSerializer(ResponseItemSerializer), value.items),
            )
        }
    }

    override fun deserialize(decoder: Decoder): ResponsesInput {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponsesInput requires kotlinx.serialization JSON")
        return when (val element = jsonDecoder.decodeJsonElement()) {
            is JsonPrimitive -> ResponsesInput.Text(element.content)
            is JsonArray -> ResponsesInput.Items(
                jsonDecoder.json.decodeFromJsonElement(ListSerializer(ResponseItemSerializer), element),
            )
            else -> error("input must be a string or array")
        }
    }
}

/**
 * A Responses request. Like [ChatRequest], `extra` is raw JSON whose keys merge at the **top
 * level** of the request object (SPEC.md "Free-form JSON fields"), so it uses a custom serializer
 * over the derived [ResponsesRequestWire].
 */
@Serializable(with = ResponsesRequestSerializer::class)
public data class ResponsesRequest(
    val model: String,
    val input: ResponsesInput,
    val instructions: String? = null,
    val stream: Boolean? = null,
    val temperature: Float? = null,
    val topP: Float? = null,
    val maxOutputTokens: Int? = null,
    val tools: List<ResponsesToolDef> = emptyList(),
    val toolChoice: ResponsesToolChoice? = null,
    val reasoning: ResponsesReasoning? = null,
    val store: Boolean? = null, // accepted but always answered `false` — llmleaf stores nothing
    val extra: RawJson? = null, // raw JSON object; keys merged at top level
)

/** The derived shape of [ResponsesRequest] minus `extra`; [ResponsesRequestSerializer] splices `extra` over it. */
@Serializable
internal data class ResponsesRequestWire(
    @SerialName("model") val model: String,
    @SerialName("input") val input: ResponsesInput,
    @SerialName("instructions") val instructions: String? = null,
    @SerialName("stream") val stream: Boolean? = null,
    @SerialName("temperature") val temperature: Float? = null,
    @SerialName("top_p") val topP: Float? = null,
    @SerialName("max_output_tokens") val maxOutputTokens: Int? = null,
    @SerialName("tools") val tools: List<ResponsesToolDef>? = null,
    @SerialName("tool_choice") val toolChoice: ResponsesToolChoice? = null,
    @SerialName("reasoning") val reasoning: ResponsesReasoning? = null,
    @SerialName("store") val store: Boolean? = null,
)

public object ResponsesRequestSerializer : KSerializer<ResponsesRequest> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ResponsesRequest")

    override fun serialize(encoder: Encoder, value: ResponsesRequest) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ResponsesRequest requires kotlinx.serialization JSON")
        val json = jsonEncoder.json
        val wire = ResponsesRequestWire(
            model = value.model,
            input = value.input,
            instructions = value.instructions,
            stream = value.stream,
            temperature = value.temperature,
            topP = value.topP,
            maxOutputTokens = value.maxOutputTokens,
            tools = value.tools.ifEmpty { null },
            toolChoice = value.toolChoice,
            reasoning = value.reasoning,
            store = value.store,
        )
        val base = json.encodeToJsonElement(ResponsesRequestWire.serializer(), wire) as JsonObject
        jsonEncoder.encodeJsonElement(mergeExtra(json, base, value.extra))
    }

    override fun deserialize(decoder: Decoder): ResponsesRequest {
        // Requests are encode-only in normal SDK use; decode is supported for round-trip tests.
        // Unknown top-level keys (i.e. `extra` passthrough) are captured back into `extra`.
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ResponsesRequest requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        val known = ResponsesRequestWire.serializer().descriptor.let { d ->
            (0 until d.elementsCount).map { d.getElementName(it) }.toSet()
        }
        val wire = jsonDecoder.json.decodeFromJsonElement(ResponsesRequestWire.serializer(), obj)
        val leftover = obj.filterKeys { it !in known }
        val extra = if (leftover.isEmpty()) null else RawJson.of(JsonObject(leftover))
        return ResponsesRequest(
            model = wire.model,
            input = wire.input,
            instructions = wire.instructions,
            stream = wire.stream,
            temperature = wire.temperature,
            topP = wire.topP,
            maxOutputTokens = wire.maxOutputTokens,
            tools = wire.tools ?: emptyList(),
            toolChoice = wire.toolChoice,
            reasoning = wire.reasoning,
            store = wire.store,
            extra = extra,
        )
    }
}

// --- Response --------------------------------------------------------------

/**
 * Token accounting in the Responses dialect's own names (`input_tokens` / `output_tokens`, not the
 * chat dialect's `prompt_tokens` / `completion_tokens`).
 */
@Serializable
public data class ResponsesUsage(
    @SerialName("input_tokens") val inputTokens: Int = 0,
    @SerialName("input_tokens_details") val inputTokensDetails: ResponsesInputTokensDetails? = null,
    @SerialName("output_tokens") val outputTokens: Int = 0,
    @SerialName("output_tokens_details") val outputTokensDetails: ResponsesOutputTokensDetails? = null,
    @SerialName("total_tokens") val totalTokens: Int = 0,
) {
    /** Input tokens served from the provider's cache this request — `0` when none reported. */
    val cachedTokens: Int
        get() = inputTokensDetails?.cachedTokens ?: 0

    /** Output tokens spent on reasoning this request — `0` when none reported. */
    val reasoningTokens: Int
        get() = outputTokensDetails?.reasoningTokens ?: 0
}

@Serializable
public data class ResponsesInputTokensDetails(
    @SerialName("cached_tokens") val cachedTokens: Int? = null,
)

@Serializable
public data class ResponsesOutputTokensDetails(
    @SerialName("reasoning_tokens") val reasoningTokens: Int? = null,
)

/** `incomplete_details`: why a response stopped short — `"max_output_tokens"` | `"content_filter"`. */
@Serializable
public data class ResponsesIncompleteDetails(
    @SerialName("reason") val reason: String,
)

/**
 * The response object (`"object":"response"`), also the snapshot carried by the
 * `response.created` / `response.in_progress` / `response.completed` stream events. Wire fields the
 * SDK doesn't type (tools, tool_choice, truncation, …) are ignored on decode.
 */
@Serializable
public data class ResponsesResponse(
    @SerialName("id") val id: String,
    @SerialName("object") val `object`: String = "response",
    @SerialName("created_at") val createdAt: Long = 0,
    @SerialName("status") val status: String,
    @SerialName("incomplete_details") val incompleteDetails: ResponsesIncompleteDetails? = null,
    @SerialName("error") val error: ErrorBody? = null,
    @SerialName("model") val model: String = "",
    @SerialName("output") val output: List<ResponseItem> = emptyList(),
    @SerialName("usage") val usage: ResponsesUsage? = null, // null on in-flight snapshots
    @SerialName("store") val store: Boolean? = null, // llmleaf always answers false
    @SerialName("instructions") val instructions: String? = null,
    @SerialName("max_output_tokens") val maxOutputTokens: Int? = null,
    @SerialName("temperature") val temperature: Float? = null,
    @SerialName("top_p") val topP: Float? = null,
    @SerialName("reasoning") val reasoning: ResponsesReasoning? = null,
) {
    /** The assembled assistant text of all `output_text` parts across output message items. */
    val outputText: String
        get() = buildString {
            for (item in output) {
                if (item is ResponseMessageItem) {
                    val content = item.content
                    if (content is ResponseContent.Parts) {
                        for (part in content.parts) {
                            if (part is ResponseContentPart.OutputText) append(part.text)
                        }
                    } else if (content is ResponseContent.Text) {
                        append(content.text)
                    }
                }
            }
        }
}

/**
 * One streaming SSE event. Unlike chat streaming there is NO `data: [DONE]` sentinel: the stream
 * ends after the terminal `response.completed` / `response.incomplete` / `response.failed` event.
 * This is a flat superset of every event's fields — [type] says which ones are meaningful.
 */
@Serializable
public data class ResponsesStreamEvent(
    @SerialName("type") val type: String,
    @SerialName("sequence_number") val sequenceNumber: Long = 0,
    @SerialName("response") val response: ResponsesResponse? = null,
    @SerialName("output_index") val outputIndex: Int? = null,
    @SerialName("item_id") val itemId: String? = null,
    @SerialName("content_index") val contentIndex: Int? = null,
    @SerialName("item") val item: ResponseItem? = null, // response.output_item.added/done
    @SerialName("part") val part: ResponseContentPart? = null, // response.content_part.added/done
    @SerialName("delta") val delta: String? = null, // *.delta events (text / reasoning / arguments)
    @SerialName("text") val text: String? = null, // *.done text events
    @SerialName("arguments") val arguments: String? = null, // function_call_arguments.done
    @SerialName("message") val message: String? = null, // "error" event
)
