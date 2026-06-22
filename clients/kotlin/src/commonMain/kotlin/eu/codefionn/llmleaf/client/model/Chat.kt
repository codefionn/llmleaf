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

/**
 * One structured reasoning ("thinking") block (OpenRouter `reasoning_details[]`). It expresses
 * both *open* reasoning — visible text, optionally signed — and *hidden* reasoning — an
 * encrypted/redacted blob the provider returns in place of the text. [type] is the wire
 * discriminator and selects which field is set:
 *
 *  - `"reasoning.text"`    → [text] (+ optional [signature]) — **open** (visible reasoning)
 *  - `"reasoning.summary"` → [summary]                        — **open** (a summarised view)
 *  - `"reasoning.encrypted"` → [data]                         — **hidden** (redacted / opaque)
 *
 * [signature] and [data] are opaque and MUST be echoed back verbatim in the next request's
 * `reasoning_details` to continue a signed/encrypted reasoning turn (the upstream rejects an
 * altered or dropped block — e.g. before a tool call). Use [isHidden] / [openText] to branch
 * without matching on the raw [type] string.
 */
@Serializable
public data class ReasoningDetail(
    @SerialName("type") val type: String,
    @SerialName("text") val text: String? = null, // "reasoning.text"
    @SerialName("summary") val summary: String? = null, // "reasoning.summary"
    @SerialName("data") val data: String? = null, // "reasoning.encrypted" (hidden)
    @SerialName("signature") val signature: String? = null, // opaque, replayed verbatim
    @SerialName("id") val id: String? = null,
    @SerialName("format") val format: String? = null, // e.g. "anthropic-claude-v1"
    @SerialName("index") val index: Int? = null,
) {
    /** Whether this block is hidden (redacted / encrypted) rather than open visible reasoning. */
    val isHidden: Boolean
        get() = type == "reasoning.encrypted" || (data != null && text == null)

    /**
     * The visible reasoning text of an open block — its [text], falling back to its [summary].
     * `null` for a hidden block.
     */
    val openText: String?
        get() = text ?: summary
}

/**
 * A chat message. `content` is the string-or-parts union; `role` serialises to its lowercase
 * wire token. A message is valid with content, tool calls (assistant), or a tool result.
 */
@Serializable
public data class ChatMessage(
    @SerialName("role") val role: Role,
    @SerialName("content") val content: MessageContent? = null,
    @SerialName("name") val name: String? = null,
    @SerialName("tool_calls") val toolCalls: List<ToolCall> = emptyList(),
    @SerialName("tool_call_id") val toolCallId: String? = null,
    /**
     * Open reasoning text the assistant emitted (OpenRouter `reasoning`), if any. The flat,
     * human-readable form; the structured [reasoningDetails] is the replay-safe one.
     */
    @SerialName("reasoning") val reasoning: String? = null,
    /**
     * Structured reasoning blocks (open and hidden, with signatures — see [ReasoningDetail]).
     * Echo these back verbatim on the next request to preserve signed reasoning across a turn.
     */
    @SerialName("reasoning_details") val reasoningDetails: List<ReasoningDetail> = emptyList(),
) {
    public companion object {
        public fun system(text: String): ChatMessage =
            ChatMessage(Role.SYSTEM, MessageContent.text(text))

        public fun user(text: String): ChatMessage =
            ChatMessage(Role.USER, MessageContent.text(text))

        public fun assistant(text: String): ChatMessage =
            ChatMessage(Role.ASSISTANT, MessageContent.text(text))

        public fun tool(toolCallId: String, text: String): ChatMessage =
            ChatMessage(Role.TOOL, MessageContent.text(text), toolCallId = toolCallId)
    }
}

/**
 * A chat-completion request. The `extra` field is raw JSON whose keys are merged at the **top
 * level** of the request object (SPEC.md "Free-form JSON fields"); the rest of the struct is
 * derived normally. Hence the custom [ChatRequestSerializer].
 */
@Serializable(with = ChatRequestSerializer::class)
public data class ChatRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val stream: Boolean? = null,
    val temperature: Float? = null,
    val topP: Float? = null,
    val maxTokens: Int? = null, // legacy name
    val maxCompletionTokens: Int? = null, // modern name (takes precedence)
    val stop: List<String> = emptyList(), // wire: string or array
    val n: Int? = null,
    val seed: Long? = null,
    val frequencyPenalty: Float? = null,
    val presencePenalty: Float? = null,
    val tools: List<ToolDef> = emptyList(),
    val toolChoice: ToolChoice? = null,
    val responseFormat: ResponseFormat? = null,
    val reasoningEffort: String? = null, // "low" | "medium" | "high"
    val extra: RawJson? = null, // raw JSON object; keys merged at top level
)

/**
 * The derived shape of [ChatRequest] minus `extra`. [ChatRequestSerializer] serialises this,
 * then splices `extra`'s keys over the result so passthrough lands at the top level.
 */
@Serializable
internal data class ChatRequestWire(
    @SerialName("model") val model: String,
    @SerialName("messages") val messages: List<ChatMessage>,
    @SerialName("stream") val stream: Boolean? = null,
    @SerialName("temperature") val temperature: Float? = null,
    @SerialName("top_p") val topP: Float? = null,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    @SerialName("max_completion_tokens") val maxCompletionTokens: Int? = null,
    @Serializable(with = StringOrListSerializer::class)
    @SerialName("stop") val stop: List<String>? = null,
    @SerialName("n") val n: Int? = null,
    @SerialName("seed") val seed: Long? = null,
    @SerialName("frequency_penalty") val frequencyPenalty: Float? = null,
    @SerialName("presence_penalty") val presencePenalty: Float? = null,
    @SerialName("tools") val tools: List<ToolDef>? = null,
    @SerialName("tool_choice") val toolChoice: ToolChoice? = null,
    @SerialName("response_format") val responseFormat: ResponseFormat? = null,
    @SerialName("reasoning_effort") val reasoningEffort: String? = null,
)

public object ChatRequestSerializer : KSerializer<ChatRequest> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ChatRequest")

    override fun serialize(encoder: Encoder, value: ChatRequest) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("ChatRequest requires kotlinx.serialization JSON")
        val json = jsonEncoder.json
        val wire = ChatRequestWire(
            model = value.model,
            messages = value.messages,
            stream = value.stream,
            temperature = value.temperature,
            topP = value.topP,
            maxTokens = value.maxTokens,
            maxCompletionTokens = value.maxCompletionTokens,
            stop = value.stop.ifEmpty { null },
            n = value.n,
            seed = value.seed,
            frequencyPenalty = value.frequencyPenalty,
            presencePenalty = value.presencePenalty,
            tools = value.tools.ifEmpty { null },
            toolChoice = value.toolChoice,
            responseFormat = value.responseFormat,
            reasoningEffort = value.reasoningEffort,
        )
        val base = json.encodeToJsonElement(ChatRequestWire.serializer(), wire) as JsonObject
        jsonEncoder.encodeJsonElement(mergeExtra(json, base, value.extra))
    }

    // `SerialDescriptor.elementsCount` / `getElementName` are still @ExperimentalSerializationApi
    // in kotlinx.serialization 1.7.3; we use them to recover the wire field names so unknown
    // top-level keys round-trip back into `extra`.
    @OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)
    override fun deserialize(decoder: Decoder): ChatRequest {
        // Requests are encode-only in normal SDK use, but BatchRequestItem nests a ChatRequest
        // so we support decode too. Unknown top-level keys (i.e. `extra` passthrough) are
        // captured back into `extra` as a raw JSON object.
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("ChatRequest requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        val known = ChatRequestWire.serializer().descriptor.let { d ->
            (0 until d.elementsCount).map { d.getElementName(it) }.toSet()
        }
        val wire = jsonDecoder.json.decodeFromJsonElement(ChatRequestWire.serializer(), obj)
        val leftover = obj.filterKeys { it !in known }
        val extra = if (leftover.isEmpty()) null else RawJson.of(JsonObject(leftover))
        return ChatRequest(
            model = wire.model,
            messages = wire.messages,
            stream = wire.stream,
            temperature = wire.temperature,
            topP = wire.topP,
            maxTokens = wire.maxTokens,
            maxCompletionTokens = wire.maxCompletionTokens,
            stop = wire.stop ?: emptyList(),
            n = wire.n,
            seed = wire.seed,
            frequencyPenalty = wire.frequencyPenalty,
            presencePenalty = wire.presencePenalty,
            tools = wire.tools ?: emptyList(),
            toolChoice = wire.toolChoice,
            responseFormat = wire.responseFormat,
            reasoningEffort = wire.reasoningEffort,
            extra = extra,
        )
    }
}

@Serializable
public data class Choice(
    @SerialName("index") val index: Int = 0,
    @SerialName("message") val message: ChatMessage,
    @SerialName("finish_reason") val finishReason: FinishReason? = null,
)

@Serializable
public data class ChatResponse(
    @SerialName("id") val id: String,
    @SerialName("object") val `object`: String = "chat.completion",
    @SerialName("created") val created: Long = 0,
    @SerialName("model") val model: String,
    @SerialName("choices") val choices: List<Choice> = emptyList(),
    @SerialName("usage") val usage: Usage? = null,
)

// --- Streaming chunk types -------------------------------------------------

@Serializable
public data class Delta(
    @SerialName("role") val role: Role? = null,
    @SerialName("content") val content: String? = null,
    @SerialName("tool_calls") val toolCalls: List<ToolCallDelta> = emptyList(),
    /** Incremental open reasoning text, if any. */
    @SerialName("reasoning") val reasoning: String? = null,
    /** Incremental structured reasoning blocks (open / hidden — see [ReasoningDetail]). */
    @SerialName("reasoning_details") val reasoningDetails: List<ReasoningDetail> = emptyList(),
)

@Serializable
public data class ChunkChoice(
    @SerialName("index") val index: Int = 0,
    @SerialName("delta") val delta: Delta,
    @SerialName("finish_reason") val finishReason: FinishReason? = null,
)

@Serializable
public data class ChatCompletionChunk(
    @SerialName("id") val id: String,
    @SerialName("object") val `object`: String = "chat.completion.chunk",
    @SerialName("created") val created: Long = 0,
    @SerialName("model") val model: String,
    @SerialName("choices") val choices: List<ChunkChoice> = emptyList(),
    @SerialName("usage") val usage: Usage? = null,
)
