package eu.codefionn.llmleaf.client.model

import eu.codefionn.llmleaf.client.RawJson
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public data class Architecture(
    @SerialName("input_modalities") val inputModalities: List<String> = emptyList(),
    @SerialName("output_modalities") val outputModalities: List<String> = emptyList(),
    @SerialName("modality") val modality: String? = null,
    @SerialName("tokenizer") val tokenizer: String = "",
    @SerialName("instruct_type") val instructType: String? = null,
)

@Serializable
public data class Pricing(
    @SerialName("prompt") val prompt: String,
    @SerialName("completion") val completion: String,
)

@Serializable
public data class TopProvider(
    @SerialName("context_length") val contextLength: Int? = null,
    @SerialName("max_completion_tokens") val maxCompletionTokens: Int? = null,
    @SerialName("is_moderated") val isModerated: Boolean = false,
    @SerialName("max_thinking_tokens") val maxThinkingTokens: Int? = null,
)

/** Admin-only fallback-chain entry (present only with a valid x-admin-token). */
@Serializable
public data class ModelEndpoint(
    @SerialName("provider") val provider: String,
    @SerialName("model") val model: String,
    @SerialName("down") val down: Boolean = false,
    @SerialName("source") val source: String, // "route" | "prefix"
)

@Serializable
public data class ModelEntry(
    @SerialName("id") val id: String,
    @SerialName("canonical_slug") val canonicalSlug: String = "",
    @SerialName("name") val name: String = "",
    @SerialName("created") val created: Long = 0,
    @SerialName("description") val description: String = "",
    @SerialName("context_length") val contextLength: Int? = null,
    @SerialName("architecture") val architecture: Architecture? = null,
    @SerialName("pricing") val pricing: Pricing? = null,
    @SerialName("top_provider") val topProvider: TopProvider? = null,
    @SerialName("supported_parameters") val supportedParameters: List<String> = emptyList(),
    @SerialName("unsupported_parameters") val unsupportedParameters: List<String> = emptyList(),
    @SerialName("default_parameters") val defaultParameters: RawJson? = null,
    @SerialName("endpoints") val endpoints: List<ModelEndpoint> = emptyList(), // admin-only
)

@Serializable
public data class ListModelsResponse(
    @SerialName("data") val data: List<ModelEntry> = emptyList(),
)

/** The `type` filter for `GET /v1/models`. */
public enum class ModelType(public val wire: String) {
    ALL("all"),
    LLM("llm"),
    TTS("tts"),
    STT("stt"),
    EMBEDDING("embedding"),
    RERANK("rerank"),
}
