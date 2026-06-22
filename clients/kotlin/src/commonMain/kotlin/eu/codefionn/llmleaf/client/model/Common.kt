package eu.codefionn.llmleaf.client.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Token accounting echoed on every response. `cost_usd` is an llmleaf addition (real-time
 * price); it is absent when the model has no known price.
 */
@Serializable
public data class Usage(
    @SerialName("prompt_tokens") val promptTokens: Int = 0,
    @SerialName("completion_tokens") val completionTokens: Int = 0,
    @SerialName("total_tokens") val totalTokens: Int = 0,
    @SerialName("cost_usd") val costUsd: Double? = null,
    /**
     * Prompt-cache hit accounting (OpenAI `usage.prompt_tokens_details`). Absent when the
     * upstream reported no caching; [cachedTokens] flattens it to a plain count.
     */
    @SerialName("prompt_tokens_details") val promptTokensDetails: PromptTokensDetails? = null,
    /**
     * Input tokens written to the provider's prompt cache this request — a cache *write*
     * (creation). An llmleaf extension (Anthropic reports it; OpenAI/OpenRouter do not); absent
     * when there were none.
     */
    @SerialName("cache_creation_tokens") val cacheCreationTokens: Int? = null,
) {
    /**
     * Prompt tokens served from the provider's cache this request — a cache *read* (hit). `0`
     * when the upstream reported no caching.
     */
    val cachedTokens: Int
        get() = promptTokensDetails?.cachedTokens ?: 0

    /**
     * Input tokens written to the provider's cache this request — a cache *write* (creation).
     * `0` when there were none (or the provider does not report writes).
     */
    val cacheWrites: Int
        get() = cacheCreationTokens ?: 0
}

/**
 * Breakdown of [Usage.promptTokens]. Today only the cache-read (hit) share is surfaced — the
 * count of prompt tokens served from the provider's cache rather than processed fresh.
 */
@Serializable
public data class PromptTokensDetails(
    @SerialName("cached_tokens") val cachedTokens: Int? = null,
)

/** The canonical error envelope: {"error":{"message":...}}. */
@Serializable
public data class ErrorResponse(
    @SerialName("error") val error: ErrorBody,
)

@Serializable
public data class ErrorBody(
    @SerialName("message") val message: String,
    @SerialName("type") val type: String? = null,
    @SerialName("code") val code: String? = null,
)
