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
