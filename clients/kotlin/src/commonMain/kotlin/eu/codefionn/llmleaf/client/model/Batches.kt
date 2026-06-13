package eu.codefionn.llmleaf.client.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public data class BatchRequestItem(
    @SerialName("custom_id") val customId: String,
    @SerialName("body") val body: ChatRequest,
)

@Serializable
public data class BatchCreateRequest(
    @SerialName("requests") val requests: List<BatchRequestItem>,
)

@Serializable
public data class BatchCounts(
    @SerialName("total") val total: Long = 0,
    @SerialName("processing") val processing: Long = 0,
    @SerialName("succeeded") val succeeded: Long = 0,
    @SerialName("errored") val errored: Long = 0,
    @SerialName("canceled") val canceled: Long = 0,
    @SerialName("expired") val expired: Long = 0,
)

@Serializable
public data class BatchHandle(
    @SerialName("id") val id: String,
    @SerialName("status") val status: BatchStatus = BatchStatus.BATCH_STATUS_UNSPECIFIED,
    @SerialName("counts") val counts: BatchCounts = BatchCounts(),
    @SerialName("created_at") val createdAt: Long? = null,
    @SerialName("expires_at") val expiresAt: Long? = null,
    @SerialName("ended_at") val endedAt: Long? = null,
    @SerialName("endpoint") val endpoint: String? = null,
)

@Serializable
public data class BatchResponse(
    @SerialName("status_code") val statusCode: Int = 0,
    @SerialName("body") val body: ChatResponse,
)

@Serializable
public data class BatchError(
    @SerialName("code") val code: String,
    @SerialName("message") val message: String,
)

/** One line of the JSONL results stream (`GET /v1/batches/{id}/results`). */
@Serializable
public data class BatchResultLine(
    @SerialName("custom_id") val customId: String,
    @SerialName("response") val response: BatchResponse? = null,
    @SerialName("error") val error: BatchError? = null,
)
