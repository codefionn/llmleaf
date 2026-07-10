package eu.codefionn.llmleaf.client.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * A rerank request. [documents] are scored against [query]; the response returns them in
 * descending relevance order. Unlike [EmbeddingRequest] there is no base64 payload and no
 * top-level `extra` splice, so a plain `@Serializable` (no custom serializer) suffices.
 *
 * [documents] is modelled as `List<String>` — the common case. Structured / multimodal
 * documents are a JSON-only extension not surfaced by this typed field.
 */
@Serializable
public data class RerankRequest(
    @SerialName("model") val model: String,
    @SerialName("query") val query: String,
    @SerialName("documents") val documents: List<String>,
    @SerialName("top_n") val topN: Int? = null,
    @SerialName("return_documents") val returnDocuments: Boolean? = null,
)

/**
 * A single rerank result: the [index] of the document in the request's `documents` array and
 * its [relevanceScore]. [document] is present only when the request set `return_documents`;
 * it is a bare string for text documents or an object for structured ones, so it is kept as a
 * raw [JsonElement].
 */
@Serializable
public data class RerankResult(
    @SerialName("index") val index: Int = 0,
    @SerialName("relevance_score") val relevanceScore: Double = 0.0,
    @SerialName("document") val document: JsonElement? = null,
)

@Serializable
public data class RerankResponse(
    @SerialName("object") val `object`: String = "list",
    @SerialName("model") val model: String,
    @SerialName("results") val results: List<RerankResult> = emptyList(),
    @SerialName("usage") val usage: Usage? = null,
)
