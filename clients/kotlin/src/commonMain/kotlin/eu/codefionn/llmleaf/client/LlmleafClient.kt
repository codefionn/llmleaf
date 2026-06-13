package eu.codefionn.llmleaf.client

import eu.codefionn.llmleaf.client.model.BatchCreateRequest
import eu.codefionn.llmleaf.client.model.BatchHandle
import eu.codefionn.llmleaf.client.model.BatchResultLine
import eu.codefionn.llmleaf.client.model.ChatCompletionChunk
import eu.codefionn.llmleaf.client.model.ChatRequest
import eu.codefionn.llmleaf.client.model.ChatResponse
import eu.codefionn.llmleaf.client.model.EmbeddingRequest
import eu.codefionn.llmleaf.client.model.EmbeddingResponse
import eu.codefionn.llmleaf.client.model.ErrorResponse
import eu.codefionn.llmleaf.client.model.ListModelsResponse
import eu.codefionn.llmleaf.client.model.ModelType
import eu.codefionn.llmleaf.client.model.SpeechRequest
import eu.codefionn.llmleaf.client.model.TranscriptionRequest
import eu.codefionn.llmleaf.client.model.TranscriptionResponse
import eu.codefionn.llmleaf.client.model.VoicesResponse
import io.ktor.client.HttpClient
import io.ktor.client.HttpClientConfig
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.engine.HttpClientEngineFactory
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.timeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.request.HttpRequestBuilder
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.forms.MultiPartFormDataContent
import io.ktor.client.request.forms.formData
import io.ktor.client.request.header
import io.ktor.client.request.parameter
import io.ktor.client.request.prepareGet
import io.ktor.client.request.prepareRequest
import io.ktor.client.request.setBody
import io.ktor.client.request.url
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsChannel
import io.ktor.client.statement.bodyAsText
import io.ktor.client.statement.readRawBytes
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import io.ktor.utils.io.ByteReadChannel
import io.ktor.utils.io.readUTF8Line
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.SerializationException

/**
 * The official Kotlin Multiplatform client for an llmleaf gateway. Wraps a [Ktor][HttpClient]
 * configured with bearer auth, content-negotiation (kotlinx.serialization JSON) and a request
 * timeout. Every call is `suspend`; streaming chat is a [Flow] of [ChatCompletionChunk].
 *
 * Construct it with `(baseUrl, apiKey)` and an optional [config]:
 * ```
 * val client = LlmleafClient("https://gateway.example.com", "sk-...") {
 *     requestTimeout = 30.seconds
 *     adminToken = "..."
 * }
 * ```
 * Close it when done ([close]); it implements no finalizer.
 */
public class LlmleafClient private constructor(
    baseUrl: String,
    private val http: HttpClient,
) {
    private val base: String = baseUrl.trimEnd('/')

    /** Primary constructor: default platform engine + a config block. */
    public constructor(
        baseUrl: String,
        apiKey: String,
        configure: LlmleafConfig.Builder.() -> Unit = {},
    ) : this(
        baseUrl,
        // A top-level factory keeps the HttpClient construction outside any instance scope:
        // a delegating `this(...)` call cannot touch the (still-uninitialised) receiver, and the
        // member-style `run { }` would bind to it. See [buildDefaultHttpClient].
        buildDefaultHttpClient(LlmleafConfig(configure), apiKey),
    )

    /** Injectable-engine-factory constructor (custom engines). */
    public constructor(
        baseUrl: String,
        apiKey: String,
        engineFactory: HttpClientEngineFactory<*>,
        config: LlmleafConfig = LlmleafConfig.DEFAULT,
    ) : this(baseUrl, HttpClient(engineFactory) { applyLlmleaf(config, apiKey) })

    /** Injectable-engine-instance constructor (tests with MockEngine, pre-built engines). */
    public constructor(
        baseUrl: String,
        apiKey: String,
        engine: HttpClientEngine,
        config: LlmleafConfig = LlmleafConfig.DEFAULT,
    ) : this(baseUrl, HttpClient(engine) { applyLlmleaf(config, apiKey) })

    /** Releases the underlying HTTP client and its connection pool. */
    public fun close() {
        http.close()
    }

    // --- Chat completions --------------------------------------------------

    /** Non-streaming chat completion. Forces `stream=false` regardless of [request]. */
    public suspend fun chat(request: ChatRequest): ChatResponse {
        val body = request.copy(stream = false)
        val response = http.prepareRequest {
            method = HttpMethod.Post
            url("$base/v1/chat/completions")
            jsonBody(body, ChatRequest.serializer())
        }.execute { resp ->
            ensureSuccess(resp)
            decode(resp.bodyAsText(), ChatResponse.serializer())
        }
        return response
    }

    /**
     * Streaming chat completion. Forces `stream=true`, parses the `text/event-stream` body and
     * emits one [ChatCompletionChunk] per `data:` frame, stopping at `data: [DONE]` (the
     * sentinel is never JSON-parsed). The flow is cold; collecting it runs the request.
     */
    public fun chatStream(request: ChatRequest): Flow<ChatCompletionChunk> = flow {
        val body = request.copy(stream = true)
        http.prepareRequest {
            method = HttpMethod.Post
            url("$base/v1/chat/completions")
            // Streaming responses must not be cut by the per-request timeout.
            timeout { requestTimeoutMillis = Long.MAX_VALUE }
            header(HttpHeaders.Accept, ContentType.Text.EventStream.toString())
            jsonBody(body, ChatRequest.serializer())
        }.execute { resp ->
            ensureSuccess(resp)
            val channel = resp.bodyAsChannel()
            while (true) {
                val line = channel.readUTF8Line() ?: break
                if (line.isEmpty()) continue // blank line between SSE events
                if (!line.startsWith(SSE_DATA_PREFIX)) continue // ignore comments / event: lines
                val data = line.substring(SSE_DATA_PREFIX.length).trim()
                if (data == SSE_DONE) break // sentinel — do NOT parse it
                emit(decode(data, ChatCompletionChunk.serializer()))
            }
        }
    }

    // --- Embeddings --------------------------------------------------------

    /**
     * Creates embeddings. When [EmbeddingRequest.encodingFormat] is `"base64"` the wire's
     * base64 little-endian f32 payloads are decoded into the returned float vectors.
     */
    public suspend fun embeddings(request: EmbeddingRequest): EmbeddingResponse =
        http.prepareRequest {
            method = HttpMethod.Post
            url("$base/v1/embeddings")
            jsonBody(request, EmbeddingRequest.serializer())
        }.execute { resp ->
            ensureSuccess(resp)
            decode(resp.bodyAsText(), EmbeddingResponse.serializer())
        }

    // --- Models ------------------------------------------------------------

    /**
     * Lists models. [type] filters by modality; [search] is a substring filter. When the client
     * was built with an `adminToken`, each entry carries its admin-only `endpoints` array.
     */
    public suspend fun listModels(
        type: ModelType? = null,
        search: String? = null,
    ): ListModelsResponse =
        http.prepareGet("$base/v1/models") {
            type?.let { parameter("type", it.wire) }
            search?.let { parameter("search", it) }
        }.execute { resp ->
            ensureSuccess(resp)
            decode(resp.bodyAsText(), ListModelsResponse.serializer())
        }

    // --- Audio -------------------------------------------------------------

    /** Text-to-speech. Returns the raw audio [SpeechResult.bytes] and reported content type. */
    public suspend fun speech(request: SpeechRequest): SpeechResult =
        http.prepareRequest {
            method = HttpMethod.Post
            url("$base/v1/audio/speech")
            jsonBody(request, SpeechRequest.serializer())
        }.execute { resp ->
            ensureSuccess(resp)
            SpeechResult(
                bytes = resp.readRawBytes(),
                contentType = resp.headers[HttpHeaders.ContentType],
            )
        }

    /** Lists the voices available for a TTS model. */
    public suspend fun voices(model: String): VoicesResponse =
        http.prepareGet("$base/v1/audio/voices") {
            parameter("model", model)
        }.execute { resp ->
            ensureSuccess(resp)
            decode(resp.bodyAsText(), VoicesResponse.serializer())
        }

    /**
     * Speech-to-text via `multipart/form-data`. [audio] is the raw file bytes sent as the `file`
     * part under [filename]; [request] supplies `model` and any optional form fields.
     *
     * For `response_format` json/verbose_json the parsed [TranscriptionResponse] has its `text`
     * populated; for text/srt/vtt the gateway returns a plain-text body, surfaced verbatim in
     * [TranscriptionResponse.text] (other fields null).
     */
    public suspend fun transcribe(
        request: TranscriptionRequest,
        audio: ByteArray,
        filename: String,
    ): TranscriptionResponse {
        val multipart = MultiPartFormDataContent(
            formData {
                append(
                    "file",
                    audio,
                    io.ktor.http.Headers.build {
                        append(HttpHeaders.ContentType, ContentType.Application.OctetStream.toString())
                        append(HttpHeaders.ContentDisposition, "filename=\"$filename\"")
                    },
                )
                append("model", request.model)
                request.language?.let { append("language", it) }
                request.prompt?.let { append("prompt", it) }
                request.responseFormat?.let { append("response_format", it) }
                request.temperature?.let { append("temperature", it.toString()) }
            },
        )
        return http.prepareRequest {
            method = HttpMethod.Post
            url("$base/v1/audio/transcriptions")
            setBody(multipart)
        }.execute { resp ->
            ensureSuccess(resp)
            val text = resp.bodyAsText()
            val ct = resp.contentType()
            if (ct != null && ct.match(ContentType.Application.Json)) {
                decode(text, TranscriptionResponse.serializer())
            } else {
                // text / srt / vtt — return the plain body verbatim.
                TranscriptionResponse(text = text)
            }
        }
    }

    // --- Batches -----------------------------------------------------------

    /** Creates a batch of chat requests. */
    public suspend fun createBatch(request: BatchCreateRequest): BatchHandle =
        http.prepareRequest {
            method = HttpMethod.Post
            url("$base/v1/batches")
            jsonBody(request, BatchCreateRequest.serializer())
        }.execute { resp ->
            ensureSuccess(resp)
            decode(resp.bodyAsText(), BatchHandle.serializer())
        }

    /** Retrieves a batch's current state. */
    public suspend fun getBatch(id: String): BatchHandle =
        http.prepareGet("$base/v1/batches/$id").execute { resp ->
            ensureSuccess(resp)
            decode(resp.bodyAsText(), BatchHandle.serializer())
        }

    /** Cancels a batch. */
    public suspend fun cancelBatch(id: String): BatchHandle =
        http.prepareRequest {
            method = HttpMethod.Post
            url("$base/v1/batches/$id/cancel")
        }.execute { resp ->
            ensureSuccess(resp)
            decode(resp.bodyAsText(), BatchHandle.serializer())
        }

    /**
     * Streams a completed batch's results as a [Flow] of [BatchResultLine], one per NDJSON line
     * (`application/x-ndjson`). Cold; collecting it runs the request.
     */
    public fun batchResults(id: String): Flow<BatchResultLine> = flow {
        http.prepareGet("$base/v1/batches/$id/results") {
            timeout { requestTimeoutMillis = Long.MAX_VALUE }
        }.execute { resp ->
            ensureSuccess(resp)
            val channel: ByteReadChannel = resp.bodyAsChannel()
            while (true) {
                val line = channel.readUTF8Line() ?: break
                if (line.isBlank()) continue
                emit(decode(line, BatchResultLine.serializer()))
            }
        }
    }

    // --- Internals ---------------------------------------------------------

    private fun <T> HttpRequestBuilder.jsonBody(value: T, serializer: kotlinx.serialization.KSerializer<T>) {
        // Serialise via the shared Json so RawJson splicing / unions are applied, then send the
        // string as a JSON body. (ContentNegotiation would also work, but explicit encoding keeps
        // the custom serializers and merge-at-top-level behaviour unambiguous.)
        val text = LenientJson.encodeToString(serializer, value)
        contentType(ContentType.Application.Json)
        setBody(text)
    }

    private suspend fun ensureSuccess(resp: HttpResponse) {
        if (resp.status.isSuccess()) return
        val raw = runCatching { resp.bodyAsText() }.getOrDefault("")
        val message = parseErrorMessage(raw) ?: defaultMessage(resp.status)
        throw ApiError(resp.status.value, message)
    }

    private fun parseErrorMessage(raw: String): String? {
        if (raw.isBlank()) return null
        return runCatching {
            LenientJson.decodeFromString(ErrorResponse.serializer(), raw).error.message
        }.getOrNull()
    }

    private fun <T> decode(raw: String, serializer: kotlinx.serialization.KSerializer<T>): T =
        try {
            LenientJson.decodeFromString(serializer, raw)
        } catch (e: SerializationException) {
            throw ApiError(0, "failed to decode response: ${e.message}")
        }

    private companion object {
        const val SSE_DATA_PREFIX = "data:"
        const val SSE_DONE = "[DONE]"

        fun defaultMessage(status: HttpStatusCode): String = when (status.value) {
            400 -> "bad request"
            401 -> "missing or invalid API key"
            403 -> "blocked or model not allowed"
            404 -> "no route for model"
            429 -> "key suspended (rate limiter)"
            502 -> "all upstreams failed"
            else -> status.description.ifBlank { "request failed" }
        }
    }
}

private const val ADMIN_TOKEN_HEADER = "x-admin-token"

/**
 * Builds the default-engine [HttpClient]. A free top-level function (not a member, not `run {}`)
 * so the primary constructor can call it inside its `this(...)` delegation without touching the
 * not-yet-initialised receiver.
 */
private fun buildDefaultHttpClient(config: LlmleafConfig, apiKey: String): HttpClient =
    HttpClient(defaultHttpClientEngine()) { applyLlmleaf(config, apiKey) }

/**
 * Shared HttpClient configuration applied by every [LlmleafClient] constructor. Declared on
 * `HttpClientConfig<*>` as a top-level function so it applies uniformly whatever the engine's
 * config type is — sidestepping the star-projection issue of abstracting over engine
 * construction. Plugins and `defaultRequest` do not depend on the engine config type.
 */
private fun HttpClientConfig<*>.applyLlmleaf(config: LlmleafConfig, apiKey: String) {
    expectSuccess = false // non-2xx is mapped to ApiError by the call sites
    install(ContentNegotiation) {
        json(LenientJson)
    }
    install(HttpTimeout) {
        requestTimeoutMillis = config.requestTimeout.inWholeMilliseconds
    }
    defaultRequest {
        bearerAuth(apiKey)
        config.adminToken?.let { header(ADMIN_TOKEN_HEADER, it) }
    }
}
