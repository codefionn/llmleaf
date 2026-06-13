// The official C# client for the llmleaf LLM proxy.
//
// The typed model is generated from clients/proto/llmleaf/v1/llmleaf.proto into src/Gen
// (Google.Protobuf classes — the schema proof). This class adds a hand-written HttpClient +
// System.Text.Json transport that (de)serialises the public records in Models.cs to and from the
// OpenAI/OpenRouter-shaped JSON the llmleaf core speaks (see clients/SPEC.md). The wire is JSON,
// never protobuf-binary.
//
// Construct a client with `new LlmleafClient(baseUrl, apiKey, options?)`, then call the endpoint
// methods. Every call takes a CancellationToken. Non-2xx responses surface as ApiException.

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Web;
using Llmleaf.Client.Wire;

namespace Llmleaf.Client;

/// <summary>Construction options for <see cref="LlmleafClient"/>.</summary>
public sealed record LlmleafClientOptions
{
    /// <summary>HTTP timeout. Applied only when <see cref="HttpClient"/> is not supplied. Defaults to 60s.</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(60);

    /// <summary>Optional <c>x-admin-token</c>; with it, GET /v1/models can include each model's <c>endpoints</c>.</summary>
    public string? AdminToken { get; init; }

    /// <summary>
    /// An injected <see cref="System.Net.Http.HttpClient"/> (proxies, transport tuning, custom TLS,
    /// pooling). When supplied it owns its own timeout and <see cref="Timeout"/> is ignored; the
    /// client does NOT dispose it.
    /// </summary>
    public HttpClient? HttpClient { get; init; }
}

/// <summary>
/// Talks to a single llmleaf gateway. Safe for concurrent use. Dispose to release the
/// internally-created <see cref="System.Net.Http.HttpClient"/> (a no-op when one was injected).
/// </summary>
public sealed class LlmleafClient : IDisposable
{
    private readonly Uri _baseUrl;
    private readonly string _apiKey;
    private readonly string? _adminToken;
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    /// <summary>
    /// Build a client for the given gateway root (e.g. <c>https://gateway.example.com</c>) and API
    /// key. The key is sent as <c>Authorization: Bearer &lt;key&gt;</c>.
    /// </summary>
    public LlmleafClient(string baseUrl, string apiKey, LlmleafClientOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(baseUrl);
        ArgumentNullException.ThrowIfNull(apiKey);
        options ??= new LlmleafClientOptions();

        // Normalise to a root with exactly one trailing slash so relative paths resolve cleanly.
        _baseUrl = new Uri(baseUrl.TrimEnd('/') + "/", UriKind.Absolute);
        _apiKey = apiKey;
        _adminToken = options.AdminToken;

        if (options.HttpClient is { } injected)
        {
            _http = injected;
            _ownsHttp = false;
        }
        else
        {
            _http = new HttpClient { Timeout = options.Timeout };
            _ownsHttp = true;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsHttp)
        {
            _http.Dispose();
        }
    }

    // ---- chat completions ----------------------------------------------

    /// <summary>
    /// Non-streaming chat completion (POST /v1/chat/completions). The wire <c>stream</c> flag is
    /// forced false; use <see cref="CreateChatCompletionStreamAsync"/> for streaming.
    /// </summary>
    public async Task<ChatResponse> CreateChatCompletionAsync(ChatRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var body = Mapper.EncodeChatRequest(request, streamOverride: false);
        var wire = await SendJsonForJsonAsync<WireChatResponse>(HttpMethod.Post, "v1/chat/completions", body, cancellationToken).ConfigureAwait(false);
        return Mapper.ChatResponseFromWire(wire);
    }

    /// <summary>
    /// Streaming chat completion (POST /v1/chat/completions with <c>stream:true</c>). Yields
    /// <see cref="ChatCompletionChunk"/> values parsed from the SSE body until the <c>data: [DONE]</c>
    /// sentinel (handled internally, never yielded). Accumulate <c>choices[].delta.content</c> for the
    /// assembled text; <c>usage</c> appears only on the terminal chunk when present.
    /// </summary>
    public async IAsyncEnumerable<ChatCompletionChunk> CreateChatCompletionStreamAsync(
        ChatRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var body = Mapper.EncodeChatRequest(request, streamOverride: true);

        using var req = BuildRequest(HttpMethod.Post, "v1/chat/completions");
        req.Content = JsonContent(body);
        req.Headers.Accept.ParseAdd("text/event-stream");

        using var resp = await _http
            .SendAsync(req, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);
        await EnsureSuccessAsync(resp, cancellationToken).ConfigureAwait(false);

#if NET8_0_OR_GREATER
        await using var stream = await resp.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
#else
        await using var stream = await resp.Content.ReadAsStreamAsync().ConfigureAwait(false);
#endif
        await foreach (var payload in LineReader.ParseSseDataAsync(stream, cancellationToken).ConfigureAwait(false))
        {
            var wire = JsonSerializer.Deserialize<WireChunk>(payload, Json.Options);
            if (wire is not null)
            {
                yield return Mapper.ChunkFromWire(wire);
            }
        }
    }

    // ---- embeddings -----------------------------------------------------

    /// <summary>
    /// Create embeddings (POST /v1/embeddings). When <see cref="EmbeddingRequest.EncodingFormat"/> is
    /// "base64", each returned vector is decoded from little-endian f32 bytes into floats for you.
    /// </summary>
    public async Task<EmbeddingResponse> CreateEmbeddingAsync(EmbeddingRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var body = Mapper.EncodeEmbeddingRequest(request);
        var wire = await SendJsonForJsonAsync<WireEmbeddingResponse>(HttpMethod.Post, "v1/embeddings", body, cancellationToken).ConfigureAwait(false);
        return Mapper.EmbeddingResponseFromWire(wire);
    }

    // ---- models ---------------------------------------------------------

    /// <summary>List the model catalog (GET /v1/models).</summary>
    public async Task<ListModelsResponse> ListModelsAsync(ListModelsOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new ListModelsOptions();
        var query = HttpUtility.ParseQueryString(string.Empty);
        if (options.Type is { } type)
        {
            query["type"] = type.ToString().ToLowerInvariant();
        }
        if (!string.IsNullOrEmpty(options.Search))
        {
            query["search"] = options.Search;
        }
        var qs = query.Count > 0 ? "?" + query : "";

        using var req = BuildRequest(HttpMethod.Get, "v1/models" + qs);
        // The admin token is only attached when the caller opts in (it widens the response).
        if (options.Admin && !string.IsNullOrEmpty(_adminToken))
        {
            req.Headers.TryAddWithoutValidation("x-admin-token", _adminToken);
        }
        var wire = await SendForJsonAsync<WireListModelsResponse>(req, cancellationToken).ConfigureAwait(false);
        return Mapper.ListModelsResponseFromWire(wire);
    }

    // ---- audio: speech (TTS) -------------------------------------------

    /// <summary>
    /// Synthesise speech (POST /v1/audio/speech). Returns the raw audio bytes plus the
    /// <c>Content-Type</c> the server reported (which reflects <c>response_format</c>).
    /// </summary>
    public async Task<SpeechResult> CreateSpeechAsync(SpeechRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var body = Mapper.EncodeSpeechRequest(request);

        using var req = BuildRequest(HttpMethod.Post, "v1/audio/speech");
        req.Content = JsonContent(body);

        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        await EnsureSuccessAsync(resp, cancellationToken).ConfigureAwait(false);

        var contentType = resp.Content.Headers.ContentType?.ToString() ?? "application/octet-stream";
#if NET8_0_OR_GREATER
        var bytes = await resp.Content.ReadAsByteArrayAsync(cancellationToken).ConfigureAwait(false);
#else
        var bytes = await resp.Content.ReadAsByteArrayAsync().ConfigureAwait(false);
#endif
        return new SpeechResult(bytes, contentType);
    }

    /// <summary>List the voices a TTS model supports (GET /v1/audio/voices?model=&lt;id&gt;).</summary>
    public async Task<VoicesResponse> ListVoicesAsync(string model, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(model);
        var path = "v1/audio/voices?model=" + Uri.EscapeDataString(model);
        var wire = await SendForJsonAsync<WireVoicesResponse>(BuildRequest(HttpMethod.Get, path), cancellationToken).ConfigureAwait(false);
        return Mapper.VoicesResponseFromWire(wire);
    }

    // ---- audio: transcription (STT) ------------------------------------

    /// <summary>
    /// Transcribe audio (POST /v1/audio/transcriptions, multipart/form-data). For
    /// <c>response_format</c> json/verbose_json the server returns a structured
    /// <see cref="TranscriptionResponse"/>; for text/srt/vtt it returns a plain-text body, surfaced
    /// here in <see cref="TranscriptionResponse.Text"/> with the other fields null.
    /// </summary>
    public async Task<TranscriptionResponse> CreateTranscriptionAsync(
        TranscriptionFile file,
        TranscriptionRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(request);

        using var form = new MultipartFormDataContent();

        var fileContent = new ByteArrayContent(file.Content);
        if (!string.IsNullOrEmpty(file.ContentType))
        {
            fileContent.Headers.ContentType = new MediaTypeHeaderValue(file.ContentType);
        }
        // Quote name + filename explicitly (RFC 7578 / OpenAI wire form). .NET otherwise writes them
        // unquoted, which some strict gateways reject.
        fileContent.Headers.ContentDisposition = new ContentDispositionHeaderValue("form-data")
        {
            Name = "\"file\"",
            FileName = "\"" + file.FileName + "\"",
        };
        form.Add(fileContent);

        AddField(form, "model", request.Model);
        if (request.Language is { } lang) AddField(form, "language", lang);
        if (request.Prompt is { } prompt) AddField(form, "prompt", prompt);
        if (request.ResponseFormat is { } fmt) AddField(form, "response_format", fmt);
        if (request.Temperature is { } temp)
        {
            AddField(form, "temperature", temp.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        using var req = BuildRequest(HttpMethod.Post, "v1/audio/transcriptions");
        req.Content = form;

        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        await EnsureSuccessAsync(resp, cancellationToken).ConfigureAwait(false);

#if NET8_0_OR_GREATER
        var text = await resp.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
#else
        var text = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
#endif
        var mediaType = resp.Content.Headers.ContentType?.MediaType;
        var trimmed = text.TrimStart();
        // JSON formats (json/verbose_json) decode into the structured response; text/srt/vtt come back
        // as a plain body -> surface verbatim in Text.
        if ((mediaType is not null && mediaType.Contains("json", StringComparison.OrdinalIgnoreCase))
            || trimmed.StartsWith('{'))
        {
            var wire = JsonSerializer.Deserialize<WireTranscriptionResponse>(text, Json.Options)
                       ?? new WireTranscriptionResponse { Text = text };
            return Mapper.TranscriptionResponseFromWire(wire);
        }
        return new TranscriptionResponse(text);
    }

    // ---- batches --------------------------------------------------------

    /// <summary>Create a batch (POST /v1/batches).</summary>
    public async Task<BatchHandle> CreateBatchAsync(BatchCreateRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var body = Mapper.EncodeBatchCreateRequest(request);
        var wire = await SendJsonForJsonAsync<WireBatchHandle>(HttpMethod.Post, "v1/batches", body, cancellationToken).ConfigureAwait(false);
        return Mapper.BatchHandleFromWire(wire);
    }

    /// <summary>Retrieve a batch's current handle (GET /v1/batches/{id}).</summary>
    public async Task<BatchHandle> RetrieveBatchAsync(string batchId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(batchId);
        var wire = await SendForJsonAsync<WireBatchHandle>(
            BuildRequest(HttpMethod.Get, "v1/batches/" + Uri.EscapeDataString(batchId)),
            cancellationToken).ConfigureAwait(false);
        return Mapper.BatchHandleFromWire(wire);
    }

    /// <summary>Cancel a batch (POST /v1/batches/{id}/cancel).</summary>
    public async Task<BatchHandle> CancelBatchAsync(string batchId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(batchId);
        var wire = await SendForJsonAsync<WireBatchHandle>(
            BuildRequest(HttpMethod.Post, "v1/batches/" + Uri.EscapeDataString(batchId) + "/cancel"),
            cancellationToken).ConfigureAwait(false);
        return Mapper.BatchHandleFromWire(wire);
    }

    /// <summary>
    /// Stream a batch's results (GET /v1/batches/{id}/results, <c>application/x-ndjson</c>). Yields one
    /// <see cref="BatchResultLine"/> per line until the body ends.
    /// </summary>
    public async IAsyncEnumerable<BatchResultLine> GetBatchResultsAsync(
        string batchId,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(batchId);
        using var req = BuildRequest(HttpMethod.Get, "v1/batches/" + Uri.EscapeDataString(batchId) + "/results");
        req.Headers.Accept.ParseAdd("application/x-ndjson");

        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        await EnsureSuccessAsync(resp, cancellationToken).ConfigureAwait(false);

#if NET8_0_OR_GREATER
        await using var stream = await resp.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
#else
        await using var stream = await resp.Content.ReadAsStreamAsync().ConfigureAwait(false);
#endif
        await foreach (var line in LineReader.ParseNdjsonAsync(stream, cancellationToken).ConfigureAwait(false))
        {
            var wire = JsonSerializer.Deserialize<WireBatchResultLine>(line, Json.Options);
            if (wire is not null)
            {
                yield return Mapper.BatchResultLineFromWire(wire);
            }
        }
    }

    // ---- transport plumbing --------------------------------------------

    private HttpRequestMessage BuildRequest(HttpMethod method, string relativePath)
    {
        var req = new HttpRequestMessage(method, new Uri(_baseUrl, relativePath));
        req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        return req;
    }

    // Add a simple text field to a multipart form with an explicitly quoted name (RFC 7578).
    private static void AddField(MultipartFormDataContent form, string name, string value)
    {
        var content = new StringContent(value);
        content.Headers.ContentDisposition = new ContentDispositionHeaderValue("form-data")
        {
            Name = "\"" + name + "\"",
        };
        form.Add(content);
    }

    private static ByteArrayContent JsonContent(byte[] body)
    {
        var content = new ByteArrayContent(body);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        return content;
    }

    // POST/PUT a JSON body and decode a JSON response into T.
    private async Task<T> SendJsonForJsonAsync<T>(HttpMethod method, string path, byte[] body, CancellationToken ct)
    {
        using var req = BuildRequest(method, path);
        req.Content = JsonContent(body);
        return await SendForJsonAsync<T>(req, ct).ConfigureAwait(false);
    }

    // Send a prepared request and decode a JSON response into T. Disposes the request.
    private async Task<T> SendForJsonAsync<T>(HttpRequestMessage req, CancellationToken ct)
    {
        using (req)
        {
            using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            await EnsureSuccessAsync(resp, ct).ConfigureAwait(false);
#if NET8_0_OR_GREATER
            await using var stream = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            var value = await JsonSerializer.DeserializeAsync<T>(stream, Json.Options, ct).ConfigureAwait(false);
#else
            await using var stream = await resp.Content.ReadAsStreamAsync().ConfigureAwait(false);
            var value = await JsonSerializer.DeserializeAsync<T>(stream, Json.Options, ct).ConfigureAwait(false);
#endif
            return value ?? throw new ApiException((int)resp.StatusCode, "llmleaf: empty response body");
        }
    }

    // Throw a typed ApiException for any non-2xx response, parsing the {"error":{"message":...}} envelope.
    private static async Task EnsureSuccessAsync(HttpResponseMessage resp, CancellationToken ct)
    {
        if (resp.IsSuccessStatusCode)
        {
            return;
        }
        var status = (int)resp.StatusCode;
        var fallback = resp.ReasonPhrase ?? $"HTTP {status}";
        string body;
        try
        {
#if NET8_0_OR_GREATER
            body = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
#else
            body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
#endif
        }
        catch
        {
            throw new ApiException(status, fallback);
        }
        throw ApiException.FromBody(status, fallback, body);
    }
}
