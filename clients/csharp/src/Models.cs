// Public, hand-written C# surface mirroring proto/llmleaf/v1/llmleaf.proto.
//
// Why records and not the generated Google.Protobuf classes directly? protoc --csharp_out emits
// classes that use the Google.Protobuf runtime, whose JSON formatter produces proto canonical
// JSON (camelCase keys, enum VALUE NAMES) — which does NOT match the OpenAI/OpenRouter wire
// (snake_case keys, lowercase tokens like "assistant"), and has no notion of the
// "free-form JSON carried as a raw string", "content is string-or-array", or "stop is
// string-or-array" conventions in SPEC.md. So these immutable records are the ergonomic public
// model; src/Wire/*.cs maps them to/from the actual wire JSON with System.Text.Json.
//
// The generated descriptors/enums (src/Gen/Llmleaf.cs) remain the committed codegen artifact and
// are the single source of truth these shapes track.

using System.Collections.Generic;

namespace Llmleaf.Client;

// ---------------------------------------------------------------------------
// Common
// ---------------------------------------------------------------------------

/// <summary>Token accounting echoed on every response.</summary>
/// <param name="PromptTokens">Tokens in the prompt.</param>
/// <param name="CompletionTokens">Tokens generated.</param>
/// <param name="TotalTokens">Prompt + completion.</param>
/// <param name="CostUsd">llmleaf addition; null when the model has no known price.</param>
public sealed record Usage(
    uint PromptTokens,
    uint CompletionTokens,
    uint TotalTokens,
    double? CostUsd = null);

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/// <summary>One content part of a multimodal message: text or an image URL.</summary>
public abstract record ContentPart;

/// <summary>A plain-text content part: <c>{"type":"text","text":"..."}</c>.</summary>
public sealed record TextPart(string Text) : ContentPart;

/// <summary>
/// An image-URL content part: <c>{"type":"image_url","image_url":{"url":"...","detail":"auto"}}</c>.
/// </summary>
public sealed record ImageUrlPart(string Url, string? Detail = null) : ContentPart;

/// <summary>
/// Message content: either plain <see cref="Text"/> or an array of <see cref="Parts"/>. Exactly one
/// is set. Wire <c>content</c> is a bare string for the former, an array for the latter.
/// </summary>
public sealed record MessageContent
{
    /// <summary>Plain text content, or null when this is a multipart message.</summary>
    public string? Text { get; init; }

    /// <summary>Multimodal content parts, or null when this is plain text.</summary>
    public IReadOnlyList<ContentPart>? Parts { get; init; }

    /// <summary>Plain-text content.</summary>
    public static MessageContent FromText(string text) => new() { Text = text };

    /// <summary>Multimodal content.</summary>
    public static MessageContent FromParts(IReadOnlyList<ContentPart> parts) => new() { Parts = parts };

    /// <summary>Convenience: implicitly wrap a string as plain-text content.</summary>
    public static implicit operator MessageContent(string text) => FromText(text);
}

/// <summary>A function the model called. <see cref="Arguments"/> is a JSON-encoded string.</summary>
public sealed record FunctionCall(string Name, string Arguments);

/// <summary>A tool call emitted by the model.</summary>
public sealed record ToolCall(string Id, string Type, FunctionCall Function);

/// <summary>A single chat message.</summary>
public sealed record ChatMessage
{
    /// <summary>Author role.</summary>
    public Role Role { get; init; }

    /// <summary>Message content (string or content parts). Null for an empty assistant tool-call turn.</summary>
    public MessageContent? Content { get; init; }

    /// <summary>Optional author name.</summary>
    public string? Name { get; init; }

    /// <summary>Tool calls the model requested (assistant turns).</summary>
    public IReadOnlyList<ToolCall>? ToolCalls { get; init; }

    /// <summary>The tool call this message answers (set when <see cref="Role"/> is <see cref="Role.Tool"/>).</summary>
    public string? ToolCallId { get; init; }

    /// <summary>Convenience constructor for a plain-text message.</summary>
    public static ChatMessage Text(Role role, string content) => new() { Role = role, Content = content };
}

/// <summary>A function the model MAY call. <see cref="Parameters"/> is a raw JSON Schema string.</summary>
public sealed record FunctionDef(string Name, string? Description = null, string? Parameters = null);

/// <summary>A tool the model may use.</summary>
public sealed record ToolDef(string Type, FunctionDef Function);

/// <summary>
/// tool_choice: a bare mode string (<c>"auto"</c>/<c>"none"</c>/<c>"required"</c>) or a named function.
/// Exactly one is set.
/// </summary>
public sealed record ToolChoice
{
    /// <summary>The mode string, or null when a named function is pinned.</summary>
    public string? Mode { get; init; }

    /// <summary>The pinned function name, or null when a mode string is used.</summary>
    public string? FunctionName { get; init; }

    /// <summary>A mode choice such as <c>"auto"</c>.</summary>
    public static ToolChoice FromMode(string mode) => new() { Mode = mode };

    /// <summary>Pin a specific function by name.</summary>
    public static ToolChoice Named(string functionName) => new() { FunctionName = functionName };
}

/// <summary>response_format. <see cref="JsonSchema"/> is a raw JSON object string when type is json_schema.</summary>
public sealed record ResponseFormat(string Type, string? JsonSchema = null);

/// <summary>POST /v1/chat/completions request body.</summary>
public sealed record ChatRequest
{
    /// <summary>Model id.</summary>
    public required string Model { get; init; }

    /// <summary>Conversation messages.</summary>
    public required IReadOnlyList<ChatMessage> Messages { get; init; }

    /// <summary>Request streaming. The client overrides this per call method; setting it here is optional.</summary>
    public bool? Stream { get; init; }

    public float? Temperature { get; init; }
    public float? TopP { get; init; }

    /// <summary>Legacy token cap.</summary>
    public uint? MaxTokens { get; init; }

    /// <summary>Modern token cap (takes precedence over <see cref="MaxTokens"/>).</summary>
    public uint? MaxCompletionTokens { get; init; }

    /// <summary>Stop sequences. Emitted as a bare string for one element, else an array.</summary>
    public IReadOnlyList<string>? Stop { get; init; }

    public uint? N { get; init; }
    public long? Seed { get; init; }
    public float? FrequencyPenalty { get; init; }
    public float? PresencePenalty { get; init; }
    public IReadOnlyList<ToolDef>? Tools { get; init; }
    public ToolChoice? ToolChoice { get; init; }
    public ResponseFormat? ResponseFormat { get; init; }

    /// <summary>"low" | "medium" | "high".</summary>
    public string? ReasoningEffort { get; init; }

    /// <summary>Dialect passthrough: a raw JSON object string, merged at the top level of the request.</summary>
    public string? Extra { get; init; }
}

/// <summary>One non-streaming choice.</summary>
public sealed record Choice(uint Index, ChatMessage Message, FinishReason? FinishReason = null);

/// <summary>POST /v1/chat/completions non-streaming response (<c>object:"chat.completion"</c>).</summary>
public sealed record ChatResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<Choice> Choices,
    Usage? Usage = null);

// Streaming

/// <summary>Incremental tool-call fragment on a streaming delta.</summary>
public sealed record FunctionCallDelta(string? Name = null, string? Arguments = null);

/// <summary>Incremental tool call on a streaming delta; fields arrive piecemeal.</summary>
public sealed record ToolCallDelta(uint Index, string? Id = null, string? Type = null, FunctionCallDelta? Function = null);

/// <summary>A streaming delta.</summary>
public sealed record Delta(Role? Role = null, string? Content = null, IReadOnlyList<ToolCallDelta>? ToolCalls = null);

/// <summary>One streaming choice.</summary>
public sealed record ChunkChoice(uint Index, Delta Delta, FinishReason? FinishReason = null);

/// <summary>One SSE frame (<c>object:"chat.completion.chunk"</c>). <see cref="Usage"/> appears only on the terminal chunk.</summary>
public sealed record ChatCompletionChunk(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<ChunkChoice> Choices,
    Usage? Usage = null);

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// <summary>POST /v1/embeddings request body.</summary>
public sealed record EmbeddingRequest
{
    public required string Model { get; init; }

    /// <summary>One or more inputs. Emitted as a bare string for one element, else an array.</summary>
    public required IReadOnlyList<string> Input { get; init; }

    public uint? Dimensions { get; init; }

    /// <summary>"float" | "base64". When "base64", the response vectors are decoded to floats for you.</summary>
    public string? EncodingFormat { get; init; }

    /// <summary>Raw JSON object passthrough, merged at the top level.</summary>
    public string? Extra { get; init; }
}

/// <summary>One embedding vector. Always decoded to floats, even when the wire used base64.</summary>
public sealed record Embedding(string Object, uint Index, IReadOnlyList<float> Vector);

/// <summary>POST /v1/embeddings response (<c>object:"list"</c>).</summary>
public sealed record EmbeddingResponse(string Object, IReadOnlyList<Embedding> Data, string Model, Usage? Usage = null);

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

/// <summary>POST /v1/audio/speech request body.</summary>
public sealed record SpeechRequest
{
    public required string Model { get; init; }
    public required string Input { get; init; }
    public required string Voice { get; init; }

    /// <summary>mp3|opus|aac|flac|wav|pcm.</summary>
    public string? ResponseFormat { get; init; }

    public float? Speed { get; init; }

    /// <summary>Raw JSON object passthrough, merged at the top level.</summary>
    public string? Extra { get; init; }
}

/// <summary>Raw audio bytes plus the Content-Type the server reported.</summary>
public sealed record SpeechResult(byte[] Bytes, string ContentType);

/// <summary>A TTS voice.</summary>
public sealed record Voice(string Id, string? Name = null, IReadOnlyList<string>? Languages = null);

/// <summary>GET /v1/audio/voices response.</summary>
public sealed record VoicesResponse(string Model, IReadOnlyList<Voice> Voices);

/// <summary>The audio bytes for a transcription, plus the filename to send as the multipart part.</summary>
/// <param name="Content">The audio bytes.</param>
/// <param name="FileName">Filename for the <c>file</c> part (extension hints the format to the server).</param>
/// <param name="ContentType">Optional MIME type; null lets the server sniff it.</param>
public sealed record TranscriptionFile(byte[] Content, string FileName, string? ContentType = null);

/// <summary>Accompanying form fields for POST /v1/audio/transcriptions (the file is sent separately).</summary>
public sealed record TranscriptionRequest
{
    public required string Model { get; init; }

    /// <summary>ISO-639-1 language hint.</summary>
    public string? Language { get; init; }

    /// <summary>Decoding bias prompt.</summary>
    public string? Prompt { get; init; }

    /// <summary>json|text|verbose_json|srt|vtt. For text/srt/vtt the result is a plain-text body.</summary>
    public string? ResponseFormat { get; init; }

    public float? Temperature { get; init; }
}

/// <summary>POST /v1/audio/transcriptions response for json/verbose_json formats.</summary>
public sealed record TranscriptionResponse(
    string Text,
    string? Task = null,
    string? Language = null,
    float? Duration = null,
    Usage? Usage = null);

// ---------------------------------------------------------------------------
// Model catalog
// ---------------------------------------------------------------------------

public sealed record Architecture(
    IReadOnlyList<string> InputModalities,
    IReadOnlyList<string> OutputModalities,
    string Tokenizer,
    string? Modality = null,
    string? InstructType = null);

/// <summary>USD per token, as decimal strings.</summary>
public sealed record Pricing(string Prompt, string Completion);

public sealed record TopProvider(
    bool IsModerated,
    uint? ContextLength = null,
    uint? MaxCompletionTokens = null,
    uint? MaxThinkingTokens = null);

/// <summary>Admin-only fallback-chain entry (present only with a valid x-admin-token).</summary>
public sealed record ModelEndpoint(string Provider, string Model, bool Down, string Source);

public sealed record ModelEntry
{
    public required string Id { get; init; }
    public string CanonicalSlug { get; init; } = "";
    public string Name { get; init; } = "";
    public long Created { get; init; }
    public string Description { get; init; } = "";
    public uint? ContextLength { get; init; }
    public Architecture? Architecture { get; init; }
    public Pricing? Pricing { get; init; }
    public TopProvider? TopProvider { get; init; }
    public IReadOnlyList<string> SupportedParameters { get; init; } = [];
    public IReadOnlyList<string> UnsupportedParameters { get; init; } = [];

    /// <summary>Raw JSON object string.</summary>
    public string? DefaultParameters { get; init; }

    /// <summary>Admin-only; populated when the request carries the admin token.</summary>
    public IReadOnlyList<ModelEndpoint> Endpoints { get; init; } = [];
}

public sealed record ListModelsResponse(IReadOnlyList<ModelEntry> Data);

/// <summary>Filter for <see cref="LlmleafClient.ListModelsAsync"/>.</summary>
public enum ModelType
{
    All,
    Llm,
    Tts,
    Stt,
    Embedding,
}

/// <summary>Options for <see cref="LlmleafClient.ListModelsAsync"/>.</summary>
public sealed record ListModelsOptions
{
    /// <summary>Catalog filter.</summary>
    public ModelType? Type { get; init; }

    /// <summary>Substring search.</summary>
    public string? Search { get; init; }

    /// <summary>When true, send the configured admin token so per-model <c>endpoints</c> are included.</summary>
    public bool Admin { get; init; }
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

public sealed record BatchRequestItem(string CustomId, ChatRequest Body);

public sealed record BatchCreateRequest(IReadOnlyList<BatchRequestItem> Requests);

public sealed record BatchCounts(
    ulong Total,
    ulong Processing,
    ulong Succeeded,
    ulong Errored,
    ulong Canceled,
    ulong Expired);

public sealed record BatchHandle(
    string Id,
    BatchStatus Status,
    BatchCounts? Counts = null,
    long? CreatedAt = null,
    long? ExpiresAt = null,
    long? EndedAt = null,
    string? Endpoint = null);

public sealed record BatchResponse(uint StatusCode, ChatResponse Body);

public sealed record BatchError(string Code, string Message);

/// <summary>One line of GET /v1/batches/{id}/results. Exactly one of <see cref="Response"/>/<see cref="Error"/> is set.</summary>
public sealed record BatchResultLine(string CustomId, BatchResponse? Response = null, BatchError? Error = null);
