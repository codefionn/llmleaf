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

/// <summary>Breakdown of <see cref="Usage.PromptTokens"/>. Today only the cache-read (hit) share is
/// surfaced — the count of prompt tokens served from the provider's cache rather than processed fresh.</summary>
/// <param name="CachedTokens">Prompt tokens served from the provider's cache (a cache <em>read</em>/hit).</param>
public sealed record PromptTokensDetails(uint? CachedTokens = null);

/// <summary>Token accounting echoed on every response.</summary>
/// <param name="PromptTokens">Tokens in the prompt.</param>
/// <param name="CompletionTokens">Tokens generated.</param>
/// <param name="TotalTokens">Prompt + completion.</param>
/// <param name="CostUsd">llmleaf addition; null when the model has no known price.</param>
/// <param name="PromptTokensDetails">Prompt-cache hit accounting (OpenAI <c>usage.prompt_tokens_details</c>);
/// null when the upstream reported no caching.</param>
/// <param name="CacheCreationTokens">Input tokens written to the provider's prompt cache this request —
/// a cache <em>write</em> (creation). An llmleaf extension (Anthropic reports it; OpenAI/OpenRouter do
/// not); null when there were none.</param>
public sealed record Usage(
    uint PromptTokens,
    uint CompletionTokens,
    uint TotalTokens,
    double? CostUsd = null,
    PromptTokensDetails? PromptTokensDetails = null,
    uint? CacheCreationTokens = null);

/// <summary>The canonical error envelope body: <c>{"message":...,"type"?:...,"code"?:...}</c>. Carried
/// inline on a failed <see cref="ResponsesResponse"/>; the transport-level equivalent is <see cref="ApiException"/>.</summary>
public sealed record ErrorBody(string Message, string? Type = null, string? Code = null);

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

/// <summary>
/// One structured reasoning ("thinking") block (OpenRouter <c>reasoning_details[]</c>). It expresses
/// both <em>open</em> reasoning — visible text, optionally signed — and <em>hidden</em> reasoning — an
/// encrypted/redacted blob the provider returns in place of the text. <see cref="Type"/> (wire
/// <c>type</c>) is the discriminator:
/// <list type="bullet">
/// <item><c>"reasoning.text"</c> → <see cref="Text"/> (+ optional <see cref="Signature"/>) — <em>open</em>.</item>
/// <item><c>"reasoning.summary"</c> → <see cref="Summary"/> — <em>open</em> (a summarised view).</item>
/// <item><c>"reasoning.encrypted"</c> → <see cref="Data"/> — <em>hidden</em> (redacted / opaque).</item>
/// </list>
/// <see cref="Signature"/> and <see cref="Data"/> are opaque and MUST be sent back verbatim in the next
/// request's <c>reasoning_details</c> to continue a signed/encrypted reasoning turn (the upstream rejects
/// an altered or dropped block — e.g. before a tool call).
/// </summary>
public sealed record ReasoningDetail
{
    /// <summary>Wire discriminator: <c>"reasoning.text"</c>, <c>"reasoning.summary"</c>, <c>"reasoning.encrypted"</c>, …</summary>
    public required string Type { get; init; }

    /// <summary>Open reasoning text (<c>"reasoning.text"</c>).</summary>
    public string? Text { get; init; }

    /// <summary>Open summarised reasoning (<c>"reasoning.summary"</c>).</summary>
    public string? Summary { get; init; }

    /// <summary>Hidden encrypted/redacted blob (<c>"reasoning.encrypted"</c>); opaque, replayed verbatim.</summary>
    public string? Data { get; init; }

    /// <summary>Opaque signature for a signed open block; replayed verbatim.</summary>
    public string? Signature { get; init; }

    public string? Id { get; init; }

    /// <summary>Provider encoding tag when known, e.g. <c>"anthropic-claude-v1"</c>.</summary>
    public string? Format { get; init; }

    public uint? Index { get; init; }

    /// <summary>Whether this block is hidden (redacted / encrypted) rather than open visible reasoning.</summary>
    public bool IsHidden => Type == "reasoning.encrypted" || (Data is not null && Text is null);

    /// <summary>The visible reasoning text of an open block — its <see cref="Text"/>, falling back to its
    /// <see cref="Summary"/>; null for a hidden block.</summary>
    public string? OpenText => Text ?? Summary;
}

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

    /// <summary>Open reasoning text the assistant emitted (OpenRouter <c>reasoning</c>), if any. The flat,
    /// human-readable form; the structured <see cref="ReasoningDetails"/> is the replay-safe one.</summary>
    public string? Reasoning { get; init; }

    /// <summary>Structured reasoning blocks (open and hidden, with signatures — see <see cref="ReasoningDetail"/>).
    /// Echo these back verbatim on the next request to preserve signed reasoning across a turn.</summary>
    public IReadOnlyList<ReasoningDetail>? ReasoningDetails { get; init; }

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

/// <summary>A streaming delta. <c>Reasoning</c> is incremental open reasoning text; <c>ReasoningDetails</c>
/// carries incremental structured reasoning blocks (open / hidden — see <see cref="ReasoningDetail"/>).</summary>
public sealed record Delta(
    Role? Role = null,
    string? Content = null,
    IReadOnlyList<ToolCallDelta>? ToolCalls = null,
    string? Reasoning = null,
    IReadOnlyList<ReasoningDetail>? ReasoningDetails = null);

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
// Responses (POST /v1/responses) — the OpenAI Responses dialect
// ---------------------------------------------------------------------------

/// <summary>One content part of a Responses message (input or output). Unlike the chat dialect,
/// <see cref="InputImagePart.ImageUrl"/> is a plain string and an <see cref="OutputTextPart"/> carries
/// an <c>annotations</c> array (emitted as <c>[]</c> when constructing one).</summary>
public abstract record ResponseContentPart;

/// <summary>An input text part: <c>{"type":"input_text","text":...}</c>.</summary>
public sealed record InputTextPart(string Text) : ResponseContentPart;

/// <summary>An input image part: <c>{"type":"input_image","image_url":"&lt;url&gt;","detail":...}</c>.
/// <see cref="ImageUrl"/> is a plain string, NOT the chat dialect's nested <c>{url}</c> object.</summary>
public sealed record InputImagePart(string ImageUrl, string? Detail = null) : ResponseContentPart;

/// <summary>An output text part: <c>{"type":"output_text","text":...,"annotations":[]}</c>.</summary>
public sealed record OutputTextPart(string Text) : ResponseContentPart;

/// <summary>Message content: either plain <see cref="Text"/> or an array of <see cref="Parts"/>. Exactly
/// one is set. Wire <c>content</c> is a bare string for the former, an array for the latter.</summary>
public sealed record ResponseContent
{
    /// <summary>Plain text content, or null when this is a multipart message.</summary>
    public string? Text { get; init; }

    /// <summary>Content parts, or null when this is plain text.</summary>
    public IReadOnlyList<ResponseContentPart>? Parts { get; init; }

    /// <summary>Plain-text content.</summary>
    public static ResponseContent FromText(string text) => new() { Text = text };

    /// <summary>Multipart content.</summary>
    public static ResponseContent FromParts(IReadOnlyList<ResponseContentPart> parts) => new() { Parts = parts };

    /// <summary>Convenience: implicitly wrap a string as plain-text content.</summary>
    public static implicit operator ResponseContent(string text) => FromText(text);
}

/// <summary>One item of the request <c>input</c> array or the response <c>output</c> array. The wire
/// discriminator is <c>"type"</c>; a <see cref="ResponseMessageItem"/> serialises as a role-keyed object
/// with no <c>"type"</c>, the others carry their <c>"type"</c>.</summary>
public abstract record ResponseItem;

/// <summary>A conversation message item. On input, <see cref="Role"/> is <c>"user"</c> | <c>"system"</c> |
/// <c>"developer"</c> | <c>"assistant"</c> and <see cref="Content"/> is a bare string or input parts; on
/// output, role is <c>"assistant"</c>, content is output_text parts, and <see cref="Status"/>/<see cref="Id"/>
/// are set. <see cref="Role"/> is a plain string (the dialect's <c>"developer"</c> role has no enum).</summary>
public sealed record ResponseMessageItem : ResponseItem
{
    /// <summary>Output-only server id.</summary>
    public string? Id { get; init; }

    /// <summary>Author role: <c>"user"</c> | <c>"system"</c> | <c>"developer"</c> | <c>"assistant"</c>.</summary>
    public required string Role { get; init; }

    /// <summary>Message content (string or content parts).</summary>
    public ResponseContent? Content { get; init; }

    /// <summary>Output only: <c>"in_progress"</c> | <c>"completed"</c>.</summary>
    public string? Status { get; init; }

    /// <summary>Convenience constructor for a message with the given role and content.</summary>
    public static ResponseMessageItem Message(string role, ResponseContent content) => new() { Role = role, Content = content };

    /// <summary>Convenience: a <c>"user"</c> message.</summary>
    public static ResponseMessageItem User(ResponseContent content) => new() { Role = "user", Content = content };
}

/// <summary>A function call the model made. <see cref="CallId"/> pairs it with its
/// <see cref="ResponseFunctionCallOutputItem"/>; <see cref="Arguments"/> is the raw JSON string.</summary>
public sealed record ResponseFunctionCallItem : ResponseItem
{
    public string? Id { get; init; }
    public required string CallId { get; init; }
    public required string Name { get; init; }
    public required string Arguments { get; init; }
    public string? Status { get; init; }
}

/// <summary>The caller's answer to a function call, replayed on the next turn.</summary>
public sealed record ResponseFunctionCallOutputItem : ResponseItem
{
    public string? Id { get; init; }
    public required string CallId { get; init; }
    public required string Output { get; init; }
}

/// <summary>One reasoning-text entry of a <see cref="ResponseReasoningItem"/>. Its wire <c>"type"</c>
/// (<c>summary_text</c> vs <c>reasoning_text</c>) is decided by the list it lives in.</summary>
public sealed record ResponseReasoningText(string Text);

/// <summary>A reasoning ("thinking") item. <see cref="Summary"/> entries serialise as
/// <c>{"type":"summary_text","text"}</c>, <see cref="Content"/> entries as <c>{"type":"reasoning_text","text"}</c>.
/// <see cref="EncryptedContent"/> is opaque and MUST be echoed back verbatim in the next request's input to
/// continue an encrypted reasoning turn.</summary>
public sealed record ResponseReasoningItem : ResponseItem
{
    public string? Id { get; init; }
    public IReadOnlyList<ResponseReasoningText>? Summary { get; init; }
    public IReadOnlyList<ResponseReasoningText>? Content { get; init; }
    public string? EncryptedContent { get; init; }
}

/// <summary>The request <c>input</c>: a bare string (one user message) or an array of items. Exactly one
/// is set. Implicitly constructible from a string.</summary>
public sealed record ResponsesInput
{
    /// <summary>Bare-string input, or null when items are used.</summary>
    public string? Text { get; init; }

    /// <summary>Item-array input, or null when a bare string is used.</summary>
    public IReadOnlyList<ResponseItem>? Items { get; init; }

    /// <summary>A single-message bare-string input.</summary>
    public static ResponsesInput FromText(string text) => new() { Text = text };

    /// <summary>An item-array input.</summary>
    public static ResponsesInput FromItems(IReadOnlyList<ResponseItem> items) => new() { Items = items };

    /// <summary>Convenience: implicitly wrap a string as bare-string input.</summary>
    public static implicit operator ResponsesInput(string text) => FromText(text);
}

/// <summary>A tool the model MAY call — FLAT in this dialect (<c>type</c>/<c>name</c>/<c>parameters</c> at
/// the top level, no nested <c>function</c> object). <see cref="Parameters"/> is a raw JSON Schema string.</summary>
public sealed record ResponsesToolDef(
    string Type,
    string Name,
    string? Description = null,
    string? Parameters = null,
    bool? Strict = null);

/// <summary>tool_choice: a bare mode string (<c>"auto"</c>/<c>"none"</c>/<c>"required"</c>) or a named
/// function (the FLAT <c>{"type":"function","name":...}</c>, unlike the chat dialect). Exactly one is set.</summary>
public sealed record ResponsesToolChoice
{
    /// <summary>The mode string, or null when a named function is pinned.</summary>
    public string? Mode { get; init; }

    /// <summary>The pinned function name, or null when a mode string is used.</summary>
    public string? FunctionName { get; init; }

    /// <summary>A mode choice such as <c>"auto"</c>.</summary>
    public static ResponsesToolChoice FromMode(string mode) => new() { Mode = mode };

    /// <summary>Pin a specific function by name.</summary>
    public static ResponsesToolChoice Named(string functionName) => new() { FunctionName = functionName };
}

/// <summary>reasoning config: <c>{"effort":...,"summary":...}</c>.</summary>
public sealed record ResponsesReasoning(string? Effort = null, string? Summary = null);

/// <summary>POST /v1/responses request body.</summary>
public sealed record ResponsesRequest
{
    /// <summary>Model id.</summary>
    public required string Model { get; init; }

    /// <summary>Input: a bare string or an array of items.</summary>
    public required ResponsesInput Input { get; init; }

    /// <summary>Prepended as a leading system message.</summary>
    public string? Instructions { get; init; }

    /// <summary>Request streaming. The client overrides this per call method; setting it here is optional.</summary>
    public bool? Stream { get; init; }

    public float? Temperature { get; init; }
    public float? TopP { get; init; }
    public uint? MaxOutputTokens { get; init; }
    public IReadOnlyList<ResponsesToolDef>? Tools { get; init; }
    public ResponsesToolChoice? ToolChoice { get; init; }
    public ResponsesReasoning? Reasoning { get; init; }

    /// <summary>Accepted but always answered <c>false</c> — llmleaf stores nothing.</summary>
    public bool? Store { get; init; }

    /// <summary>Dialect passthrough: a raw JSON object string, merged at the top level of the request.</summary>
    public string? Extra { get; init; }
}

/// <summary>Breakdown of <see cref="ResponsesUsage.InputTokens"/>.</summary>
public sealed record ResponsesInputTokensDetails(uint? CachedTokens = null);

/// <summary>Breakdown of <see cref="ResponsesUsage.OutputTokens"/>.</summary>
public sealed record ResponsesOutputTokensDetails(uint? ReasoningTokens = null);

/// <summary>Token accounting in the Responses dialect's own names (<c>input_tokens</c>/<c>output_tokens</c>).</summary>
public sealed record ResponsesUsage(
    uint InputTokens,
    uint OutputTokens,
    uint TotalTokens,
    ResponsesInputTokensDetails? InputTokensDetails = null,
    ResponsesOutputTokensDetails? OutputTokensDetails = null);

/// <summary>status <c>"incomplete"</c> refinement: <c>"max_output_tokens"</c> | <c>"content_filter"</c>.</summary>
public sealed record ResponsesIncompleteDetails(string Reason);

/// <summary>POST /v1/responses non-streaming response (<c>object:"response"</c>); also the snapshot carried
/// by the <c>response.created</c>/<c>in_progress</c>/<c>completed</c> stream events. <see cref="Status"/> is
/// a plain string (<c>"completed"</c> | <c>"in_progress"</c> | <c>"incomplete"</c> | <c>"failed"</c>);
/// <see cref="Store"/> is always <c>false</c> (llmleaf is stateless).</summary>
public sealed record ResponsesResponse
{
    public required string Id { get; init; }
    public required string Object { get; init; }
    public long CreatedAt { get; init; }
    public required string Status { get; init; }
    public ResponsesIncompleteDetails? IncompleteDetails { get; init; }
    public ErrorBody? Error { get; init; }
    public required string Model { get; init; }
    public IReadOnlyList<ResponseItem> Output { get; init; } = [];

    /// <summary>Token accounting; null on in-flight snapshots.</summary>
    public ResponsesUsage? Usage { get; init; }

    public bool? Store { get; init; }
    public string? Instructions { get; init; }
    public uint? MaxOutputTokens { get; init; }
    public float? Temperature { get; init; }
    public float? TopP { get; init; }
    public ResponsesReasoning? Reasoning { get; init; }
}

/// <summary>One streaming SSE event. Unlike chat streaming there is NO <c>data: [DONE]</c> sentinel — the
/// stream ends after the terminal <c>response.completed</c>/<c>response.incomplete</c>/<c>response.failed</c>
/// event (<see cref="IsTerminal"/>). This is a flat superset of every event's fields; <see cref="Type"/>
/// says which are meaningful. The <c>"error"</c> event is surfaced as a frame with <see cref="Message"/> set.</summary>
public sealed record ResponsesStreamEvent
{
    /// <summary>Event type, e.g. <c>"response.created"</c>, <c>"response.output_text.delta"</c>, <c>"error"</c>.</summary>
    public required string Type { get; init; }

    /// <summary>Strictly increasing across the stream.</summary>
    public ulong SequenceNumber { get; init; }

    /// <summary>Response snapshot (<c>response.created</c>/<c>in_progress</c>/<c>completed</c>/<c>incomplete</c>/<c>failed</c>).</summary>
    public ResponsesResponse? Response { get; init; }

    public uint? OutputIndex { get; init; }
    public string? ItemId { get; init; }
    public uint? ContentIndex { get; init; }

    /// <summary>The item (<c>response.output_item.added</c>/<c>done</c>).</summary>
    public ResponseItem? Item { get; init; }

    /// <summary>The content part (<c>response.content_part.added</c>/<c>done</c>).</summary>
    public ResponseContentPart? Part { get; init; }

    /// <summary>Incremental text on a <c>*.delta</c> event (text / reasoning / arguments).</summary>
    public string? Delta { get; init; }

    /// <summary>The assembled text on a <c>*.done</c> event (<c>output_text.done</c> / <c>reasoning_text.done</c>).</summary>
    public string? Text { get; init; }

    /// <summary>The assembled arguments on <c>response.function_call_arguments.done</c>.</summary>
    public string? Arguments { get; init; }

    /// <summary>The message on an <c>"error"</c> event.</summary>
    public string? Message { get; init; }

    /// <summary>True for the terminal <c>response.completed</c>/<c>response.incomplete</c>/<c>response.failed</c>
    /// events, after which the stream ends.</summary>
    public bool IsTerminal => Type is "response.completed" or "response.incomplete" or "response.failed";
}

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
