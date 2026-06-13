// Internal System.Text.Json DTOs mirroring the proto messages with the EXACT OpenAI/OpenRouter
// wire shape: snake_case keys (explicit [JsonPropertyName]), lowercase enum tokens, content/stop
// as string-or-array, free-form JSON spliced as raw values. The public records in Models.cs map
// to/from these; the transport never (de)serialises the public records directly.

using System.Collections.Generic;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace Llmleaf.Client.Wire;

// ---- content parts -------------------------------------------------------

internal sealed class WireImageUrl
{
    [JsonPropertyName("url")] public string Url { get; set; } = "";
    [JsonPropertyName("detail")] public string? Detail { get; set; }
}

// A content part on the wire is one of {type:text,text} | {type:image_url,image_url:{...}}.
// Custom (de)serialisation lives in ContentPartConverter.
[JsonConverter(typeof(ContentPartConverter))]
internal sealed class WireContentPart
{
    public string? Text { get; set; }
    public WireImageUrl? ImageUrl { get; set; }
}

// ---- tool calls ----------------------------------------------------------

internal sealed class WireFunctionCall
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("arguments")] public string Arguments { get; set; } = "";
}

internal sealed class WireToolCall
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("type")] public string Type { get; set; } = "";
    [JsonPropertyName("function")] public WireFunctionCall Function { get; set; } = new();
}

internal sealed class WireFunctionCallDelta
{
    [JsonPropertyName("name")] public string? Name { get; set; }
    [JsonPropertyName("arguments")] public string? Arguments { get; set; }
}

internal sealed class WireToolCallDelta
{
    [JsonPropertyName("index")] public uint Index { get; set; }
    [JsonPropertyName("id")] public string? Id { get; set; }
    [JsonPropertyName("type")] public string? Type { get; set; }
    [JsonPropertyName("function")] public WireFunctionCallDelta? Function { get; set; }
}

// ---- chat message --------------------------------------------------------

internal sealed class WireChatMessage
{
    [JsonPropertyName("role")] public string? Role { get; set; }

    // string or array of content parts, or null/absent.
    [JsonPropertyName("content")] public JsonNode? Content { get; set; }

    [JsonPropertyName("name")] public string? Name { get; set; }

    [JsonPropertyName("tool_calls")] public List<WireToolCall>? ToolCalls { get; set; }

    [JsonPropertyName("tool_call_id")] public string? ToolCallId { get; set; }
}

// ---- tools / tool_choice / response_format ------------------------------

internal sealed class WireFunctionDef
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("description")] public string? Description { get; set; }

    // raw JSON object (JSON Schema), spliced verbatim.
    [JsonPropertyName("parameters")] public JsonNode? Parameters { get; set; }
}

internal sealed class WireToolDef
{
    [JsonPropertyName("type")] public string Type { get; set; } = "";
    [JsonPropertyName("function")] public WireFunctionDef Function { get; set; } = new();
}

internal sealed class WireResponseFormat
{
    [JsonPropertyName("type")] public string Type { get; set; } = "";
    [JsonPropertyName("json_schema")] public JsonNode? JsonSchema { get; set; }
}

// ---- chat request --------------------------------------------------------

// Every ChatRequest field except `extra`, which is merged at the top level by the mapper.
internal sealed class WireChatRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("messages")] public List<WireChatMessage> Messages { get; set; } = [];
    [JsonPropertyName("stream")] public bool? Stream { get; set; }
    [JsonPropertyName("temperature")] public float? Temperature { get; set; }
    [JsonPropertyName("top_p")] public float? TopP { get; set; }
    [JsonPropertyName("max_tokens")] public uint? MaxTokens { get; set; }
    [JsonPropertyName("max_completion_tokens")] public uint? MaxCompletionTokens { get; set; }

    // bare string for one element, else an array.
    [JsonPropertyName("stop")] public JsonNode? Stop { get; set; }

    [JsonPropertyName("n")] public uint? N { get; set; }
    [JsonPropertyName("seed")] public long? Seed { get; set; }
    [JsonPropertyName("frequency_penalty")] public float? FrequencyPenalty { get; set; }
    [JsonPropertyName("presence_penalty")] public float? PresencePenalty { get; set; }
    [JsonPropertyName("tools")] public List<WireToolDef>? Tools { get; set; }

    // bare mode string or a named-function object.
    [JsonPropertyName("tool_choice")] public JsonNode? ToolChoice { get; set; }

    [JsonPropertyName("response_format")] public WireResponseFormat? ResponseFormat { get; set; }
    [JsonPropertyName("reasoning_effort")] public string? ReasoningEffort { get; set; }
}

// ---- chat response (decode) ---------------------------------------------

internal sealed class WireUsage
{
    [JsonPropertyName("prompt_tokens")] public uint PromptTokens { get; set; }
    [JsonPropertyName("completion_tokens")] public uint CompletionTokens { get; set; }
    [JsonPropertyName("total_tokens")] public uint TotalTokens { get; set; }
    [JsonPropertyName("cost_usd")] public double? CostUsd { get; set; }
}

internal sealed class WireChoice
{
    [JsonPropertyName("index")] public uint Index { get; set; }
    [JsonPropertyName("message")] public WireChatMessage Message { get; set; } = new();
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; set; }
}

internal sealed class WireChatResponse
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("object")] public string Object { get; set; } = "";
    [JsonPropertyName("created")] public long Created { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("choices")] public List<WireChoice> Choices { get; set; } = [];
    [JsonPropertyName("usage")] public WireUsage? Usage { get; set; }
}

// ---- streaming chunk (decode) -------------------------------------------

internal sealed class WireDelta
{
    [JsonPropertyName("role")] public string? Role { get; set; }
    [JsonPropertyName("content")] public string? Content { get; set; }
    [JsonPropertyName("tool_calls")] public List<WireToolCallDelta>? ToolCalls { get; set; }
}

internal sealed class WireChunkChoice
{
    [JsonPropertyName("index")] public uint Index { get; set; }
    [JsonPropertyName("delta")] public WireDelta Delta { get; set; } = new();
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; set; }
}

internal sealed class WireChunk
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("object")] public string Object { get; set; } = "";
    [JsonPropertyName("created")] public long Created { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("choices")] public List<WireChunkChoice> Choices { get; set; } = [];
    [JsonPropertyName("usage")] public WireUsage? Usage { get; set; }
}

// ---- embeddings ----------------------------------------------------------

internal sealed class WireEmbeddingRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = "";

    // string for one element, else an array.
    [JsonPropertyName("input")] public JsonNode? Input { get; set; }

    [JsonPropertyName("dimensions")] public uint? Dimensions { get; set; }
    [JsonPropertyName("encoding_format")] public string? EncodingFormat { get; set; }
}

internal sealed class WireEmbedding
{
    [JsonPropertyName("object")] public string Object { get; set; } = "";
    [JsonPropertyName("index")] public uint Index { get; set; }

    // float array (encoding_format "float") or a base64 string of little-endian f32 bytes.
    [JsonPropertyName("embedding")] public JsonNode? Embedding { get; set; }
}

internal sealed class WireEmbeddingResponse
{
    [JsonPropertyName("object")] public string Object { get; set; } = "";
    [JsonPropertyName("data")] public List<WireEmbedding> Data { get; set; } = [];
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("usage")] public WireUsage? Usage { get; set; }
}

// ---- speech / voices -----------------------------------------------------

internal sealed class WireSpeechRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("input")] public string Input { get; set; } = "";
    [JsonPropertyName("voice")] public string Voice { get; set; } = "";
    [JsonPropertyName("response_format")] public string? ResponseFormat { get; set; }
    [JsonPropertyName("speed")] public float? Speed { get; set; }
}

internal sealed class WireVoice
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("name")] public string? Name { get; set; }
    [JsonPropertyName("languages")] public List<string>? Languages { get; set; }
}

internal sealed class WireVoicesResponse
{
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("voices")] public List<WireVoice> Voices { get; set; } = [];
}

// ---- transcription -------------------------------------------------------

internal sealed class WireTranscriptionResponse
{
    [JsonPropertyName("text")] public string Text { get; set; } = "";
    [JsonPropertyName("task")] public string? Task { get; set; }
    [JsonPropertyName("language")] public string? Language { get; set; }
    [JsonPropertyName("duration")] public float? Duration { get; set; }
    [JsonPropertyName("usage")] public WireUsage? Usage { get; set; }
}

// ---- models catalog ------------------------------------------------------

internal sealed class WireArchitecture
{
    [JsonPropertyName("input_modalities")] public List<string>? InputModalities { get; set; }
    [JsonPropertyName("output_modalities")] public List<string>? OutputModalities { get; set; }
    [JsonPropertyName("modality")] public string? Modality { get; set; }
    [JsonPropertyName("tokenizer")] public string Tokenizer { get; set; } = "";
    [JsonPropertyName("instruct_type")] public string? InstructType { get; set; }
}

internal sealed class WirePricing
{
    [JsonPropertyName("prompt")] public string Prompt { get; set; } = "";
    [JsonPropertyName("completion")] public string Completion { get; set; } = "";
}

internal sealed class WireTopProvider
{
    [JsonPropertyName("context_length")] public uint? ContextLength { get; set; }
    [JsonPropertyName("max_completion_tokens")] public uint? MaxCompletionTokens { get; set; }
    [JsonPropertyName("is_moderated")] public bool IsModerated { get; set; }
    [JsonPropertyName("max_thinking_tokens")] public uint? MaxThinkingTokens { get; set; }
}

internal sealed class WireModelEndpoint
{
    [JsonPropertyName("provider")] public string Provider { get; set; } = "";
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("down")] public bool Down { get; set; }
    [JsonPropertyName("source")] public string Source { get; set; } = "";
}

internal sealed class WireModelEntry
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("canonical_slug")] public string CanonicalSlug { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("created")] public long Created { get; set; }
    [JsonPropertyName("description")] public string Description { get; set; } = "";
    [JsonPropertyName("context_length")] public uint? ContextLength { get; set; }
    [JsonPropertyName("architecture")] public WireArchitecture? Architecture { get; set; }
    [JsonPropertyName("pricing")] public WirePricing? Pricing { get; set; }
    [JsonPropertyName("top_provider")] public WireTopProvider? TopProvider { get; set; }
    [JsonPropertyName("supported_parameters")] public List<string>? SupportedParameters { get; set; }
    [JsonPropertyName("unsupported_parameters")] public List<string>? UnsupportedParameters { get; set; }
    [JsonPropertyName("default_parameters")] public JsonNode? DefaultParameters { get; set; }
    [JsonPropertyName("endpoints")] public List<WireModelEndpoint>? Endpoints { get; set; }
}

internal sealed class WireListModelsResponse
{
    [JsonPropertyName("data")] public List<WireModelEntry> Data { get; set; } = [];
}

// ---- batches -------------------------------------------------------------

internal sealed class WireBatchRequestItem
{
    [JsonPropertyName("custom_id")] public string CustomId { get; set; } = "";

    // The ChatRequest body, already encoded (with extra merged) and spliced verbatim.
    [JsonPropertyName("body")] public JsonNode? Body { get; set; }
}

internal sealed class WireBatchCreateRequest
{
    [JsonPropertyName("requests")] public List<WireBatchRequestItem> Requests { get; set; } = [];
}

internal sealed class WireBatchCounts
{
    [JsonPropertyName("total")] public ulong Total { get; set; }
    [JsonPropertyName("processing")] public ulong Processing { get; set; }
    [JsonPropertyName("succeeded")] public ulong Succeeded { get; set; }
    [JsonPropertyName("errored")] public ulong Errored { get; set; }
    [JsonPropertyName("canceled")] public ulong Canceled { get; set; }
    [JsonPropertyName("expired")] public ulong Expired { get; set; }
}

internal sealed class WireBatchHandle
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("status")] public string? Status { get; set; }
    [JsonPropertyName("counts")] public WireBatchCounts? Counts { get; set; }
    [JsonPropertyName("created_at")] public long? CreatedAt { get; set; }
    [JsonPropertyName("expires_at")] public long? ExpiresAt { get; set; }
    [JsonPropertyName("ended_at")] public long? EndedAt { get; set; }
    [JsonPropertyName("endpoint")] public string? Endpoint { get; set; }
}

internal sealed class WireBatchResponse
{
    [JsonPropertyName("status_code")] public uint StatusCode { get; set; }
    [JsonPropertyName("body")] public WireChatResponse Body { get; set; } = new();
}

internal sealed class WireBatchError
{
    [JsonPropertyName("code")] public string Code { get; set; } = "";
    [JsonPropertyName("message")] public string Message { get; set; } = "";
}

internal sealed class WireBatchResultLine
{
    [JsonPropertyName("custom_id")] public string CustomId { get; set; } = "";
    [JsonPropertyName("response")] public WireBatchResponse? Response { get; set; }
    [JsonPropertyName("error")] public WireBatchError? Error { get; set; }
}
