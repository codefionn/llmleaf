//! Generated typed model for the llmleaf client SDK — the Zig mirror of
//! `clients/proto/llmleaf/v1/llmleaf.proto` (package `llmleaf.v1`).
//!
//! THE WIRE IS JSON, NOT PROTOBUF-BINARY. These are the canonical request /
//! response shapes the SDK exposes; `src/wire.zig` serialises them to / parses
//! them from the OpenAI/OpenRouter-shaped JSON the llmleaf core speaks.
//!
//! Codegen story (see README + scripts/gen.sh): there is no official protoc
//! plugin for Zig, so this file is the AUTHORITATIVE hand-maintained mirror.
//! `scripts/gen.sh` documents how a contributor can regenerate it via the
//! third-party `protoc-gen-zig` plugin if they vendor one; the committed file
//! is what the build uses.
//!
//! Two conventions make the proto<->JSON mapping mechanical and identical to
//! every other client:
//!   1. Field names are snake_case and already match the JSON keys on the wire.
//!   2. Closed-set enums map to their wire token by LOWERCASING the value name
//!      (`TOOL_CALLS` <-> "tool_calls"). The `*_UNSPECIFIED` zero value means
//!      "field absent on the wire" and is modelled as `null` (an optional enum)
//!      everywhere it appears.
//!
//! Free-form JSON fields (`ChatRequest.extra`, `FunctionDef.parameters`,
//! `ResponseFormat.json_schema`, `EmbeddingRequest.extra`, `SpeechRequest.extra`,
//! `ModelEntry.default_parameters`) are carried as raw-JSON `[]const u8` strings
//! and spliced verbatim into / captured back from the body.
//!
//! Ownership: the SDK never frees a value the *caller* built. Values the SDK
//! *returns* (chat responses, model lists, chunks, ...) own heap memory backed
//! by an arena; free them with their accompanying `deinit`/`Parsed.deinit`.
//! See `src/root.zig` and the per-call docs.

const std = @import("std");

// ---------------------------------------------------------------------------
// Common
// ---------------------------------------------------------------------------

/// Breakdown of `Usage.prompt_tokens`. Today only the cache-read (hit) share is surfaced —
/// the count of prompt tokens served from the provider's cache rather than processed fresh.
pub const PromptTokensDetails = struct {
    cached_tokens: ?u32 = null,
};

/// Token accounting echoed on every response. `cost_usd` is an llmleaf addition.
pub const Usage = struct {
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    total_tokens: u32 = 0,
    cost_usd: ?f64 = null,
    /// Prompt-cache hit accounting (OpenAI `usage.prompt_tokens_details`). Absent when the
    /// upstream reported no caching; `cachedTokens` flattens it to a plain count.
    prompt_tokens_details: ?PromptTokensDetails = null,
    /// Input tokens written to the provider's prompt cache this request — a cache *write*
    /// (creation). An llmleaf extension (Anthropic reports it; OpenAI/OpenRouter do not);
    /// absent when there were none.
    cache_creation_tokens: ?u32 = null,

    /// Prompt tokens served from the provider's cache this request — a cache *read* (hit).
    /// `0` when the upstream reported no caching.
    pub fn cachedTokens(self: Usage) u32 {
        const d = self.prompt_tokens_details orelse return 0;
        return d.cached_tokens orelse 0;
    }

    /// Input tokens written to the provider's cache this request — a cache *write* (creation).
    /// `0` when there were none (or the provider does not report writes).
    pub fn cacheWrites(self: Usage) u32 {
        return self.cache_creation_tokens orelse 0;
    }
};

/// Canonical error envelope:  {"error":{"message":...}}.
pub const ErrorBody = struct {
    message: []const u8,
    type: ?[]const u8 = null,
    code: ?[]const u8 = null,
};

// ---------------------------------------------------------------------------
// Enums  (wire token = lowercase(value name); UNSPECIFIED <-> absent/null)
// ---------------------------------------------------------------------------

pub const Role = enum {
    system,
    user,
    assistant,
    tool,
};

pub const FinishReason = enum {
    stop,
    length,
    tool_calls,
    content_filter,
};

pub const BatchStatus = enum {
    validating,
    in_progress,
    finalizing,
    completed,
    failed,
    expired,
    canceling,
    canceled,
};

// ---------------------------------------------------------------------------
// Chat completions  (POST /v1/chat/completions)
// ---------------------------------------------------------------------------

pub const TextPart = struct {
    text: []const u8,
};

pub const ImageUrlPart = struct {
    url: []const u8,
    detail: ?[]const u8 = null, // "auto" | "low" | "high"
};

/// One content part of a multimodal message.
pub const ContentPart = union(enum) {
    text: TextPart,
    image_url: ImageUrlPart,
};

/// Wire `content`: a plain string, or an array of content parts.
pub const Content = union(enum) {
    text: []const u8,
    parts: []const ContentPart,
};

pub const FunctionCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON-encoded string (OpenAI shape)
};

pub const ToolCall = struct {
    id: []const u8,
    type: []const u8 = "function",
    function: FunctionCall,
};

pub const FunctionCallDelta = struct {
    name: ?[]const u8 = null,
    arguments: ?[]const u8 = null,
};

pub const ToolCallDelta = struct {
    index: u32 = 0,
    id: ?[]const u8 = null,
    type: ?[]const u8 = null,
    function: ?FunctionCallDelta = null,
};

/// One structured reasoning ("thinking") block (OpenRouter `reasoning_details[]`). It
/// expresses both *open* reasoning — visible text, optionally signed — and *hidden*
/// reasoning — an encrypted/redacted blob the provider returns in place of the text.
/// `type` is the wire discriminator:
///   "reasoning.text"      -> text (+ optional signature)  — OPEN  (visible)
///   "reasoning.summary"   -> summary                       — OPEN  (a summarised view)
///   "reasoning.encrypted" -> data                          — HIDDEN (redacted / opaque)
/// `signature` and `data` are opaque and MUST be echoed back verbatim in the next request's
/// `reasoning_details` to continue a signed/encrypted reasoning turn. Use `isHidden` /
/// `openText` to branch without matching on the raw `type` string.
pub const ReasoningDetail = struct {
    type: []const u8,
    text: ?[]const u8 = null, // "reasoning.text"
    summary: ?[]const u8 = null, // "reasoning.summary"
    data: ?[]const u8 = null, // "reasoning.encrypted" (hidden)
    signature: ?[]const u8 = null, // opaque, replayed verbatim
    id: ?[]const u8 = null,
    format: ?[]const u8 = null, // e.g. "anthropic-claude-v1"
    index: ?u32 = null,

    /// Whether this block is hidden (redacted / encrypted) rather than open visible reasoning.
    pub fn isHidden(self: ReasoningDetail) bool {
        return std.mem.eql(u8, self.type, "reasoning.encrypted") or
            (self.data != null and self.text == null);
    }

    /// The visible reasoning text of an open block — its `text`, falling back to its
    /// `summary`. `null` for a hidden block.
    pub fn openText(self: ReasoningDetail) ?[]const u8 {
        return self.text orelse self.summary;
    }
};

pub const ChatMessage = struct {
    role: Role,
    content: ?Content = null,
    name: ?[]const u8 = null,
    tool_calls: []const ToolCall = &.{},
    tool_call_id: ?[]const u8 = null, // set when role == .tool
    /// Open reasoning text the assistant emitted (OpenRouter `reasoning`), if any. The flat,
    /// human-readable form; the structured `reasoning_details` is the replay-safe one.
    reasoning: ?[]const u8 = null,
    /// Structured reasoning blocks (open and hidden, with signatures — see `ReasoningDetail`).
    /// Echo these back verbatim on the next request to preserve signed reasoning across a turn.
    reasoning_details: []const ReasoningDetail = &.{},

    /// Convenience constructor for the common plain-text message.
    pub fn textMsg(role: Role, text: []const u8) ChatMessage {
        return .{ .role = role, .content = .{ .text = text } };
    }
};

/// A function the model MAY call. `parameters` is a raw JSON Schema string.
pub const FunctionDef = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    parameters: ?[]const u8 = null, // raw JSON object
};

pub const ToolDef = struct {
    type: []const u8 = "function",
    function: FunctionDef,
};

pub const FunctionName = struct {
    name: []const u8,
};

pub const NamedToolChoice = struct {
    type: []const u8 = "function",
    function: FunctionName,
};

/// tool_choice: a mode string ("auto"|"none"|"required") or a named object.
pub const ToolChoice = union(enum) {
    mode: []const u8,
    named: NamedToolChoice,
};

/// response_format: {"type": "text"|"json_object"|"json_schema", ...}.
pub const ResponseFormat = struct {
    type: []const u8,
    json_schema: ?[]const u8 = null, // raw JSON object when type == "json_schema"
};

pub const ChatRequest = struct {
    model: []const u8,
    messages: []const ChatMessage,
    stream: ?bool = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    max_tokens: ?u32 = null, // legacy name
    max_completion_tokens: ?u32 = null, // modern name (takes precedence)
    stop: []const []const u8 = &.{}, // wire accepts string or array
    n: ?u32 = null,
    seed: ?i64 = null,
    frequency_penalty: ?f32 = null,
    presence_penalty: ?f32 = null,
    tools: []const ToolDef = &.{},
    tool_choice: ?ToolChoice = null,
    response_format: ?ResponseFormat = null,
    reasoning_effort: ?[]const u8 = null, // "low" | "medium" | "high"
    extra: ?[]const u8 = null, // raw JSON object, merged at the top level
};

pub const Choice = struct {
    index: u32 = 0,
    message: ChatMessage,
    finish_reason: ?FinishReason = null,
};

pub const ChatResponse = struct {
    id: []const u8,
    object: []const u8, // "chat.completion"
    created: i64 = 0,
    model: []const u8,
    choices: []const Choice,
    usage: ?Usage = null,
};

// Streaming SSE frame: `data: <ChatCompletionChunk JSON>`, ended by `data: [DONE]`.

pub const Delta = struct {
    role: ?Role = null, // first chunk only
    content: ?[]const u8 = null, // incremental text
    tool_calls: []const ToolCallDelta = &.{},
    /// Incremental open reasoning text, if any.
    reasoning: ?[]const u8 = null,
    /// Incremental structured reasoning blocks (open / hidden — see `ReasoningDetail`).
    reasoning_details: []const ReasoningDetail = &.{},
};

pub const ChunkChoice = struct {
    index: u32 = 0,
    delta: Delta = .{},
    finish_reason: ?FinishReason = null,
};

pub const ChatCompletionChunk = struct {
    id: []const u8,
    object: []const u8, // "chat.completion.chunk"
    created: i64 = 0,
    model: []const u8,
    choices: []const ChunkChoice,
    usage: ?Usage = null, // terminal chunk only
};

// ---------------------------------------------------------------------------
// Embeddings  (POST /v1/embeddings)
// ---------------------------------------------------------------------------

pub const EmbeddingRequest = struct {
    model: []const u8,
    input: []const []const u8, // wire accepts string or array of strings
    dimensions: ?u32 = null,
    encoding_format: ?[]const u8 = null, // "float" | "base64"
    extra: ?[]const u8 = null, // raw JSON object passthrough
};

pub const Embedding = struct {
    object: []const u8 = "embedding",
    index: u32 = 0,
    /// Always the decoded float vector — even when the wire used base64, the
    /// transport decodes it before returning.
    embedding: []const f32,
};

pub const EmbeddingResponse = struct {
    object: []const u8, // "list"
    data: []const Embedding,
    model: []const u8,
    usage: ?Usage = null,
};

// ---------------------------------------------------------------------------
// Audio — text to speech / voices
// ---------------------------------------------------------------------------

pub const SpeechRequest = struct {
    model: []const u8,
    input: []const u8,
    voice: []const u8,
    response_format: ?[]const u8 = null, // mp3|opus|aac|flac|wav|pcm
    speed: ?f32 = null,
    extra: ?[]const u8 = null, // raw JSON object passthrough
};

/// What `speech` returns: raw audio bytes + the Content-Type the server set.
pub const Speech = struct {
    bytes: []const u8,
    content_type: []const u8,
};

pub const Voice = struct {
    id: []const u8,
    name: ?[]const u8 = null,
    languages: []const []const u8 = &.{},
};

pub const VoicesResponse = struct {
    model: []const u8,
    voices: []const Voice,
};

// ---------------------------------------------------------------------------
// Audio — speech to text  (multipart/form-data)
// ---------------------------------------------------------------------------

/// The audio bytes are passed separately as the multipart `file` part; these
/// are the accompanying form fields.
pub const TranscriptionRequest = struct {
    model: []const u8,
    language: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    response_format: ?[]const u8 = null, // json|text|verbose_json|srt|vtt
    temperature: ?f32 = null,
};

pub const TranscriptionResponse = struct {
    text: []const u8,
    task: ?[]const u8 = null,
    language: ?[]const u8 = null,
    duration: ?f32 = null,
    usage: ?Usage = null,
};

// ---------------------------------------------------------------------------
// Model catalog  (GET /v1/models)  — OpenRouter-shaped
// ---------------------------------------------------------------------------

pub const Architecture = struct {
    input_modalities: []const []const u8 = &.{},
    output_modalities: []const []const u8 = &.{},
    modality: ?[]const u8 = null,
    tokenizer: []const u8 = "",
    instruct_type: ?[]const u8 = null,
};

pub const Pricing = struct {
    prompt: []const u8 = "",
    completion: []const u8 = "",
};

pub const TopProvider = struct {
    context_length: ?u32 = null,
    max_completion_tokens: ?u32 = null,
    is_moderated: bool = false,
    max_thinking_tokens: ?u32 = null, // llmleaf extension
};

/// Admin-only fallback-chain entry (present only with a valid x-admin-token).
pub const ModelEndpoint = struct {
    provider: []const u8,
    model: []const u8,
    down: bool = false,
    source: []const u8, // "route" | "prefix"
};

pub const ModelEntry = struct {
    id: []const u8,
    canonical_slug: []const u8 = "",
    name: []const u8 = "",
    created: i64 = 0,
    description: []const u8 = "",
    context_length: ?u32 = null,
    architecture: ?Architecture = null,
    pricing: ?Pricing = null,
    top_provider: ?TopProvider = null,
    supported_parameters: []const []const u8 = &.{},
    unsupported_parameters: []const []const u8 = &.{},
    default_parameters: ?[]const u8 = null, // raw JSON object
    endpoints: []const ModelEndpoint = &.{}, // admin-only
};

pub const ListModelsResponse = struct {
    data: []const ModelEntry,
};

pub const ModelType = enum {
    all,
    llm,
    tts,
    stt,
    embedding,
};

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

pub const BatchRequestItem = struct {
    custom_id: []const u8,
    body: ChatRequest,
};

pub const BatchCreateRequest = struct {
    requests: []const BatchRequestItem,
};

pub const BatchCounts = struct {
    total: u64 = 0,
    processing: u64 = 0,
    succeeded: u64 = 0,
    errored: u64 = 0,
    canceled: u64 = 0,
    expired: u64 = 0,
};

pub const BatchHandle = struct {
    id: []const u8,
    status: ?BatchStatus = null,
    counts: BatchCounts = .{},
    created_at: ?i64 = null,
    expires_at: ?i64 = null,
    ended_at: ?i64 = null,
    endpoint: ?[]const u8 = null,
};

pub const BatchResponse = struct {
    status_code: u32 = 0,
    body: ChatResponse,
};

pub const BatchError = struct {
    code: []const u8,
    message: []const u8,
};

/// One line of the JSONL results stream. The wire key is `error`; in Zig the
/// field is `@"error"` (a keyword) and serialises to/from `"error"` on the wire.
pub const BatchResultLine = struct {
    custom_id: []const u8,
    response: ?BatchResponse = null,
    @"error": ?BatchError = null,
};

// ---------------------------------------------------------------------------
// Mechanical enum <-> wire token helpers (SPEC.md: lowercase the value name).
// Because these enums already use lowercase Zig identifiers that match the wire
// token exactly, the mapping is `@tagName` one way and a name-match the other.
// Keep these as the single pair every (de)serialiser reuses — do not hand-map.
// ---------------------------------------------------------------------------

/// enum -> wire token. The `*_UNSPECIFIED` zero value is modelled as `null` by
/// callers, so this only ever receives a concrete tag.
pub fn enumToWire(value: anytype) []const u8 {
    return @tagName(value);
}

/// wire token -> enum, or `null` for an absent/unknown token (UNSPECIFIED).
pub fn enumFromWire(comptime E: type, token: []const u8) ?E {
    inline for (@typeInfo(E).@"enum".fields) |f| {
        if (std.mem.eql(u8, f.name, token)) return @field(E, f.name);
    }
    return null;
}

test "enum wire roundtrip" {
    try std.testing.expectEqualStrings("tool_calls", enumToWire(FinishReason.tool_calls));
    try std.testing.expectEqualStrings("assistant", enumToWire(Role.assistant));
    try std.testing.expectEqualStrings("in_progress", enumToWire(BatchStatus.in_progress));
    try std.testing.expectEqual(@as(?Role, .assistant), enumFromWire(Role, "assistant"));
    try std.testing.expectEqual(@as(?Role, null), enumFromWire(Role, "nope"));
    try std.testing.expectEqual(@as(?FinishReason, .content_filter), enumFromWire(FinishReason, "content_filter"));
}
