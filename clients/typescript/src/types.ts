// Public, hand-written TypeScript surface mirroring proto/llmleaf/v1/llmleaf.proto.
//
// Why interfaces and not the generated protobuf-es classes directly? protobuf-es's
// own JSON codec emits camelCase keys and STRING enum NAMES (e.g. "ASSISTANT"), which
// do NOT match the OpenAI/OpenRouter wire (snake_case keys + lowercase tokens like
// "assistant"), and it has no notion of the "free-form JSON carried as a raw string"
// or "content is string-or-array" conventions in SPEC.md. So these plain interfaces
// are the ergonomic public model; src/wire.ts maps them to/from the actual wire JSON.
//
// The generated descriptors/enums remain the committed codegen artifact (src/gen) and
// are the single source of truth these shapes track. The three closed-set enums are
// re-used straight from the generated file via src/enums.ts.

import type { Role, FinishReason, BatchStatus } from "./enums.js";

export type { Role, FinishReason, BatchStatus };

// ---------------------------------------------------------------------------
// Common
// ---------------------------------------------------------------------------

export interface Usage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  /** llmleaf addition; absent when the model has no known price. */
  costUsd?: number;
  /**
   * Prompt-cache hit accounting (OpenAI `usage.prompt_tokens_details`). Absent when the
   * upstream reported no caching; {@link cachedTokens} flattens it to a plain count.
   */
  promptTokensDetails?: PromptTokensDetails;
  /**
   * Input tokens written to the provider's prompt cache this request — a cache *write*
   * (creation). An llmleaf extension (Anthropic reports it; OpenAI/OpenRouter do not);
   * absent when there were none.
   */
  cacheCreationTokens?: number;
}

/**
 * Breakdown of {@link Usage.promptTokens}. Today only the cache-read (hit) share is
 * surfaced — the count of prompt tokens served from the provider's cache rather than
 * processed fresh.
 */
export interface PromptTokensDetails {
  cachedTokens?: number;
}

/**
 * Prompt tokens served from the provider's cache this request — a cache *read* (hit).
 * `0` when the upstream reported no caching.
 */
export function cachedTokens(usage: Usage): number {
  return usage.promptTokensDetails?.cachedTokens ?? 0;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface TextPart {
  type: "text";
  text: string;
}

export interface ImageUrlPart {
  type: "image_url";
  imageUrl: {
    url: string;
    /** "auto" | "low" | "high" */
    detail?: string;
  };
}

export type ContentPart = TextPart | ImageUrlPart;

/** Wire `content` is either a plain string or an array of content parts. */
export type MessageContent = string | ContentPart[];

export interface FunctionCall {
  name: string;
  /** JSON-encoded arguments string (OpenAI shape). */
  arguments: string;
}

export interface ToolCall {
  id: string;
  /** always "function" today */
  type: string;
  function: FunctionCall;
}

/**
 * One structured reasoning ("thinking") block (OpenRouter `reasoning_details[]`). It
 * expresses both *open* reasoning — visible text, optionally signed — and *hidden*
 * reasoning — an encrypted/redacted blob the provider returns in place of the text.
 * `type` is the wire discriminator and selects which field is set:
 *
 * - `"reasoning.text"` → {@link text} (+ optional {@link signature}) — **open** (visible)
 * - `"reasoning.summary"` → {@link summary} — **open** (a summarised view)
 * - `"reasoning.encrypted"` → {@link data} — **hidden** (redacted / opaque)
 *
 * {@link signature} and {@link data} are opaque to the client and MUST be echoed back
 * verbatim in the next request's `reasoning_details` to continue a signed/encrypted
 * reasoning turn (the upstream rejects an altered or dropped block — e.g. before a tool
 * call). {@link format} tags the provider encoding when known.
 */
export interface ReasoningDetail {
  type: string;
  /** set for `"reasoning.text"` */
  text?: string;
  /** set for `"reasoning.summary"` */
  summary?: string;
  /** set for `"reasoning.encrypted"` (hidden) */
  data?: string;
  /** opaque, replayed verbatim */
  signature?: string;
  id?: string;
  /** e.g. "anthropic-claude-v1" */
  format?: string;
  index?: number;
}

export interface ChatMessage {
  role: Role;
  content?: MessageContent;
  name?: string;
  toolCalls?: ToolCall[];
  /** set when role == TOOL */
  toolCallId?: string;
  /**
   * Open reasoning text the assistant emitted (OpenRouter `reasoning`), if any. The flat,
   * human-readable form; {@link reasoningDetails} is the structured, replay-safe one.
   */
  reasoning?: string;
  /**
   * Structured reasoning blocks (open and hidden, with signatures — see
   * {@link ReasoningDetail}). Echo these back verbatim on the next request to preserve
   * signed reasoning across a turn.
   */
  reasoningDetails?: ReasoningDetail[];
}

export interface FunctionDef {
  name: string;
  description?: string;
  /** raw JSON Schema object, as a JSON string. */
  parameters?: string;
}

export interface ToolDef {
  /** "function" */
  type: string;
  function: FunctionDef;
}

/** "auto" | "none" | "required", or a named-function object. */
export type ToolChoice =
  | string
  | {
      type: "function";
      function: { name: string };
    };

export interface ResponseFormat {
  /** "text" | "json_object" | "json_schema" */
  type: string;
  /** raw JSON object as a JSON string when type == "json_schema". */
  jsonSchema?: string;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  topP?: number;
  /** legacy name */
  maxTokens?: number;
  /** modern name (takes precedence) */
  maxCompletionTokens?: number;
  stop?: string[];
  n?: number;
  seed?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  tools?: ToolDef[];
  toolChoice?: ToolChoice;
  responseFormat?: ResponseFormat;
  /** "low" | "medium" | "high" */
  reasoningEffort?: string;
  /** dialect-specific passthrough, raw JSON object as a JSON string, merged at the top level. */
  extra?: string;
}

export interface Choice {
  index: number;
  message: ChatMessage;
  finishReason?: FinishReason;
}

export interface ChatResponse {
  id: string;
  /** "chat.completion" */
  object: string;
  /** unix seconds */
  created: number;
  model: string;
  choices: Choice[];
  usage?: Usage;
}

// Streaming

export interface FunctionCallDelta {
  name?: string;
  arguments?: string;
}

export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: string;
  function?: FunctionCallDelta;
}

export interface Delta {
  /** first chunk only */
  role?: Role;
  /** incremental text */
  content?: string;
  toolCalls?: ToolCallDelta[];
  /** incremental open reasoning text, if any */
  reasoning?: string;
  /** incremental structured reasoning blocks (open / hidden — see {@link ReasoningDetail}). */
  reasoningDetails?: ReasoningDetail[];
}

export interface ChunkChoice {
  index: number;
  delta: Delta;
  finishReason?: FinishReason;
}

export interface ChatCompletionChunk {
  id: string;
  /** "chat.completion.chunk" */
  object: string;
  created: number;
  model: string;
  choices: ChunkChoice[];
  /** terminal chunk only */
  usage?: Usage;
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

export interface EmbeddingRequest {
  model: string;
  /** wire accepts string or array of strings. */
  input: string[];
  dimensions?: number;
  /** "float" | "base64" */
  encodingFormat?: string;
  /** raw JSON object passthrough as a JSON string. */
  extra?: string;
}

export interface Embedding {
  /** "embedding" */
  object: string;
  index: number;
  /** Always decoded to floats, even when encoding_format == "base64". */
  embedding: number[];
}

export interface EmbeddingResponse {
  /** "list" */
  object: string;
  data: Embedding[];
  model: string;
  usage?: Usage;
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

export interface SpeechRequest {
  model: string;
  input: string;
  voice: string;
  /** mp3|opus|aac|flac|wav|pcm */
  responseFormat?: string;
  speed?: number;
  /** raw JSON object passthrough as a JSON string. */
  extra?: string;
}

/** Raw audio bytes plus the Content-Type the server reported. */
export interface SpeechResult {
  bytes: Uint8Array;
  contentType: string;
}

export interface Voice {
  /** value to put in SpeechRequest.voice */
  id: string;
  name?: string;
  /** BCP-47 tags */
  languages: string[];
}

export interface VoicesResponse {
  model: string;
  voices: Voice[];
}

export interface TranscriptionRequest {
  model: string;
  /** ISO-639-1 hint */
  language?: string;
  /** decoding bias */
  prompt?: string;
  /** json|text|verbose_json|srt|vtt */
  responseFormat?: string;
  temperature?: number;
}

export interface TranscriptionResponse {
  text: string;
  /** "transcribe" (verbose_json) */
  task?: string;
  language?: string;
  duration?: number;
  usage?: Usage;
}

// ---------------------------------------------------------------------------
// Model catalog
// ---------------------------------------------------------------------------

export interface Architecture {
  inputModalities: string[];
  outputModalities: string[];
  /** "text->text" | "text->audio" | ... */
  modality?: string;
  tokenizer: string;
  instructType?: string;
}

export interface Pricing {
  /** USD per token, decimal string */
  prompt: string;
  completion: string;
}

export interface TopProvider {
  contextLength?: number;
  maxCompletionTokens?: number;
  isModerated: boolean;
  /** llmleaf extension */
  maxThinkingTokens?: number;
}

/** Admin-only fallback-chain entry (present only with a valid admin token). */
export interface ModelEndpoint {
  provider: string;
  model: string;
  down: boolean;
  /** "route" | "prefix" */
  source: string;
}

export interface ModelEntry {
  id: string;
  canonicalSlug: string;
  name: string;
  created: number;
  description: string;
  contextLength?: number;
  architecture?: Architecture;
  pricing?: Pricing;
  topProvider?: TopProvider;
  supportedParameters: string[];
  unsupportedParameters: string[];
  /** raw JSON object as a JSON string. */
  defaultParameters?: string;
  /** admin-only */
  endpoints: ModelEndpoint[];
}

export interface ListModelsResponse {
  data: ModelEntry[];
}

/** Filter for {@link "./client".LlmleafClient.listModels}. */
export type ModelType = "all" | "llm" | "tts" | "stt" | "embedding";

export interface ListModelsOptions {
  type?: ModelType;
  /** substring search */
  search?: string;
  /** when true, send the admin token so per-model `endpoints` are included. */
  admin?: boolean;
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

export interface BatchRequestItem {
  customId: string;
  body: ChatRequest;
}

export interface BatchCreateRequest {
  requests: BatchRequestItem[];
}

export interface BatchCounts {
  total: number;
  processing: number;
  succeeded: number;
  errored: number;
  canceled: number;
  expired: number;
}

export interface BatchHandle {
  id: string;
  status: BatchStatus;
  counts?: BatchCounts;
  createdAt?: number;
  expiresAt?: number;
  endedAt?: number;
  /** e.g. "/v1/chat/completions" */
  endpoint?: string;
}

export interface BatchResponse {
  statusCode: number;
  body: ChatResponse;
}

export interface BatchError {
  code: string;
  message: string;
}

export interface BatchResultLine {
  customId: string;
  response?: BatchResponse;
  error?: BatchError;
}
