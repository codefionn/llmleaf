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

/**
 * The canonical error envelope body (`{"error":{"message":...}}`). Any non-2xx HTTP
 * response is raised as an {@link "./error".ApiError}; this typed shape is the `error`
 * carried *inside* a `failed` {@link ResponsesResponse} snapshot.
 */
export interface ErrorBody {
  message: string;
  /** present on some dialects; absent on the llmleaf core envelope. */
  type?: string;
  code?: string;
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

// ---------------------------------------------------------------------------
// Responses (POST /v1/responses) — the OpenAI Responses dialect
// ---------------------------------------------------------------------------
//
// llmleaf serves this dialect statelessly: `store` is accepted but always answered
// `false`, `previousResponseId`/`background:true` are rejected (400), and there is no
// retrieval call (`GET /v1/responses/{id}` is an explained 404). Dialect vocabulary
// that would collide with the chat enums (statuses "completed"/"in_progress"/…, roles
// incl. "developer") stays a plain wire string here rather than an enum.

/** `{"type":"input_text","text":...}` */
export interface ResponseInputTextPart {
  type: "input_text";
  text: string;
}

/**
 * `{"type":"input_image","image_url":"<url>","detail":...}`. Unlike the chat dialect's
 * nested `{url}` object, {@link imageUrl} is a plain string here.
 */
export interface ResponseInputImagePart {
  type: "input_image";
  imageUrl: string;
  /** "auto" | "low" | "high" */
  detail?: string;
}

/** `{"type":"output_text","text":...,"annotations":[]}` (annotations emitted as `[]`). */
export interface ResponseOutputTextPart {
  type: "output_text";
  text: string;
}

export type ResponseContentPart =
  | ResponseInputTextPart
  | ResponseInputImagePart
  | ResponseOutputTextPart;

/** A message item's `content`: a bare string or an array of content parts. */
export type ResponseMessageContent = string | ResponseContentPart[];

/**
 * A conversation message item. On input, {@link role} is "user" | "system" |
 * "developer" | "assistant" and serialises as a bare role-keyed object (no `"type"`);
 * on output, role is "assistant" with `output_text` parts and {@link status}/{@link id}.
 */
export interface ResponseMessageItem {
  type: "message";
  id?: string;
  /** "user" | "system" | "developer" | "assistant" */
  role: string;
  content?: ResponseMessageContent;
  /** output only: "in_progress" | "completed" */
  status?: string;
}

/**
 * A function call the model made. {@link callId} pairs it with its
 * {@link ResponseFunctionCallOutputItem}; {@link arguments} is the raw JSON string.
 */
export interface ResponseFunctionCallItem {
  type: "function_call";
  id?: string;
  callId: string;
  name: string;
  arguments: string;
  status?: string;
}

/** The caller's answer to a function call, replayed on the next turn. */
export interface ResponseFunctionCallOutputItem {
  type: "function_call_output";
  id?: string;
  callId: string;
  output: string;
}

/** One entry of a reasoning item's `summary[]` or `content[]` list. */
export interface ResponseReasoningText {
  text: string;
}

/**
 * A reasoning ("thinking") item. {@link summary} entries serialise as
 * `{"type":"summary_text","text"}`, {@link content} entries as
 * `{"type":"reasoning_text","text"}` — the list decides the wire token.
 * {@link encryptedContent} is opaque and MUST be echoed back verbatim in the next
 * request's input to continue an encrypted reasoning turn.
 */
export interface ResponseReasoningItem {
  type: "reasoning";
  id?: string;
  summary?: ResponseReasoningText[];
  content?: ResponseReasoningText[];
  encryptedContent?: string;
}

/**
 * One item of the request `input` array or the response `output` array. The `type`
 * field is the discriminator (a message item serialises without one — see
 * {@link ResponseMessageItem}).
 */
export type ResponseItem =
  | ResponseMessageItem
  | ResponseFunctionCallItem
  | ResponseFunctionCallOutputItem
  | ResponseReasoningItem;

/** `input`: a bare string (one user message) or an array of items. */
export type ResponsesInput = string | ResponseItem[];

/**
 * A tool the model MAY call — FLAT in this dialect (`type`/`name`/`parameters` at the
 * top level, no nested `function` object).
 */
export interface ResponsesToolDef {
  /** "function" */
  type: string;
  name: string;
  description?: string;
  /** raw JSON Schema object, as a JSON string. */
  parameters?: string;
  /** defaults TRUE upstream; llmleaf's own edge pins false. */
  strict?: boolean;
}

/**
 * "auto" | "none" | "required", or the FLAT named object `{type:"function",name}`
 * (no nested `function`, unlike the chat dialect).
 */
export type ResponsesToolChoice =
  | string
  | {
      type: "function";
      name: string;
    };

/** `reasoning`: `{"effort":...,"summary":...}`. */
export interface ResponsesReasoning {
  /** "minimal" | "low" | "medium" | "high" | ... */
  effort?: string;
  summary?: string;
}

export interface ResponsesRequest {
  model: string;
  input: ResponsesInput;
  /** becomes a leading system message */
  instructions?: string;
  stream?: boolean;
  temperature?: number;
  topP?: number;
  maxOutputTokens?: number;
  tools?: ResponsesToolDef[];
  toolChoice?: ResponsesToolChoice;
  reasoning?: ResponsesReasoning;
  /** accepted, but llmleaf stores nothing and always answers `false`. */
  store?: boolean;
  /** dialect-specific passthrough, raw JSON object as a JSON string, merged at the top level. */
  extra?: string;
}

export interface ResponsesInputTokensDetails {
  cachedTokens?: number;
}

export interface ResponsesOutputTokensDetails {
  reasoningTokens?: number;
}

/**
 * Token accounting in the Responses dialect's own names (`input_tokens`/`output_tokens`,
 * not the chat dialect's `prompt_tokens`/`completion_tokens`).
 */
export interface ResponsesUsage {
  inputTokens: number;
  inputTokensDetails?: ResponsesInputTokensDetails;
  outputTokens: number;
  outputTokensDetails?: ResponsesOutputTokensDetails;
  totalTokens: number;
}

/** `status:"incomplete"` refinement: "max_output_tokens" | "content_filter". */
export interface ResponsesIncompleteDetails {
  reason: string;
}

/**
 * The response object (`"object":"response"`), also the snapshot carried by the
 * `response.created` / `response.in_progress` / `response.completed` stream events.
 */
export interface ResponsesResponse {
  id: string;
  /** "response" */
  object: string;
  /** unix seconds */
  createdAt: number;
  /** "completed" | "in_progress" | "incomplete" | "failed" */
  status: string;
  incompleteDetails?: ResponsesIncompleteDetails;
  /** carried by a `failed` snapshot */
  error?: ErrorBody;
  model: string;
  output: ResponseItem[];
  /** null on in-flight snapshots */
  usage?: ResponsesUsage;
  /** llmleaf always answers `false`. */
  store?: boolean;
  instructions?: string;
  maxOutputTokens?: number;
  temperature?: number;
  topP?: number;
  reasoning?: ResponsesReasoning;
}

/**
 * One streaming SSE event. Unlike chat streaming there is NO `data: [DONE]` sentinel —
 * the stream ends after the terminal `response.completed` / `response.incomplete` /
 * `response.failed` event. This is a flat superset of every event's fields; {@link type}
 * says which are meaningful. {@link "./client".LlmleafClient.responsesStream} skips event
 * types it doesn't recognise (the dialect grows by adding types).
 */
export interface ResponsesStreamEvent {
  /** "response.created", "response.output_text.delta", "error", ... */
  type: string;
  sequenceNumber: number;
  /** response.created / in_progress / completed / incomplete / failed */
  response?: ResponsesResponse;
  outputIndex?: number;
  itemId?: string;
  contentIndex?: number;
  /** response.output_item.added / done */
  item?: ResponseItem;
  /** response.content_part.added / done */
  part?: ResponseContentPart;
  /** *.delta events (text / reasoning / arguments) */
  delta?: string;
  /** response.output_text.done / reasoning_text.done */
  text?: string;
  /** response.function_call_arguments.done */
  arguments?: string;
  /** "error" event */
  message?: string;
}
