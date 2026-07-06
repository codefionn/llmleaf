// Serialization layer: map the public TypeScript model (src/types.ts) to/from the
// OpenAI/OpenRouter-shaped JSON the llmleaf core speaks (SPEC.md). This is the only
// place that knows the wire key names (snake_case), the enum<->token mapping, the
// "content is string-or-array" rule, the "stop is bare-string-or-array" rule, and the
// "free-form JSON fields are raw JSON strings spliced verbatim" rule.

import {
  roleToWire,
  roleFromWire,
  finishReasonToWire,
  finishReasonFromWire,
  batchStatusFromWire,
} from "./enums.js";
import type {
  ChatRequest,
  ChatMessage,
  ContentPart,
  MessageContent,
  ReasoningDetail,
  ToolCall,
  ToolDef,
  ToolChoice,
  ResponseFormat,
  ChatResponse,
  Choice,
  ChatCompletionChunk,
  ChunkChoice,
  Delta,
  ToolCallDelta,
  Usage,
  PromptTokensDetails,
  EmbeddingRequest,
  EmbeddingResponse,
  Embedding,
  SpeechRequest,
  VoicesResponse,
  Voice,
  TranscriptionResponse,
  ListModelsResponse,
  ModelEntry,
  Architecture,
  Pricing,
  TopProvider,
  ModelEndpoint,
  BatchCreateRequest,
  BatchHandle,
  BatchCounts,
  BatchResultLine,
  BatchResponse,
  BatchError,
  ErrorBody,
  ResponsesRequest,
  ResponsesInput,
  ResponseItem,
  ResponseMessageItem,
  ResponseMessageContent,
  ResponseContentPart,
  ResponseFunctionCallItem,
  ResponseFunctionCallOutputItem,
  ResponseReasoningItem,
  ResponseReasoningText,
  ResponsesToolDef,
  ResponsesToolChoice,
  ResponsesReasoning,
  ResponsesResponse,
  ResponsesUsage,
  ResponsesInputTokensDetails,
  ResponsesOutputTokensDetails,
  ResponsesIncompleteDetails,
  ResponsesStreamEvent,
} from "./types.js";

// A plain JSON object on the wire.
type Json = Record<string, unknown>;

// ---------------------------------------------------------------------------
// Small read helpers (defensive: the server is authoritative, but be lenient).
// ---------------------------------------------------------------------------

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}
function num(v: unknown, fallback = 0): number {
  return typeof v === "number" ? v : fallback;
}
function optStr(v: unknown): string | undefined {
  return typeof v === "string" ? v : undefined;
}
function optNum(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}
function optBool(v: unknown): boolean | undefined {
  return typeof v === "boolean" ? v : undefined;
}
function arr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}
function obj(v: unknown): Json | undefined {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : undefined;
}

/** Parse a raw-JSON-string field; throws a clear error if the caller passed bad JSON. */
function parseRawJson(field: string, raw: string | undefined): unknown {
  if (raw === undefined) return undefined;
  try {
    return JSON.parse(raw);
  } catch (e) {
    throw new TypeError(`${field} must be a valid JSON string: ${(e as Error).message}`);
  }
}

/** Capture a wire sub-value back as a raw JSON string (free-form decode). */
function captureRawJson(v: unknown): string | undefined {
  return v === undefined ? undefined : JSON.stringify(v);
}

/** Set key only when value is defined (keeps request bodies minimal/byte-clean). */
function put(o: Json, key: string, value: unknown): void {
  if (value !== undefined) o[key] = value;
}

// ===========================================================================
// Chat — encode
// ===========================================================================

function encodeContent(content: MessageContent | undefined): unknown {
  if (content === undefined) return undefined;
  if (typeof content === "string") return content;
  return content.map(encodeContentPart);
}

function encodeContentPart(p: ContentPart): Json {
  if (p.type === "text") {
    return { type: "text", text: p.text };
  }
  const imageUrl: Json = { url: p.imageUrl.url };
  put(imageUrl, "detail", p.imageUrl.detail);
  return { type: "image_url", image_url: imageUrl };
}

function encodeToolCall(tc: ToolCall): Json {
  return {
    id: tc.id,
    type: tc.type,
    function: { name: tc.function.name, arguments: tc.function.arguments },
  };
}

/**
 * Encode one reasoning_details[] entry. `signature` and `data` are opaque and must
 * round-trip verbatim, so they are emitted exactly as given when present.
 */
function encodeReasoningDetail(d: ReasoningDetail): Json {
  const out: Json = { type: d.type };
  put(out, "text", d.text);
  put(out, "summary", d.summary);
  put(out, "data", d.data);
  put(out, "signature", d.signature);
  put(out, "id", d.id);
  put(out, "format", d.format);
  put(out, "index", d.index);
  return out;
}

function encodeMessage(m: ChatMessage): Json {
  const out: Json = { role: roleToWire(m.role) ?? "user" };
  const content = encodeContent(m.content);
  // content is required on the wire for most roles; emit it (possibly empty string)
  // whenever it was provided, including the empty string.
  if (content !== undefined) out["content"] = content;
  put(out, "name", m.name);
  if (m.toolCalls && m.toolCalls.length > 0) {
    out["tool_calls"] = m.toolCalls.map(encodeToolCall);
  }
  put(out, "tool_call_id", m.toolCallId);
  // Reasoning (OpenRouter): echo a prior assistant turn back verbatim to preserve
  // signed/encrypted reasoning across a turn (SPEC.md / ReasoningDetail).
  put(out, "reasoning", m.reasoning);
  if (m.reasoningDetails && m.reasoningDetails.length > 0) {
    out["reasoning_details"] = m.reasoningDetails.map(encodeReasoningDetail);
  }
  return out;
}

function encodeToolDef(t: ToolDef): Json {
  const fn: Json = { name: t.function.name };
  put(fn, "description", t.function.description);
  const params = parseRawJson("FunctionDef.parameters", t.function.parameters);
  put(fn, "parameters", params);
  return { type: t.type, function: fn };
}

function encodeToolChoice(tc: ToolChoice): unknown {
  if (typeof tc === "string") return tc;
  return { type: tc.type, function: { name: tc.function.name } };
}

function encodeResponseFormat(rf: ResponseFormat): Json {
  const out: Json = { type: rf.type };
  put(out, "json_schema", parseRawJson("ResponseFormat.json_schema", rf.jsonSchema));
  return out;
}

/** `stop`: bare string when single-element, else array (SPEC.md). */
function encodeStop(stop: string[] | undefined): unknown {
  if (!stop || stop.length === 0) return undefined;
  return stop.length === 1 ? stop[0] : stop;
}

/** Build the chat request body. `forceStream` overrides `stream` for the streaming call. */
export function encodeChatRequest(req: ChatRequest, forceStream?: boolean): Json {
  const out: Json = {
    model: req.model,
    messages: req.messages.map(encodeMessage),
  };
  const stream = forceStream !== undefined ? forceStream : req.stream;
  put(out, "stream", stream);
  put(out, "temperature", req.temperature);
  put(out, "top_p", req.topP);
  // Prefer max_completion_tokens; still send max_tokens if only that was set (SPEC.md).
  put(out, "max_completion_tokens", req.maxCompletionTokens);
  if (req.maxTokens !== undefined) put(out, "max_tokens", req.maxTokens);
  put(out, "stop", encodeStop(req.stop));
  put(out, "n", req.n);
  put(out, "seed", req.seed);
  put(out, "frequency_penalty", req.frequencyPenalty);
  put(out, "presence_penalty", req.presencePenalty);
  if (req.tools && req.tools.length > 0) out["tools"] = req.tools.map(encodeToolDef);
  put(out, "tool_choice", req.toolChoice && encodeToolChoice(req.toolChoice));
  put(out, "response_format", req.responseFormat && encodeResponseFormat(req.responseFormat));
  put(out, "reasoning_effort", req.reasoningEffort);

  // `extra`: parsed and merged at the top level of the request object (SPEC.md).
  const extra = parseRawJson("ChatRequest.extra", req.extra);
  if (extra !== undefined) {
    if (typeof extra !== "object" || extra === null || Array.isArray(extra)) {
      throw new TypeError("ChatRequest.extra must be a JSON object");
    }
    for (const [k, v] of Object.entries(extra as Json)) out[k] = v;
  }
  return out;
}

// ===========================================================================
// Chat — decode
// ===========================================================================

function decodePromptTokensDetails(v: unknown): PromptTokensDetails | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const details: PromptTokensDetails = {};
  const cached = optNum(o["cached_tokens"]);
  if (cached !== undefined) details.cachedTokens = cached;
  return details;
}

export function decodeUsage(v: unknown): Usage | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const usage: Usage = {
    promptTokens: num(o["prompt_tokens"]),
    completionTokens: num(o["completion_tokens"]),
    totalTokens: num(o["total_tokens"]),
  };
  const cost = optNum(o["cost_usd"]);
  if (cost !== undefined) usage.costUsd = cost;
  const ptd = decodePromptTokensDetails(o["prompt_tokens_details"]);
  if (ptd !== undefined) usage.promptTokensDetails = ptd;
  const cacheCreation = optNum(o["cache_creation_tokens"]);
  if (cacheCreation !== undefined) usage.cacheCreationTokens = cacheCreation;
  return usage;
}

function decodeContent(v: unknown): MessageContent | undefined {
  if (v === undefined || v === null) return undefined;
  if (typeof v === "string") return v;
  if (Array.isArray(v)) {
    const parts: ContentPart[] = [];
    for (const raw of v) {
      const p = obj(raw);
      if (!p) continue;
      const type = str(p["type"]);
      if (type === "image_url") {
        const iu = obj(p["image_url"]) ?? {};
        parts.push({
          type: "image_url",
          imageUrl: { url: str(iu["url"]), detail: optStr(iu["detail"]) },
        });
      } else {
        parts.push({ type: "text", text: str(p["text"]) });
      }
    }
    return parts;
  }
  return undefined;
}

function decodeToolCall(v: unknown): ToolCall | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const fn = obj(o["function"]) ?? {};
  return {
    id: str(o["id"]),
    type: str(o["type"], "function"),
    function: { name: str(fn["name"]), arguments: str(fn["arguments"]) },
  };
}

function decodeReasoningDetail(v: unknown): ReasoningDetail | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const d: ReasoningDetail = { type: str(o["type"]) };
  // `signature` and `data` are opaque blobs replayed verbatim — preserve them as-is.
  const text = optStr(o["text"]);
  if (text !== undefined) d.text = text;
  const summary = optStr(o["summary"]);
  if (summary !== undefined) d.summary = summary;
  const data = optStr(o["data"]);
  if (data !== undefined) d.data = data;
  const signature = optStr(o["signature"]);
  if (signature !== undefined) d.signature = signature;
  const id = optStr(o["id"]);
  if (id !== undefined) d.id = id;
  const format = optStr(o["format"]);
  if (format !== undefined) d.format = format;
  const index = optNum(o["index"]);
  if (index !== undefined) d.index = index;
  return d;
}

function decodeReasoningDetails(v: unknown): ReasoningDetail[] {
  return arr(v)
    .map(decodeReasoningDetail)
    .filter((x): x is ReasoningDetail => x !== undefined);
}

export function decodeMessage(v: unknown): ChatMessage {
  const o = obj(v) ?? {};
  const msg: ChatMessage = { role: roleFromWire(optStr(o["role"])) };
  const content = decodeContent(o["content"]);
  if (content !== undefined) msg.content = content;
  const name = optStr(o["name"]);
  if (name !== undefined) msg.name = name;
  const tcs = arr(o["tool_calls"])
    .map(decodeToolCall)
    .filter((x): x is ToolCall => x !== undefined);
  if (tcs.length > 0) msg.toolCalls = tcs;
  const tcid = optStr(o["tool_call_id"]);
  if (tcid !== undefined) msg.toolCallId = tcid;
  const reasoning = optStr(o["reasoning"]);
  if (reasoning !== undefined) msg.reasoning = reasoning;
  const rds = decodeReasoningDetails(o["reasoning_details"]);
  if (rds.length > 0) msg.reasoningDetails = rds;
  return msg;
}

function decodeChoice(v: unknown): Choice {
  const o = obj(v) ?? {};
  const choice: Choice = {
    index: num(o["index"]),
    message: decodeMessage(o["message"]),
  };
  const fr = o["finish_reason"];
  if (typeof fr === "string") choice.finishReason = finishReasonFromWire(fr);
  return choice;
}

export function decodeChatResponse(v: unknown): ChatResponse {
  const o = obj(v) ?? {};
  return {
    id: str(o["id"]),
    object: str(o["object"], "chat.completion"),
    created: num(o["created"]),
    model: str(o["model"]),
    choices: arr(o["choices"]).map(decodeChoice),
    usage: decodeUsage(o["usage"]),
  };
}

// Streaming chunk decode

function decodeToolCallDelta(v: unknown): ToolCallDelta | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const d: ToolCallDelta = { index: num(o["index"]) };
  const id = optStr(o["id"]);
  if (id !== undefined) d.id = id;
  const type = optStr(o["type"]);
  if (type !== undefined) d.type = type;
  const fn = obj(o["function"]);
  if (fn) {
    d.function = {};
    const name = optStr(fn["name"]);
    if (name !== undefined) d.function.name = name;
    const args = optStr(fn["arguments"]);
    if (args !== undefined) d.function.arguments = args;
  }
  return d;
}

function decodeDelta(v: unknown): Delta {
  const o = obj(v) ?? {};
  const delta: Delta = {};
  const role = optStr(o["role"]);
  if (role !== undefined) delta.role = roleFromWire(role);
  const content = optStr(o["content"]);
  if (content !== undefined) delta.content = content;
  const tcs = arr(o["tool_calls"])
    .map(decodeToolCallDelta)
    .filter((x): x is ToolCallDelta => x !== undefined);
  if (tcs.length > 0) delta.toolCalls = tcs;
  const reasoning = optStr(o["reasoning"]);
  if (reasoning !== undefined) delta.reasoning = reasoning;
  const rds = decodeReasoningDetails(o["reasoning_details"]);
  if (rds.length > 0) delta.reasoningDetails = rds;
  return delta;
}

function decodeChunkChoice(v: unknown): ChunkChoice {
  const o = obj(v) ?? {};
  const cc: ChunkChoice = {
    index: num(o["index"]),
    delta: decodeDelta(o["delta"]),
  };
  const fr = o["finish_reason"];
  if (typeof fr === "string") cc.finishReason = finishReasonFromWire(fr);
  return cc;
}

export function decodeChatCompletionChunk(v: unknown): ChatCompletionChunk {
  const o = obj(v) ?? {};
  const chunk: ChatCompletionChunk = {
    id: str(o["id"]),
    object: str(o["object"], "chat.completion.chunk"),
    created: num(o["created"]),
    model: str(o["model"]),
    choices: arr(o["choices"]).map(decodeChunkChoice),
  };
  const usage = decodeUsage(o["usage"]);
  if (usage !== undefined) chunk.usage = usage;
  return chunk;
}

// ===========================================================================
// Embeddings
// ===========================================================================

export function encodeEmbeddingRequest(req: EmbeddingRequest): Json {
  const out: Json = {
    model: req.model,
    // wire accepts string or array; emit a bare string for a single input.
    input: req.input.length === 1 ? req.input[0] : req.input,
  };
  put(out, "dimensions", req.dimensions);
  put(out, "encoding_format", req.encodingFormat);
  const extra = parseRawJson("EmbeddingRequest.extra", req.extra);
  if (extra !== undefined) {
    if (typeof extra !== "object" || extra === null || Array.isArray(extra)) {
      throw new TypeError("EmbeddingRequest.extra must be a JSON object");
    }
    for (const [k, v] of Object.entries(extra as Json)) out[k] = v;
  }
  return out;
}

/** Decode a base64 string of little-endian f32 bytes into a number[] (SPEC.md). */
function decodeBase64Floats(b64: string): number[] {
  // atob is available in Node 16+, Deno, Bun and browsers.
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  const floats = new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    Math.floor(len / 4),
  );
  return Array.from(floats);
}

function decodeEmbedding(v: unknown): Embedding {
  const o = obj(v) ?? {};
  const raw = o["embedding"];
  let embedding: number[];
  if (typeof raw === "string") {
    embedding = decodeBase64Floats(raw);
  } else if (Array.isArray(raw)) {
    embedding = raw.map((x) => num(x));
  } else {
    embedding = [];
  }
  return { object: str(o["object"], "embedding"), index: num(o["index"]), embedding };
}

export function decodeEmbeddingResponse(v: unknown): EmbeddingResponse {
  const o = obj(v) ?? {};
  return {
    object: str(o["object"], "list"),
    data: arr(o["data"]).map(decodeEmbedding),
    model: str(o["model"]),
    usage: decodeUsage(o["usage"]),
  };
}

// ===========================================================================
// Audio
// ===========================================================================

export function encodeSpeechRequest(req: SpeechRequest): Json {
  const out: Json = { model: req.model, input: req.input, voice: req.voice };
  put(out, "response_format", req.responseFormat);
  put(out, "speed", req.speed);
  const extra = parseRawJson("SpeechRequest.extra", req.extra);
  if (extra !== undefined) {
    if (typeof extra !== "object" || extra === null || Array.isArray(extra)) {
      throw new TypeError("SpeechRequest.extra must be a JSON object");
    }
    for (const [k, v] of Object.entries(extra as Json)) out[k] = v;
  }
  return out;
}

function decodeVoice(v: unknown): Voice {
  const o = obj(v) ?? {};
  const voice: Voice = {
    id: str(o["id"]),
    languages: arr(o["languages"]).map((x) => str(x)),
  };
  const name = optStr(o["name"]);
  if (name !== undefined) voice.name = name;
  return voice;
}

export function decodeVoicesResponse(v: unknown): VoicesResponse {
  const o = obj(v) ?? {};
  return { model: str(o["model"]), voices: arr(o["voices"]).map(decodeVoice) };
}

export function decodeTranscriptionResponse(v: unknown): TranscriptionResponse {
  const o = obj(v) ?? {};
  const out: TranscriptionResponse = { text: str(o["text"]) };
  const task = optStr(o["task"]);
  if (task !== undefined) out.task = task;
  const language = optStr(o["language"]);
  if (language !== undefined) out.language = language;
  const duration = optNum(o["duration"]);
  if (duration !== undefined) out.duration = duration;
  const usage = decodeUsage(o["usage"]);
  if (usage !== undefined) out.usage = usage;
  return out;
}

// ===========================================================================
// Model catalog
// ===========================================================================

function decodeArchitecture(v: unknown): Architecture | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const a: Architecture = {
    inputModalities: arr(o["input_modalities"]).map((x) => str(x)),
    outputModalities: arr(o["output_modalities"]).map((x) => str(x)),
    tokenizer: str(o["tokenizer"]),
  };
  const modality = optStr(o["modality"]);
  if (modality !== undefined) a.modality = modality;
  const instruct = optStr(o["instruct_type"]);
  if (instruct !== undefined) a.instructType = instruct;
  return a;
}

function decodePricing(v: unknown): Pricing | undefined {
  const o = obj(v);
  if (!o) return undefined;
  return { prompt: str(o["prompt"]), completion: str(o["completion"]) };
}

function decodeTopProvider(v: unknown): TopProvider | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const tp: TopProvider = { isModerated: optBool(o["is_moderated"]) ?? false };
  const cl = optNum(o["context_length"]);
  if (cl !== undefined) tp.contextLength = cl;
  const mct = optNum(o["max_completion_tokens"]);
  if (mct !== undefined) tp.maxCompletionTokens = mct;
  const mtt = optNum(o["max_thinking_tokens"]);
  if (mtt !== undefined) tp.maxThinkingTokens = mtt;
  return tp;
}

function decodeModelEndpoint(v: unknown): ModelEndpoint {
  const o = obj(v) ?? {};
  return {
    provider: str(o["provider"]),
    model: str(o["model"]),
    down: optBool(o["down"]) ?? false,
    source: str(o["source"]),
  };
}

function decodeModelEntry(v: unknown): ModelEntry {
  const o = obj(v) ?? {};
  const entry: ModelEntry = {
    id: str(o["id"]),
    canonicalSlug: str(o["canonical_slug"]),
    name: str(o["name"]),
    created: num(o["created"]),
    description: str(o["description"]),
    supportedParameters: arr(o["supported_parameters"]).map((x) => str(x)),
    unsupportedParameters: arr(o["unsupported_parameters"]).map((x) => str(x)),
    endpoints: arr(o["endpoints"]).map(decodeModelEndpoint),
  };
  const cl = optNum(o["context_length"]);
  if (cl !== undefined) entry.contextLength = cl;
  const arch = decodeArchitecture(o["architecture"]);
  if (arch !== undefined) entry.architecture = arch;
  const pricing = decodePricing(o["pricing"]);
  if (pricing !== undefined) entry.pricing = pricing;
  const tp = decodeTopProvider(o["top_provider"]);
  if (tp !== undefined) entry.topProvider = tp;
  const dp = captureRawJson(o["default_parameters"]);
  if (dp !== undefined && o["default_parameters"] !== undefined) {
    entry.defaultParameters = dp;
  }
  return entry;
}

export function decodeListModelsResponse(v: unknown): ListModelsResponse {
  const o = obj(v) ?? {};
  return { data: arr(o["data"]).map(decodeModelEntry) };
}

// ===========================================================================
// Batches
// ===========================================================================

export function encodeBatchCreateRequest(req: BatchCreateRequest): Json {
  return {
    requests: req.requests.map((item) => ({
      custom_id: item.customId,
      body: encodeChatRequest(item.body),
    })),
  };
}

function decodeBatchCounts(v: unknown): BatchCounts | undefined {
  const o = obj(v);
  if (!o) return undefined;
  return {
    total: num(o["total"]),
    processing: num(o["processing"]),
    succeeded: num(o["succeeded"]),
    errored: num(o["errored"]),
    canceled: num(o["canceled"]),
    expired: num(o["expired"]),
  };
}

export function decodeBatchHandle(v: unknown): BatchHandle {
  const o = obj(v) ?? {};
  const handle: BatchHandle = {
    id: str(o["id"]),
    status: batchStatusFromWire(optStr(o["status"])),
  };
  const counts = decodeBatchCounts(o["counts"]);
  if (counts !== undefined) handle.counts = counts;
  const createdAt = optNum(o["created_at"]);
  if (createdAt !== undefined) handle.createdAt = createdAt;
  const expiresAt = optNum(o["expires_at"]);
  if (expiresAt !== undefined) handle.expiresAt = expiresAt;
  const endedAt = optNum(o["ended_at"]);
  if (endedAt !== undefined) handle.endedAt = endedAt;
  const endpoint = optStr(o["endpoint"]);
  if (endpoint !== undefined) handle.endpoint = endpoint;
  return handle;
}

function decodeBatchResponse(v: unknown): BatchResponse | undefined {
  const o = obj(v);
  if (!o) return undefined;
  return { statusCode: num(o["status_code"]), body: decodeChatResponse(o["body"]) };
}

function decodeBatchError(v: unknown): BatchError | undefined {
  const o = obj(v);
  if (!o) return undefined;
  return { code: str(o["code"]), message: str(o["message"]) };
}

export function decodeBatchResultLine(v: unknown): BatchResultLine {
  const o = obj(v) ?? {};
  const line: BatchResultLine = { customId: str(o["custom_id"]) };
  const response = decodeBatchResponse(o["response"]);
  if (response !== undefined) line.response = response;
  const error = decodeBatchError(o["error"]);
  if (error !== undefined) line.error = error;
  return line;
}

// ===========================================================================
// Responses (POST /v1/responses) — encode
// ===========================================================================

function encodeResponseContentPart(p: ResponseContentPart): Json {
  switch (p.type) {
    case "input_text":
      return { type: "input_text", text: p.text };
    case "input_image": {
      // image_url is a plain STRING here, not the chat dialect's nested {url} object.
      const out: Json = { type: "input_image", image_url: p.imageUrl };
      put(out, "detail", p.detail);
      return out;
    }
    case "output_text":
      // A constructed output_text part carries an (empty) annotations array (SPEC.md).
      return { type: "output_text", text: p.text, annotations: [] };
  }
}

function encodeResponseMessageContent(content: ResponseMessageContent | undefined): unknown {
  if (content === undefined) return undefined;
  if (typeof content === "string") return content;
  return content.map(encodeResponseContentPart);
}

/** A plain message item is a bare role-keyed object with NO `"type"` (SPEC.md). */
function encodeResponseMessageItem(m: ResponseMessageItem): Json {
  const out: Json = { role: m.role };
  put(out, "id", m.id);
  const content = encodeResponseMessageContent(m.content);
  if (content !== undefined) out["content"] = content;
  put(out, "status", m.status);
  return out;
}

function encodeResponseFunctionCallItem(fc: ResponseFunctionCallItem): Json {
  const out: Json = {
    type: "function_call",
    call_id: fc.callId,
    name: fc.name,
    arguments: fc.arguments,
  };
  put(out, "id", fc.id);
  put(out, "status", fc.status);
  return out;
}

function encodeResponseFunctionCallOutputItem(fo: ResponseFunctionCallOutputItem): Json {
  const out: Json = {
    type: "function_call_output",
    call_id: fo.callId,
    output: fo.output,
  };
  put(out, "id", fo.id);
  return out;
}

function encodeResponseReasoningItem(r: ResponseReasoningItem): Json {
  const out: Json = { type: "reasoning" };
  put(out, "id", r.id);
  // The list an entry lives in decides its wire token: summary -> "summary_text",
  // content -> "reasoning_text" (SPEC.md).
  if (r.summary && r.summary.length > 0) {
    out["summary"] = r.summary.map((s) => ({ type: "summary_text", text: s.text }));
  }
  if (r.content && r.content.length > 0) {
    out["content"] = r.content.map((c) => ({ type: "reasoning_text", text: c.text }));
  }
  // Opaque blob replayed verbatim to continue an encrypted reasoning turn.
  put(out, "encrypted_content", r.encryptedContent);
  return out;
}

function encodeResponseItem(item: ResponseItem): Json {
  switch (item.type) {
    case "function_call":
      return encodeResponseFunctionCallItem(item);
    case "function_call_output":
      return encodeResponseFunctionCallOutputItem(item);
    case "reasoning":
      return encodeResponseReasoningItem(item);
    case "message":
      return encodeResponseMessageItem(item);
  }
}

function encodeResponsesInput(input: ResponsesInput): unknown {
  // A bare string is one user message; otherwise an array of items.
  if (typeof input === "string") return input;
  return input.map(encodeResponseItem);
}

function encodeResponsesToolDef(t: ResponsesToolDef): Json {
  // FLAT: type/name/parameters at the top level, no nested `function` object.
  const out: Json = { type: t.type, name: t.name };
  put(out, "description", t.description);
  put(out, "parameters", parseRawJson("ResponsesToolDef.parameters", t.parameters));
  put(out, "strict", t.strict);
  return out;
}

function encodeResponsesToolChoice(tc: ResponsesToolChoice): unknown {
  // FLAT named object {type,name}, unlike the chat dialect's nested `function`.
  if (typeof tc === "string") return tc;
  return { type: tc.type, name: tc.name };
}

function encodeResponsesReasoning(r: ResponsesReasoning): Json {
  const out: Json = {};
  put(out, "effort", r.effort);
  put(out, "summary", r.summary);
  return out;
}

/** Build the responses request body. `forceStream` overrides `stream` for the streaming call. */
export function encodeResponsesRequest(req: ResponsesRequest, forceStream?: boolean): Json {
  const out: Json = {
    model: req.model,
    input: encodeResponsesInput(req.input),
  };
  put(out, "instructions", req.instructions);
  const stream = forceStream !== undefined ? forceStream : req.stream;
  put(out, "stream", stream);
  put(out, "temperature", req.temperature);
  put(out, "top_p", req.topP);
  put(out, "max_output_tokens", req.maxOutputTokens);
  if (req.tools && req.tools.length > 0) out["tools"] = req.tools.map(encodeResponsesToolDef);
  put(out, "tool_choice", req.toolChoice && encodeResponsesToolChoice(req.toolChoice));
  put(out, "reasoning", req.reasoning && encodeResponsesReasoning(req.reasoning));
  put(out, "store", req.store);

  // `extra`: parsed and merged at the top level of the request object (SPEC.md).
  const extra = parseRawJson("ResponsesRequest.extra", req.extra);
  if (extra !== undefined) {
    if (typeof extra !== "object" || extra === null || Array.isArray(extra)) {
      throw new TypeError("ResponsesRequest.extra must be a JSON object");
    }
    for (const [k, v] of Object.entries(extra as Json)) out[k] = v;
  }
  return out;
}

// ===========================================================================
// Responses — decode
// ===========================================================================

function decodeResponseContentPart(v: unknown): ResponseContentPart | undefined {
  const o = obj(v);
  if (!o) return undefined;
  switch (str(o["type"])) {
    case "input_text":
      return { type: "input_text", text: str(o["text"]) };
    case "input_image": {
      const part: ResponseInputImagePartLocal = {
        type: "input_image",
        imageUrl: str(o["image_url"]),
      };
      const detail = optStr(o["detail"]);
      if (detail !== undefined) part.detail = detail;
      return part;
    }
    case "output_text":
      return { type: "output_text", text: str(o["text"]) };
    default:
      return undefined; // unknown content part — skip
  }
}

// A local alias so the decoder can build the image part incrementally.
type ResponseInputImagePartLocal = Extract<ResponseContentPart, { type: "input_image" }>;

function decodeResponseMessageContent(v: unknown): ResponseMessageContent | undefined {
  if (v === undefined || v === null) return undefined;
  if (typeof v === "string") return v;
  if (Array.isArray(v)) {
    const parts: ResponseContentPart[] = [];
    for (const raw of v) {
      const p = decodeResponseContentPart(raw);
      if (p !== undefined) parts.push(p);
    }
    return parts;
  }
  return undefined;
}

function decodeResponseMessageItem(o: Json): ResponseMessageItem {
  const item: ResponseMessageItem = { type: "message", role: str(o["role"]) };
  const id = optStr(o["id"]);
  if (id !== undefined) item.id = id;
  const content = decodeResponseMessageContent(o["content"]);
  if (content !== undefined) item.content = content;
  const status = optStr(o["status"]);
  if (status !== undefined) item.status = status;
  return item;
}

function decodeResponseFunctionCallItem(o: Json): ResponseFunctionCallItem {
  const item: ResponseFunctionCallItem = {
    type: "function_call",
    callId: str(o["call_id"]),
    name: str(o["name"]),
    arguments: str(o["arguments"]),
  };
  const id = optStr(o["id"]);
  if (id !== undefined) item.id = id;
  const status = optStr(o["status"]);
  if (status !== undefined) item.status = status;
  return item;
}

function decodeResponseFunctionCallOutputItem(o: Json): ResponseFunctionCallOutputItem {
  const item: ResponseFunctionCallOutputItem = {
    type: "function_call_output",
    callId: str(o["call_id"]),
    output: str(o["output"]),
  };
  const id = optStr(o["id"]);
  if (id !== undefined) item.id = id;
  return item;
}

function decodeReasoningTextList(v: unknown): ResponseReasoningText[] {
  return arr(v)
    .map((raw): ResponseReasoningText | undefined => {
      const o = obj(raw);
      return o ? { text: str(o["text"]) } : undefined;
    })
    .filter((x): x is ResponseReasoningText => x !== undefined);
}

function decodeResponseReasoningItem(o: Json): ResponseReasoningItem {
  const item: ResponseReasoningItem = { type: "reasoning" };
  const id = optStr(o["id"]);
  if (id !== undefined) item.id = id;
  const summary = decodeReasoningTextList(o["summary"]);
  if (summary.length > 0) item.summary = summary;
  const content = decodeReasoningTextList(o["content"]);
  if (content.length > 0) item.content = content;
  const enc = optStr(o["encrypted_content"]);
  if (enc !== undefined) item.encryptedContent = enc;
  return item;
}

function decodeResponseItem(v: unknown): ResponseItem | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const type = optStr(o["type"]);
  // A role-keyed object with no `type` is a plain message item (SPEC.md).
  if (type === undefined || type === "message") return decodeResponseMessageItem(o);
  switch (type) {
    case "function_call":
      return decodeResponseFunctionCallItem(o);
    case "function_call_output":
      return decodeResponseFunctionCallOutputItem(o);
    case "reasoning":
      return decodeResponseReasoningItem(o);
    default:
      return undefined; // unknown item type — skip
  }
}

function decodeErrorBody(v: unknown): ErrorBody | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const body: ErrorBody = { message: str(o["message"]) };
  const type = optStr(o["type"]);
  if (type !== undefined) body.type = type;
  const code = optStr(o["code"]);
  if (code !== undefined) body.code = code;
  return body;
}

function decodeResponsesReasoning(v: unknown): ResponsesReasoning | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const r: ResponsesReasoning = {};
  const effort = optStr(o["effort"]);
  if (effort !== undefined) r.effort = effort;
  const summary = optStr(o["summary"]);
  if (summary !== undefined) r.summary = summary;
  return r;
}

function decodeResponsesIncompleteDetails(
  v: unknown,
): ResponsesIncompleteDetails | undefined {
  const o = obj(v);
  if (!o) return undefined;
  return { reason: str(o["reason"]) };
}

function decodeResponsesInputTokensDetails(
  v: unknown,
): ResponsesInputTokensDetails | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const d: ResponsesInputTokensDetails = {};
  const cached = optNum(o["cached_tokens"]);
  if (cached !== undefined) d.cachedTokens = cached;
  return d;
}

function decodeResponsesOutputTokensDetails(
  v: unknown,
): ResponsesOutputTokensDetails | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const d: ResponsesOutputTokensDetails = {};
  const reasoning = optNum(o["reasoning_tokens"]);
  if (reasoning !== undefined) d.reasoningTokens = reasoning;
  return d;
}

export function decodeResponsesUsage(v: unknown): ResponsesUsage | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const usage: ResponsesUsage = {
    inputTokens: num(o["input_tokens"]),
    outputTokens: num(o["output_tokens"]),
    totalTokens: num(o["total_tokens"]),
  };
  const itd = decodeResponsesInputTokensDetails(o["input_tokens_details"]);
  if (itd !== undefined) usage.inputTokensDetails = itd;
  const otd = decodeResponsesOutputTokensDetails(o["output_tokens_details"]);
  if (otd !== undefined) usage.outputTokensDetails = otd;
  return usage;
}

export function decodeResponsesResponse(v: unknown): ResponsesResponse {
  const o = obj(v) ?? {};
  const resp: ResponsesResponse = {
    id: str(o["id"]),
    object: str(o["object"], "response"),
    createdAt: num(o["created_at"]),
    status: str(o["status"]),
    model: str(o["model"]),
    output: arr(o["output"])
      .map(decodeResponseItem)
      .filter((x): x is ResponseItem => x !== undefined),
  };
  const inc = decodeResponsesIncompleteDetails(o["incomplete_details"]);
  if (inc !== undefined) resp.incompleteDetails = inc;
  const err = decodeErrorBody(o["error"]);
  if (err !== undefined) resp.error = err;
  const usage = decodeResponsesUsage(o["usage"]);
  if (usage !== undefined) resp.usage = usage;
  const store = optBool(o["store"]);
  if (store !== undefined) resp.store = store;
  const instructions = optStr(o["instructions"]);
  if (instructions !== undefined) resp.instructions = instructions;
  const maxOut = optNum(o["max_output_tokens"]);
  if (maxOut !== undefined) resp.maxOutputTokens = maxOut;
  const temp = optNum(o["temperature"]);
  if (temp !== undefined) resp.temperature = temp;
  const topP = optNum(o["top_p"]);
  if (topP !== undefined) resp.topP = topP;
  const reasoning = decodeResponsesReasoning(o["reasoning"]);
  if (reasoning !== undefined) resp.reasoning = reasoning;
  return resp;
}

// Streaming: typed events, NO `[DONE]` sentinel. The stream ends after the terminal
// event; SDKs skip event types they don't recognise (the dialect grows by adding types).
const TERMINAL_RESPONSES_EVENT_TYPES = new Set<string>([
  "response.completed",
  "response.incomplete",
  "response.failed",
]);

const KNOWN_RESPONSES_EVENT_TYPES = new Set<string>([
  "response.created",
  "response.in_progress",
  "response.completed",
  "response.incomplete",
  "response.failed",
  "response.output_item.added",
  "response.output_item.done",
  "response.content_part.added",
  "response.content_part.done",
  "response.output_text.delta",
  "response.output_text.done",
  "response.refusal.delta",
  "response.refusal.done",
  "response.reasoning_text.delta",
  "response.reasoning_text.done",
  "response.reasoning_summary_part.added",
  "response.reasoning_summary_part.done",
  "response.reasoning_summary_text.delta",
  "response.reasoning_summary_text.done",
  "response.function_call_arguments.delta",
  "response.function_call_arguments.done",
  "error",
]);

/** True for `response.completed` / `response.incomplete` / `response.failed`. */
export function isTerminalResponsesEvent(type: string): boolean {
  return TERMINAL_RESPONSES_EVENT_TYPES.has(type);
}

/**
 * Decode one streaming SSE frame's JSON into a {@link ResponsesStreamEvent}, or
 * `undefined` when the `type` is one the SDK doesn't recognise (the caller skips it).
 */
export function decodeResponsesStreamEvent(v: unknown): ResponsesStreamEvent | undefined {
  const o = obj(v);
  if (!o) return undefined;
  const type = str(o["type"]);
  if (!KNOWN_RESPONSES_EVENT_TYPES.has(type)) return undefined; // unknown event — skip (SPEC.md)
  const event: ResponsesStreamEvent = {
    type,
    sequenceNumber: num(o["sequence_number"]),
  };
  if (obj(o["response"])) event.response = decodeResponsesResponse(o["response"]);
  const outputIndex = optNum(o["output_index"]);
  if (outputIndex !== undefined) event.outputIndex = outputIndex;
  const itemId = optStr(o["item_id"]);
  if (itemId !== undefined) event.itemId = itemId;
  const contentIndex = optNum(o["content_index"]);
  if (contentIndex !== undefined) event.contentIndex = contentIndex;
  const item = decodeResponseItem(o["item"]);
  if (item !== undefined) event.item = item;
  const part = decodeResponseContentPart(o["part"]);
  if (part !== undefined) event.part = part;
  const delta = optStr(o["delta"]);
  if (delta !== undefined) event.delta = delta;
  const text = optStr(o["text"]);
  if (text !== undefined) event.text = text;
  const args = optStr(o["arguments"]);
  if (args !== undefined) event.arguments = args;
  const message = optStr(o["message"]);
  if (message !== undefined) event.message = message;
  return event;
}
