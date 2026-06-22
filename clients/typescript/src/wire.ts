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
