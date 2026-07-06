// LlmleafClient — the fetch-based HTTP transport. One method per SPEC.md endpoint.
//
// Runtime-agnostic: relies only on global fetch / Request / Response / FormData /
// Blob / AbortController, all present in Node 20+, Deno, Bun and browsers. A custom
// fetch can be injected (testing, proxies, custom agents).

import { ApiError, apiErrorFromResponse } from "./error.js";
import { parseSseData, parseNdjson } from "./stream.js";
import {
  encodeChatRequest,
  decodeChatResponse,
  decodeChatCompletionChunk,
  encodeResponsesRequest,
  decodeResponsesResponse,
  decodeResponsesStreamEvent,
  isTerminalResponsesEvent,
  encodeEmbeddingRequest,
  decodeEmbeddingResponse,
  encodeSpeechRequest,
  decodeVoicesResponse,
  decodeTranscriptionResponse,
  decodeListModelsResponse,
  encodeBatchCreateRequest,
  decodeBatchHandle,
  decodeBatchResultLine,
} from "./wire.js";
import type {
  ChatRequest,
  ChatResponse,
  ChatCompletionChunk,
  ResponsesRequest,
  ResponsesResponse,
  ResponsesStreamEvent,
  EmbeddingRequest,
  EmbeddingResponse,
  SpeechRequest,
  SpeechResult,
  VoicesResponse,
  TranscriptionRequest,
  TranscriptionResponse,
  ListModelsResponse,
  ListModelsOptions,
  BatchCreateRequest,
  BatchHandle,
  BatchResultLine,
} from "./types.js";

/** The subset of the WHATWG `fetch` signature this client uses. */
export type FetchLike = (
  input: string | URL,
  init?: RequestInit,
) => Promise<Response>;

export interface LlmleafClientOptions {
  /** Gateway base URL, e.g. "https://gateway.example.com". */
  baseUrl: string;
  /** API key sent as `Authorization: Bearer <apiKey>`. */
  apiKey: string;
  /** Per-request timeout in milliseconds. 0 / omitted disables the timeout. */
  timeoutMs?: number;
  /** Optional admin token; sent as `x-admin-token` when an endpoint opts in. */
  adminToken?: string;
  /** Inject a custom fetch (defaults to the global `fetch`). */
  fetch?: FetchLike;
}

/** Audio file input for transcriptions: bytes + a filename. */
export interface TranscriptionFile {
  /** A Blob/File, or raw bytes. */
  data: Blob | Uint8Array | ArrayBuffer;
  /** Filename for the multipart part (e.g. "audio.mp3"). */
  filename: string;
  /** Optional content type when `data` is not already a Blob. */
  contentType?: string;
}

const SPEECH_CONTENT_TYPES: Record<string, string> = {
  mp3: "audio/mpeg",
  wav: "audio/wav",
  opus: "audio/ogg",
  aac: "audio/aac",
  flac: "audio/flac",
  pcm: "audio/pcm",
};

export class LlmleafClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly adminToken?: string;
  private readonly fetchImpl: FetchLike;

  constructor(options: LlmleafClientOptions) {
    if (!options.baseUrl) throw new TypeError("LlmleafClient: baseUrl is required");
    if (!options.apiKey) throw new TypeError("LlmleafClient: apiKey is required");
    // Strip a single trailing slash so path joins are predictable.
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 0;
    this.adminToken = options.adminToken;
    const f = options.fetch ?? (globalThis.fetch as FetchLike | undefined);
    if (!f) {
      throw new TypeError(
        "LlmleafClient: no global fetch available; pass `fetch` in the options (Node <18, or a non-fetch runtime).",
      );
    }
    this.fetchImpl = f;
  }

  // -------------------------------------------------------------------------
  // Internal request plumbing
  // -------------------------------------------------------------------------

  private url(path: string, query?: Record<string, string | undefined>): string {
    let u = this.baseUrl + path;
    if (query) {
      const params = new URLSearchParams();
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined) params.set(k, v);
      }
      const qs = params.toString();
      if (qs) u += "?" + qs;
    }
    return u;
  }

  private headers(extra?: Record<string, string>): Headers {
    const h = new Headers({ authorization: `Bearer ${this.apiKey}` });
    if (extra) for (const [k, v] of Object.entries(extra)) h.set(k, v);
    return h;
  }

  /** Issue a request with the configured timeout; returns the raw Response. */
  private async send(url: string, init: RequestInit): Promise<Response> {
    if (this.timeoutMs <= 0) {
      return this.fetchImpl(url, init);
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      return await this.fetchImpl(url, { ...init, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  /** Send a JSON request and return the parsed JSON body, or throw ApiError. */
  private async sendJson(
    url: string,
    body: unknown,
    headers?: Record<string, string>,
  ): Promise<unknown> {
    const res = await this.send(url, {
      method: "POST",
      headers: this.headers({ "content-type": "application/json", ...headers }),
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await apiErrorFromResponse(res);
    return res.json();
  }

  // -------------------------------------------------------------------------
  // Chat completions
  // -------------------------------------------------------------------------

  /** POST /v1/chat/completions (non-streaming). */
  async chat(req: ChatRequest): Promise<ChatResponse> {
    const body = encodeChatRequest(req, false);
    const json = await this.sendJson(this.url("/v1/chat/completions"), body);
    return decodeChatResponse(json);
  }

  /**
   * POST /v1/chat/completions with `stream:true`. Returns an async iterable of
   * {@link ChatCompletionChunk}; stops at the `data: [DONE]` sentinel.
   */
  async *chatStream(
    req: ChatRequest,
  ): AsyncGenerator<ChatCompletionChunk, void, unknown> {
    const body = encodeChatRequest(req, true);
    const res = await this.send(this.url("/v1/chat/completions"), {
      method: "POST",
      headers: this.headers({
        "content-type": "application/json",
        accept: "text/event-stream",
      }),
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await apiErrorFromResponse(res);
    if (!res.body) {
      throw new ApiError(res.status, "streaming response had no body");
    }
    for await (const payload of parseSseData(res.body)) {
      yield decodeChatCompletionChunk(JSON.parse(payload));
    }
  }

  // -------------------------------------------------------------------------
  // Responses (OpenAI Responses dialect)
  // -------------------------------------------------------------------------

  /** POST /v1/responses (non-streaming). */
  async responses(req: ResponsesRequest): Promise<ResponsesResponse> {
    const body = encodeResponsesRequest(req, false);
    const json = await this.sendJson(this.url("/v1/responses"), body);
    return decodeResponsesResponse(json);
  }

  /**
   * POST /v1/responses with `stream:true`. Returns an async iterable of
   * {@link ResponsesStreamEvent}. Unlike {@link chatStream} there is no `[DONE]`
   * sentinel: the iterator stops after the terminal `response.completed` /
   * `response.incomplete` / `response.failed` event (or when the connection closes).
   * Event types the SDK doesn't recognise are skipped (the dialect grows by adding types).
   */
  async *responsesStream(
    req: ResponsesRequest,
  ): AsyncGenerator<ResponsesStreamEvent, void, unknown> {
    const body = encodeResponsesRequest(req, true);
    const res = await this.send(this.url("/v1/responses"), {
      method: "POST",
      headers: this.headers({
        "content-type": "application/json",
        accept: "text/event-stream",
      }),
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await apiErrorFromResponse(res);
    if (!res.body) {
      throw new ApiError(res.status, "streaming response had no body");
    }
    for await (const payload of parseSseData(res.body)) {
      const event = decodeResponsesStreamEvent(JSON.parse(payload));
      if (event === undefined) continue; // unrecognised event type — skip (SPEC.md)
      yield event;
      if (isTerminalResponsesEvent(event.type)) return;
    }
  }

  // -------------------------------------------------------------------------
  // Embeddings
  // -------------------------------------------------------------------------

  /** POST /v1/embeddings. Base64 embeddings are decoded to float arrays. */
  async embeddings(req: EmbeddingRequest): Promise<EmbeddingResponse> {
    const body = encodeEmbeddingRequest(req);
    const json = await this.sendJson(this.url("/v1/embeddings"), body);
    return decodeEmbeddingResponse(json);
  }

  // -------------------------------------------------------------------------
  // Model catalog
  // -------------------------------------------------------------------------

  /** GET /v1/models. Pass `admin:true` to include per-model `endpoints`. */
  async listModels(opts: ListModelsOptions = {}): Promise<ListModelsResponse> {
    const headers: Record<string, string> = {};
    if (opts.admin) {
      if (!this.adminToken) {
        throw new TypeError(
          "listModels({ admin:true }) requires an adminToken in the client options",
        );
      }
      headers["x-admin-token"] = this.adminToken;
    }
    const url = this.url("/v1/models", { type: opts.type, search: opts.search });
    const res = await this.send(url, {
      method: "GET",
      headers: this.headers(headers),
    });
    if (!res.ok) throw await apiErrorFromResponse(res);
    return decodeListModelsResponse(await res.json());
  }

  // -------------------------------------------------------------------------
  // Audio
  // -------------------------------------------------------------------------

  /** POST /v1/audio/speech. Returns the raw audio bytes + Content-Type. */
  async speech(req: SpeechRequest): Promise<SpeechResult> {
    const body = encodeSpeechRequest(req);
    const res = await this.send(this.url("/v1/audio/speech"), {
      method: "POST",
      headers: this.headers({ "content-type": "application/json" }),
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await apiErrorFromResponse(res);
    const buf = await res.arrayBuffer();
    const contentType =
      res.headers.get("content-type") ??
      (req.responseFormat ? SPEECH_CONTENT_TYPES[req.responseFormat] : undefined) ??
      "application/octet-stream";
    return { bytes: new Uint8Array(buf), contentType };
  }

  /** GET /v1/audio/voices?model=<id>. */
  async voices(model: string): Promise<VoicesResponse> {
    const url = this.url("/v1/audio/voices", { model });
    const res = await this.send(url, { method: "GET", headers: this.headers() });
    if (!res.ok) throw await apiErrorFromResponse(res);
    return decodeVoicesResponse(await res.json());
  }

  /**
   * POST /v1/audio/transcriptions (multipart/form-data).
   *
   * For `responseFormat` json/verbose_json the structured
   * {@link TranscriptionResponse} is returned; for text/srt/vtt the response is a
   * plain-text body and the returned object carries it in `.text` (other fields empty).
   */
  async transcribe(
    file: TranscriptionFile,
    req: TranscriptionRequest,
  ): Promise<TranscriptionResponse> {
    const form = new FormData();
    const blob =
      file.data instanceof Blob
        ? file.data
        : new Blob(
            [file.data instanceof Uint8Array ? bytesToArrayBuffer(file.data) : file.data],
            file.contentType ? { type: file.contentType } : undefined,
          );
    form.set("file", blob, file.filename);
    form.set("model", req.model);
    if (req.language !== undefined) form.set("language", req.language);
    if (req.prompt !== undefined) form.set("prompt", req.prompt);
    if (req.responseFormat !== undefined) form.set("response_format", req.responseFormat);
    if (req.temperature !== undefined) form.set("temperature", String(req.temperature));

    // Do NOT set content-type: the runtime sets it with the multipart boundary.
    const res = await this.send(this.url("/v1/audio/transcriptions"), {
      method: "POST",
      headers: this.headers(),
      body: form,
    });
    if (!res.ok) throw await apiErrorFromResponse(res);

    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      return decodeTranscriptionResponse(await res.json());
    }
    // text / srt / vtt: a plain-text body. Surface it directly in `.text`.
    return { text: await res.text() };
  }

  // -------------------------------------------------------------------------
  // Batches
  // -------------------------------------------------------------------------

  /** POST /v1/batches. */
  async createBatch(req: BatchCreateRequest): Promise<BatchHandle> {
    const body = encodeBatchCreateRequest(req);
    const json = await this.sendJson(this.url("/v1/batches"), body);
    return decodeBatchHandle(json);
  }

  /** GET /v1/batches/{id}. */
  async getBatch(id: string): Promise<BatchHandle> {
    const res = await this.send(this.url(`/v1/batches/${encodeURIComponent(id)}`), {
      method: "GET",
      headers: this.headers(),
    });
    if (!res.ok) throw await apiErrorFromResponse(res);
    return decodeBatchHandle(await res.json());
  }

  /** POST /v1/batches/{id}/cancel. */
  async cancelBatch(id: string): Promise<BatchHandle> {
    const res = await this.send(
      this.url(`/v1/batches/${encodeURIComponent(id)}/cancel`),
      { method: "POST", headers: this.headers() },
    );
    if (!res.ok) throw await apiErrorFromResponse(res);
    return decodeBatchHandle(await res.json());
  }

  /**
   * GET /v1/batches/{id}/results — an async iterable of {@link BatchResultLine}
   * (one per NDJSON line).
   */
  async *batchResults(
    id: string,
  ): AsyncGenerator<BatchResultLine, void, unknown> {
    const res = await this.send(
      this.url(`/v1/batches/${encodeURIComponent(id)}/results`),
      { method: "GET", headers: this.headers({ accept: "application/x-ndjson" }) },
    );
    if (!res.ok) throw await apiErrorFromResponse(res);
    if (!res.body) throw new ApiError(res.status, "batch results had no body");
    for await (const line of parseNdjson(res.body)) {
      yield decodeBatchResultLine(line);
    }
  }
}

/** Copy a Uint8Array's bytes into a fresh ArrayBuffer (handles non-zero byteOffset). */
function bytesToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}
