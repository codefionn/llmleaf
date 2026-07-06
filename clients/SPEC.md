# llmleaf client wire spec

This is the contract every client SDK under `clients/` implements. The typed model is
generated from [`proto/llmleaf/v1/llmleaf.proto`](proto/llmleaf/v1/llmleaf.proto); this
document describes how those types map onto the **OpenAI/OpenRouter-shaped JSON over
HTTP** that the llmleaf core actually speaks. The wire is JSON, never protobuf-binary.

## Base

- **Base URL**: operator-provided, e.g. `https://gateway.example.com`. All paths below are
  relative to it. The SDK takes the base URL + an API key at construction.
- **Auth**: every endpoint except `GET /healthz` and `GET /v1/openapi.json` requires
  `Authorization: Bearer <api-key>`.
- **Content type**: request bodies are `application/json` unless noted (transcriptions are
  `multipart/form-data`). Responses are JSON unless noted (speech = raw audio bytes;
  streaming chat = `text/event-stream`; batch results = `application/x-ndjson`).
- **Errors**: any non-2xx response carries the envelope `{"error":{"message":"..."}}`.
  Clients parse it into a typed `ApiError { status: int, message: string }` and raise/return it.
  Status codes: 400 bad request · 401 missing/invalid key · 403 blocked or model-not-allowed
  · 404 no route for model · 429 key suspended (limiter) · 502 all upstreams failed.

## Enum ⇄ wire mapping

Every closed-set enum maps to its wire token by lowercasing the value name:
`TOOL_CALLS`→`"tool_calls"`, `ASSISTANT`→`"assistant"`, `IN_PROGRESS`→`"in_progress"`.
The `*_UNSPECIFIED` zero value ⇔ field absent. Implement one helper pair per SDK
(`enumToWire`/`enumFromWire`) and reuse it; do not hand-map each enum.

## Free-form JSON fields

`ChatRequest.extra`, `FunctionDef.parameters`, `ResponseFormat.json_schema`,
`EmbeddingRequest.extra`, `SpeechRequest.extra`, `ModelEntry.default_parameters` are raw
JSON strings. On encode, splice the parsed JSON value into the body at that key (do not
double-encode it as a string). On decode, capture the sub-object back as a raw JSON string.
`extra` keys are merged at the top level of the request object.

## Endpoints

### POST /v1/chat/completions
Body = `ChatRequest`. Notes:
- `content` is `string` when the message has plain text, else an array of content parts.
- `stop` serialises as a bare string if it has one element, else an array (either is accepted; emitting an array is fine).
- Prefer `max_completion_tokens`; still send `max_tokens` if the caller set only that.
- `tool_choice` is a bare string (`"auto"`/`"none"`/`"required"`) or the named object.

**Non-streaming** (`stream` absent/false): response = `ChatResponse` (`object:"chat.completion"`).

**Streaming** (`stream:true`): response is SSE. Each event is a line `data: <json>` followed
by a blank line; `<json>` is a `ChatCompletionChunk`. The stream ends with the literal line
`data: [DONE]`. Clients expose this as an async iterator / channel / callback of chunks and
MUST stop on `[DONE]` without trying to JSON-parse it. Accumulate `choices[].delta.content`
for the assembled text; `usage` appears only on the terminal chunk (when present).

### POST /v1/responses
Body = `ResponsesRequest` — the OpenAI Responses dialect on the same canonical core. Notes:
- `input` is a bare string (one user message) or an array of items; each item's `"type"` selects
  the `ResponseItem` variant (`"message"` may be implied by a bare role-keyed object — emit
  role-keyed objects without `"type"` for plain messages, typed objects for the rest).
- `content` on a message item is a string or an array of parts; `input_image.image_url` is a
  plain **string** (not the chat dialect's nested `{url}` object). Output text parts carry an
  `annotations` array — emit `[]` when constructing one.
- Tools and the named `tool_choice` are **flat** (`type`/`name` at the top level, no nested
  `function` object).
- Reasoning items: `summary[]` entries serialise as `{"type":"summary_text","text"}` and
  `content[]` entries as `{"type":"reasoning_text","text"}` — the list an entry lives in decides
  its wire token. `encrypted_content` is opaque: echo it back verbatim in the next request's
  input to continue an encrypted reasoning turn.
- llmleaf is stateless: `store` is accepted but the response always reports `"store": false`;
  `previous_response_id` and `background: true` are rejected with 400. Do not expose a
  retrieval call — `GET /v1/responses/{id}` is an explained 404 by design.

**Non-streaming** (`stream` absent/false): response = `ResponsesResponse` (`object:"response"`).

**Streaming** (`stream:true`): SSE, but **typed events and no `[DONE]` sentinel** — each frame is
`event: <type>` + `data: <json>`, the JSON self-describing via its `type` field (parse `data:`
lines only; the `event:` line is redundant). Decode each into `ResponsesStreamEvent`, ignore
unrecognised types, and stop after the terminal `response.completed` / `response.incomplete` /
`response.failed` event (also stop if the connection closes without one). `sequence_number` is
strictly increasing. Accumulate `response.output_text.delta` deltas for the assembled text; the
terminal event's `response` snapshot carries the full output and usage. A mid-stream `error`
event surfaces idiomatically, matching how the SDK's chat stream reports failures: either
raised as the typed `ApiError` (status 0/502 — there is no HTTP status mid-stream) or yielded
as the decoded event for the caller to inspect; it must never be silently dropped.

### POST /v1/embeddings
Body = `EmbeddingRequest`. Response = `EmbeddingResponse` (`object:"list"`, `data[].object:"embedding"`).
If `encoding_format:"base64"`, each `data[].embedding` is a base64 string of little-endian
f32 bytes — decode it into the float vector before returning.

### GET /v1/models
Query: `type` (`all|llm|tts|stt|embedding`), `search` (substring). Optional header
`x-admin-token` adds the per-model `endpoints` array. Response = `ListModelsResponse`.

### POST /v1/audio/speech
Body = `SpeechRequest`. Response = **raw audio bytes**; `Content-Type` reflects
`response_format` (`mp3`→`audio/mpeg`, `wav`→`audio/wav`, `opus`→`audio/ogg`, `aac`→`audio/aac`,
`flac`→`audio/flac`, `pcm`→`audio/pcm`). Return the bytes + content-type to the caller.

### GET /v1/audio/voices?model=<id>
Response = `VoicesResponse`.

### POST /v1/audio/transcriptions
`multipart/form-data`. Parts: `file` (the audio bytes + filename, required), `model` (required),
and any set of `language` / `prompt` / `response_format` / `temperature`. For
`response_format` json/verbose_json the response is `TranscriptionResponse`; for `text`/`srt`/`vtt`
it is a plain-text body — return the text directly.

### POST /v1/batches · GET /v1/batches/{id} · POST /v1/batches/{id}/cancel
Create body = `BatchCreateRequest` (`{"requests":[{"custom_id","body":<ChatRequest>}]}`).
The three return a `BatchHandle`.

### GET /v1/batches/{id}/results
`application/x-ndjson` — one JSON object per line, each a `BatchResultLine`
(`response` on success, `error` on failure). Expose as an iterator/stream of `BatchResultLine`.

## What every SDK must ship

1. Generated types from the proto (real codegen wired into the build — see each client README).
2. A `Client` constructed from `(baseUrl, apiKey, opts)` with a pluggable HTTP timeout and an
   optional `x-admin-token`.
3. The nine calls above, with streaming chat and streaming responses surfaced idiomatically
   (async iterator / channel / `Flow` / callback) — chat streaming handles the `[DONE]`
   sentinel; responses streaming stops on the terminal `response.completed` / `incomplete` /
   `failed` event instead (there is no sentinel).
4. Typed `ApiError`.
5. One runnable example (non-stream chat, stream chat, list models) and a README with install +
   regen-from-proto instructions.

## Hand-test against a live gateway (optional, all clients identical)

```
curl -s $BASE/v1/models -H "Authorization: Bearer $KEY" | head
curl -s $BASE/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'
```
The SDK calls must produce byte-identical request bodies.
