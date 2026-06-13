# @llmleaf/client

Official TypeScript / JavaScript client for the [llmleaf](../../) LLM proxy.

The wire is **OpenAI/OpenRouter-shaped JSON over HTTP** (see [`../SPEC.md`](../SPEC.md)),
not protobuf-binary. The [proto](../proto/llmleaf/v1/llmleaf.proto) is the typed source of
truth; the `protobuf-es` codegen artifact is committed under [`src/gen`](src/gen) and a
hand-written `fetch`-based transport maps the public model to/from the wire JSON.

Zero runtime dependencies beyond the `@bufbuild/protobuf` runtime (used for the generated
enum/descriptor types). Works in Node 20+, Deno, Bun and browsers — anywhere a global
`fetch` / `ReadableStream` / `FormData` / `Blob` exists.

## Install

```sh
npm install @llmleaf/client
```

## Quickstart

```ts
import { LlmleafClient, Role } from "@llmleaf/client";

const client = new LlmleafClient({
  baseUrl: "https://gateway.example.com",
  apiKey: process.env.LLMLEAF_API_KEY!,
  timeoutMs: 30_000,        // optional per-request timeout (0/omitted = no timeout)
  adminToken: "…",          // optional; enables admin-only fields
  // fetch: customFetch,    // optional; defaults to the global fetch
});

// Non-streaming chat
const res = await client.chat({
  model: "gpt-4o-mini",
  messages: [{ role: Role.USER, content: "Say hi." }],
});
console.log(res.choices[0]?.message.content);

// Streaming chat — async iterable of ChatCompletionChunk, stops at `data: [DONE]`
for await (const chunk of client.chatStream({
  model: "gpt-4o-mini",
  messages: [{ role: Role.USER, content: "Count to 5." }],
})) {
  process.stdout.write(chunk.choices[0]?.delta.content ?? "");
}
```

### Every endpoint

```ts
await client.chat(req);                       // POST /v1/chat/completions
client.chatStream(req);                        // POST /v1/chat/completions (stream:true)
await client.embeddings(req);                  // POST /v1/embeddings (base64 decoded to floats)
await client.listModels({ type: "llm" });      // GET  /v1/models  (admin:true adds endpoints)
await client.speech(req);                       // POST /v1/audio/speech -> { bytes, contentType }
await client.voices("tts-1");                   // GET  /v1/audio/voices
await client.transcribe(file, req);             // POST /v1/audio/transcriptions (multipart)
await client.createBatch(req);                  // POST /v1/batches
await client.getBatch(id);                      // GET  /v1/batches/{id}
await client.cancelBatch(id);                   // POST /v1/batches/{id}/cancel
client.batchResults(id);                        // GET  /v1/batches/{id}/results (async iterable)
```

Speech returns the raw audio bytes plus the server's content type:

```ts
const { bytes, contentType } = await client.speech({
  model: "tts-1", input: "hello", voice: "alloy", responseFormat: "mp3",
});
```

Transcription takes the audio as a `Blob`/`Uint8Array`/`ArrayBuffer` + filename:

```ts
const tr = await client.transcribe(
  { data: audioBytes, filename: "audio.mp3", contentType: "audio/mpeg" },
  { model: "whisper-1", responseFormat: "json" },
);
// json / verbose_json -> structured TranscriptionResponse
// text / srt / vtt     -> plain text in tr.text
```

### Free-form JSON fields

`ChatRequest.extra`, `FunctionDef.parameters`, `ResponseFormat.jsonSchema`,
`EmbeddingRequest.extra`, `SpeechRequest.extra` and `ModelEntry.defaultParameters` are
**raw JSON strings** — you pass a `JSON.stringify(...)`d object, the transport splices the
parsed value into the wire body (and `extra` keys are merged at the top level). On decode,
sub-objects are captured back as raw JSON strings.

```ts
await client.chat({
  model: "gpt-4o-mini",
  messages: [{ role: Role.USER, content: "hi" }],
  responseFormat: { type: "json_schema", jsonSchema: JSON.stringify({ type: "object" }) },
  extra: JSON.stringify({ provider: { order: ["openai"] } }),
});
```

### Errors

Any non-2xx response throws a typed `ApiError`:

```ts
import { ApiError } from "@llmleaf/client";

try {
  await client.chat(/* … */);
} catch (e) {
  if (e instanceof ApiError) {
    console.error(e.status, e.message, e.code); // 403 "model not allowed" "blocked"
  }
}
```

Status codes: 400 bad request · 401 missing/invalid key · 403 blocked or model-not-allowed
· 404 no route for model · 429 key suspended · 502 all upstreams failed.

## Example

[`examples/basic.ts`](examples/basic.ts) runs non-streaming chat, streaming chat and a model
list. It reads `LLMLEAF_BASE_URL` / `LLMLEAF_API_KEY` (and optional `LLMLEAF_MODEL`):

```sh
LLMLEAF_BASE_URL=https://gateway.example.com LLMLEAF_API_KEY=sk-… npx tsx examples/basic.ts
```

## Regenerate the typed model from the proto

The proto is the source of truth. The generated file (`src/gen/llmleaf/v1/llmleaf_pb.ts`)
is committed; regenerate it after the proto changes:

```sh
npm install      # fetch the @bufbuild/protoc-gen-es plugin into node_modules/.bin
npm run gen      # runs scripts/gen.sh
```

`scripts/gen.sh` invokes:

```sh
protoc --plugin=protoc-gen-es=./node_modules/.bin/protoc-gen-es \
  --es_out=src/gen --es_opt=target=ts \
  --proto_path=../proto ../proto/llmleaf/v1/llmleaf.proto
```

**Toolchain:** `protoc` (libprotoc 35) on `PATH`, plus `@bufbuild/protoc-gen-es`
(devDependency — run `npm install` first). From the repo root, `make gen-ts` delegates here.

## Scripts

| Command            | What it does                                              |
|--------------------|----------------------------------------------------------|
| `npm run gen`      | Regenerate `src/gen` from the proto.                     |
| `npm run typecheck`| `tsc --noEmit` over `src` + `examples`.                  |
| `npm run build`    | Emit ESM + `.d.ts` to `dist/` (library only, Node-free). |
| `npm run example`  | Run `examples/basic.ts` with `tsx`.                      |

## Caveats (honest)

- **Wire mapping is hand-written.** protobuf-es's own JSON codec emits camelCase keys and
  `SCREAMING_CASE` enum names, which do **not** match the OpenAI wire (snake_case keys +
  lowercase tokens like `"assistant"`). So the public surface is plain TS interfaces and
  [`src/wire.ts`](src/wire.ts) does the mapping; the generated descriptors stay available
  under the `gen` export for advanced reflection use.
- **Decoders are lenient.** Responses are parsed defensively (missing/extra fields tolerated,
  the server is authoritative); they do not strictly validate every field.
- **No retries / backoff.** One request per call. Wrap calls yourself if you need retries.
- **Streaming needs a `ReadableStream` body.** Some exotic `fetch` polyfills buffer the whole
  response; use a streaming-capable fetch for `chatStream` / `batchResults`.
- **Realtime WebSocket is out of scope** for all llmleaf SDKs (per the repo README).
- **Node < 18** has no global `fetch`/`FormData`; pass a `fetch` implementation and run on
  Node 20+ for `FormData`/`Blob` in transcriptions.
