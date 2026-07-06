# @codefionn/llmleaf-client

TypeScript / JavaScript client for the [llmleaf](../../) LLM proxy.

Speaks llmleaf's OpenAI/OpenRouter-shaped JSON over HTTP (see [`../SPEC.md`](../SPEC.md)).
Zero runtime dependencies beyond the `@bufbuild/protobuf` runtime, and it runs anywhere a
global `fetch` / `ReadableStream` / `FormData` exists — Node 20+, Deno, Bun, browsers.

## Install

```sh
npm install @codefionn/llmleaf-client
```

## Example

```ts
import { LlmleafClient, Role } from "@codefionn/llmleaf-client";

const client = new LlmleafClient({
  baseUrl: "https://gateway.example.com",
  apiKey: process.env.LLMLEAF_API_KEY!,
  timeoutMs: 30_000,   // optional; 0/omitted = no timeout. Also: adminToken, fetch
});

// Non-streaming chat
const res = await client.chat({
  model: "gpt-4o-mini",
  messages: [{ role: Role.USER, content: "Say hi." }],
});
console.log(res.choices[0]?.message.content);

// Streaming chat — async iterable of chunks, stops at `data: [DONE]`
for await (const chunk of client.chatStream({
  model: "gpt-4o-mini",
  messages: [{ role: Role.USER, content: "Count to 5." }],
})) {
  process.stdout.write(chunk.choices[0]?.delta.content ?? "");
}

// Responses dialect (POST /v1/responses) — `input` is a bare string or an item array
const resp = await client.responses({
  model: "gpt-4o-mini",
  input: [{ type: "message", role: "user", content: "Say hi." }],
});

// Streaming responses — typed events, NO `[DONE]`; stops on the terminal
// response.completed / .incomplete / .failed event; unknown event types are skipped
for await (const event of client.responsesStream({
  model: "gpt-4o-mini",
  input: "Count to 5.",
})) {
  if (event.type === "response.output_text.delta") process.stdout.write(event.delta ?? "");
}
```

## Endpoints

```ts
await client.chat(req);              // POST /v1/chat/completions
client.chatStream(req);              // stream:true, async iterable of chunks
await client.responses(req);         // POST /v1/responses (OpenAI Responses dialect)
client.responsesStream(req);         // stream:true, async iterable of typed events (no [DONE])
await client.embeddings(req);        // base64 vectors decoded to floats
await client.listModels({ type: "llm" });
await client.speech(req);            // -> { bytes, contentType }
await client.voices("tts-1");
await client.transcribe(file, req);  // multipart; json/verbose_json -> object, text/srt/vtt -> tr.text
await client.createBatch(req);
await client.getBatch(id);
await client.cancelBatch(id);
client.batchResults(id);             // async iterable of result lines
```

Free-form fields (`extra`, `responseFormat.jsonSchema`, `functionDef.parameters`, …) are raw
JSON strings — pass `JSON.stringify(obj)` and the transport splices the parsed value in;
`extra` keys merge at the top level of the request.

## Errors

Any non-2xx response throws a typed `ApiError`:

```ts
import { ApiError } from "@codefionn/llmleaf-client";

try {
  await client.chat(req);
} catch (e) {
  if (e instanceof ApiError) console.error(e.status, e.message); // 403 "model not allowed"
}
```

## Run the example

[`examples/basic.ts`](examples/basic.ts) does chat, streaming, and a model list:

```sh
LLMLEAF_BASE_URL=https://gateway.example.com LLMLEAF_API_KEY=sk-… npx tsx examples/basic.ts
```

## Regenerate from the proto

The generated file (`src/gen/llmleaf/v1/llmleaf_pb.ts`) is committed; regenerate it after the
proto changes:

```sh
npm install      # fetch the @bufbuild/protoc-gen-es plugin
npm run gen      # runs scripts/gen.sh (needs protoc on PATH)
```

## Notes

- The wire mapping is hand-written (`src/wire.ts`) because protobuf-es's JSON codec emits
  camelCase + `SCREAMING_CASE`, which isn't the OpenAI wire (snake_case + lowercase tokens).
  The generated descriptors stay available under the `gen` export for reflection.
- Decoders are lenient (extra/missing fields tolerated). No retries. `chatStream` /
  `batchResults` need a streaming-capable `fetch`. Realtime WebSocket is out of scope.
