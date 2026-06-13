# llmleaf — official Zig client SDK

A std-only Zig SDK for the [llmleaf](../../README.md) LLM proxy. It speaks
llmleaf's **OpenAI/OpenRouter-shaped JSON over HTTP** wire (see
[`../SPEC.md`](../SPEC.md)); the typed model mirrors
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto).

- **Zig version:** built and tested against **Zig 0.16.0**. The `std.http.Client`
  and `std.json` APIs (and the new `std.Io` interface) differ across releases —
  this SDK targets 0.16.0 exactly.
- **Dependencies:** none. `std` only.

## Build & run

```sh
cd clients/zig
zig build                # builds the `llmleaf` module + static lib + example
zig build test           # runs the unit tests (no network)
zig build example        # runs examples/basic.zig against a live gateway
```

`zig build example` reads its configuration from the environment:

| Variable           | Default                   | Meaning                         |
|--------------------|---------------------------|---------------------------------|
| `LLMLEAF_API_KEY`  | *(required)*              | your API key                    |
| `LLMLEAF_BASE_URL` | `http://localhost:8080`   | gateway base URL                |
| `LLMLEAF_MODEL`    | `gpt-4o-mini`             | model id for the example calls  |

```sh
LLMLEAF_BASE_URL=https://gateway.example.com \
LLMLEAF_API_KEY=sk-... \
zig build example
```

## Use as a dependency

```sh
zig fetch --save "git+https://github.com/codefionn/llmleaf#main:clients/zig"
```

then in your `build.zig`:

```zig
const llmleaf = b.dependency("llmleaf", .{ .target = target, .optimize = optimize });
your_module.addImport("llmleaf", llmleaf.module("llmleaf"));
```

```zig
const std = @import("std");
const llmleaf = @import("llmleaf");

pub fn main() !void {
    var dbg: std.heap.DebugAllocator(.{}) = .init;
    defer _ = dbg.deinit();
    const gpa = dbg.allocator();

    const client = try llmleaf.Client.init(gpa, "https://gateway.example.com", "sk-...", .{});
    defer client.deinit();

    var api_err: ?llmleaf.ApiError = null;
    const resp = client.chat(.{
        .model = "gpt-4o-mini",
        .messages = &.{ llmleaf.ChatMessage.textMsg(.user, "hello") },
    }, &api_err) catch |err| {
        if (api_err) |e| {
            std.debug.print("API error {d}: {s}\n", .{ e.status, e.message });
            gpa.free(e.message); // ApiError.message is owned by you
        }
        return err;
    };
    defer resp.deinit();
    std.debug.print("{s}\n", .{resp.value.choices[0].message.content.?.text});
}
```

## Surface

Every endpoint in [`../SPEC.md`](../SPEC.md) that is reasonable in synchronous Zig:

| Method                              | Endpoint |
|-------------------------------------|----------|
| `chat`                              | `POST /v1/chat/completions` (non-streaming) |
| `chatStream` / `chatStreamCallback` | `POST /v1/chat/completions` (SSE streaming) |
| `embeddings`                        | `POST /v1/embeddings` (handles `encoding_format:"base64"`) |
| `listModels`                        | `GET /v1/models` (`type` + `search`; `x-admin-token` adds `endpoints`) |
| `speech`                            | `POST /v1/audio/speech` (TTS → bytes + content-type) |
| `voices`                            | `GET /v1/audio/voices` |
| `transcribe`                        | `POST /v1/audio/transcriptions` (STT, multipart `file` part) |
| `createBatch` / `getBatch` / `cancelBatch` | `POST /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` |
| `batchResults`                      | `GET /v1/batches/{id}/results` (NDJSON iterator) |

### Streaming

Zig's std is synchronous, so streaming is **pull-based**:

```zig
var stream = try client.chatStream(req, &api_err);
defer stream.deinit();
while (try stream.next()) |chunk| {
    if (chunk.choices.len > 0)
        if (chunk.choices[0].delta.content) |delta|
            std.debug.print("{s}", .{delta});
}
```

`next()` returns the next `ChatCompletionChunk` or `null` at the `data: [DONE]`
sentinel (the sentinel is **never** JSON-parsed, per SPEC.md). A callback form is
also provided:

```zig
const text = try client.chatStreamCallback(req, &api_err, ctx, myCallback);
defer gpa.free(text); // assembled choices[0].delta.content
```

`batchResults(id)` is the same shape over line-delimited NDJSON, yielding
`BatchResultLine` values.

> **Chunk lifetime:** the chunk returned by `ChatStream.next()` (and the line
> from `BatchResults.next()`) borrows from an internal arena that is **reset on
> the next `next()` call**. Copy out anything you need to keep past the next
> iteration.

## Errors

Zig error values cannot carry a payload, so non-2xx responses are surfaced two ways:

1. the call returns a typed error from `llmleaf.Error`
   (`BadRequest` 400, `Unauthorized` 401, `Forbidden` 403, `NotFound` 404,
   `RateLimited` 429, `UpstreamFailed` 502, `Api`/`UnexpectedStatus` otherwise); and
2. if you pass a `*?llmleaf.ApiError` out-pointer, the structured
   `{ status, message }` is written there. Pass `null` to ignore it.

`ApiError.message` is **duplicated into your allocator** — free it with
`gpa.free(api_err.message)` once read.

## Allocator & ownership

- `Client.init(gpa, base_url, api_key, opts)` **heap-allocates** the client (its
  embedded HTTP transport holds self-referential pointers and must not move) and
  **borrows** `base_url`, `api_key`, and any `opts.admin_token` — they must
  outlive the client. Always `defer client.deinit()`.
- Calls that return parsed data return an **`Owned(T)`**: `value` borrows from an
  arena the wrapper owns. Call `.deinit()` when done.
- `speech()` returns owned `bytes` + `content_type` (free both).
- `transcribe()` returns a `TranscriptionResult` union — `.json` (an `Owned(...)`)
  for `json`/`verbose_json`, `.text` (an owned slice) for `text`/`srt`/`vtt`.
  Its `deinit(gpa)` frees whichever it is.
- Streams (`ChatStream`, `BatchResults`) own a connection + arena; **must** be
  `deinit`'d to return the connection to the pool.

## Options

```zig
.{
    .admin_token = "...",      // optional x-admin-token (adds admin-only fields)
    .timeout_ms = 0,           // see caveat below
    .user_agent = "my-app/1.0",
}
```

## Regenerating the typed model (third-party plugin note)

The proto is the single source of truth. **There is no official protoc plugin for
Zig**, so unlike the Go/Rust/TS clients, this SDK's typed model is a committed,
hand-maintained mirror:

- [`src/gen/llmleaf.zig`](src/gen/llmleaf.zig) is the **authoritative** mirror of
  the proto, shaped for the JSON wire (snake_case fields, enums whose Zig names
  equal their lowercase wire tokens, free-form JSON as raw `[]const u8`). This is
  what `zig build` compiles.
- [`scripts/gen.sh`](scripts/gen.sh) documents and *attempts* regeneration via a
  **third-party** `protoc-gen-zig` plugin (e.g.
  [Arwalk/zig-protobuf](https://github.com/Arwalk/zig-protobuf)) **if a
  contributor vendors one**. Because such plugins emit protobuf-binary-oriented
  types rather than the JSON-wire mirror this SDK needs, their output is a *diff
  aid* to reconcile by hand — never a drop-in replacement for `src/gen/llmleaf.zig`.

When the proto changes: edit `src/gen/llmleaf.zig` (and the
encode/decode mapping in `src/wire.zig`) to match, then `zig build test`.

## Layout

```
zig/
  build.zig            module + static lib + `example`/`test` steps
  build.zig.zon        package manifest (std-only, no deps)
  scripts/gen.sh       regen story + third-party plugin note
  src/
    root.zig           public entry (@import("llmleaf"))
    client.zig         the Client + every endpoint + streaming iterators
    wire.zig           typed model <-> OpenAI/OpenRouter JSON edge mapping
    http.zig           std.http.Client transport (buffered + streaming)
    errors.zig         ApiError + typed error set
    gen/llmleaf.zig    authoritative proto mirror (the typed model)
  examples/basic.zig   non-stream chat, stream chat, list models
```

## Caveats / honest corners

- **`options.timeout_ms` is advisory only.** Zig 0.16's `std.http.Client` does
  not yet expose a per-socket connect/read timeout knob through the public API,
  so the field is accepted and documented but not currently enforced. Wrap calls
  with your own watchdog if you need a hard deadline.
- **TLS** relies on `std.crypto` + the system certificate bundle (loaded by
  `std.http.Client` on first HTTPS request). No custom CA injection is exposed.
- **Streaming line size.** SSE/NDJSON lines are read through a 1 MiB line buffer;
  a single chunk or result line larger than that yields `error.StreamTooLong`.
  Normal chat chunks and batch result lines are far smaller.
- **Response decoding is tolerant.** Decoders pull known fields out of the JSON
  and ignore unknown ones (forward-compatible), defaulting missing scalars — they
  do not reject extra keys.
- **Tested corners.** The wire mapping (encode/decode, base64 embeddings,
  multipart, SSE prefix/sentinel handling) is covered by `zig build test`, and
  chat (stream + non-stream), model listing, and the 401 error path were
  exercised end-to-end against a local mock gateway. TTS/STT/voices/batches share
  the same transport and decoders but were validated at the unit level rather
  than against a live audio/batch backend.

## License

Copyright (C) 2026 Fionn Langhans. Licensed under either of
[Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option (`MIT OR Apache-2.0`).
