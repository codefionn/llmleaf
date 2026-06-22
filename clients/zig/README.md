# llmleaf — Zig client

A std-only Zig SDK for the [llmleaf](../../README.md) LLM proxy. Speaks llmleaf's
OpenAI/OpenRouter-shaped JSON over HTTP (see [`../SPEC.md`](../SPEC.md)). No dependencies —
`std` only. Built and tested against **Zig 0.16.0** (the `std.http` / `std.json` / `std.Io`
APIs shift between releases, so it targets 0.16.0 exactly).

## Build & run

```sh
cd clients/zig
zig build                # module + static lib + example
zig build test           # unit tests (no network)

LLMLEAF_BASE_URL=https://gateway.example.com \
LLMLEAF_API_KEY=sk-... \
  zig build example      # runs examples/basic.zig against a live gateway
```

## Use as a dependency

```sh
zig fetch --save "git+https://github.com/codefionn/llmleaf#main:clients/zig"
```

```zig
// build.zig
const llmleaf = b.dependency("llmleaf", .{ .target = target, .optimize = optimize });
your_module.addImport("llmleaf", llmleaf.module("llmleaf"));
```

## Example

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
        .messages = &.{ llmleaf.ChatMessage.textMsg(.user, "Say hi") },
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

Streaming is pull-based (Zig's std is synchronous):

```zig
var stream = try client.chatStream(req, &api_err);
defer stream.deinit();
while (try stream.next()) |chunk| {
    if (chunk.choices.len > 0)
        if (chunk.choices[0].delta.content) |delta|
            std.debug.print("{s}", .{delta});
}
```

`next()` returns `null` at the `data: [DONE]` sentinel. The chunk borrows an internal arena
that is **reset on the next `next()`** — copy out anything you need to keep. A callback form
(`chatStreamCallback`) is also provided.

## Surface

| Method | Endpoint |
|--------|----------|
| `chat` / `chatStream` / `chatStreamCallback` | `POST /v1/chat/completions` |
| `embeddings` | `POST /v1/embeddings` (handles base64 vectors) |
| `listModels` | `GET /v1/models` |
| `speech` | `POST /v1/audio/speech` (bytes + content-type) |
| `voices` | `GET /v1/audio/voices` |
| `transcribe` | `POST /v1/audio/transcriptions` (multipart) |
| `createBatch` / `getBatch` / `cancelBatch` / `batchResults` | `/v1/batches…` (NDJSON iterator) |

## Errors & ownership

Zig errors can't carry a payload, so non-2xx is surfaced two ways: the call returns a typed
error from `llmleaf.Error` (`BadRequest`, `Unauthorized`, `Forbidden`, …), and if you pass a
`*?llmleaf.ApiError` out-pointer the `{ status, message }` is written there (pass `null` to
ignore). `ApiError.message` is duplicated into your allocator — `gpa.free` it once read.

Most calls return an `Owned(T)` whose `value` borrows an arena the wrapper owns — `defer
.deinit()`. The client is heap-allocated and borrows `base_url` / `api_key` (they must outlive
it). Streams own a connection + arena and must be `deinit`'d to return the connection.

## Regenerate from the proto

There's no official protoc plugin for Zig, so [`src/gen/llmleaf.zig`](src/gen/llmleaf.zig) is a
committed, hand-maintained mirror of the proto (shaped for the JSON wire). When the proto
changes, edit it and the mapping in `src/wire.zig`, then `zig build test`.
[`scripts/gen.sh`](scripts/gen.sh) documents an optional third-party plugin whose output is a
diff aid, never a drop-in.

## Notes

- `options.timeout_ms` is advisory only — Zig 0.16's `std.http.Client` exposes no per-socket
  timeout. Wrap calls with your own watchdog if you need a hard deadline.
- SSE/NDJSON lines read through a 1 MiB buffer (`error.StreamTooLong` past that). Decoders are
  tolerant of unknown fields. The wire mapping plus chat / models / the 401 path were exercised
  against a local mock gateway; TTS / STT / voices / batches are unit-tested but not run against
  a live backend.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT). Copyright (C) 2026
Fionn Langhans.
