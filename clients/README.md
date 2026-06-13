# llmleaf clients

Official client libraries for the llmleaf proxy, one per language. Per
[SOUL.md](../SOUL.md), they are **generated from a single Protocol Buffers schema** —
[`proto/llmleaf/v1/llmleaf.proto`](proto/llmleaf/v1/llmleaf.proto) is the source of truth for
the request/response shapes — while on the wire they speak llmleaf's existing
**OpenAI/OpenRouter-shaped JSON over HTTP** (the core gains no protobuf surface).

| Language | Directory | Codegen | Build verified here |
|----------|-----------|---------|---------------------|
| Go | [`go/`](go/) | `protoc-gen-go` | ✅ |
| Rust | [`rust/`](rust/) | `prost-build` (build.rs) | ✅ |
| TypeScript / JavaScript | [`typescript/`](typescript/) | `protobuf-es` | ✅ |
| Zig | [`zig/`](zig/) | vendored `protoc-gen-zig` | ✅ |
| C# / .NET | [`csharp/`](csharp/) | `protoc --csharp_out` (Google.Protobuf) | ✅ |
| Kotlin Multiplatform | [`kotlin/`](kotlin/) | Square **Wire** (Gradle) | ✅ |

## Layout

```
clients/
  proto/llmleaf/v1/llmleaf.proto   the contract — edit here, regenerate everywhere
  SPEC.md                          how the proto maps onto the OpenAI/OpenRouter HTTP wire
  buf.yaml, buf.gen.yaml           `buf generate` regen path (Go, TS, Rust)
  Makefile                         `make gen` / `make gen-<lang>` / `make proto-check`
  <lang>/                          one self-contained SDK per language
```

## Regenerate the typed model

The proto is the source of truth: change it, then regenerate.

```sh
cd clients
make proto-check       # validate the schema (protoc)
make gen               # regenerate all five clients
# or, the Go/TS/Rust subset in one shot if you have buf:
buf generate
```

Each client also carries a `scripts/gen.sh` that regenerates only that language and is wired
into its native build; see the per-client README for the toolchain it needs.

## Surface

All five cover the same endpoints (see [SPEC.md](SPEC.md)): chat completions (streaming +
non-streaming), embeddings, model catalog, text-to-speech, speech-to-text (multipart), and
batches. Auth is `Authorization: Bearer <key>`; errors surface as a typed `ApiError`. The
realtime WebSocket surface is intentionally out of scope for these SDKs.
