# llmleaf — Kotlin Multiplatform client

Official [Kotlin Multiplatform](https://kotlinlang.org/docs/multiplatform.html) SDK for the
**llmleaf** LLM proxy. It speaks llmleaf's OpenAI/OpenRouter-shaped JSON over HTTP (see
[`../SPEC.md`](../SPEC.md)); the typed model is generated from the single source of truth
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto).

> **Not compiled in the authoring environment.** This project was written without a local
> Gradle/Kotlin toolchain available, so it has **not** been built or test-run here. It targets
> "compiles on first try" — but please run `./gradlew build` once on a normal machine to
> confirm before depending on it. Versions are pinned to recent stable releases (Kotlin 2.0.21,
> Ktor 3.0.3, kotlinx.serialization 1.7.3, kotlinx.coroutines 1.9.0, Square Wire 5.1.0).

## Two kinds of "codegen", on purpose

The wire is JSON, not protobuf-binary, so this SDK uses both:

1. **Square Wire** (`com.squareup.wire`, applied in `build.gradle.kts`, pointed at `../proto`)
   genuinely compiles the proto into Kotlin types under `eu.codefionn.llmleaf.v1`. This is the
   real proto codegen — proof the schema compiles — and is what `make gen-kotlin` /
   `scripts/gen.sh` run (`./gradlew generateProtos`).
2. **kotlinx.serialization** `@Serializable` data classes (under
   `src/commonMain/kotlin/.../client/model/`) mirror the proto and drive the **HTTP wire**. They
   produce the exact OpenAI/OpenRouter JSON in `SPEC.md` (snake_case via `@SerialName`, enums
   lowercased to wire tokens, string-or-array unions, free-form JSON spliced inline). Wire's
   runtime types are not shaped for that JSON, so Wire compiles the proto while
   kotlinx.serialization owns the bytes on the wire.

## Supported targets

`build.gradle.kts` declares:

- **JVM** (Java 17) — Ktor **CIO** engine (actual in `src/jvmMain`).
- **linuxX64** (native) — Ktor **CIO** engine (actual in `src/linuxX64Main`). Add `macosArm64()`,
  `mingwX64()`, `iosArm64()`, etc. the same way — give each its CIO dependency and an `actual`
  (or hoist the `actual` into a shared `nativeMain` source set).
- **JS** (Node, IR) — Ktor **Js** engine (actual in `src/jsMain`). `browser()` is left off by
  default so a fresh `./gradlew build` needs no headless browser; add it in `build.gradle.kts`
  if you want it.

The platform HTTP engine is selected by `defaultHttpClientEngine()` (an `expect`/`actual`).

## Build & run

```sh
cd clients/kotlin

# Build + test every target. Wire codegen runs as a dependency of the build.
./gradlew build

# Regenerate only the proto types (parity with the Makefile's `gen-kotlin`):
./gradlew generateProtos        # or: ./scripts/gen.sh

# Run the example (JVM). It reads the gateway from the environment:
LLMLEAF_BASE_URL=https://gateway.example.com \
LLMLEAF_API_KEY=sk-... \
LLMLEAF_MODEL=gpt-4o-mini \
  ./gradlew :example:run
```

The Gradle **wrapper scripts** (`gradlew`, `gradlew.bat`) and
`gradle/wrapper/gradle-wrapper.properties` are checked in and pin Gradle 8.11.1. The
`gradle-wrapper.jar` is intentionally **not** vendored — the first `./gradlew` invocation needs
network access to fetch it (and the Gradle distribution). If you work offline, copy a
`gradle-wrapper.jar` from any Gradle 8.11.x checkout into `gradle/wrapper/`, or run with a
system Gradle 8.11+ once to materialise it.

## Usage

```kotlin
import eu.codefionn.llmleaf.client.LlmleafClient
import eu.codefionn.llmleaf.client.model.ChatMessage
import eu.codefionn.llmleaf.client.model.ChatRequest
import eu.codefionn.llmleaf.client.model.MessageContent
import kotlin.time.Duration.Companion.seconds

val client = LlmleafClient("https://gateway.example.com", "sk-...") {
    requestTimeout = 30.seconds
    adminToken = null            // set to add admin-only `endpoints` to model listings
}

// Non-streaming chat
val resp = client.chat(
    ChatRequest(
        model = "gpt-4o-mini",
        messages = listOf(ChatMessage.user("Hello")),
    ),
)
(resp.choices.first().message.content as? MessageContent.Text)?.let { println(it.text) }

// Streaming chat — a cold Flow of chunks; stops at the `data: [DONE]` sentinel
client.chatStream(ChatRequest("gpt-4o-mini", listOf(ChatMessage.user("Count to 5"))))
    .collect { chunk -> chunk.choices.firstOrNull()?.delta?.content?.let(::print) }

client.close()
```

### Every endpoint in SPEC.md

| SDK call | Endpoint |
|----------|----------|
| `chat(req)` | `POST /v1/chat/completions` (non-streaming) |
| `chatStream(req): Flow<ChatCompletionChunk>` | `POST /v1/chat/completions` (SSE, stops at `[DONE]`) |
| `embeddings(req)` | `POST /v1/embeddings` (decodes `base64` little-endian f32) |
| `listModels(type, search)` | `GET /v1/models` |
| `speech(req): SpeechResult` | `POST /v1/audio/speech` (raw bytes + content-type) |
| `voices(model)` | `GET /v1/audio/voices` |
| `transcribe(req, audio, filename)` | `POST /v1/audio/transcriptions` (multipart `file` part) |
| `createBatch` / `getBatch` / `cancelBatch` | `POST /v1/batches` · `GET /v1/batches/{id}` · `POST .../cancel` |
| `batchResults(id): Flow<BatchResultLine>` | `GET /v1/batches/{id}/results` (NDJSON) |

All calls are `suspend`; streaming surfaces are `kotlinx.coroutines.flow.Flow`.

### Errors

Any non-2xx response is parsed from `{"error":{"message":...}}` and thrown as
`ApiError(status: Int, message: String)`.

### Free-form JSON passthrough

Fields that carry arbitrary JSON (`ChatRequest.extra`, `FunctionDef.parameters`,
`ResponseFormat.jsonSchema`, `EmbeddingRequest.extra`, `SpeechRequest.extra`,
`ModelEntry.defaultParameters`) are typed as `RawJson` — a raw JSON string that is **spliced**
into the body verbatim on encode (never double-encoded) and captured back on decode. For
`extra`, the object's keys are merged at the **top level** of the request; explicit struct
fields win on collisions.

## Regenerate from the proto

The proto is the source of truth. Edit `../proto/llmleaf/v1/llmleaf.proto`, then:

```sh
cd clients/kotlin && ./scripts/gen.sh     # runs ./gradlew generateProtos (Wire)
# or, from clients/:  make gen-kotlin
```

If you change the message shapes, mirror the change in the kotlinx.serialization classes under
`src/commonMain/kotlin/eu/codefionn/llmleaf/client/model/` (Wire generates the proof-of-codegen
types; these hand-written ones drive the wire).

## Layout

```
kotlin/
  build.gradle.kts            KMP + Wire + kotlinx.serialization; pinned versions
  settings.gradle.kts         root build; includes :example
  gradle.properties           daemon/JVM args, KMP flags
  gradlew, gradlew.bat        wrapper scripts (jar fetched on first run)
  gradle/wrapper/…            wrapper properties (Gradle 8.11.1)
  scripts/gen.sh              ./gradlew generateProtos (Makefile parity)
  src/commonMain/…/client/    LlmleafClient, RawJson, ApiError, config, engine expect
  src/commonMain/…/client/model/  @Serializable wire types + custom serializers
  src/{jvm,linuxX64,js}Main/… actual HTTP engines (CIO / CIO / Js)
  src/commonTest/…            MockEngine wire tests (no network needed)
  example/                    runnable JVM demo (:example:run)
```
