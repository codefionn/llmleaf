# llmleaf — Kotlin Multiplatform client

[Kotlin Multiplatform](https://kotlinlang.org/docs/multiplatform.html) SDK for the **llmleaf**
LLM proxy. Speaks llmleaf's OpenAI/OpenRouter-shaped JSON over HTTP (see
[`../SPEC.md`](../SPEC.md)) on JVM, native (linuxX64), and JS (Node) via Ktor. All calls are
`suspend`; streaming surfaces are `kotlinx.coroutines.flow.Flow`.

> **Not built in the authoring environment** — written without a local Gradle/Kotlin
> toolchain, so run `./gradlew build` once before depending on it. Versions are pinned to
> recent stable releases (Kotlin 2.0.21, Ktor 3.0.3, kotlinx.serialization 1.7.3,
> coroutines 1.9.0, Square Wire 5.1.0).

## Build & run

```sh
cd clients/kotlin
./gradlew build                 # build + test every target (Wire codegen runs as a dependency)

LLMLEAF_BASE_URL=https://gateway.example.com \
LLMLEAF_API_KEY=sk-... \
  ./gradlew :example:run        # runs the JVM demo
```

The Gradle wrapper pins 8.11.1; `gradle-wrapper.jar` is not vendored, so the first `./gradlew`
needs network access to fetch it (copy one from any Gradle 8.11.x checkout if you're offline).

## Example

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
    ChatRequest("gpt-4o-mini", listOf(ChatMessage.user("Say hi"))),
)
(resp.choices.first().message.content as? MessageContent.Text)?.let { println(it.text) }

// Streaming chat — a cold Flow of chunks, stops at the `data: [DONE]` sentinel
client.chatStream(ChatRequest("gpt-4o-mini", listOf(ChatMessage.user("Count to 5"))))
    .collect { chunk -> chunk.choices.firstOrNull()?.delta?.content?.let(::print) }

client.close()
```

### Responses (`POST /v1/responses`)

The OpenAI **Responses** dialect on the same core. `input` is a bare string (one user message) or
an array of items; tools and `tool_choice` are **flat**; llmleaf is stateless, so the response
always reports `store = false`.

```kotlin
import eu.codefionn.llmleaf.client.model.ResponsesInput
import eu.codefionn.llmleaf.client.model.ResponsesRequest

// Non-streaming — `outputText` flattens every output_text part.
val r = client.responses(
    ResponsesRequest("gpt-4o-mini", ResponsesInput.text("Say hi"), instructions = "Be concise."),
)
println(r.outputText)

// Streaming — typed events, NO `[DONE]` sentinel: the Flow stops on the terminal
// response.completed / response.incomplete / response.failed event; unknown event types are
// ignored, and a mid-stream `error` event throws ApiError.
client.responsesStream(ResponsesRequest("gpt-4o-mini", ResponsesInput.text("Count to 5")))
    .collect { event -> if (event.type == "response.output_text.delta") print(event.delta) }
```

## Endpoints

| SDK call | Endpoint |
|----------|----------|
| `chat` / `chatStream` | `POST /v1/chat/completions` (stream → `Flow<ChatCompletionChunk>`) |
| `responses` / `responsesStream` | `POST /v1/responses` (stream → `Flow<ResponsesStreamEvent>`, no `[DONE]`) |
| `embeddings` | `POST /v1/embeddings` (decodes base64 little-endian f32) |
| `listModels` | `GET /v1/models` |
| `speech` | `POST /v1/audio/speech` → `SpeechResult` |
| `voices` | `GET /v1/audio/voices` |
| `transcribe` | `POST /v1/audio/transcriptions` (multipart) |
| `createBatch` / `getBatch` / `cancelBatch` / `batchResults` | `/v1/batches…` (results → `Flow<BatchResultLine>`) |

Non-2xx responses throw `ApiError(status, message)`. Free-form JSON fields (`extra`,
`parameters`, `jsonSchema`, …) are typed `RawJson` — spliced into the body verbatim on encode
and captured back on decode; `extra` keys merge at the top level (explicit fields win).

## Regenerate from the proto

Square **Wire** compiles the proto into Kotlin types (the codegen proof);
**kotlinx.serialization** `@Serializable` classes mirror it and drive the actual JSON wire.
After editing the proto:

```sh
cd clients/kotlin && ./scripts/gen.sh     # ./gradlew generateProtos   (or: make gen-kotlin)
```

If you change message shapes, mirror the change in the kotlinx.serialization classes under
`src/commonMain/kotlin/eu/codefionn/llmleaf/client/model/` — those own the bytes on the wire.
