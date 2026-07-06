//! The benchmark cases: every core loop and component SOUL.md calls hot, plus the discrete pieces
//! they are built from.
//!
//! The naming is `area/case`:
//!   - `hotpath/*` — the whole consumer edge for one request (`parse → authorize → route → stream →
//!     map out`), the loop SOUL.md principle 1 calls sacred, driven through an inline provider so it
//!     runs fully offline. `edge_nonstream` collects to a JSON body; `edge_stream` encodes SSE frames.
//!   - `engine/*` — just the engine's middle (`route → stream → emit events → collect`), no edge mapping.
//!   - `collect/*` — the stream-fold that *defines* the non-streaming response (principle 4).
//!   - `openai/*` — the chat dialect mappings at the edge (principle 3), in and out.
//!   - `embeddings/*`, `speech/*`, `transcription/*`, `batch/*` — the other three modalities and the
//!     async-batch envelope (the modalities memo): their edge mappings, plus the `hotpath/*` and
//!     `engine/*` entries that run them through the same shared `dispatch` the chat path uses.
//!   - `keys/*`, `router/*`, `health/*`, `pricing/*` — the per-request lookups (principle 5): each is
//!     meant to be a map probe and a comparison, never arithmetic, and the allocation count proves it.
//!   - `events/*` — the one output channel; `serde/*` — the optional full-payload serialisation cost.
//!   - `provider/*` — a *real* first-party provider's build-request + response-parse path, driven over
//!     the injectable fake transport (`llmleaf-providers`'s `test-support` feature): no network, no
//!     keys, just the work the provider edge does mapping a canonical request out to the wire and a
//!     canned upstream response back in. This is the one area that names a concrete provider crate —
//!     deliberately, to *measure* an edge SOUL.md pushes the work to (principle 2 / the decision
//!     filter); the core and its hot-path cases above stay provider-oblivious.
//!
//! Everything outside `provider/*` uses only public APIs and an *inline* provider, so the benchmark —
//! like the core itself and the e2e test — never names or depends on a concrete provider crate
//! (principle 2). The `provider/*` cases are the explicit exception, and they live entirely in the QA
//! crate, never in the core.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on;
use futures::StreamExt;
use llmleaf_core::compat::openai::{self, ChunkEncoder};
use llmleaf_core::compat::{batch, embeddings, speech, transcription};
use llmleaf_core::{
    build_state, Config, Event, EventBus, HealthTable, RateLimitConfig, RateLimiter,
};
use llmleaf_model::{
    collect, collect_audio, collect_chunks, AudioChunk, AudioStream, BatchCounts, BatchHandle,
    BatchOutcome, BatchResult, BatchStatus, ChatRequest, ChatResponse, Choice, Embedding,
    EmbeddingRequest, EmbeddingResponse, FinishReason, Message, ModelError, ResponseStream, Role,
    SpeechRequest, StreamChunk, ToolCall, ToolCallDelta, TranscriptionRequest,
    TranscriptionResponse, Usage,
};
use llmleaf_pricing::Pricing;
use llmleaf_provider::{Provider, ProviderCx, ProviderRegistry};
use llmleaf_providers::fake::FakeHttpTransport;
use llmleaf_providers::{Brand, OpenAiCompatProvider, Transports};
use serde_json::{json, Value};

use crate::harness::Suite;

/// A fixed, caller-supplied "now" — the core holds no clock for decisions (principle 9), so the
/// benchmark supplies one constant and every time-comparison reads it.
const NOW: u64 = 1_000_000;

/// Consumer tokens: `base64(key-id:password)`, password "s3cret" throughout (see [`CONFIG`]). The
/// harness warms each case before measuring, so the password KDF runs during warm-up and the measured
/// `authorize` is the cached warm-path lookup — exactly the steady state the ceilings legislate.
const TOKEN_BENCH: &str = "YmVuY2g6czNjcmV0"; // base64("bench:s3cret")
const TOKEN_NARROW: &str = "bmFycm93OnMzY3JldA=="; // base64("narrow:s3cret")
const TOKEN_UNKNOWN: &str = "Z2hvc3Q6czNjcmV0"; // base64("ghost:s3cret") — a well-formed, unknown id

/// An inline provider that returns short, fully canonical responses for every modality. Mirrors the
/// e2e test's `MockProvider` and the simulation's `SimProvider`: it keeps the benchmark off the network
/// and off any provider crate, exercising the engine over a real `futures::stream::iter` of chunks. It
/// overrides all four modalities (the trait's defaults are `Unsupported`) so the engine's shared
/// `dispatch` is measured for embeddings/speech/transcription as well as chat.
struct BenchProvider;

#[async_trait]
impl Provider for BenchProvider {
    fn name(&self) -> &str {
        "bench"
    }

    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
            Ok(StreamChunk::Start {
                id: "bench-1".into(),
                model: req.model.clone(),
            }),
            Ok(StreamChunk::Content {
                index: 0,
                delta: "Hello, world!".into(),
            }),
            Ok(StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            }),
            Ok(StreamChunk::Usage(Usage {
                prompt_tokens: 10,
                completion_tokens: 3,
                total_tokens: 13,
                cost_usd: None,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            })),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        _cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        // One short vector per input, indices in order — the canonical batch object the edge maps out.
        let embeddings = req
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| Embedding {
                index: i as u32,
                vector: bench_vector(),
            })
            .collect();
        Ok(EmbeddingResponse {
            model: req.model.clone(),
            embeddings,
            usage: Usage {
                prompt_tokens: 8,
                completion_tokens: 0,
                total_tokens: 8,
                cost_usd: None,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
        })
    }

    async fn speech(
        &self,
        _req: SpeechRequest,
        _cx: &ProviderCx,
    ) -> Result<AudioStream, ModelError> {
        Ok(Box::pin(futures::stream::iter(audio_chunks(4))))
    }

    async fn transcribe(
        &self,
        _req: TranscriptionRequest,
        _cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        Ok(TranscriptionResponse {
            text: "Hello, world! This is a representative transcript.".into(),
            language: Some("en".into()),
            duration: Some(1.5),
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                cost_usd: None,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
        })
    }
}

/// Register every case into `suite` and run them. The model `gpt-4o` is routed to the inline provider
/// and is present in the bundled pricing dataset, so the engine path also exercises a real cost lookup.
pub fn register(suite: &mut Suite) {
    let state = bench_state(false);

    register_hot_path(suite, &state);
    register_engine(suite, &state);
    register_collect(suite);
    register_openai(suite);
    register_embeddings(suite, &state);
    register_speech(suite, &state);
    register_transcription(suite, &state);
    register_batch(suite);
    register_lookups(suite, &state);
    register_pricing(suite);
    register_events(suite);
    register_serde(suite);
    register_provider(suite);
}

/// Build the shared engine state. `include_payloads` flips the opt-in full-payload event emission —
/// the one knob that changes the engine's per-request allocation profile (it serializes the whole
/// request into the `RequestStarted` event).
fn bench_state(include_payloads: bool) -> llmleaf_core::AppState {
    let mut config = Config::from_toml_str(CONFIG).expect("bench config parses");
    config.server.include_payloads = include_payloads;
    let mut registry = ProviderRegistry::new();
    registry.register("bench", Arc::new(BenchProvider));
    build_state(&config, Arc::new(registry)).expect("bench state builds")
}

// ---------------------------------------------------------------------------------------------
// hotpath/* — the whole edge for one request
// ---------------------------------------------------------------------------------------------

fn register_hot_path(suite: &mut Suite, state: &llmleaf_core::AppState) {
    // The benchmark begins at the canonical/dialect boundary. The two steps before it on the real
    // edge — `bearer()` extracting the token from the `Authorization` header (server.rs), and axum's
    // JSON body deserialisation — are excluded deliberately: `bearer` is a private, single-allocation
    // helper (one `String` for the token), and body deserialisation is serde_json's, not the core's.
    // `serde/request_to_value` covers the core's own (de)serialisation cost separately.

    // Non-streaming: dialect in → canonical → engine → collect → dialect out. The complete journey
    // a `POST /v1/chat/completions` body makes, minus the HTTP framing and body (de)serialisation.
    {
        let engine = state.engine.clone();
        let keys = state.keys.clone();
        suite.bench("hotpath/edge_nonstream", small_openai_value, move |value| {
            block_on(async {
                let req = openai::parse_chat_request(value).expect("parse");
                let model = req.model.clone();
                let key = keys.authorize(TOKEN_BENCH, &model, NOW).expect("authorize");
                let stream = engine
                    .run(req, key, "bench-req".into(), NOW)
                    .await
                    .expect("run");
                let resp = collect(stream).await.expect("collect");
                serde_json::to_vec(&openai::response_to_openai(&resp, NOW)).expect("serialise")
            })
        });
    }

    // Streaming: same front half, but the map-out is the per-chunk encode-then-serialize loop. Each
    // canonical chunk becomes an OpenAI frame and then a wire `String` — exactly what the SSE surface
    // emits (`SseEvent::default().data(frame.to_string())` in server.rs), minus axum's `SseEvent`.
    {
        let engine = state.engine.clone();
        let keys = state.keys.clone();
        suite.bench("hotpath/edge_stream", small_openai_value, move |value| {
            block_on(async {
                let req = openai::parse_chat_request(value).expect("parse");
                let model = req.model.clone();
                let key = keys.authorize(TOKEN_BENCH, &model, NOW).expect("authorize");
                let mut stream = engine
                    .run(req, key, "bench-req".into(), NOW)
                    .await
                    .expect("run");
                let encoder = ChunkEncoder::new("chatcmpl-bench", model, NOW);
                // Mirror server.rs: one buffer reused across the stream, refilled per frame.
                let mut buf: Vec<u8> = Vec::with_capacity(256);
                let mut frames = 0u64;
                while let Some(item) = stream.next().await {
                    let chunk = item.expect("chunk");
                    if encoder.encode_into(&chunk, &mut buf) {
                        std::hint::black_box(&buf);
                        frames += 1;
                    }
                }
                frames
            })
        });
    }

    // The same non-streaming edge with full-payload event emission turned on (`include_payloads`):
    // `RequestStarted` now carries the whole request, serialized — the opt-in cost SOUL.md says the
    // operator pays knowingly. The gap to `edge_nonstream` is exactly that serialisation.
    {
        let payload_state = bench_state(true);
        let engine = payload_state.engine.clone();
        let keys = payload_state.keys.clone();
        suite.bench(
            "hotpath/edge_nonstream_payloads",
            small_openai_value,
            move |value| {
                block_on(async {
                    let req = openai::parse_chat_request(value).expect("parse");
                    let model = req.model.clone();
                    let key = keys.authorize(TOKEN_BENCH, &model, NOW).expect("authorize");
                    let stream = engine
                        .run(req, key, "bench-req".into(), NOW)
                        .await
                        .expect("run");
                    let resp = collect(stream).await.expect("collect");
                    serde_json::to_vec(&openai::response_to_openai(&resp, NOW)).expect("serialise")
                })
            },
        );
    }
}

// ---------------------------------------------------------------------------------------------
// engine/* — the engine's middle, no edge mapping
// ---------------------------------------------------------------------------------------------

fn register_engine(suite: &mut Suite, state: &llmleaf_core::AppState) {
    let engine = state.engine.clone();
    suite.bench("engine/run_collect", small_request, move |req| {
        block_on(async {
            let stream = engine
                .run(req, "bench".into(), "bench-req".into(), NOW)
                .await
                .expect("run");
            collect(stream).await.expect("collect")
        })
    });
}

// ---------------------------------------------------------------------------------------------
// collect/* — the stream fold that defines non-streaming (principle 4)
// ---------------------------------------------------------------------------------------------

fn register_collect(suite: &mut Suite) {
    suite.bench(
        "collect/small",
        || make_stream(stream_chunks(1)),
        |stream| block_on(collect(stream)).expect("collect"),
    );
    suite.bench(
        "collect/large",
        || make_stream(stream_chunks(256)),
        |stream| block_on(collect(stream)).expect("collect"),
    );
    // The heavier fold: tool-call fragments accumulated across choice indices — the `BTreeMap` paths
    // (`tools`, multi-index choices) that single-text streams never reach. This is the branch the
    // simulation harness makes its central correctness case; here it is measured for cost.
    suite.bench(
        "collect/tools_multichoice",
        || make_stream(tool_call_chunks()),
        |stream| block_on(collect(stream)).expect("collect"),
    );
    // The synchronous fold core that `collect` wraps (`collect_chunks`): the same accumulation with the
    // stream and its per-item poll removed. The batch JSONL line mapper drives this directly — it has a
    // whole upstream response in hand and never spins up a stream. The gap to `collect/small` (same
    // chunk sequence) is exactly the cost of the stream machinery the async path carries.
    suite.bench(
        "collect/chunks_sync",
        || {
            stream_chunks(1)
                .into_iter()
                .map(|c| c.expect("chunk"))
                .collect::<Vec<_>>()
        },
        collect_chunks,
    );
    // The error-termination path: a stream that emits content and then fails upstream. `collect`'s `?`
    // returns at the first `Err`, so this measures how cheaply a mid-stream failure propagates out of
    // the fold (the partial chunks already accumulated are simply dropped).
    suite.bench(
        "collect/error",
        || make_stream(erroring_chunks()),
        |stream| block_on(collect(stream)).expect_err("errors"),
    );
}

// ---------------------------------------------------------------------------------------------
// openai/* — the dialect mappings at the edge (principle 3)
// ---------------------------------------------------------------------------------------------

fn register_openai(suite: &mut Suite) {
    // In: OpenAI request JSON → canonical ChatRequest.
    suite.bench("openai/parse_small", small_openai_value, |value| {
        openai::parse_chat_request(value).expect("parse")
    });
    suite.bench("openai/parse_large", large_openai_value, |value| {
        openai::parse_chat_request(value).expect("parse")
    });

    // Out (streaming): canonical chunks → OpenAI `chat.completion.chunk` frames. One representative
    // sequence, mirroring the per-chunk map-out loop.
    {
        let encoder = ChunkEncoder::new("chatcmpl-bench", "gpt-4o", NOW);
        let chunks = representative_chunks();
        // `prepare` hands `run` a fresh buffer (excluded from the count); `run` encodes the whole
        // sequence into it, reused per frame — so this measures the encode itself, not buffer setup.
        suite.bench(
            "openai/encode_stream",
            || Vec::<u8>::with_capacity(256),
            move |mut buf| {
                let mut frames = 0u64;
                for chunk in &chunks {
                    if encoder.encode_into(chunk, &mut buf) {
                        std::hint::black_box(&buf);
                        frames += 1;
                    }
                }
                frames
            },
        );
    }

    // Out (collected): canonical ChatResponse → OpenAI `chat.completion` object.
    {
        let resp = sample_response();
        suite.bench(
            "openai/response_collect",
            || (),
            move |()| {
                serde_json::to_vec(&openai::response_to_openai(&resp, NOW)).expect("serialise")
            },
        );
    }
}

// ---------------------------------------------------------------------------------------------
// embeddings/* — the embeddings modality (principle 3 mappings + the engine middle + full edge)
// ---------------------------------------------------------------------------------------------

fn register_embeddings(suite: &mut Suite, state: &llmleaf_core::AppState) {
    // In: OpenAI embeddings request JSON → canonical EmbeddingRequest (a small batch of inputs).
    suite.bench("embeddings/parse", small_embeddings_value, |value| {
        embeddings::parse_embedding_request(value).expect("parse")
    });
    // The same mapping over a large batch — embeddings APIs routinely embed many inputs in one call.
    // The batch analogue of `openai/parse_large`: shows the inbound cost scaling with the input array.
    suite.bench("embeddings/parse_large", large_embeddings_value, |value| {
        embeddings::parse_embedding_request(value).expect("parse")
    });

    // Out: canonical EmbeddingResponse → OpenAI `{ object: "list", data }`. Two encodings: the default
    // float arrays, and base64 (the hand-rolled little-endian f32 encoder — a distinct allocation path).
    {
        let resp = sample_embedding_response();
        suite.bench(
            "embeddings/response_float",
            || (),
            move |()| {
                serde_json::to_vec(&embeddings::response_to_openai(&resp, Some("float")))
                    .expect("serialise")
            },
        );
    }
    {
        let resp = sample_embedding_response();
        suite.bench(
            "embeddings/response_base64",
            || (),
            move |()| {
                serde_json::to_vec(&embeddings::response_to_openai(&resp, Some("base64")))
                    .expect("serialise")
            },
        );
    }

    // Engine middle: route → dispatch → price → emit batch tail (Usage + Completed), no edge mapping.
    {
        let engine = state.engine.clone();
        suite.bench("engine/embed", small_embedding_request, move |req| {
            block_on(async {
                engine
                    .embed(req, "bench".into(), "bench-req".into(), NOW)
                    .await
                    .expect("embed")
            })
        });
    }

    // The whole embeddings edge for one request: dialect in → canonical → engine → dialect out — the
    // batch-modality analogue of `hotpath/edge_nonstream`.
    {
        let engine = state.engine.clone();
        let keys = state.keys.clone();
        suite.bench("hotpath/embeddings", small_embeddings_value, move |value| {
            block_on(async {
                let req = embeddings::parse_embedding_request(value).expect("parse");
                let model = req.model.clone();
                let key = keys.authorize(TOKEN_BENCH, &model, NOW).expect("authorize");
                let resp = engine
                    .embed(req, key, "bench-req".into(), NOW)
                    .await
                    .expect("embed");
                serde_json::to_vec(&embeddings::response_to_openai(&resp, None)).expect("serialise")
            })
        });
    }
}

// ---------------------------------------------------------------------------------------------
// speech/* — text-to-speech (a real stream per principle 4: parse, the collect fold, the full edge)
// ---------------------------------------------------------------------------------------------

fn register_speech(suite: &mut Suite, state: &llmleaf_core::AppState) {
    // In: OpenAI speech request JSON → canonical SpeechRequest.
    suite.bench("speech/parse", small_speech_value, |value| {
        speech::parse_speech_request(value).expect("parse")
    });

    // The per-request edge lookup that maps the requested `response_format` to the streamed response's
    // Content-Type — a match over static strings returning a `&'static str`, never an allocation. A
    // principle-5-style lookup on the speech edge (cf. `router/resolve`, `pricing/price`).
    suite.bench(
        "speech/content_type",
        || (),
        |()| std::hint::black_box(speech::content_type_for_format(Some("mp3"))),
    );

    // The audio-stream fold that defines a non-streaming speech response (the AudioChunk analogue of
    // `collect/small`): concatenate the data frames into one `AudioBytes`.
    suite.bench(
        "speech/collect",
        || make_audio_stream(audio_chunks(64)),
        |stream| block_on(collect_audio(stream)).expect("collect_audio"),
    );

    // The whole streaming-speech edge: dialect in → canonical → engine → drain the instrumented audio
    // stream relaying bytes (the server sets Content-Type from the first chunk and forwards the rest).
    {
        let engine = state.engine.clone();
        let keys = state.keys.clone();
        suite.bench("hotpath/speech_stream", small_speech_value, move |value| {
            block_on(async {
                let req = speech::parse_speech_request(value).expect("parse");
                let model = req.model.clone();
                let key = keys.authorize(TOKEN_BENCH, &model, NOW).expect("authorize");
                let mut stream = engine
                    .speech(req, key, "bench-req".into(), NOW)
                    .await
                    .expect("speech");
                let mut bytes = 0u64;
                while let Some(item) = stream.next().await {
                    if let AudioChunk::Data(b) = item.expect("chunk") {
                        bytes += b.len() as u64;
                    }
                }
                bytes
            })
        });
    }
}

// ---------------------------------------------------------------------------------------------
// transcription/* — speech-to-text (form → canonical, the transcript renderings, the full edge)
// ---------------------------------------------------------------------------------------------

fn register_transcription(suite: &mut Suite, state: &llmleaf_core::AppState) {
    // In: the decoded multipart form → canonical TranscriptionRequest (validates + parses temperature).
    suite.bench("transcription/build_request", bench_form, |form| {
        transcription::build_request(form).expect("build_request")
    });

    // Out: the transcript rendered per `response_format`. The default `json` is a one-field object; the
    // `verbose_json` shape carries language + duration too.
    {
        let resp = sample_transcription_response();
        suite.bench(
            "transcription/encode_json",
            || (),
            move |()| match transcription::encode_transcription(&resp, None) {
                transcription::TranscriptionBody::Json(v) => v,
                transcription::TranscriptionBody::Text(_) => unreachable!("json default"),
            },
        );
    }
    {
        let resp = sample_transcription_response();
        suite.bench(
            "transcription/encode_verbose",
            || (),
            move |()| match transcription::encode_transcription(&resp, Some("verbose_json")) {
                transcription::TranscriptionBody::Json(v) => v,
                transcription::TranscriptionBody::Text(_) => unreachable!("verbose_json"),
            },
        );
    }
    // The third rendering: `text` (and the `srt`/`vtt` fallback) returns the transcript verbatim — one
    // string clone, no JSON object built. The lightest of the three response-format branches.
    {
        let resp = sample_transcription_response();
        suite.bench(
            "transcription/encode_text",
            || (),
            move |()| match transcription::encode_transcription(&resp, Some("text")) {
                transcription::TranscriptionBody::Text(t) => t,
                transcription::TranscriptionBody::Json(_) => unreachable!("text format"),
            },
        );
    }

    // Engine middle: route → dispatch → price → emit batch tail, no edge mapping (cf. `engine/embed`).
    {
        let engine = state.engine.clone();
        suite.bench(
            "engine/transcribe",
            sample_transcription_request,
            move |req| {
                block_on(async {
                    engine
                        .transcribe(req, "bench".into(), "bench-req".into(), NOW)
                        .await
                        .expect("transcribe")
                })
            },
        );
    }

    // The whole transcription edge for one request: form → canonical → engine → rendered transcript.
    {
        let engine = state.engine.clone();
        let keys = state.keys.clone();
        suite.bench("hotpath/transcription", bench_form, move |form| {
            block_on(async {
                let req = transcription::build_request(form).expect("build_request");
                let model = req.model.clone();
                let format = req.response_format.clone();
                let key = keys.authorize(TOKEN_BENCH, &model, NOW).expect("authorize");
                let resp = engine
                    .transcribe(req, key, "bench-req".into(), NOW)
                    .await
                    .expect("transcribe");
                match transcription::encode_transcription(&resp, format.as_deref()) {
                    transcription::TranscriptionBody::Json(v) => std::hint::black_box(v.is_null()),
                    transcription::TranscriptionBody::Text(t) => std::hint::black_box(t.is_empty()),
                }
            })
        });
    }
}

// ---------------------------------------------------------------------------------------------
// batch/* — the asynchronous-batch envelope mappings (principle 3: one internal model, only the
// envelope is new — each item body reuses the chat mapper).
// ---------------------------------------------------------------------------------------------

fn register_batch(suite: &mut Suite) {
    // In: an inline batch-create body (an array of OpenAI chat requests tagged with custom_ids) →
    // canonical BatchSpec — each body parsed by the very same `openai::parse_chat_request`.
    suite.bench(
        "batch/parse_create",
        || batch_create_value(8),
        |value| batch::parse_batch_create(value).expect("parse_batch_create"),
    );

    // Out: the job handle → the OpenAI batch object (status + the superset of request counts).
    {
        let handle = sample_batch_handle();
        suite.bench(
            "batch/handle_to_json",
            || (),
            move |()| batch::handle_to_json(&handle),
        );
    }

    // Out: one result → a JSONL output line. The success arm re-renders the chat response with the same
    // `openai::response_to_openai` the live surface uses.
    {
        let result = sample_batch_result();
        suite.bench(
            "batch/result_line",
            || (),
            move |()| serde_json::to_vec(&batch::result_to_line(&result)).expect("serialise"),
        );
    }

    // The error arm of the same mapper: a failed item renders an `{ error: { code, message } }` line
    // with a null body, instead of re-rendering a chat response. A distinct path through `result_to_line`.
    {
        let result = sample_batch_error_result();
        suite.bench(
            "batch/result_line_error",
            || (),
            move |()| serde_json::to_vec(&batch::result_to_line(&result)).expect("serialise"),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// keys/*, router/*, health/* — the per-request lookups (principle 5)
// ---------------------------------------------------------------------------------------------

fn register_lookups(suite: &mut Suite, state: &llmleaf_core::AppState) {
    // keys/authorize — the hot-path check: a lookup and a comparison, never arithmetic.
    {
        let keys = state.keys.clone();
        suite.bench(
            "keys/authorize_hit",
            || (),
            move |()| keys.authorize(TOKEN_BENCH, "gpt-4o", NOW),
        );
    }
    {
        let keys = state.keys.clone();
        suite.bench(
            "keys/authorize_restricted",
            || (),
            move |()| keys.authorize(TOKEN_NARROW, "gpt-4o", NOW),
        );
    }
    {
        let keys = state.keys.clone();
        // The allow-list *rejection* counterpart to `authorize_restricted`: the same narrow key, a
        // model outside its allow-list. The token is warm (identity proven during warm-up), so this is
        // the cached warm path; the model gate then rejects with no key-id to clone — a lookup and a
        // `contains`, never an allocation. (`authorize_restricted` allocates the one id it returns; the
        // denial allocates nothing.)
        suite.bench(
            "keys/authorize_denied",
            || (),
            move |()| keys.authorize(TOKEN_NARROW, "gpt-3.5-turbo", NOW),
        );
    }
    {
        let keys = state.keys.clone();
        suite.bench(
            "keys/authorize_miss",
            || (),
            move |()| keys.authorize(TOKEN_UNKNOWN, "gpt-4o", NOW),
        );
    }

    // router/resolve — a logical model to its ordered fallback chain. An explicit route resolves to
    // a borrow (no allocation); an unrouted, non-prefixed model resolves to `None`.
    {
        let router = state.engine.router();
        suite.bench(
            "router/resolve_hit",
            || (),
            move |()| {
                std::hint::black_box(router.resolve("gpt-4o"));
            },
        );
        suite.bench(
            "router/resolve_miss",
            || (),
            move |()| {
                std::hint::black_box(router.resolve("not-a-model"));
            },
        );
    }

    // health — node-local cooldown: the happy path (nothing penalised) is a read lock and return;
    // `penalize` is a single map insert; `clear` on an unpenalised provider takes only the read lock.
    {
        let health = HealthTable::new();
        suite.bench(
            "health/is_down_clear",
            || (),
            move |()| std::hint::black_box(health.is_down("bench", NOW)),
        );
    }
    {
        let health = HealthTable::new();
        suite.bench(
            "health/penalize",
            || (),
            move |()| {
                health.penalize("bench", NOW, 15);
            },
        );
    }
    {
        let health = HealthTable::new();
        suite.bench(
            "health/clear_noop",
            || (),
            move |()| {
                health.clear("bench");
            },
        );
    }

    // ratelimit/admit — node-local admission (the rate-limit sibling of the health check): a borrowed-key
    // map lookup, a per-entry mutex critical section for the request token + a tokens/min floor read, and
    // a semaphore permit (an `Arc` clone, not a heap allocation). The guard is dropped each iteration so
    // the permit is released, and the buckets are sized so they never drain — measuring the steady
    // happy-path admission cost (principle 1: it must stay cheap).
    {
        use llmleaf_core::config::ProviderConfig;
        let provider = ProviderConfig {
            name: "bench".into(),
            kind: "bench".into(),
            endpoint: None,
            credential: None,
            prefix: None,
            settings: Default::default(),
            limits: Some(RateLimitConfig {
                requests_per_min: Some(1_000_000_000),
                tokens_per_min: Some(1_000_000_000),
                max_concurrent: Some(1_000_000),
            }),
            model_limits: Default::default(),
        };
        let rl = RateLimiter::new(std::slice::from_ref(&provider));
        suite.bench(
            "ratelimit/admit",
            || (),
            move |()| {
                let guard = rl.try_admit("bench", "model", tokio::time::Instant::now());
                std::hint::black_box(&guard);
            },
        );
    }
}

// ---------------------------------------------------------------------------------------------
// pricing/* — the bundled-dataset cost lookup (a get and two multiplies)
// ---------------------------------------------------------------------------------------------

fn register_pricing(suite: &mut Suite) {
    let pricing = Pricing::bundled().expect("bundled pricing parses");
    let usage = Usage {
        prompt_tokens: 1_234,
        completion_tokens: 567,
        total_tokens: 1_801,
        cost_usd: None,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
    };
    {
        let pricing = pricing.clone();
        suite.bench(
            "pricing/price_known",
            || (),
            move |()| std::hint::black_box(pricing.price("gpt-4o", usage)),
        );
    }
    {
        let pricing = pricing.clone();
        suite.bench(
            "pricing/price_unknown",
            || (),
            move |()| std::hint::black_box(pricing.price("model-not-in-dataset", usage)),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// events/* — the one output channel (an Arc + a fire-and-forget broadcast send)
// ---------------------------------------------------------------------------------------------

fn register_events(suite: &mut Suite) {
    {
        let bus = EventBus::new(1024);
        suite.bench("events/emit_no_subscriber", completed_event, move |event| {
            bus.emit(event)
        });
    }
    {
        let bus = EventBus::new(1024);
        // Hold a live subscriber so the emit path writes into a retained ring slot.
        let _rx = bus.subscribe();
        suite.bench(
            "events/emit_with_subscriber",
            completed_event,
            move |event| {
                let _keep = &_rx;
                bus.emit(event)
            },
        );
    }
}

// ---------------------------------------------------------------------------------------------
// serde/* — the optional full-payload serialisation (include_payloads = true)
// ---------------------------------------------------------------------------------------------

fn register_serde(suite: &mut Suite) {
    let req = small_request();
    suite.bench(
        "serde/request_to_value",
        || (),
        move |()| serde_json::to_value(&req).expect("serialise"),
    );
}

// ---------------------------------------------------------------------------------------------
// provider/* — a real provider's build-request + response-parse path over the fake transport
// ---------------------------------------------------------------------------------------------

/// Measure the work a *real* first-party provider does at the edge: map a canonical request out to the
/// wire, hand it to the (fake) transport, and parse the canned upstream response back to canonical —
/// the build+parse path SOUL.md pushes off the core (principle 2). The OpenAI-compatible family is the
/// representative brand (`Brand::for_kind("openai")`); the transport is a [`FakeHttpTransport`] returning
/// a fixed canned body, so there is no network and no key. Both provider and `cx` are built once and
/// captured by `move` — exactly like the engine/keys handles the other cases share — so `prepare` only
/// builds the fresh per-iter request, keeping that construction out of the measured `run`.
fn register_provider(suite: &mut Suite) {
    // Chat over SSE: build the OpenAI-wire `stream:true` body, hand it to the transport, then parse the
    // canned `text/event-stream` body incrementally and *collect* the canonical stream — the full
    // build-request → SSE-parse → fold path one streaming chat request drives through the provider.
    {
        let provider = openai_provider(FakeHttpTransport::sse(canned_chat_sse()));
        // The openai brand now defaults to the Responses wire; pin this instance back to chat
        // completions so the case keeps measuring exactly the wire its label names.
        let mut cx = bench_provider_cx();
        cx.settings
            .insert("chat_api".into(), json!("chat_completions"));
        suite.bench("provider/openai_chat_sse", small_request, move |req| {
            block_on(async {
                let stream = provider.chat(req, &cx).await.expect("chat");
                collect(stream).await.expect("collect")
            })
        });
    }

    // Chat over the Responses wire — the openai brand's *default* chat path: build the `POST /responses`
    // body, hand it to the transport, then parse the canned typed-event SSE incrementally and collect
    // the canonical stream. The Responses sibling of the chat-completions case above.
    {
        let provider = openai_provider(FakeHttpTransport::sse(canned_responses_sse()));
        let cx = bench_provider_cx();
        suite.bench("provider/openai_responses_sse", small_request, move |req| {
            block_on(async {
                let stream = provider.chat(req, &cx).await.expect("chat");
                collect(stream).await.expect("collect")
            })
        });
    }

    // Embeddings: build the OpenAI-wire embeddings body, hand it to the transport, then parse the canned
    // JSON `{ object: "list", data }` response back to a canonical `EmbeddingResponse` (a collected
    // modality, no stream) — the batch-modality analogue of the chat case above.
    {
        let provider = openai_provider(FakeHttpTransport::json(canned_embeddings_json()));
        let cx = bench_provider_cx();
        suite.bench(
            "provider/openai_embed",
            small_embedding_request,
            move |req| block_on(async { provider.embed(req, &cx).await.expect("embed") }),
        );
    }
}

/// A real OpenAI-compatible provider wired to `http` as its HTTP transport (and the no-frame fake
/// realtime transport, never exercised here). `Brand::for_kind("openai")` is the representative brand.
fn openai_provider(http: FakeHttpTransport) -> OpenAiCompatProvider {
    let transports = Transports {
        http: std::sync::Arc::new(http),
        realtime: std::sync::Arc::new(llmleaf_providers::fake::FakeRealtimeTransport::scripted(
            Vec::new(),
        )),
    };
    OpenAiCompatProvider::new(
        Brand::for_kind("openai").expect("openai brand"),
        &transports,
    )
}

/// The per-call provider context: a resolved credential and the default endpoint. Built fresh per case
/// (cheap), captured by `move`, so it is excluded from the measured `run` like every other shared handle.
fn bench_provider_cx() -> ProviderCx {
    ProviderCx {
        request_id: "bench-req".into(),
        credential: Some("bench-key".into()),
        ..Default::default()
    }
}

/// A canned OpenAI-wire SSE chat response: an opening frame (id/model), two content deltas, a tool-call
/// delta, the `finish_reason` frame, the `include_usage` tail, and the `[DONE]` sentinel — the
/// representative shape `openai_sse_to_stream` parses (cf. the inline `representative_chunks`).
fn canned_chat_sse() -> Bytes {
    let frames = [
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "Hello" } }]
        }),
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": { "content": ", world!" } }]
        }),
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"NYC\"}" }
                }] }
            }]
        }),
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "tool_calls" }]
        }),
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o",
            "choices": [],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        }),
    ];
    let mut body = String::new();
    for frame in &frames {
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(frame).expect("frame serialises"));
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    Bytes::from(body)
}

/// A canned OpenAI Responses SSE stream: `response.created`, two `output_text` deltas, a
/// `function_call` item with an arguments delta, and the terminal `response.completed` carrying usage —
/// the representative typed-event shape `openai_responses_sse_to_stream` parses (the Responses mirror
/// of [`canned_chat_sse`]; no `[DONE]` sentinel on this dialect).
fn canned_responses_sse() -> Bytes {
    let frames = [
        json!({
            "type": "response.created",
            "response": { "id": "resp-bench", "model": "gpt-4o", "status": "in_progress" }
        }),
        json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello" }),
        json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": ", world!" }),
        json!({
            "type": "response.output_item.added",
            "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "" }
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": "{\"city\":\"NYC\"}"
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp-bench",
                "model": "gpt-4o",
                "status": "completed",
                "usage": { "input_tokens": 10, "output_tokens": 5, "total_tokens": 15 }
            }
        }),
    ];
    let mut body = String::new();
    for frame in &frames {
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(frame).expect("frame serialises"));
        body.push_str("\n\n");
    }
    Bytes::from(body)
}

/// A canned OpenAI-wire embeddings response: an `{ object: "list", data: [ … ], usage }` body with two
/// short float vectors — the shape `openai_to_embeddings` parses back to a canonical `EmbeddingResponse`.
fn canned_embeddings_json() -> Value {
    let data: Vec<Value> = (0..2)
        .map(|i| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": bench_vector(),
            })
        })
        .collect();
    json!({
        "object": "list",
        "model": "text-embedding-3-small",
        "data": data,
        "usage": { "prompt_tokens": 8, "total_tokens": 8 }
    })
}

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

const CONFIG: &str = r#"
[server]
listen = "127.0.0.1:0"
event_buffer = 1024

[[providers]]
name = "bench"
kind = "bench"

[[routes]]
model = "gpt-4o"
targets = [{ provider = "bench", model = "gpt-4o" }]

[[keys]]
id = "bench"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
name = "bench"

[[keys]]
id = "narrow"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
name = "narrow"
allowed_models = ["gpt-4o"]
"#;

/// A minimal canonical request.
fn small_request() -> ChatRequest {
    ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message::text(Role::User, "Hello, world!")],
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: Vec::new(),
        stream: false,
        tools: Vec::new(),
        tool_choice: None,
        thinking: None,
        extra: Default::default(),
    }
}

/// A minimal OpenAI request body.
fn small_openai_value() -> Value {
    json!({
        "model": "gpt-4o",
        "messages": [{ "role": "user", "content": "Hello, world!" }]
    })
}

/// A chunky OpenAI request body: a multi-turn history, array content, tools, and unknown fields that
/// must ride through to `extra` — the realistic upper end of the inbound mapping's work.
fn large_openai_value() -> Value {
    let mut messages = vec![json!({ "role": "system", "content": "You are a helpful assistant." })];
    for i in 0..16 {
        messages.push(json!({
            "role": "user",
            "content": [{ "type": "text", "text": format!("Question {i}: please consider the following at some length.") }]
        }));
        messages.push(json!({
            "role": "assistant",
            "content": format!("Answer {i}: here is a correspondingly substantial reply to consider.")
        }));
    }
    json!({
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "stop": ["\n\n"],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
            }
        }],
        "tool_choice": "auto",
        "seed": 42,
        "logit_bias": { "50256": -100 }
    })
}

/// A canonical chunk sequence: a `Start`, `n_content` content deltas, a `Finish`, and a `Usage`.
fn stream_chunks(n_content: usize) -> Vec<Result<StreamChunk, ModelError>> {
    let mut chunks: Vec<Result<StreamChunk, ModelError>> = Vec::with_capacity(n_content + 3);
    chunks.push(Ok(StreamChunk::Start {
        id: "bench-1".into(),
        model: "gpt-4o".into(),
    }));
    for i in 0..n_content {
        chunks.push(Ok(StreamChunk::Content {
            index: 0,
            delta: format!("token{i} "),
        }));
    }
    chunks.push(Ok(StreamChunk::Finish {
        index: 0,
        reason: FinishReason::Stop,
    }));
    chunks.push(Ok(StreamChunk::Usage(Usage {
        prompt_tokens: 10,
        completion_tokens: n_content as u64,
        total_tokens: 10 + n_content as u64,
        cost_usd: None,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
    })));
    chunks
}

/// A stream that drives collect()'s tool-call and multi-choice folding: two choice indices, each with
/// text and tool-call fragments (the args split across two deltas, as a real provider streams them).
fn tool_call_chunks() -> Vec<Result<StreamChunk, ModelError>> {
    vec![
        Ok(StreamChunk::Start {
            id: "bench-1".into(),
            model: "gpt-4o".into(),
        }),
        Ok(StreamChunk::Content {
            index: 0,
            delta: "Let me check the weather.".into(),
        }),
        Ok(StreamChunk::ToolCall {
            index: 0,
            call: ToolCallDelta {
                index: 0,
                id: Some("call_1".into()),
                name: Some("get_weather".into()),
                arguments: Some("{\"ci".into()),
            },
        }),
        Ok(StreamChunk::ToolCall {
            index: 0,
            call: ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments: Some("ty\":\"NYC\"}".into()),
            },
        }),
        Ok(StreamChunk::Content {
            index: 1,
            delta: "And the time too.".into(),
        }),
        Ok(StreamChunk::ToolCall {
            index: 1,
            call: ToolCallDelta {
                index: 0,
                id: Some("call_2".into()),
                name: Some("get_time".into()),
                arguments: Some("{\"tz\":\"UTC\"}".into()),
            },
        }),
        Ok(StreamChunk::Finish {
            index: 0,
            reason: FinishReason::ToolCalls,
        }),
        Ok(StreamChunk::Finish {
            index: 1,
            reason: FinishReason::ToolCalls,
        }),
        Ok(StreamChunk::Usage(Usage {
            prompt_tokens: 12,
            completion_tokens: 8,
            total_tokens: 20,
            cost_usd: None,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        })),
    ]
}

/// A stream that fails partway through: a `Start`, a content delta, then an upstream error — the
/// input that drives `collect`'s `?` early-return (the `collect/error` case). The error's message is
/// built here in `prepare`, so the measured `run` is charged only for propagating the failure.
fn erroring_chunks() -> Vec<Result<StreamChunk, ModelError>> {
    vec![
        Ok(StreamChunk::Start {
            id: "bench-1".into(),
            model: "gpt-4o".into(),
        }),
        Ok(StreamChunk::Content {
            index: 0,
            delta: "partial answer before the failure".into(),
        }),
        Err(ModelError::Upstream {
            status: 503,
            message: "upstream unavailable".into(),
        }),
    ]
}

fn make_stream(chunks: Vec<Result<StreamChunk, ModelError>>) -> ResponseStream {
    Box::pin(futures::stream::iter(chunks))
}

/// One chunk of every variant — the inputs the outbound encoder must handle.
fn representative_chunks() -> Vec<StreamChunk> {
    vec![
        StreamChunk::Start {
            id: "bench-1".into(),
            model: "gpt-4o".into(),
        },
        StreamChunk::Content {
            index: 0,
            delta: "Hello".into(),
        },
        StreamChunk::Content {
            index: 0,
            delta: ", world".into(),
        },
        StreamChunk::ToolCall {
            index: 0,
            call: ToolCallDelta {
                index: 0,
                id: Some("call_1".into()),
                name: Some("get_weather".into()),
                arguments: Some("{\"city\":\"NYC\"}".into()),
            },
        },
        StreamChunk::Usage(Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cost_usd: Some(0.001),
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        }),
        StreamChunk::Finish {
            index: 0,
            reason: FinishReason::Stop,
        },
    ]
}

/// A collected response with text and a tool call — the input to the collect-mode outbound mapping.
fn sample_response() -> ChatResponse {
    ChatResponse {
        id: "chatcmpl-bench".into(),
        model: "gpt-4o".into(),
        choices: vec![Choice {
            index: 0,
            text: "Hello, world! This is a representative assistant reply.".into(),
            thinking: vec![],
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\"}".into(),
            }],
            finish_reason: Some(FinishReason::ToolCalls),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 8,
            total_tokens: 18,
            cost_usd: Some(0.0012),
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        },
    }
}

fn completed_event() -> Event {
    Event::RequestCompleted {
        id: "bench-req".into(),
        finish: Some(FinishReason::Stop),
    }
}

// --- embeddings fixtures ---

/// A short, fixed embedding vector — the per-input payload the provider returns and the edge maps out.
fn bench_vector() -> Vec<f32> {
    (0..16).map(|i| i as f32 * 0.0625 - 0.5).collect()
}

/// A canonical embeddings request: a small batch of inputs over the routed model.
fn small_embedding_request() -> EmbeddingRequest {
    EmbeddingRequest {
        model: "gpt-4o".into(),
        input: vec!["Hello, world!".into(), "A second input to embed.".into()],
        dimensions: None,
        encoding_format: None,
        extra: Default::default(),
    }
}

/// A minimal OpenAI embeddings request body (array input — the batch shape).
fn small_embeddings_value() -> Value {
    json!({
        "model": "gpt-4o",
        "input": ["Hello, world!", "A second input to embed."]
    })
}

/// A large OpenAI embeddings request body: a batch of many inputs — the upper end of the inbound
/// mapping's work, where the cost scales with the input array (cf. `large_openai_value`).
fn large_embeddings_value() -> Value {
    let input: Vec<Value> = (0..64)
        .map(|i| {
            json!(format!(
                "Input number {i} to embed, of a representative sentence length."
            ))
        })
        .collect();
    json!({ "model": "gpt-4o", "input": input })
}

/// A collected embeddings response with two vectors — the input to the outbound mapping.
fn sample_embedding_response() -> EmbeddingResponse {
    EmbeddingResponse {
        model: "gpt-4o".into(),
        embeddings: vec![
            Embedding {
                index: 0,
                vector: bench_vector(),
            },
            Embedding {
                index: 1,
                vector: bench_vector(),
            },
        ],
        usage: Usage {
            prompt_tokens: 8,
            completion_tokens: 0,
            total_tokens: 8,
            cost_usd: Some(0.0001),
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        },
    }
}

// --- speech fixtures ---

/// A minimal OpenAI speech (TTS) request body.
fn small_speech_value() -> Value {
    json!({
        "model": "gpt-4o",
        "input": "Hello, world!",
        "voice": "alloy"
    })
}

/// A canonical audio stream: a `Start` carrying the MIME type, `n` data frames, a `Usage`, a `Finish`.
fn audio_chunks(n_data: usize) -> Vec<Result<AudioChunk, ModelError>> {
    let mut chunks: Vec<Result<AudioChunk, ModelError>> = Vec::with_capacity(n_data + 3);
    chunks.push(Ok(AudioChunk::Start {
        content_type: "audio/mpeg".into(),
    }));
    for _ in 0..n_data {
        chunks.push(Ok(AudioChunk::Data(Bytes::from_static(&[0u8; 64]))));
    }
    chunks.push(Ok(AudioChunk::Usage(Usage {
        prompt_tokens: 4,
        completion_tokens: 0,
        total_tokens: 4,
        cost_usd: None,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
    })));
    chunks.push(Ok(AudioChunk::Finish));
    chunks
}

fn make_audio_stream(chunks: Vec<Result<AudioChunk, ModelError>>) -> AudioStream {
    Box::pin(futures::stream::iter(chunks))
}

// --- transcription fixtures ---

/// A decoded transcription multipart form — the shape the server edge hands `build_request`.
fn bench_form() -> transcription::TranscriptionForm {
    transcription::TranscriptionForm {
        model: Some("gpt-4o".into()),
        audio: Some(Bytes::from_static(&[0u8; 256])),
        filename: Some("speech.mp3".into()),
        language: Some("en".into()),
        prompt: None,
        response_format: Some("json".into()),
        temperature: Some("0.0".into()),
    }
}

/// A canonical transcription request — the input to the engine middle (`engine/transcribe`).
fn sample_transcription_request() -> TranscriptionRequest {
    TranscriptionRequest {
        model: "gpt-4o".into(),
        audio: Bytes::from_static(&[0u8; 256]),
        filename: "speech.mp3".into(),
        language: Some("en".into()),
        prompt: None,
        response_format: Some("json".into()),
        temperature: Some(0.0),
        extra: Default::default(),
    }
}

/// A collected transcription response — the input to the outbound renderings.
fn sample_transcription_response() -> TranscriptionResponse {
    TranscriptionResponse {
        text: "Hello, world! This is a representative transcript.".into(),
        language: Some("en".into()),
        duration: Some(1.5),
        usage: Usage::default(),
    }
}

// --- batch fixtures ---

/// An inline batch-create body: `n` OpenAI chat requests, each tagged with a `custom_id`.
fn batch_create_value(n: usize) -> Value {
    let requests: Vec<Value> = (0..n)
        .map(|i| {
            json!({
                "custom_id": format!("req-{i}"),
                "body": {
                    "model": "gpt-4o",
                    "messages": [{ "role": "user", "content": format!("Question {i}?") }]
                }
            })
        })
        .collect();
    json!({ "requests": requests })
}

/// A batch job handle mid-flight — the input to the handle rendering.
fn sample_batch_handle() -> BatchHandle {
    BatchHandle {
        id: "batch_opaque_token".into(),
        status: BatchStatus::InProgress,
        counts: BatchCounts {
            total: 8,
            processing: 5,
            succeeded: 3,
            ..Default::default()
        },
        created_at: Some(NOW),
        expires_at: Some(NOW + 86_400),
        ended_at: None,
        endpoint: Some("/v1/chat/completions".into()),
        extra: Default::default(),
    }
}

/// One successful batch result — its success arm re-renders a chat response via the chat mapper.
fn sample_batch_result() -> BatchResult {
    BatchResult {
        custom_id: "req-0".into(),
        outcome: BatchOutcome::Succeeded(sample_response()),
    }
}

/// One failed batch result — the error arm renders an `{ error: { code, message } }` line with a null
/// body instead of re-rendering a response (cf. `sample_batch_result`).
fn sample_batch_error_result() -> BatchResult {
    BatchResult {
        custom_id: "req-1".into(),
        outcome: BatchOutcome::Errored {
            status: 429,
            message: "rate limited upstream".into(),
        },
    }
}
