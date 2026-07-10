//! Simulation of a **real** provider's upstream-failure handling, driven through a **seeded** fake
//! transport.
//!
//! Where [`crate::dispatch`] and [`crate::engine`] use a hand-written [`llmleaf_provider::Provider`]
//! mock to exercise the engine's routing spine, this family does the inverse: it drives the *actual*
//! first-party [`OpenAiCompatProvider`] (its real request building + response parsing) over the
//! injectable [`FakeHttpTransport`] seam, with the transport scripted by the seed to return a typical
//! upstream condition — a healthy response, a transport failure, a 429/500 status, or a malformed
//! body. The point (SOUL.md principle 8, "providers go down; llmleaf doesn't"): prove the real provider
//! **never panics** and maps every typical upstream failure onto the documented canonical
//! [`ModelError`] taxonomy, so the engine's fallback logic always sees a clean, classified error.
//!
//! As in the other families, an independent reference ([`expect`]) predicts the outcome purely from the
//! seeded fault class, and is written without consulting the provider's own classification code (which
//! lives in `llmleaf-providers::http`). The mapping it asserts:
//!   - `Healthy`       → `Ok(..)` carrying the exact canned content/usage,
//!   - `TransportFail` → `Err(ModelError::Unavailable(_))` (connect failure — fallback-eligible),
//!   - `Http429`/`Http500` → `Err(ModelError::Upstream { status, .. })` with the scripted status,
//!   - `MalformedJson` → `Err(ModelError::Mapping(_))` (a 200 whose body would not parse).
//!
//! Determinism is load-bearing: a fresh [`FakeHttpTransport`] is built per op for exactly the seeded
//! fault (no shared mutable responder state), and every canned body is a fixed `json!`/string literal,
//! so a seed replays the identical scenario forever.

use llmleaf_model::{
    ChatRequest, EmbeddingRequest, Message, ModelError, RerankDocument, RerankRequest, Role, Usage,
};
use llmleaf_provider::{Provider, ProviderCx};
use llmleaf_providers::fake::FakeHttpTransport;
use llmleaf_providers::transport::Transports;
use llmleaf_providers::{Brand, OpenAiCompatProvider};

use crate::ensure;
use crate::rng::Rng;

/// The typical upstream conditions a provider must survive, scripted into the fake transport per op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    /// The transport returns a valid canned response (SSE for chat, JSON for embeddings).
    Healthy,
    /// The transport fails before a response (a connect/read failure → `Unavailable`).
    TransportFail,
    /// HTTP 429 (rate limited) with an error body → `Upstream { status: 429, .. }`.
    Http429,
    /// HTTP 500 (server error) with an error body → `Upstream { status: 500, .. }`.
    Http500,
    /// A 200 whose body is not JSON, on a JSON endpoint → `Mapping(_)`.
    MalformedJson,
}

/// A stable, generation-ordered fault list. Every RNG pick iterates this slice, never a set, so the
/// choice never depends on iteration order (determinism is load-bearing).
const ALL_FAULTS: [Fault; 5] = [
    Fault::Healthy,
    Fault::TransportFail,
    Fault::Http429,
    Fault::Http500,
    Fault::MalformedJson,
];

/// The three modalities this family drives the real provider through. Chat takes the brand's *streaming*
/// path (the `openai` brand serves real SSE), so a healthy chat body is canned SSE; embeddings and
/// rerank take the collect-then-parse `post_json` path, so their healthy bodies are canned JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Chat,
    Embed,
    Rerank,
}

const ALL_OPS: [Op; 3] = [Op::Chat, Op::Embed, Op::Rerank];

/// Which chat wire the provider is pinned to for one op. The `openai` brand serves two chat dialects —
/// the classic `/chat/completions` SSE and the Responses API's typed-event SSE — and both must survive
/// every scripted fault identically, so the wire is one more seeded dimension. Pinned explicitly via
/// `settings.chat_api` (never left to the brand default) so the scenario stays a pure function of the
/// seed even if the brand's default wire changes again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wire {
    Completions,
    Responses,
}

const ALL_WIRES: [Wire; 2] = [Wire::Completions, Wire::Responses];

/// The canonical content + usage a healthy chat response carries — the single source of truth the canned
/// SSE body and the reference assertion both read, so they can never drift apart.
const CHAT_ID: &str = "chatcmpl-sim";
const CHAT_MODEL: &str = "sim-model";
const CHAT_CONTENT: &str = "hello from the sim";
const CHAT_USAGE: Usage = Usage {
    prompt_tokens: 11,
    completion_tokens: 7,
    total_tokens: 18,
    cost_usd: None,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
};

/// The canonical vector + usage a healthy embeddings response carries (the source of truth for both the
/// canned JSON and the reference assertion).
const EMBED_MODEL: &str = "sim-embed-model";
const EMBED_VECTOR: [f32; 4] = [0.5, -0.25, 0.125, 0.0];
const EMBED_USAGE: Usage = Usage {
    prompt_tokens: 4,
    completion_tokens: 0,
    total_tokens: 4,
    cost_usd: None,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
};

/// The canonical model + ranked result + usage a healthy rerank response carries (the source of truth
/// for both the canned JSON and the reference assertion). Jina/OpenRouter report `usage.total_tokens`,
/// which `openai_to_rerank` relays into *both* `prompt_tokens` and `total_tokens` (the billed count —
/// principle 5: relayed, never computed), so `RERANK_USAGE` mirrors that below. The score is chosen
/// exactly representable in `f32` so it round-trips through the canned JSON for the equality check.
const RERANK_MODEL: &str = "sim-rerank-model";
const RERANK_INDEX: u32 = 0;
const RERANK_SCORE: f32 = 0.5;
const RERANK_DOCUMENT: &str = "the ranked document";
const RERANK_USAGE: Usage = Usage {
    prompt_tokens: 6,
    completion_tokens: 0,
    total_tokens: 6,
    cost_usd: None,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
};

/// A canned, well-formed OpenAI-wire SSE chat stream: a start frame (id/model), one content delta, a
/// finish frame, a terminal `include_usage` frame, then `[DONE]`. Parsed by the real provider's
/// [`llmleaf_providers::openai_wire::openai_sse_to_stream`] exactly like a live upstream's SSE.
fn healthy_chat_sse() -> String {
    let start = serde_json::json!({
        "id": CHAT_ID,
        "model": CHAT_MODEL,
        "choices": [{ "index": 0, "delta": { "role": "assistant" } }],
    });
    let content = serde_json::json!({
        "choices": [{ "index": 0, "delta": { "content": CHAT_CONTENT } }],
    });
    let finish = serde_json::json!({
        "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
    });
    let usage = serde_json::json!({
        "choices": [],
        "usage": {
            "prompt_tokens": CHAT_USAGE.prompt_tokens,
            "completion_tokens": CHAT_USAGE.completion_tokens,
            "total_tokens": CHAT_USAGE.total_tokens,
        },
    });
    format!(
        "data: {start}\n\ndata: {content}\n\ndata: {finish}\n\ndata: {usage}\n\ndata: [DONE]\n\n"
    )
}

/// A canned, well-formed Responses-API typed-event SSE stream carrying the very same id/model/content/
/// usage truths as [`healthy_chat_sse`]: `response.created`, one `output_text` delta, then the terminal
/// `response.completed` with usage (no `[DONE]` on this dialect). Parsed by the real provider's
/// [`llmleaf_providers::openai_responses_wire::openai_responses_sse_to_stream`].
fn healthy_responses_sse() -> String {
    let created = serde_json::json!({
        "type": "response.created",
        "response": { "id": CHAT_ID, "model": CHAT_MODEL, "status": "in_progress" },
    });
    let delta = serde_json::json!({
        "type": "response.output_text.delta",
        "item_id": "msg_sim",
        "delta": CHAT_CONTENT,
    });
    let completed = serde_json::json!({
        "type": "response.completed",
        "response": {
            "id": CHAT_ID,
            "model": CHAT_MODEL,
            "status": "completed",
            "usage": {
                "input_tokens": CHAT_USAGE.prompt_tokens,
                "output_tokens": CHAT_USAGE.completion_tokens,
                "total_tokens": CHAT_USAGE.total_tokens,
            },
        },
    });
    format!("data: {created}\n\ndata: {delta}\n\ndata: {completed}\n\n")
}

/// A canned, well-formed OpenAI-wire embeddings response (one float vector + usage), parsed by the real
/// provider's [`llmleaf_providers::openai_wire::openai_to_embeddings`].
fn healthy_embed_json() -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "model": EMBED_MODEL,
        "data": [{ "object": "embedding", "index": 0, "embedding": EMBED_VECTOR.to_vec() }],
        "usage": {
            "prompt_tokens": EMBED_USAGE.prompt_tokens,
            "total_tokens": EMBED_USAGE.total_tokens,
        },
    })
}

/// A canned, well-formed Jina/OpenRouter-wire rerank response (one ranked result with its echoed
/// document + usage), parsed by the real provider's
/// [`llmleaf_providers::openai_wire::openai_to_rerank`] exactly like a live upstream's JSON.
fn healthy_rerank_json() -> serde_json::Value {
    serde_json::json!({
        "model": RERANK_MODEL,
        "results": [{
            "index": RERANK_INDEX,
            "relevance_score": RERANK_SCORE,
            "document": RERANK_DOCUMENT,
        }],
        "usage": { "total_tokens": RERANK_USAGE.total_tokens },
    })
}

/// Build a fresh [`FakeHttpTransport`] scripted for exactly `fault` on `op` over `wire`. A fresh
/// transport per op (no shared mutable responder) keeps behaviour a pure function of the seed.
fn transport_for(op: Op, wire: Wire, fault: Fault) -> FakeHttpTransport {
    match fault {
        Fault::Healthy => match op {
            Op::Chat => match wire {
                Wire::Completions => FakeHttpTransport::sse(healthy_chat_sse()),
                Wire::Responses => FakeHttpTransport::sse(healthy_responses_sse()),
            },
            Op::Embed => FakeHttpTransport::json(healthy_embed_json()),
            Op::Rerank => FakeHttpTransport::json(healthy_rerank_json()),
        },
        Fault::TransportFail => {
            FakeHttpTransport::error(ModelError::Unavailable("connection refused".into()))
        }
        Fault::Http429 => {
            FakeHttpTransport::status(429, br#"{"error":{"message":"rate limited"}}"#.to_vec())
        }
        Fault::Http500 => {
            FakeHttpTransport::status(500, br#"{"error":{"message":"server error"}}"#.to_vec())
        }
        // A 200 with a body the JSON/SSE parser cannot make sense of. For embeddings this hits
        // `post_json`'s JSON parse (→ Mapping); for chat the SSE frame is non-JSON after `data:`
        // (→ Mapping from `openai_sse_to_stream` and `openai_responses_sse_to_stream` alike — the
        // classification must not depend on the wire).
        Fault::MalformedJson => match op {
            Op::Chat => {
                FakeHttpTransport::sse("data: this is not json at all\n\n".as_bytes().to_vec())
            }
            Op::Embed | Op::Rerank => {
                FakeHttpTransport::status(200, b"this is not json at all".to_vec())
            }
        },
    }
}

/// What the reference expects for an `(op, fault)` pair — the spec-derived classification, written
/// independently of the provider's own `http`/`openai_wire` code. `Other` carries the offending error's
/// display so an unexpected canonical variant (`Unsupported`/`Timeout`/`Canceled`, none of which the
/// scripted faults can produce) fails the equality check loudly with the seed instead of being coerced
/// into a known bucket.
#[derive(Debug, PartialEq, Eq)]
enum Expect {
    OkChat,
    OkEmbed,
    OkRerank,
    Unavailable,
    Upstream(u16),
    Mapping,
    Other(String),
}

fn expect(op: Op, fault: Fault) -> Expect {
    match fault {
        Fault::Healthy => match op {
            Op::Chat => Expect::OkChat,
            Op::Embed => Expect::OkEmbed,
            Op::Rerank => Expect::OkRerank,
        },
        Fault::TransportFail => Expect::Unavailable,
        Fault::Http429 => Expect::Upstream(429),
        Fault::Http500 => Expect::Upstream(500),
        Fault::MalformedJson => Expect::Mapping,
    }
}

/// A `ProviderCx` with a credential and endpoint, so the real provider builds a fully-formed request
/// (the fake transport ignores the URL, but the provider's auth/URL building still runs). The chat wire
/// is pinned explicitly per scenario pick — never left to the brand default (see [`Wire`]).
fn cx(wire: Wire) -> ProviderCx {
    let chat_api = match wire {
        Wire::Completions => "chat_completions",
        Wire::Responses => "responses",
    };
    let mut settings = serde_json::Map::new();
    settings.insert("chat_api".into(), serde_json::json!(chat_api));
    // The `openai` brand does not serve rerank by default; opt this instance in so the healthy rerank
    // op reaches the real `post_json` parse path (chat/embed ignore this setting).
    settings.insert("rerank_api".into(), serde_json::json!(true));
    ProviderCx {
        credential: Some("k".into()),
        endpoint: Some("https://example.test".into()),
        settings,
        ..Default::default()
    }
}

/// Drive one op against `provider` and reduce its result to the comparable [`Expect`] shape, asserting
/// the *content* of a healthy response matches the canned source of truth. Returns the seed-stamped
/// report on any divergence. A `Result` (never a panic) here is itself the property under test: the
/// real provider must never panic on any scripted upstream condition.
async fn drive(
    provider: &OpenAiCompatProvider,
    op: Op,
    wire: Wire,
    seed: u64,
    step: u64,
) -> Result<Expect, String> {
    match op {
        Op::Chat => {
            let req = ChatRequest {
                model: "logical-model".into(),
                messages: vec![Message::text(Role::User, "ping")],
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: Vec::new(),
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                thinking: None,
                extra: Default::default(),
            };
            match provider.chat(req, &cx(wire)).await {
                // `chat` may fail at the transport stage (TransportFail / non-2xx Upstream) or only once
                // the SSE stream is consumed (a malformed frame). Collecting the stream surfaces the
                // latter, so this exercises the whole parse path (principle 4).
                Ok(stream) => match llmleaf_model::collect(stream).await {
                    Ok(resp) => {
                        let text = resp.choices.first().map(|c| c.text.as_str()).unwrap_or("");
                        ensure!(
                            resp.id == CHAT_ID && resp.model == CHAT_MODEL,
                            "seed={seed} step={step}: healthy chat id/model mismatch: id={:?} model={:?}",
                            resp.id,
                            resp.model
                        );
                        ensure!(
                            text == CHAT_CONTENT,
                            "seed={seed} step={step}: healthy chat content mismatch: got {text:?}, want {CHAT_CONTENT:?}"
                        );
                        ensure!(
                            resp.usage == CHAT_USAGE,
                            "seed={seed} step={step}: healthy chat usage mismatch: got {:?}, want {CHAT_USAGE:?}",
                            resp.usage
                        );
                        Ok(Expect::OkChat)
                    }
                    Err(e) => Ok(classify(e)),
                },
                Err(e) => Ok(classify(e)),
            }
        }
        Op::Embed => {
            let req = EmbeddingRequest {
                model: "logical-model".into(),
                input: vec!["ping".into()],
                dimensions: None,
                encoding_format: None,
                extra: Default::default(),
            };
            match provider.embed(req, &cx(wire)).await {
                Ok(resp) => {
                    ensure!(
                        resp.model == EMBED_MODEL,
                        "seed={seed} step={step}: healthy embed model mismatch: got {:?}",
                        resp.model
                    );
                    ensure!(
                        resp.embeddings.len() == 1 && resp.embeddings[0].vector == EMBED_VECTOR,
                        "seed={seed} step={step}: healthy embed vector mismatch: got {:?}",
                        resp.embeddings
                    );
                    ensure!(
                        resp.usage == EMBED_USAGE,
                        "seed={seed} step={step}: healthy embed usage mismatch: got {:?}, want {EMBED_USAGE:?}",
                        resp.usage
                    );
                    Ok(Expect::OkEmbed)
                }
                Err(e) => Ok(classify(e)),
            }
        }
        Op::Rerank => {
            let req = RerankRequest {
                model: "logical-model".into(),
                query: "ping".into(),
                documents: vec![RerankDocument::Text("doc".into())],
                top_n: None,
                return_documents: None,
                extra: Default::default(),
            };
            match provider.rerank(req, &cx(wire)).await {
                Ok(resp) => {
                    ensure!(
                        resp.model == RERANK_MODEL,
                        "seed={seed} step={step}: healthy rerank model mismatch: got {:?}",
                        resp.model
                    );
                    ensure!(
                        resp.results.len() == 1
                            && resp.results[0].index == RERANK_INDEX
                            && resp.results[0].relevance_score == RERANK_SCORE
                            && resp.results[0].document == Some(serde_json::json!(RERANK_DOCUMENT)),
                        "seed={seed} step={step}: healthy rerank results mismatch: got {:?}",
                        resp.results
                    );
                    ensure!(
                        resp.usage == RERANK_USAGE,
                        "seed={seed} step={step}: healthy rerank usage mismatch: got {:?}, want {RERANK_USAGE:?}",
                        resp.usage
                    );
                    Ok(Expect::OkRerank)
                }
                Err(e) => Ok(classify(e)),
            }
        }
    }
}

/// Reduce a real [`ModelError`] to the comparable [`Expect`] error shape (the canonical taxonomy under
/// test). `Unsupported`/`Timeout`/`Canceled` cannot arise from the scripted faults; if one ever does it
/// is mapped to `Expect::Other`, which the caller's equality check then reports as a seed-stamped finding.
fn classify(err: ModelError) -> Expect {
    match err {
        ModelError::Unavailable(_) => Expect::Unavailable,
        ModelError::Upstream { status, .. } => Expect::Upstream(status),
        ModelError::Mapping(_) => Expect::Mapping,
        // Surface anything unexpected as a distinct shape so the equality check fails loudly with the
        // seed rather than silently coercing it into a known bucket.
        other => Expect::Other(other.to_string()),
    }
}

/// Run one full provider-resilience scenario for `seed`. Over ~40-90 ops it picks a `(modality, fault)`
/// pair from the seeded RNG, builds a real `OpenAiCompatProvider` wired to a fresh fake transport for
/// exactly that fault, drives the op, and asserts the outcome equals the independent reference.
pub async fn run_scenario(seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);

    let nops = rng.range(40, 90);
    for step in 0..nops {
        let op = *rng.pick(&ALL_OPS);
        let wire = *rng.pick(&ALL_WIRES);
        let fault = *rng.pick(&ALL_FAULTS);

        // A real provider, freshly built with a transport scripted for exactly this seed's fault — so
        // behaviour is a pure function of the seed (no shared mutable responder across ops).
        let transports = Transports {
            http: std::sync::Arc::new(transport_for(op, wire, fault)),
            ..Transports::fake()
        };
        let brand = Brand::for_kind("openai").expect("openai is a known brand");
        let provider = OpenAiCompatProvider::new(brand, &transports);

        let actual = drive(&provider, op, wire, seed, step).await?;
        let expected = expect(op, fault);

        ensure!(
            actual == expected,
            "seed={seed} step={step}: op={op:?} wire={wire:?} fault={fault:?}\n  actual:   {actual:?}\n  expected: {expected:?}"
        );
    }

    Ok(())
}
