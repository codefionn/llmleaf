//! Integration-style round-trip tests for the real provider implementations over the fake transport.
//!
//! Each case constructs a provider with a [`Transports`] whose HTTP (or realtime) side is a
//! [`crate::fake`] returning a canned upstream response or a seeded error, then calls the actual
//! [`Provider`] method. This exercises the genuine request-build + response-parse path — the same
//! [`request_to_openai`]/[`openai_sse_to_stream`]/[`request_to_anthropic`]/[`anthropic_to_chunks`] code
//! that runs against a live upstream — without a network or live credentials. The unit tests already
//! cover the pure mappers in isolation; these prove the wiring *through the transport seam* and the
//! shared failure taxonomy ([`crate::http`]) classify exactly as documented.

use std::sync::Arc;

use llmleaf_model::{
    collect, ChatRequest, EmbeddingRequest, FinishReason, Message, ModelError, Role,
};
use llmleaf_provider::Provider;
use llmleaf_provider::{ProviderCx, RealtimeParams, RealtimePeer, RealtimeWire};
use serde_json::json;
use tokio::sync::mpsc;

use crate::fake::{FakeHttpTransport, FakeRealtimeTransport, FakeResponse};
use crate::transport::{HttpBody, HttpRequest, RealtimeTransport, Transports};
use crate::{AnthropicProvider, OpenAiCompatProvider};

/// Build a [`Transports`] whose HTTP side is `http` and whose realtime side is a no-op (the chat /
/// embeddings cases never touch realtime). Mirrors the task's "inject a specific HTTP response" recipe.
fn http_transports(http: FakeHttpTransport) -> Transports {
    Transports {
        http: Arc::new(http),
        realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
    }
}

/// Build a [`Transports`] whose realtime side is `realtime` and whose HTTP side is the no-op fake (the
/// realtime cases never touch HTTP).
fn realtime_transports(realtime: FakeRealtimeTransport) -> Transports {
    Transports {
        http: Arc::new(FakeHttpTransport::json(json!({}))),
        realtime: Arc::new(realtime),
    }
}

/// A populated `ProviderCx` — every provider edge reads `credential`/`endpoint`, so supply both. The
/// endpoint is irrelevant (the fake ignores the URL) but exercises the real URL-building path.
fn cx() -> ProviderCx {
    ProviderCx {
        credential: Some("test-key".into()),
        endpoint: Some("https://example.test".into()),
        ..Default::default()
    }
}

fn user_chat(model: &str, text: &str) -> ChatRequest {
    ChatRequest {
        model: model.into(),
        messages: vec![Message::text(Role::User, text)],
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: vec![],
        stream: false,
        tools: vec![],
        tool_choice: None,
        thinking: None,
        extra: Default::default(),
    }
}

// ---------------------------------------------------------------------------------------------
// 1. Chat (streaming SSE) success — openai-compat
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn openai_chat_sse_roundtrips_to_canonical_stream() {
    // Two `chat.completion.chunk` frames (content split across them) plus the terminal include_usage
    // frame and the `[DONE]` sentinel — the exact shape OpenAI streams when `stream_options.include_usage`
    // is set (which `request_to_openai` requests for a streaming brand).
    let body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\", world\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,\"total_tokens\":12}}\n\n",
        "data: [DONE]\n\n",
    );
    let transports = http_transports(FakeHttpTransport::sse(body));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();

    // The stock `openai` brand now defaults to the Responses API, so pin it back to chat completions to
    // exercise the classic `chat.completion.chunk` SSE round-trip here (the Responses SSE is covered by
    // `openai_chat_posts_responses_shape_and_parses_responses_sse`).
    let cx = ProviderCx {
        settings: serde_json::from_value(json!({ "chat_api": "chat_completions" })).unwrap(),
        ..cx()
    };
    let stream = provider
        .chat(user_chat("gpt-4o", "hi"), &cx)
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    // The two content deltas concatenate; id/model come from the opening frame.
    assert_eq!(resp.id, "chatcmpl-1");
    assert_eq!(resp.model, "gpt-4o");
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].text, "Hello, world");
    assert_eq!(resp.choices[0].finish_reason, Some(FinishReason::Stop));
    // Usage is relayed verbatim from the terminal frame (principle 5: the core never counts).
    assert_eq!(resp.usage.prompt_tokens, 9);
    assert_eq!(resp.usage.completion_tokens, 3);
    assert_eq!(resp.usage.total_tokens, 12);
}

// ---------------------------------------------------------------------------------------------
// 2. Embeddings success — openai-compat
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn openai_embeddings_roundtrips_to_canonical_vectors() {
    let canned = json!({
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            { "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] },
            { "object": "embedding", "index": 1, "embedding": [-0.4, 0.5] }
        ],
        "usage": { "prompt_tokens": 7, "total_tokens": 7 }
    });
    let transports = http_transports(FakeHttpTransport::json(canned));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();

    let req = EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: vec!["a".into(), "b".into()],
        dimensions: None,
        encoding_format: None,
        extra: Default::default(),
    };
    let resp = provider.embed(req, &cx()).await.expect("embed succeeds");

    assert_eq!(resp.model, "text-embedding-3-small");
    assert_eq!(resp.embeddings.len(), 2);
    assert_eq!(resp.embeddings[0].index, 0);
    assert_eq!(resp.embeddings[0].vector, vec![0.1_f32, 0.2, 0.3]);
    assert_eq!(resp.embeddings[1].index, 1);
    assert_eq!(resp.embeddings[1].vector, vec![-0.4_f32, 0.5]);
    assert_eq!(resp.usage.prompt_tokens, 7);
    assert_eq!(resp.usage.total_tokens, 7);
}

// ---------------------------------------------------------------------------------------------
// 2b. Responses API (openai brand default) — endpoint + body shape + SSE parse, and the per-request
//     downgrade to /chat/completions when the request carries a chat-only field.
// ---------------------------------------------------------------------------------------------

/// The JSON body a captured [`HttpRequest`] carried (panics if it was not a JSON body).
fn json_body(req: &HttpRequest) -> serde_json::Value {
    match &req.body {
        HttpBody::Json(v) => v.clone(),
        other => panic!("expected a JSON body, got {other:?}"),
    }
}

#[tokio::test]
async fn openai_chat_posts_responses_shape_and_parses_responses_sse() {
    use std::sync::Mutex;

    // A realistic Responses SSE: created → reasoning delta → two output_text deltas → completed+usage.
    // The `event:` lines are decoration (`sse_payloads` reads only the self-describing `data:` JSON).
    let sse = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\"}}\n\n",
        "event: response.reasoning_text.delta\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"mull\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\", world\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":9,\"output_tokens\":3,\"total_tokens\":12}}}\n\n",
    );
    let captured: Arc<Mutex<Option<HttpRequest>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        *cap.lock().unwrap() = Some(req.clone());
        Ok(FakeResponse::ok_bytes("text/event-stream", sse))
    });
    let provider = OpenAiCompatProvider::for_kind("openai", &http_transports(http)).unwrap();

    let stream = provider
        .chat(user_chat("gpt-5", "hi"), &cx())
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    // The stock openai brand posted to `/responses` with a Responses-shaped body (stateless defaults on).
    let req = captured
        .lock()
        .unwrap()
        .clone()
        .expect("a request was sent");
    assert!(req.url.ends_with("/responses"), "posted to {}", req.url);
    let body = json_body(&req);
    assert!(
        body.get("input").is_some(),
        "Responses body has `input`, not `messages`"
    );
    assert!(body.get("messages").is_none());
    assert_eq!(body["store"], false); // llmleaf never lets the upstream store the payload
    assert_eq!(
        body["include"],
        serde_json::json!(["reasoning.encrypted_content"])
    );
    assert_eq!(body["stream"], true);

    // The Responses SSE parsed into the canonical stream: id/model, reasoning, the two content deltas.
    assert_eq!(resp.id, "resp_1");
    assert_eq!(resp.model, "gpt-5");
    assert_eq!(resp.choices[0].text, "Hello, world");
    assert!(matches!(
        resp.choices[0].thinking.first(),
        Some(llmleaf_model::ContentPart::Thinking { thinking, .. }) if thinking == "mull"
    ));
    assert_eq!(resp.usage.prompt_tokens, 9);
    assert_eq!(resp.usage.total_tokens, 12);
}

#[tokio::test]
async fn openai_chat_with_stop_downgrades_to_chat_completions() {
    use std::sync::Mutex;

    // A canonical `stop` has no Responses representation, so the stock openai brand serves this request
    // over `/chat/completions` instead (the documented per-request downgrade).
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let captured: Arc<Mutex<Option<HttpRequest>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        *cap.lock().unwrap() = Some(req.clone());
        Ok(FakeResponse::ok_bytes("text/event-stream", sse))
    });
    let provider = OpenAiCompatProvider::for_kind("openai", &http_transports(http)).unwrap();

    let mut req = user_chat("gpt-5", "hi");
    req.stop = vec!["\n".into()];
    let stream = provider
        .chat(req, &cx())
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    let sent = captured
        .lock()
        .unwrap()
        .clone()
        .expect("a request was sent");
    assert!(
        sent.url.ends_with("/chat/completions"),
        "chat-only request must downgrade to chat completions, posted to {}",
        sent.url
    );
    let body = json_body(&sent);
    assert!(
        body.get("messages").is_some(),
        "chat body carries `messages`"
    );
    assert_eq!(body["stop"], serde_json::json!(["\n"]));
    assert_eq!(resp.choices[0].text, "hi");
}

#[tokio::test]
async fn openrouter_responses_opt_in_posts_beta_endpoint_and_replays_signed_reasoning() {
    use llmleaf_model::ContentPart;
    use std::sync::Mutex;

    // OpenRouter's beta `POST /responses`: same typed-event SSE, plus the OpenRouter extras — a
    // trailing `[DONE]` sentinel, a per-item reasoning `signature`, and `usage.cost`.
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"gen-1\",\"model\":\"anthropic/claude-4.5-sonnet\"}}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"ponder\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"signature\":\"SIG2\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"sunny\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"gen-1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"total_tokens\":10,\"cost\":0.00021}}}\n\n",
        "data: [DONE]\n\n",
    );
    let captured: Arc<Mutex<Option<HttpRequest>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        *cap.lock().unwrap() = Some(req.clone());
        Ok(FakeResponse::ok_bytes("text/event-stream", sse))
    });
    let provider = OpenAiCompatProvider::for_kind("openrouter", &http_transports(http)).unwrap();

    // Opt the brand into its beta Responses endpoint; no endpoint override, so the brand default URL
    // builds the documented beta path.
    let cx = ProviderCx {
        credential: Some("test-key".into()),
        settings: serde_json::from_value(json!({ "chat_api": "responses" })).unwrap(),
        ..Default::default()
    };

    // A second turn replaying a prior signed thinking block — the OpenRouter-flavor request shape.
    let mut req = user_chat("anthropic/claude-4.5-sonnet", "weather?");
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Thinking {
                thinking: "prior reasoning".into(),
                signature: Some("SIG1".into()),
            },
            ContentPart::Text {
                text: "checking".into(),
            },
        ],
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
    });
    let stream = provider
        .chat(req, &cx)
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    // The documented beta endpoint, verbatim.
    let sent = captured
        .lock()
        .unwrap()
        .clone()
        .expect("a request was sent");
    assert_eq!(sent.url, "https://openrouter.ai/api/v1/responses");
    // The prior signed thinking replays as an OpenRouter-flavor reasoning item (content + signature);
    // the stock-OpenAI flavor would have dropped it.
    let body = json_body(&sent);
    let input = body["input"].as_array().unwrap();
    assert_eq!(
        input[1],
        json!({
            "type": "reasoning",
            "summary": [],
            "content": [{ "type": "reasoning_text", "text": "prior reasoning" }],
            "signature": "SIG1",
        })
    );
    // The OpenRouter extras parsed back: signed thinking on the choice, upstream cost on usage; the
    // `[DONE]` sentinel was skipped, not decoded.
    assert!(matches!(
        resp.choices[0].thinking.first(),
        Some(ContentPart::Thinking { thinking, signature })
            if thinking == "ponder" && signature.as_deref() == Some("SIG2")
    ));
    assert_eq!(resp.choices[0].text, "sunny");
    assert_eq!(resp.usage.cost_usd, Some(0.00021));
    assert_eq!(resp.usage.total_tokens, 10);
}

#[tokio::test]
async fn groq_responses_opt_in_posts_beta_endpoint_with_unsigned_open_reasoning() {
    use llmleaf_model::ContentPart;
    use std::sync::Mutex;

    // Groq's beta `POST /responses`: the same typed-event SSE as stock OpenAI, with open (plaintext)
    // reasoning deltas and no signature/encrypted extras.
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_g1\",\"model\":\"openai/gpt-oss-120b\"}}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"mull\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"sunny\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_g1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"total_tokens\":10}}}\n\n",
    );
    let captured: Arc<Mutex<Option<HttpRequest>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        *cap.lock().unwrap() = Some(req.clone());
        Ok(FakeResponse::ok_bytes("text/event-stream", sse))
    });
    let provider = OpenAiCompatProvider::for_kind("groq", &http_transports(http)).unwrap();

    // Opt the brand into its beta Responses endpoint; no endpoint override, so the brand default URL
    // builds the documented path.
    let cx = ProviderCx {
        credential: Some("test-key".into()),
        settings: serde_json::from_value(json!({ "chat_api": "responses" })).unwrap(),
        ..Default::default()
    };

    // A second turn replaying prior open thinking (signed by a foreign vendor) plus a redacted block —
    // the Groq flavor must replay the plaintext unsigned and drop the encrypted block.
    let mut req = user_chat("openai/gpt-oss-120b", "weather?");
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Thinking {
                thinking: "prior reasoning".into(),
                signature: Some("SIG1".into()),
            },
            ContentPart::RedactedThinking { data: "ENC".into() },
            ContentPart::Text {
                text: "checking".into(),
            },
        ],
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
    });
    let stream = provider
        .chat(req, &cx)
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    // The documented beta endpoint, verbatim (the brand base already carries `/openai/v1`).
    let sent = captured
        .lock()
        .unwrap()
        .clone()
        .expect("a request was sent");
    assert_eq!(sent.url, "https://api.groq.com/openai/v1/responses");
    let body = json_body(&sent);
    // Groq documents `include` as unsupported: the statelessness default is omitted; `store: false`
    // still goes out (Groq accepts false/null).
    assert!(body.get("include").is_none());
    assert_eq!(body["store"], json!(false));
    // The prior thinking replays as a plaintext reasoning item with NO signature; the redacted block
    // has no Groq representation and produced no item.
    let input = body["input"].as_array().unwrap();
    assert_eq!(
        input[1],
        json!({
            "type": "reasoning",
            "summary": [],
            "content": [{ "type": "reasoning_text", "text": "prior reasoning" }],
        })
    );
    assert!(!input
        .iter()
        .any(|item| item.get("encrypted_content").is_some()));
    // The open reasoning and answer parsed back into the canonical stream.
    assert!(matches!(
        resp.choices[0].thinking.first(),
        Some(ContentPart::Thinking { thinking, signature })
            if thinking == "mull" && signature.is_none()
    ));
    assert_eq!(resp.choices[0].text, "sunny");
    assert_eq!(resp.usage.total_tokens, 10);
}

#[tokio::test]
async fn xai_chat_defaults_to_responses_endpoint() {
    use llmleaf_model::ContentPart;
    use std::sync::Mutex;

    // xAI's Responses API is its documented-preferred chat surface, so the brand defaults to it — the
    // same typed-event SSE as stock OpenAI, encrypted reasoning included for stateless replay.
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_x1\",\"model\":\"grok-4\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"sunny\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"XENC\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_x1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"total_tokens\":10}}}\n\n",
    );
    let captured: Arc<Mutex<Option<HttpRequest>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        *cap.lock().unwrap() = Some(req.clone());
        Ok(FakeResponse::ok_bytes("text/event-stream", sse))
    });
    let provider = OpenAiCompatProvider::for_kind("xai", &http_transports(http)).unwrap();

    // No endpoint override and no settings: the brand default builds xAI's documented URL and wire.
    let cx = ProviderCx {
        credential: Some("test-key".into()),
        ..Default::default()
    };
    let stream = provider
        .chat(user_chat("grok-4", "hi"), &cx)
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    // The documented endpoint, verbatim, with the stock-OpenAI statelessness defaults on (xAI
    // documents both `store` and `include: ["reasoning.encrypted_content"]`).
    let sent = captured
        .lock()
        .unwrap()
        .clone()
        .expect("a request was sent");
    assert_eq!(sent.url, "https://api.x.ai/v1/responses");
    let body = json_body(&sent);
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    // The answer parsed; the encrypted reasoning block survives for next-turn stateless replay.
    assert_eq!(resp.choices[0].text, "sunny");
    assert!(resp.choices[0]
        .thinking
        .iter()
        .any(|p| matches!(p, ContentPart::RedactedThinking { data } if data == "XENC")));
    assert_eq!(resp.usage.total_tokens, 10);
}

#[tokio::test]
async fn azure_responses_opt_in_posts_v1_resource_url() {
    use std::sync::Mutex;

    // Azure's Responses API serves on the *v1* surface: resource-scoped URL (never deployment-scoped
    // like its chat completions), deployment name in the body `model`, `api-key` header auth.
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_az1\",\"model\":\"gpt-5-deploy\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_az1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":9,\"output_tokens\":3,\"total_tokens\":12}}}\n\n",
    );
    let captured: Arc<Mutex<Option<HttpRequest>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        *cap.lock().unwrap() = Some(req.clone());
        Ok(FakeResponse::ok_bytes("text/event-stream", sse))
    });
    let provider = OpenAiCompatProvider::for_kind("azure", &http_transports(http)).unwrap();

    // The v1 surface is region/feature-gated, so Responses is an explicit opt-in for azure.
    let cx = ProviderCx {
        credential: Some("azure-key".into()),
        endpoint: Some("https://my-res.openai.azure.com".into()),
        settings: serde_json::from_value(json!({ "chat_api": "responses" })).unwrap(),
        ..Default::default()
    };
    let stream = provider
        .chat(user_chat("gpt-5-deploy", "hi"), &cx)
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    let sent = captured
        .lock()
        .unwrap()
        .clone()
        .expect("a request was sent");
    // Resource-scoped v1 URL — no deployment segment, no api-version.
    assert_eq!(
        sent.url,
        "https://my-res.openai.azure.com/openai/v1/responses"
    );
    // Azure authenticates with the `api-key` header, not a bearer token.
    assert!(
        sent.headers
            .iter()
            .any(|(k, v)| k == "api-key" && v == "azure-key"),
        "api-key header must carry the credential: {:?}",
        sent.headers
    );
    // The deployment name rides in the body `model`; the statelessness defaults still apply.
    let body = json_body(&sent);
    assert_eq!(body["model"], "gpt-5-deploy");
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(resp.choices[0].text, "Hello");
    assert_eq!(resp.usage.total_tokens, 12);
}

// ---------------------------------------------------------------------------------------------
// 2c. Responses → /chat/completions endpoint fallback: a brand opted into Responses whose upstream
//     answers `POST /responses` with 404/405 ("no such endpoint") transparently retries the same
//     request over `/chat/completions`; every other status keeps its meaning and does NOT downgrade.
// ---------------------------------------------------------------------------------------------

/// A canned chat-completions SSE the fallback path parses (one content delta + finish + `[DONE]`).
const CHAT_FALLBACK_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// Drive the stock `openai` brand (Responses default) against a transport that answers `/responses`
/// with `responses_status` and `/chat/completions` with [`CHAT_FALLBACK_SSE`], recording every URL it
/// saw in order. Shared by the 404 and 405 cases, which differ only in the status.
async fn responses_status_downgrades(responses_status: u16) {
    use std::sync::Mutex;

    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_c = seen.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        seen_c.lock().unwrap().push(req.url.clone());
        if req.url.ends_with("/responses") {
            Ok(FakeResponse::status(
                responses_status,
                "{\"error\":{\"message\":\"no such endpoint\"}}",
            ))
        } else if req.url.ends_with("/chat/completions") {
            Ok(FakeResponse::ok_bytes(
                "text/event-stream",
                CHAT_FALLBACK_SSE,
            ))
        } else {
            panic!("unexpected url {}", req.url)
        }
    });
    let provider = OpenAiCompatProvider::for_kind("openai", &http_transports(http)).unwrap();

    let stream = provider
        .chat(user_chat("gpt-5", "hi"), &cx())
        .await
        .expect("chat downgrades to /chat/completions and returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");
    // The canonical chunks came from the chat-completions body, not the (absent) responses one.
    assert_eq!(resp.id, "chatcmpl-1");
    assert_eq!(resp.choices[0].text, "Hello");
    assert_eq!(resp.choices[0].finish_reason, Some(FinishReason::Stop));

    // Both endpoints were tried, responses first then chat/completions — one `chat()`, two upstream hits.
    let urls = seen.lock().unwrap().clone();
    assert_eq!(urls.len(), 2, "exactly two upstream requests: {urls:?}");
    assert!(
        urls[0].ends_with("/responses"),
        "responses is tried first: {urls:?}"
    );
    assert!(
        urls[1].ends_with("/chat/completions"),
        "then it downgrades to chat/completions: {urls:?}"
    );
}

#[tokio::test]
async fn responses_404_downgrades_to_chat_completions() {
    responses_status_downgrades(404).await;
}

#[tokio::test]
async fn responses_405_downgrades_to_chat_completions() {
    responses_status_downgrades(405).await;
}

#[tokio::test]
async fn responses_500_does_not_downgrade() {
    use std::sync::Mutex;

    // A 500 (like 401/429/5xx) keeps its meaning: it is NOT "no such endpoint", so `chat()` fails once
    // with the upstream status and never touches `/chat/completions`.
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_c = seen.clone();
    let http = FakeHttpTransport::new(move |req: &HttpRequest| {
        seen_c.lock().unwrap().push(req.url.clone());
        assert!(
            req.url.ends_with("/responses"),
            "a 500 must not trigger the chat/completions retry, but saw {}",
            req.url
        );
        Ok(FakeResponse::status(500, "internal error"))
    });
    let provider = OpenAiCompatProvider::for_kind("openai", &http_transports(http)).unwrap();

    // `chat` yields a non-`Debug` stream, so match the result directly (cf. `upstream_429_...`).
    match provider.chat(user_chat("gpt-5", "hi"), &cx()).await {
        Err(ModelError::Upstream { status, message }) => {
            assert_eq!(status, 500);
            assert!(
                message.contains("internal error"),
                "body relayed: {message}"
            );
        }
        Err(other) => panic!("expected Upstream{{500}}, got {other:?}"),
        Ok(_) => panic!("a 500 from /responses must surface as an error, not downgrade"),
    }
    let urls = seen.lock().unwrap().clone();
    assert_eq!(
        urls.len(),
        1,
        "only the responses endpoint was hit: {urls:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// 3. Anthropic chat success
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_chat_roundtrips_to_canonical_stream() {
    // A canned Anthropic Messages response. The provider collects it (Anthropic's non-streaming chat
    // path) and maps it through `anthropic_to_chunks`; here we drive `request_to_anthropic` (the request
    // build) and that mapping over the transport seam end to end.
    let canned = json!({
        "id": "msg_01abc",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [{ "type": "text", "text": "Bonjour" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 11, "output_tokens": 4 }
    });
    let transports = http_transports(FakeHttpTransport::json(canned));
    let provider = AnthropicProvider::new(&transports);

    let stream = provider
        .chat(user_chat("claude-sonnet-4", "bonjour"), &cx())
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");

    assert_eq!(resp.id, "msg_01abc");
    assert_eq!(resp.model, "claude-sonnet-4");
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].text, "Bonjour");
    assert_eq!(resp.choices[0].finish_reason, Some(FinishReason::Stop));
    // Anthropic reports input/output tokens; the mapper derives the total.
    assert_eq!(resp.usage.prompt_tokens, 11);
    assert_eq!(resp.usage.completion_tokens, 4);
    assert_eq!(resp.usage.total_tokens, 15);
}

// ---------------------------------------------------------------------------------------------
// 4. Error modes — each maps to the exact ModelError variant via the shared taxonomy (crate::http)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn upstream_429_maps_to_upstream_status() {
    // A streaming chat (send_checked path) hitting a 429: the body rides through in the message.
    let transports = http_transports(FakeHttpTransport::status(
        429,
        "{\"error\":{\"message\":\"rate limited\"}}",
    ));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();

    // `chat` returns a non-`Debug` `ResponseStream`, so match the result directly rather than
    // `expect_err` (which would require the Ok type to be `Debug`).
    match provider.chat(user_chat("gpt-4o", "hi"), &cx()).await {
        Err(ModelError::Upstream { status, message }) => {
            assert_eq!(status, 429);
            assert!(message.contains("rate limited"), "body relayed: {message}");
        }
        Err(other) => panic!("expected Upstream{{429}}, got {other:?}"),
        Ok(_) => panic!("expected a 429 status to surface as an error"),
    }
}

#[tokio::test]
async fn upstream_500_maps_to_upstream_status() {
    // A 500 on the embeddings (post_json) path: same Upstream classification, different status.
    let transports = http_transports(FakeHttpTransport::status(500, "internal error"));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();

    let req = EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: vec!["a".into()],
        dimensions: None,
        encoding_format: None,
        extra: Default::default(),
    };
    let err = provider
        .embed(req, &cx())
        .await
        .expect_err("a 500 status must surface as an error");
    match err {
        ModelError::Upstream { status, message } => {
            assert_eq!(status, 500);
            assert!(
                message.contains("internal error"),
                "body relayed: {message}"
            );
        }
        other => panic!("expected Upstream{{500}}, got {other:?}"),
    }
}

#[tokio::test]
async fn non_json_2xx_body_on_json_endpoint_maps_to_mapping() {
    // A 200 whose body is not JSON on a JSON endpoint (embeddings → post_json): the status is fine, so
    // it is a parse failure, classified as Mapping (not Upstream).
    let transports = http_transports(FakeHttpTransport::status(200, "not json at all"));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();

    let req = EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: vec!["a".into()],
        dimensions: None,
        encoding_format: None,
        extra: Default::default(),
    };
    let err = provider
        .embed(req, &cx())
        .await
        .expect_err("an undecodable 200 body must surface as a mapping error");
    assert!(
        matches!(err, ModelError::Mapping(_)),
        "expected Mapping, got {err:?}"
    );
}

#[tokio::test]
async fn transport_error_propagates_as_unavailable() {
    // A transport-level failure before any response: surfaces as the seeded ModelError verbatim
    // (Unavailable is the fallback-eligible class).
    let transports = http_transports(FakeHttpTransport::error(ModelError::Unavailable(
        "boom".into(),
    )));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();

    match provider.chat(user_chat("gpt-4o", "hi"), &cx()).await {
        Err(ModelError::Unavailable(msg)) => assert_eq!(msg, "boom"),
        Err(other) => panic!("expected Unavailable, got {other:?}"),
        Ok(_) => panic!("expected a transport error to surface"),
    }
}

// ---------------------------------------------------------------------------------------------
// 4b. MiniMax's `base_resp` envelope — an error smuggled inside a 2xx still classifies as Upstream
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn minimax_200_json_base_resp_error_maps_to_upstream() {
    // MiniMax answers a (streaming) chat request with HTTP 200 + a plain JSON `base_resp` error —
    // its envelope quirk. The provider detects the JSON body by content type and classifies the
    // business code as the HTTP status it stands for (1002 → 429), so the engine retries/falls back
    // exactly as for an honest 429.
    let transports = http_transports(FakeHttpTransport::json(json!({
        "base_resp": { "status_code": 1002, "status_msg": "rate limit" }
    })));
    let provider = OpenAiCompatProvider::for_kind("minimax", &transports).unwrap();

    match provider.chat(user_chat("MiniMax-M3", "hi"), &cx()).await {
        Err(ModelError::Upstream { status, message }) => {
            assert_eq!(status, 429);
            assert!(message.contains("1002"), "code relayed: {message}");
            assert!(message.contains("rate limit"), "msg relayed: {message}");
        }
        Err(other) => panic!("expected Upstream{{429}}, got {other:?}"),
        Ok(_) => panic!("a non-zero base_resp at HTTP 200 must surface as an error"),
    }
}

#[tokio::test]
async fn minimax_sse_frame_with_base_resp_error_fails_the_stream() {
    // Output moderation / quota can trip mid-stream inside a 200 SSE: a frame carrying a non-zero
    // `base_resp` ends the stream with the classified error instead of silently truncating.
    let body = concat!(
        "data: {\"id\":\"mm-1\",\"model\":\"MiniMax-M3\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
        "data: {\"base_resp\":{\"status_code\":1027,\"status_msg\":\"output flagged\"}}\n\n",
    );
    let transports = http_transports(FakeHttpTransport::sse(body));
    let provider = OpenAiCompatProvider::for_kind("minimax", &transports).unwrap();

    let stream = provider
        .chat(user_chat("MiniMax-M3", "hi"), &cx())
        .await
        .expect("the SSE response opens as a stream");
    match collect(stream).await {
        Err(ModelError::Upstream { status, message }) => {
            assert_eq!(status, 400);
            assert!(message.contains("output flagged"), "msg relayed: {message}");
        }
        Err(other) => panic!("expected Upstream{{400}}, got {other:?}"),
        Ok(_) => panic!("a mid-stream base_resp error must fail the stream"),
    }
}

#[tokio::test]
async fn minimax_chat_sse_roundtrips_despite_benign_base_resp() {
    // Success frames may carry `base_resp.status_code: 0` — the per-frame check must treat that as
    // success and the stream must round-trip like any other OpenAI-wire SSE.
    let body = concat!(
        "data: {\"id\":\"mm-2\",\"model\":\"MiniMax-M3\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}],\"base_resp\":{\"status_code\":0,\"status_msg\":\"\"}}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\", world\"},\"finish_reason\":\"stop\"}],\"base_resp\":{\"status_code\":0}}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n",
        "data: [DONE]\n\n",
    );
    let transports = http_transports(FakeHttpTransport::sse(body));
    let provider = OpenAiCompatProvider::for_kind("minimax", &transports).unwrap();

    let stream = provider
        .chat(user_chat("MiniMax-M3", "hi"), &cx())
        .await
        .expect("chat returns a stream");
    let resp = collect(stream).await.expect("stream collects cleanly");
    assert_eq!(resp.id, "mm-2");
    assert_eq!(resp.choices[0].text, "Hello, world");
    assert_eq!(resp.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(resp.usage.total_tokens, 10);
}

// ---------------------------------------------------------------------------------------------
// 5. Realtime — scripted frames arrive on the consumer side; a failing transport surfaces the error
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn realtime_scripted_frames_reach_the_consumer() {
    let scripted = vec![
        RealtimeWire::Text("{\"type\":\"x\"}".into()),
        RealtimeWire::Close,
    ];
    let transports = realtime_transports(FakeRealtimeTransport::scripted(scripted));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();
    assert!(
        provider.supports_realtime(),
        "openai has a native realtime upstream"
    );

    // The core builds two channels: the provider's `outbound` sender feeds the consumer's receiver
    // (`out_rx`), and the consumer's `in_tx` feeds the provider's `inbound` receiver.
    let (_in_tx, in_rx) = mpsc::channel::<RealtimeWire>(64);
    let (out_tx, mut out_rx) = mpsc::channel::<RealtimeWire>(64);
    let peer = RealtimePeer {
        inbound: in_rx,
        outbound: out_tx,
    };

    provider
        .realtime(
            RealtimeParams {
                model: "gpt-4o-realtime".into(),
            },
            peer,
            &cx(),
        )
        .await
        .expect("scripted realtime session completes cleanly");

    // The provider pumped the upstream frames verbatim (principle 7) to the consumer side, in order.
    assert_eq!(
        out_rx.recv().await,
        Some(RealtimeWire::Text("{\"type\":\"x\"}".into()))
    );
    assert_eq!(out_rx.recv().await, Some(RealtimeWire::Close));
    // Then the session ended: the channel closes (no further frames).
    assert_eq!(out_rx.recv().await, None);
}

#[tokio::test]
async fn realtime_failing_transport_surfaces_the_error() {
    let transports = realtime_transports(FakeRealtimeTransport::failing(ModelError::Unavailable(
        "dial failed".into(),
    )));
    let provider = OpenAiCompatProvider::for_kind("openai", &transports).unwrap();

    let (_in_tx, in_rx) = mpsc::channel::<RealtimeWire>(64);
    let (out_tx, _out_rx) = mpsc::channel::<RealtimeWire>(64);
    let peer = RealtimePeer {
        inbound: in_rx,
        outbound: out_tx,
    };

    let err = provider
        .realtime(
            RealtimeParams {
                model: "gpt-4o-realtime".into(),
            },
            peer,
            &cx(),
        )
        .await
        .expect_err("a failing realtime transport must surface its error");
    match err {
        ModelError::Unavailable(msg) => assert_eq!(msg, "dial failed"),
        other => panic!("expected Unavailable, got {other:?}"),
    }
}

// A tiny sanity assertion that the no-op realtime fake also satisfies the trait object shape the
// transports use — keeps the `RealtimeTransport` import honest and documents the no-op path.
#[tokio::test]
async fn noop_realtime_transport_is_usable() {
    let t: Arc<dyn RealtimeTransport> = Arc::new(FakeRealtimeTransport::scripted(Vec::new()));
    let (_in_tx, in_rx) = mpsc::channel::<RealtimeWire>(1);
    let (out_tx, mut out_rx) = mpsc::channel::<RealtimeWire>(1);
    let peer = RealtimePeer {
        inbound: in_rx,
        outbound: out_tx,
    };
    t.run(String::new(), Vec::new(), peer)
        .await
        .expect("no-op session ok");
    assert_eq!(out_rx.recv().await, None);
}
