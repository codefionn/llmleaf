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

use crate::fake::{FakeHttpTransport, FakeRealtimeTransport};
use crate::transport::{RealtimeTransport, Transports};
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

    let stream = provider
        .chat(user_chat("gpt-4o", "hi"), &cx())
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
