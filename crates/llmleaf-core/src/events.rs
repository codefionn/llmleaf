//! The event bus — the core's one output channel for everything downstream (SOUL.md principle 5
//! and the Bolt-ons section).
//!
//! The core *observes*: it emits usage and lifecycle events and forgets them. It never stores,
//! aggregates, or counts. The in-process bus is tapped by the control crate's usage reporter, which
//! batches and *pushes* the events out to the configured sink; downstream observers and the limiter
//! act back only by serving what the core next pulls. The events must carry enough — configurably
//! including full payloads — that whole capabilities (accounting, security screening, prompt archival,
//! replay) bolt on without touching the core.
//!
//! Transport is an in-memory broadcast ring: bounded, lossy for a slow reporter, and *never*
//! back-pressuring the hot path. A request never waits for the reporter.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use llmleaf_model::{FinishReason, Usage};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::broadcast;

/// A lifecycle/usage event. Tagged for self-describing SSE/WS payloads.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// A request was authorized and is entering the pipeline.
    RequestStarted {
        id: String,
        key: String,
        model: String,
        /// Full request payload — included only when the stream is configured to carry payloads.
        #[serde(skip_serializing_if = "Option::is_none")]
        request: Option<Value>,
    },
    /// A target was selected and accepted the request.
    RequestRouted {
        id: String,
        provider: String,
        upstream_model: String,
    },
    /// Provider-reported usage. The core relays; it does not compute (cost is filled by lookup).
    Usage {
        id: String,
        key: String,
        model: String,
        usage: Usage,
    },
    /// The request finished successfully.
    RequestCompleted {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        finish: Option<FinishReason>,
    },
    /// The request failed (all targets exhausted, or a fatal error).
    RequestFailed { id: String, error: String },
    /// A node-local health observation about a provider (principle 9: an observation, not a global flag).
    ProviderHealth { provider: String, status: String },
}

/// An event with the wall-clock instant it was emitted, in unix milliseconds.
#[derive(Debug, Clone, Serialize)]
pub struct Envelope {
    pub ts_ms: u64,
    #[serde(flatten)]
    pub event: Event,
}

/// A clone-able handle to the event broadcast. Cloning shares the same channel.
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Arc<Envelope>>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity.max(1));
        EventBus { tx }
    }

    /// Emit an event. Stamps the wall clock and fires it into the ring. If there are no subscribers
    /// (or all are lagging) the event is simply dropped — the core does not care who listens.
    pub fn emit(&self, event: Event) {
        let env = Arc::new(Envelope {
            ts_ms: now_ms(),
            event,
        });
        // Err only means "no receivers" — expected and ignored.
        let _ = self.tx.send(env);
    }

    /// Subscribe to the stream. Used by the control crate's usage reporter and any in-process observer.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<Envelope>> {
        self.tx.subscribe()
    }

    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn emit_reaches_subscriber() {
        let bus = EventBus::new(16);
        let mut rx = bus.subscribe();
        bus.emit(Event::RequestCompleted {
            id: "r1".into(),
            finish: Some(FinishReason::Stop),
        });
        let env = rx.recv().await.unwrap();
        match &env.event {
            Event::RequestCompleted { id, .. } => assert_eq!(id, "r1"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn emit_without_subscribers_is_silent() {
        let bus = EventBus::new(16);
        bus.emit(Event::RequestFailed {
            id: "x".into(),
            error: "boom".into(),
        });
        assert_eq!(bus.subscriber_count(), 0);
    }
}
