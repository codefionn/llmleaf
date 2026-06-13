//! The native OpenAI Realtime WebSocket client — the production [`RealtimeTransport`].
//!
//! This is the upstream half of the realtime dialect — the analogue of [`crate::openai_wire`] but for
//! a persistent bidirectional socket. It dials the realtime endpoint and pumps frames between the
//! consumer (a [`RealtimePeer`] handed in by the core) and the upstream socket **verbatim**: text
//! frames carry the realtime JSON events, binary frames carry audio. The core never inspects them
//! (principle 7: transparent transport), so audio and any future protocol additions pass straight
//! through. The transport is dialect-neutral — it dials whatever `wss://…` URL and headers the provider
//! hands it; the OpenAI-specific knowledge (the wss URL shape, the GA auth headers, the dropped
//! `OpenAI-Beta` header) lives in the provider that builds them ([`crate::compat`]), never here and
//! never in the core (principle 2). A fake [`RealtimeTransport`] can stand in for tests/simulation.

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use llmleaf_model::ModelError;
use llmleaf_provider::{RealtimePeer, RealtimeWire};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

use crate::transport::RealtimeTransport;

/// The production realtime transport: a `tokio-tungstenite` WebSocket client. The only place
/// `tokio-tungstenite` appears now (mirrored by `reqwest` in [`crate::transport::ReqwestTransport`]).
pub struct TungsteniteRealtimeTransport;

#[async_trait]
impl RealtimeTransport for TungsteniteRealtimeTransport {
    /// Dial `url` with `headers`, then pump frames between `peer` and the upstream until either side
    /// closes. The session is a trusted server-to-upstream link, so the raw credential rides in the
    /// caller-supplied `Authorization` header.
    async fn run(
        &self,
        url: String,
        headers: Vec<(String, String)>,
        peer: RealtimePeer,
    ) -> Result<(), ModelError> {
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|e| ModelError::Unavailable(format!("realtime request: {e}")))?;
        {
            let req_headers = request.headers_mut();
            for (name, value) in &headers {
                let header_name: tokio_tungstenite::tungstenite::http::HeaderName =
                    match name.parse() {
                        Ok(n) => n,
                        Err(_) => continue,
                    };
                if let Ok(v) = HeaderValue::from_str(value) {
                    req_headers.insert(header_name, v);
                }
            }
        }

        let (ws, _resp) = connect_async(request)
            .await
            .map_err(|e| ModelError::Unavailable(format!("realtime connect: {e}")))?;
        let (mut write, mut read) = ws.split();

        let RealtimePeer {
            mut inbound,
            outbound,
        } = peer;

        // One task drives both directions; whichever side closes first ends the session.
        loop {
            tokio::select! {
                // Consumer → upstream.
                msg = inbound.recv() => match msg {
                    Some(RealtimeWire::Text(t)) => {
                        if write.send(Message::text(t)).await.is_err() {
                            break;
                        }
                    }
                    Some(RealtimeWire::Binary(b)) => {
                        if write.send(Message::binary(b)).await.is_err() {
                            break;
                        }
                    }
                    // Consumer closed (or the core dropped the sender): close upstream and finish.
                    Some(RealtimeWire::Close) | None => {
                        let _ = write.send(Message::Close(None)).await;
                        break;
                    }
                },
                // Upstream → consumer.
                frame = read.next() => match frame {
                    Some(Ok(Message::Text(t))) => {
                        if outbound.send(RealtimeWire::Text(t.as_str().to_owned())).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Binary(b))) => {
                        if outbound.send(RealtimeWire::Binary(b)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        let _ = outbound.send(RealtimeWire::Close).await;
                        break;
                    }
                    // tungstenite answers pings itself; control frames carry nothing to forward.
                    Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {}
                    Some(Err(e)) => {
                        return Err(ModelError::Unavailable(format!("realtime upstream: {e}")));
                    }
                    None => {
                        let _ = outbound.send(RealtimeWire::Close).await;
                        break;
                    }
                },
            }
        }

        Ok(())
    }
}

/// Derive the realtime `wss://…/realtime?model=…` URL from a provider's base endpoint (`https`→`wss`,
/// `http`→`ws`). Model ids are simple tokens (no spaces), so they are appended without escaping.
/// `pub(crate)` so the OpenAI-wire provider builds the URL before handing it to the transport.
pub(crate) fn build_ws_url(endpoint: &str, model: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    let ws_base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    format!("{ws_base}/realtime?model={model}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_wss_url_from_https_endpoint() {
        assert_eq!(
            build_ws_url("https://api.openai.com/v1", "gpt-realtime"),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        );
        // Trailing slash trimmed; http maps to ws (e.g. a local gateway).
        assert_eq!(
            build_ws_url("http://localhost:8080/", "m"),
            "ws://localhost:8080/realtime?model=m"
        );
    }
}
