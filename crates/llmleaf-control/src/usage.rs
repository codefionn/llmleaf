//! Usage reporter: tap the in-process event bus and PUSH batches to the configured sink
//! (`[control.usage]`). Lossy by design — the broadcast ring drops oldest for a slow reporter, so the
//! hot path never waits (SOUL.md principle 5). A push failure drops the batch (downstream's problem).

use std::sync::Arc;
use std::time::Duration;

use compio::runtime::JoinHandle;
use compio::time::interval;
use llmleaf_core::{Envelope, ResolvedAuth, UsageSink};
use serde::Serialize;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

/// Subscribes to the event bus, batches events, and POSTs `{"events": [Envelope, ...]}` to the sink.
pub struct UsageReporter {
    http: cyper::Client,
    url: String,
    auth: Option<ResolvedAuth>,
    flush_every: Duration,
    batch_max: usize,
    timeout: Duration,
    rx: broadcast::Receiver<Arc<Envelope>>,
    shutdown: CancellationToken,
}

#[derive(Serialize)]
struct UsageBatch<'a> {
    events: Vec<&'a Envelope>,
}

impl UsageReporter {
    pub fn new(
        http: cyper::Client,
        cfg: &UsageSink,
        auth: Option<ResolvedAuth>,
        rx: broadcast::Receiver<Arc<Envelope>>,
        shutdown: CancellationToken,
    ) -> Self {
        UsageReporter {
            http,
            url: cfg.url.clone(),
            auth,
            flush_every: Duration::from_millis(cfg.batch_ms.max(1)),
            batch_max: cfg.batch_max.max(1),
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
            rx,
            shutdown,
        }
    }

    pub fn spawn(self) -> JoinHandle<()> {
        // Destructure so the recv future borrows a local (`rx`, `&mut`) without colliding with the
        // immutable borrows the flush helper and the shutdown future need inside the same `select!`.
        let UsageReporter {
            http,
            url,
            auth,
            flush_every,
            batch_max,
            timeout,
            mut rx,
            shutdown,
        } = self;

        compio::runtime::spawn(async move {
            let mut batch: Vec<Arc<Envelope>> = Vec::with_capacity(batch_max);
            let mut flush = interval(flush_every);
            loop {
                tokio::select! {
                    _ = shutdown.cancelled() => {
                        send_batch(&http, &url, auth.as_ref(), timeout, &mut batch).await;
                        break;
                    }
                    _ = flush.tick() => {
                        send_batch(&http, &url, auth.as_ref(), timeout, &mut batch).await;
                    }
                    r = rx.recv() => match r {
                        Ok(env) => {
                            batch.push(env);
                            if batch.len() >= batch_max {
                                send_batch(&http, &url, auth.as_ref(), timeout, &mut batch).await;
                            }
                        }
                        // The bus is lossy by design: a slow reporter loses the oldest frames. Log and
                        // keep going — never block the bus, never grow unbounded.
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(dropped = n, "usage reporter lagged; dropped events");
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            send_batch(&http, &url, auth.as_ref(), timeout, &mut batch).await;
                            break;
                        }
                    }
                }
            }
            tracing::info!(url = %url, "usage reporter stopped");
        })
    }
}

/// POST the current batch, then clear it. A failure drops the batch (the sink is downstream's problem).
async fn send_batch(
    http: &cyper::Client,
    url: &str,
    auth: Option<&ResolvedAuth>,
    timeout: Duration,
    batch: &mut Vec<Arc<Envelope>>,
) {
    if batch.is_empty() {
        return;
    }
    let body = UsageBatch {
        events: batch.iter().map(Arc::as_ref).collect(),
    };
    let result = async {
        let req = crate::apply_auth(http.post(url).map_err(|e| e.to_string())?, auth)
            .map_err(|e| e.to_string())?
            .json(&body)
            .map_err(|e| e.to_string())?;
        let response = compio::time::timeout(timeout, req.send())
            .await
            .map_err(|_| "usage request timed out".to_string())?
            .map_err(|e| e.to_string())?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err("usage sink returned a non-success status".to_string())
        }
    }
    .await;
    match result {
        Ok(()) => tracing::debug!(count = batch.len(), url = %url, "pushed usage batch"),
        Err(e) => {
            tracing::warn!(error = %e, count = batch.len(), url = %url, "usage push failed; dropping batch")
        }
    }
    batch.clear();
}
