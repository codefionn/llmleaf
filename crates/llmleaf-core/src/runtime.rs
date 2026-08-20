//! Small adapters that keep Axum-facing futures `Send` while using Compio for runtime-bound I/O.

use std::time::Duration;

/// Sleep on a local Compio runtime and surface completion through a `Send` channel. Axum requires every
/// request future to be `Send`, whereas Compio timer futures are deliberately local to their runtime.
pub(crate) async fn sleep(duration: Duration) {
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let _ = std::thread::Builder::new()
        .name("llmleaf-delay".to_string())
        .spawn(move || {
            if let Ok(runtime) = compio::runtime::Runtime::new() {
                runtime.block_on(compio::time::sleep(duration));
            }
            let _ = done_tx.send(());
        });
    let _ = done_rx.await;
}
