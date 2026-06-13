//! llmleaf-web — the Leptos control-plane app (SOUL.md "Web server").
//!
//! Two artifacts, one crate (cargo-leptos):
//!   - the **server** (`ssr`): an axum app that (a) serves the human UI, (b) serves the machine
//!     control endpoints the core PULLS — identity + verdicts, (c) ingests the usage events the core
//!     PUSHES, and (d) calls the core's read-only admin GETs. It has NO inbound path to mutate the
//!     core; it only serves what the core chooses to pull.
//!   - the **wasm bundle** (`hydrate`): hydrates the server-rendered DOM.
//!
//! Server-only modules are gated behind `ssr` so the wasm build never compiles axum/sqlx/reqwest.

pub mod app;
pub mod dto;
pub mod wire;

#[cfg(feature = "ssr")]
pub mod admin;
#[cfg(feature = "ssr")]
pub mod auth;
#[cfg(feature = "ssr")]
pub mod config;
#[cfg(feature = "ssr")]
pub mod control;
#[cfg(feature = "ssr")]
pub mod db;
#[cfg(feature = "ssr")]
pub mod limiter;
#[cfg(feature = "ssr")]
pub mod state;

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    use crate::app::*;
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(App);
}
