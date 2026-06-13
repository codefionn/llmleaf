//! The OpenAI Realtime WebSocket consumer surface (SOUL.md principle 3 — a dialect at the edge).
//!
//! Consumers connect to `/v1/realtime` speaking OpenAI's GA Realtime protocol. This module is the
//! realtime sibling of [`crate::compat::openai`]: [`wire`] maps the dialect ⇄ canonical, and
//! [`session`] runs the bridge that turns a stateful Realtime session into a series of canonical
//! [`crate::engine::Engine::run`] turns when the routed provider has no native realtime transport.
//! When a provider *does* serve native realtime, the [`crate::server`] handler proxies frames straight
//! to it instead — never touching this dialect logic. The axum handler and native-vs-bridge dispatch
//! live in [`crate::server`] alongside the other surfaces and their shared auth/id helpers.

pub mod session;
pub mod wire;
