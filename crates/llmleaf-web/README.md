# llmleaf-web

The **control-plane app for humans** — a [Leptos](https://leptos.dev) full-stack (SSR) application,
served by axum. It is a *separate component* from the llmleaf core (SOUL.md, "Web server"): the core
never depends on it, and it has no privileged backdoor into the core.

It is the operator-facing other half of llmleaf's **inverted control plane**. The core is always the
client — it *pulls* what it needs and *pushes* what it produces. This app:

- **Serves** the endpoints the core **pulls**:
  - `GET /llmleaf/keys` → the consumer-key roster (identity).
  - `GET /llmleaf/verdicts` → the per-key verdict overlay (block / suspend / narrow models).
- **Receives** what the core **pushes**:
  - `POST /llmleaf/usage` ← batched usage/lifecycle events, stored for dashboards and accounting.
- **Observes** the core through its read-only admin GETs only (`/admin/routes`, `/admin/health`).

All three machine endpoints are guarded by a shared bearer token (`[control].token`).

## What operators get

A signed-in UI to:

- **Dashboard** — 24h/all-time totals, an hourly usage chart, top models.
- **Keys** — issue consumer keys (bearer token shown once), and set verdicts by hand: block, suspend,
  resume, delete. This is the *limiter role* done manually.
- **Accounting** — per-key and per-model usage/cost over a selectable window.
- **Core** — a read-only mirror of the core's routes and provider health.
- **Events** — the recent lifecycle/usage stream the core pushed.

Plus an **automated limiter** (optional): a background loop that suspends keys exceeding a rolling
30-day cost cap or 24h request cap, and lifts those suspensions when they fall back under — only ever
touching suspensions it set, never an operator's.

## Auth

Operators sign in with a **master password** (a bcrypt hash in config) and/or **OIDC SSO**
(authorization-code flow + PKCE, id_token verified against the issuer JWKS). Sessions are server-side;
the browser cookie carries only an opaque token.

## Architecture

- `wire.rs` — the exact JSON contract the core pulls/pushes (mirrors the core's types; this crate does
  not link the core). Tests pin the shapes to the core's literal payloads.
- `control/` — the machine endpoints (axum, bearer-guarded).
- `db/` — SQLite (sqlx): the key roster + verdict overlay, the event store, sessions, OIDC flow state.
- `auth/` — sessions, the route gate, master-password + OIDC login.
- `limiter.rs` — the usage→verdict loop and housekeeping.
- `admin.rs` — the read-only client for the core's admin GETs.
- `app/` — the Leptos UI (`pages.rs`) and its typed server functions (`server.rs`).

Server-only modules are gated behind the `ssr` feature so the hydration wasm bundle never compiles
axum/sqlx/reqwest.

## Running

```sh
# Dev (rebuilds + serves UI + API on http://127.0.0.1:3000):
cargo leptos watch

# Production-style build:
cargo leptos build --release
LLMLEAF_WEB_CONFIG=./llmleaf-web.toml ./target/release/llmleaf-web
```

Configuration: copy [`llmleaf-web.example.toml`](./llmleaf-web.example.toml) to `llmleaf-web.toml`.
With no config file, a DEV fallback runs (SQLite `./llmleaf-web.db`, **open** control endpoints, master
password `llmleaf-dev`) and logs a warning — never use it in production.
