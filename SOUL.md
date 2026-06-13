# SOUL.md

This document is the constitution of **llmleaf**. It is written for the humans and AI agents
who build it. When a design decision is unclear, this file is the tiebreaker. Code that
contradicts this document is wrong, even if it works.

## Identity

llmleaf is a high-efficiency LLM proxy written in Rust. It stands as one endpoint in front of
every model provider: consumers speak a familiar dialect (OpenAI, OpenRouter), llmleaf speaks
its own minimal internal language, and extensions translate to whatever each provider needs.

The name is the architecture: a **leaf** is small, light, and does one thing — it converts.
llmleaf converts API dialects and routes tokens, as close to wire speed as Rust allows.

## Why it exists

Every model provider speaks its own dialect, with its own auth, its own failure modes, and its
own outages. Consumers don't want N integrations; they want one stable endpoint that survives
provider failures, enforces their keys and budgets, and gets out of the way. Existing gateways
solve this with heavyweight platforms. llmleaf solves it with a small, fast core and sharp
boundaries.

## Core principles

These are decision rules, not aspirations. Apply them literally.

1. **The hot path is sacred.** A request's life is: authenticate → map in → route → stream →
   map out → emit events. Nothing else runs per-request in the core. If a feature wants to be
   on the hot path, it must justify every allocation. The single sanctioned insertion is a
   sync interceptor (see Bolt-ons) — explicit opt-in, and the operator pays its latency
   knowingly.

2. **The core knows no provider.** Provider-specific knowledge lives only in extensions.
   First-party providers are compiled-in Rust trait implementations (zero overhead).
   Third-party providers are WASM plugins loaded at runtime (sandboxed, no fork required).
   If core code mentions a provider by name, it's a bug.

3. **One internal model, many dialects.** llmleaf has its own efficient canonical
   request/response representation. OpenAI compatibility, OpenRouter compatibility, and any
   future surface (Anthropic, etc.) are mappings at the edge. No external dialect is "native";
   none gets privileged shortcuts through the core.

4. **Streaming is the default.** The internal representation is a stream. A non-streaming
   response is a collected stream, never the other way around.

5. **The core observes; others account — and gatekeep.** The core *pushes* usage and lifecycle
   events out to a configured sink — batched and asynchronous, never back-pressuring the hot
   path. It never stores, aggregates, or even counts usage itself — not even to enforce limits.
   Usage limits are an asynchronous loop, now outbound: the core *pulls* per-key verdicts from a
   configured limiter on an interval and caches them node-locally — restricting a key to specific
   models, or suspending it entirely, optionally until a given time. On each request the core only
   enforces the cached verdict: a lookup, never arithmetic, never a network round-trip. The limiter
   declares; the core obeys what it last pulled. There is no inbound mutation surface. If a feature
   needs a database, it does not belong in the core.

   **Cold-start auth is the exception to fail-open.** Failing open on the *identity* pull would admit
   unauthenticated callers — so identity fails *closed* on a cold cache (reject until the first
   successful pull, which runs at startup before the listener opens) even though *limits* fail open.
   Availability never trumps authentication. Once warm, a node always serves its last-good identity
   cache through a control-plane outage. This is the one place principle 8 yields to security.

6. **Config is the base; the pulled control plane is a layer.** The core is fully defined by its
   config file: providers, routes, compat surfaces, base keys, and *which* control endpoints to
   pull from and push to. Runtime state — the pulled key roster and verdict overlay — layers on top
   of the config base, refreshed on an interval into a node-local cache that is dropped freely and
   rebuilt on the next pull. The config *names* the control endpoints; it never depends on them being
   up. Omit the `[control]` section entirely and the core is completely operable from the config file
   alone — the file `[[keys]]` are then the whole roster.

7. **Transparent by default.** llmleaf never silently mutates prompts or responses. Every
   transformation is either a documented dialect mapping or explicit configuration.

8. **Fail toward availability.** Providers go down; llmleaf doesn't. Fallback chains, retries,
   and intelligent switchover (health-aware routing away from degraded providers) are core
   behavior, driven by the same event signals the core already produces.

9. **Multi-node is trivial; HA decisions are local.** Running N core nodes behind a plain
   load balancer must Just Work — no consensus, no leader election, no shared state, no
   inter-node chatter. This falls out of the core holding no durable state (principles 5
   and 6) and must stay that way: a feature that requires nodes to know about each other
   does not belong in the core. Each node makes its own HA decisions — self-contained
   (from its own observations only), fast (a local lookup or comparison, never a network
   round-trip), and specific (a verdict targets a concrete provider, route, or key — never
   a cluster-wide mode or global flag). Nodes may converge on the same conclusions because
   they see the same world — each pulls the same roster and verdicts from the same control
   endpoints — never because they coordinated. The verdict cache is exactly principle-9
   node-local state: self-contained, fast (a local lookup), specific (targets a concrete key).
   The single sanctioned exception is sync interceptors: every node is configured to call the
   same intercept endpoint, so consistent in-flight screening falls out of identical config,
   not inter-node chatter. Keep coordination in the control endpoints the nodes pull from — it
   never becomes a general-purpose channel between nodes.

## Architecture soul

Two planes, strictly separated:

**Data plane — the core.** The proxy itself. Compat surfaces (OpenAI, OpenRouter) on the front,
extension boundary (traits + WASM) on the back, routing/fallback/caching/key-enforcement in
between, usage events pushed out the side. Small enough to hold in your head.

**Control plane — everything else, reached outbound.** The core is always the client: it *pulls*
what it needs (identity, verdicts) and *pushes* what it produces (usage). It never exposes an
inbound mutation surface — nothing reaches in to change runtime state.
- **Identity & limits sources (PULL)**: the core polls configured endpoints for the key roster and
  per-key verdicts and caches them node-locally; the hot path is a cache lookup. Identity fails
  closed on a cold cache (an unknown caller must never be admitted); limits fail open (a blip keeps
  paying keys serving). The only inbound HTTP left is read-only: `/healthz`, the unauthenticated
  self-description `GET /v1/openapi.json` (the static OpenAPI 3.1 contract for the consumer surface —
  it names no provider and carries no runtime, tenant, or topology data, so it is public by the same
  transparency that governs the rest of the surface, P7), and optionally
  `GET /admin/routes`, `/admin/health`, `/admin/keys` for observability. The consumer model-catalog
  surface `GET /v1/models` (OpenRouter-shaped, bearer-authed) carries the same observability concession
  under the *same* `x-admin-token`: with the admin token it adds each model's provider/fallback chain
  and node-local health (an `endpoints` array); without it, the public view exposes only ids,
  capabilities, and pricing — never provider identity or topology (P2/P7). It remains read-only and
  pulls nothing in; it is a view, not a mutation surface.
- **Bolt-ons** (a pattern, not a component): the pushed event stream and the pulled control
  endpoints are deliberately rich enough that whole capabilities bolt on without touching the core.
  Bolt-ons come in two couplings:
  - **Observers (async)** — receive the pushed usage/lifecycle batches, apply any external logic,
    and influence the core only by what it next *pulls* (e.g. write a verdict the limiter then
    serves) or by re-entering through the normal consumer surface. Security screening after the
    fact, prompt collection (archive requests for audit or datasets), history replay (re-issue
    captured requests as an ordinary client), alerting, A/B analysis. A request never waits for an
    observer; the push is fire-and-forget.
  - **Interceptors (sync)** — an external service named in config to sit in-flight: the core POSTs
    it the canonical request or response and waits for a verdict — pass, block (e.g. deny a tool
    call before it executes), or rewrite (e.g. censor private or security-sensitive content).
    Interceptors are the one configurable insertion point on the hot path, always explicit opt-in
    per route or key, and they pay their own latency bill. The core does no analysis itself; it
    transports payloads and enforces verdicts.

  The pushed events must carry enough, configurably including full payloads, to make this possible.
  The core neither knows nor cares what's bolted on. Prefer an observer; reach for an interceptor
  only when the verdict must land before the request proceeds.
- **Gatekeeper / limiter** (a role, not a component): the canonical bolt-on. Any external service
  can be the limiter — it ingests the pushed usage events, applies whatever accounting or policy it
  likes, and *serves* the resulting key verdicts (restrict to models, suspend until a time, block)
  at the endpoint the core pulls. The core neither knows nor cares who computes the verdicts; it
  only knows the URL it polls. The verdict shape is fixed; the policy behind it is the limiter's
  business.
- **Web server** (separate component): a Leptos app for humans. Authenticates via master password,
  OAuth 2, or OpenID Connect SSO. It does not talk *to* the core to mutate state — there is no
  inbound mutation surface. Instead it implements the control endpoints the core pulls (identity,
  verdicts) and receives the pushed usage events for dashboards, accounting, and optionally the
  limiter role. Its only calls *into* the core are the read-only admin GETs. It has no privileged
  backdoor.

The core never depends on the control plane being reachable. Take the limiter and sink down and the
proxy keeps proxying on its last-good cache (identity excepted: a cold node with no identity yet
fails closed).

**One sidecar crate — pricing.** A dedicated crate collects API pricing data from the
providers. The resulting dataset is stored and distributed with the core, which uses it to
tell clients the real-time cost of their requests (in responses and usage events) and to make
cost-aware decisions — routing, fallback choice — when configured to. Collection happens
offline in the pricing crate; the core only ever reads the bundled dataset. It never fetches
pricing at request time: consumption is a lookup, like everything else on the hot path.

**Client libraries — under `clients/`.** Official consumer client libraries live in `clients/`,
one per language: Kotlin Multiplatform, Go, Rust, TypeScript (JavaScript), and Zig. They are
generated from a single Protocol Buffers schema — `clients/proto/` is the source of truth for the
canonical request/response shapes the libraries expose — so every language binding stays in
lockstep. The schema is a *definition and codegen* source only: on the wire the libraries still
speak llmleaf's existing OpenAI/OpenRouter HTTP surface (P3), so the core gains no protobuf surface
and no new dependency. Clients sit downstream of the core, never on its hot path, and a client is a
convenience, never a requirement — the HTTP dialects remain sufficient on their own (as with the
web UI).

## In scope

- Model routing, fallback chains, retries, intelligent switchover
- Trivial multi-node HA: stateless core nodes behind any load balancer, each making its own
  fast, node-local, narrowly-scoped HA decisions — no clustering, no coordination protocol
- Virtual API keys for consumers, enforced in the core: each key can be restricted to
  specific models; identity and usage-limit verdicts are *pulled* on an interval from configured
  control endpoints and cached node-locally; provider credential handling
- Response caching at the proxy
- OpenAI- and OpenRouter-compatible consumer surfaces; more dialects as edge mappings
- Provider extensions: compiled traits (first-party) and WASM plugins (third-party)
- Official client libraries under `clients/` (Kotlin Multiplatform, Go, Rust, TypeScript/
  JavaScript, Zig), generated from a single Protocol Buffers schema; they speak the existing HTTP
  consumer surface and add no protobuf surface to the core
- Outbound control integration designed as a bolt-on surface: security screening, prompt
  collection, history replay, and the like live outside the core, built on the *pull verdicts /
  push usage* loop. Read-only admin GETs (routes, health, keys) for observability
- Sync interceptors: opt-in, in-flight external screening that can block tool-call
  execution or censor private/security-sensitive content before a request or response
  proceeds
- Usage/lifecycle event push to a configured sink, batched and async, configurably including
  full payloads so bolt-ons have everything they need
- A dedicated pricing crate that collects provider API pricing; its dataset ships with the
  core for real-time cost reporting to clients and cost-aware routing decisions
- A dedicated control crate (`llmleaf-control`), wired by the binary, that owns all outbound
  HTTP: the pull refreshers, the usage push reporter, and the sync interceptor client. The core
  stays HTTP-client-free (principle 2)
- The Leptos web server, as a separate component that serves the pulled control endpoints and
  receives the pushed usage events

## Non-goals

- **The core is not a database.** No usage storage, no aggregation, no reporting in the core —
  ever. That work belongs downstream of the event stream.
- **Not an inference engine.** llmleaf never runs models, only routes to them.
- **Not a prompt framework.** No templating, no agent orchestration, no chain logic.
- **The web UI is never required.** A config file alone (with the `[control]` section omitted)
  must always be sufficient to run.
- **No silent magic.** No automatic prompt rewriting, no invisible "optimizations" of payloads.

## Decision filter

When in doubt:

- Core simplicity beats feature richness. Push work to the edges — into extensions, the
  control plane, or downstream event consumers — before adding it to the core.
- New capability? First ask: can it be a bolt-on (consume the pushed events, serve a pulled
  control endpoint)? The answer should almost always be yes. If the pushed events or the pulled
  control shapes aren't rich enough to support it, enrich them — don't put the capability in the
  core.
- Observer or interceptor? Async observer by default; sync interceptor only when the
  verdict must arrive before the request proceeds. Eventual enforcement is usually enough.
- If a feature needs state, ask where it lives: config file (immutable base), a pulled control
  endpoint (node-local cached layer), or the pushed event stream (downstream's problem). "In the
  core's memory forever" is not an option.
- Dialect mapping fidelity beats convenience. Match the compat surface's documented behavior
  exactly, even when it's awkward.
- Measure before optimizing, but design so there's nothing to optimize: fewer allocations,
  fewer copies, fewer locks on the hot path.
- A provider quirk goes in that provider's extension, never in the core.
- If a feature makes core nodes aware of each other, redesign it. Coordination belongs in
  the control plane or nowhere.
