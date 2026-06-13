-- llmleaf-web schema. This app IS allowed a database (it is the separate control-plane component, not
-- the core — SOUL.md: "If a feature needs a database, it does not belong in the core"). It stores the
-- key roster + verdict overlay it serves to the core, the usage events the core pushes, and operator
-- sessions.

-- The consumer-key roster the core PULLS as identity, with the verdict overlay it PULLS as limits.
CREATE TABLE IF NOT EXISTS keys (
    id                TEXT PRIMARY KEY,
    -- crypt(3) MCF hash (bcrypt), never plaintext. Served verbatim as `pw_hash` on the identity pull.
    pw_hash           TEXT NOT NULL,
    name              TEXT,
    -- JSON array of model ids; NULL/absent => all routed models (the identity base allow-list).
    allowed_models    TEXT,
    -- Verdict overlay (operator- or limiter-set) ↓
    blocked           INTEGER NOT NULL DEFAULT 0,
    suspended_until   INTEGER,            -- unix SECONDS; NULL => not suspended
    verdict_models    TEXT,               -- JSON array; NULL => no runtime narrowing
    -- Provenance of the current suspension, so the operator can tell an auto-suspension from a manual
    -- one and the limiter can clear only what it set. 'manual' | 'limiter' | NULL.
    verdict_source    TEXT,
    created_at_ms     INTEGER NOT NULL,
    updated_at_ms     INTEGER NOT NULL
);

-- Every lifecycle/usage event the core pushed. Append-only; the dashboards aggregate over it.
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms             INTEGER NOT NULL,
    kind              TEXT NOT NULL,      -- request_started|request_routed|usage|request_completed|request_failed|provider_health|unknown
    request_id        TEXT,
    key_id            TEXT,
    model             TEXT,
    provider          TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    cost              REAL    NOT NULL DEFAULT 0,
    detail            TEXT                -- error text / finish reason / health status
);
CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(ts_ms);
CREATE INDEX IF NOT EXISTS idx_events_key   ON events(key_id, ts_ms);
CREATE INDEX IF NOT EXISTS idx_events_kind  ON events(kind, ts_ms);
CREATE INDEX IF NOT EXISTS idx_events_model ON events(model, ts_ms);

-- Operator sessions. Opaque random token in the browser cookie; server-side lookup is the source of
-- truth (so logout/expiry are authoritative and the cookie carries no claims).
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    method      TEXT NOT NULL,           -- 'password' | 'oidc'
    created_ms  INTEGER NOT NULL,
    expires_ms  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_ms);

-- Transient OIDC authorization-code-flow state, keyed by the `state` param. Holds the PKCE verifier and
-- nonce between the redirect to the IdP and the callback. Rows are short-lived and deleted on use.
CREATE TABLE IF NOT EXISTS oidc_flows (
    state         TEXT PRIMARY KEY,
    code_verifier TEXT NOT NULL,
    nonce         TEXT NOT NULL,
    redirect_to   TEXT,                  -- where to send the operator after a successful login
    created_ms    INTEGER NOT NULL
);
