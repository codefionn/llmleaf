//! Example: list the models a stood-up llmleaf node knows about (`GET /v1/models`).
//!
//! A *consumer* of the proxy's model-catalog surface. The response is shaped like OpenRouter's
//! `GET /api/v1/models` (`{ "data": [ … ] }`): each entry is a model the node serves — the union of
//! explicitly-routed models and the bundled catalog — enriched with its **modality**, **max context**,
//! **max output**, **max thinking budget**, pricing, and any **rejected / default sampling parameters**
//! (e.g. a reasoning model that 400s on `temperature`). Filter by modality (`all|llm|tts|stt|
//! embedding`) and substring-search the id; both filters run server-side.
//!
//! Two views. By default you get the public catalog (ids, capabilities, pricing — no provider names).
//! With `--internal` (and an admin token available) the node adds each model's provider/fallback chain
//! and node-local health, which this example shows as a dimmed line under each routed model. The
//! provider topology is gated behind the same `x-admin-token` as the `/admin/*` surface; `down` is this
//! node's view, never a cluster verdict.
//!
//! Against the bundled `llmleaf.toml` (the `demo` route on the built-in `echo` provider, plus the
//! seed catalog) it works fully offline — no network, no credentials beyond the dev key.
//!
//! Run it (with the server up via `cargo run -p llmleaf`); `--help` lists every flag:
//!
//! ```text
//! cargo run -p llmleaf --example list-models                   # all modalities, public view
//! cargo run -p llmleaf --example list-models -- llm            # only LLM/chat models
//! cargo run -p llmleaf --example list-models -- embedding      # only embedding models
//! cargo run -p llmleaf --example list-models -- --search gpt   # substring filter over the id
//! cargo run -p llmleaf --example list-models -- --json         # raw JSON, for scripting
//! cargo run -p llmleaf --example list-models -- --internal     # add provider chain + health
//! cargo run -p llmleaf --example list-models -- --help         # full usage
//! ```
//!
//! Every flag falls back to its `LLMLEAF_*` env var (`--<modality>`/`LLMLEAF_MODALITY`,
//! `--search`/`LLMLEAF_SEARCH`, `--internal`/`LLMLEAF_INTERNAL`); the admin token for `--internal` is
//! read from `LLMLEAF_ADMIN_TOKEN`, else the config `[server].admin_token` (resolved, so `env:VAR`
//! works). Auth/base-url/config follow the same overrides documented in [`common`].

#[path = "common/mod.rs"]
mod common;

use std::io::IsTerminal;

use clap::Parser;
use common::BoxError;
use serde_json::Value;

/// List the models a stood-up llmleaf server serves (GET /v1/models), OpenRouter-shaped.
#[derive(Parser)]
#[command(name = "list-models")]
struct Args {
    /// Modality filter [default: all]. One of: all | llm | tts | stt | embedding.
    #[arg(value_enum, default_value_t = Modality::All, env = "LLMLEAF_MODALITY")]
    modality: Modality,

    /// Case-insensitive substring to match against model ids.
    #[arg(short, long, env = "LLMLEAF_SEARCH")]
    search: Option<String>,

    /// Print the raw JSON response instead of the formatted table.
    #[arg(short, long)]
    json: bool,

    /// Request the admin-enhanced view (provider chain + health). Sends the admin token as `x-admin-token`.
    #[arg(long, env = "LLMLEAF_INTERNAL")]
    internal: bool,

    /// Disable ANSI colour even when stdout is a TTY.
    #[arg(long)]
    no_color: bool,
}

/// The `?type=` filter, as a CLI value-enum. `All` sends no filter.
#[derive(Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
enum Modality {
    #[default]
    All,
    Llm,
    Tts,
    Stt,
    Embedding,
}

impl Modality {
    /// The `?type=` value to send, or `None` for "no filter".
    fn as_query(self) -> Option<&'static str> {
        match self {
            Modality::All => None,
            Modality::Llm => Some("llm"),
            Modality::Tts => Some("tts"),
            Modality::Stt => Some("stt"),
            Modality::Embedding => Some("embedding"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let conn = common::connect()?;

    // Build the query string. `type` is a fixed lowercase whitelist (never needs encoding); the search
    // value is percent-encoded inline so the example pulls no urlencoding crate (mirrors voices.rs).
    let mut params: Vec<String> = Vec::new();
    if let Some(t) = args.modality.as_query() {
        params.push(format!("type={t}"));
    }
    if let Some(s) = &args.search {
        params.push(format!("search={}", encode_query(s)));
    }
    let mut url = format!("{}/v1/models", conn.base_url);
    if !params.is_empty() {
        url.push('?');
        url.push_str(&params.join("&"));
    }

    // `--internal` requests the admin view; resolve the admin token and present it on top of the bearer.
    let mut req = reqwest::Client::new().get(&url).bearer_auth(&conn.bearer);
    if args.internal {
        let token = admin_token(&conn).ok_or(
            "--internal: no admin token (set LLMLEAF_ADMIN_TOKEN or [server].admin_token in the config)",
        )?;
        req = req.header("x-admin-token", token);
    }

    let resp = req.send().await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        let detail = common::error_detail(&body);
        // A rejected admin token surfaces as 401/403 — point at the likely cause.
        if args.internal && (status.as_u16() == 401 || status.as_u16() == 403) {
            return Err(format!("{status}: {detail} (admin token rejected for --internal)").into());
        }
        return Err(format!("{status}: {detail}").into());
    }

    if args.json {
        println!("{body}");
        return Ok(());
    }

    let parsed: Value =
        serde_json::from_str(&body).map_err(|e| format!("200 OK but invalid JSON: {e}"))?;
    let models = parsed
        .get("data")
        .and_then(Value::as_array)
        .ok_or("response had no `data` array")?;

    let painter = Painter::detect(args.no_color);
    let view = if args.internal { "internal" } else { "catalog" };
    println!(
        "list-models → {}   key: {}   ({} model{}, {view} view)\n",
        conn.base_url,
        conn.id_label,
        models.len(),
        if models.len() == 1 { "" } else { "s" },
    );
    if models.is_empty() {
        println!("(no models match)");
        return Ok(());
    }

    // Pad the id column so the numeric/pricing columns line up (mirrors voices.rs).
    let id_w = models
        .iter()
        .filter_map(|m| m.get("id").and_then(Value::as_str))
        .map(str::len)
        .max()
        .unwrap_or(0)
        .max("id".len());

    let header = format!(
        "  {:<9}  {:<id_w$}  {:>8}  {:>7}  {:>8}  {}",
        "modality", "id", "context", "output", "thinking", "pricing",
    );
    println!("{}", painter.dim(&header));

    for m in models {
        let id = m.get("id").and_then(Value::as_str).unwrap_or("?");
        println!(
            "  {:<9}  {:<id_w$}  {:>8}  {:>7}  {:>8}  {}",
            modality_label(m),
            id,
            num(&m["context_length"]),
            num(&m["top_provider"]["max_completion_tokens"]),
            num(&m["top_provider"]["max_thinking_tokens"]),
            pricing_label(m),
        );
        // A dimmed line naming any sampling params the model rejects + the defaults it pins.
        if let Some(line) = params_line(m, &painter) {
            println!("    {line}");
        }
        // Admin view: a dimmed provider-chain + health line under each model that has endpoints.
        if let Some(line) = internal_line(m, &painter) {
            println!("    {line}");
        }
    }

    Ok(())
}

/// Resolve the admin token for `--internal`: `LLMLEAF_ADMIN_TOKEN` wins, else the config's
/// `[server].admin_token` (via `Secret::resolve`, which also honours `env:VAR`). Empty ⇒ absent.
fn admin_token(conn: &common::Conn) -> Option<String> {
    std::env::var("LLMLEAF_ADMIN_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            conn.config
                .server
                .admin_token
                .as_ref()
                .and_then(|s| s.resolve())
                .filter(|s| !s.is_empty())
        })
}

/// A short modality label derived from the OpenRouter `architecture` block (the server's `?type=`
/// filter is authoritative; this is display-only, and tolerant enough for a real OpenRouter response).
fn modality_label(m: &Value) -> String {
    let arch = &m["architecture"];
    let list = |key: &str| -> Vec<String> {
        arch[key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    let out = list("output_modalities");
    let inp = list("input_modalities");
    let has = |v: &[String], s: &str| v.iter().any(|x| x == s);

    if has(&out, "embeddings") {
        "embedding".into()
    } else if has(&out, "audio") || has(&out, "speech") {
        "tts".into()
    } else if has(&inp, "audio") || has(&out, "transcription") {
        "stt".into()
    } else if has(&out, "text") {
        "llm".into()
    } else if let Some(s) = arch["modality"].as_str() {
        // Fall back to the arrow form's output side (e.g. "text->audio" → "audio").
        s.rsplit("->").next().unwrap_or("?").to_string()
    } else {
        "?".into()
    }
}

/// Render a numeric field, or `–` when it is null/absent (unknown — never shown as 0).
fn num(v: &Value) -> String {
    v.as_u64()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "–".into())
}

/// Render pricing as `$<in>/$<out> /Mtok`. The wire carries per-TOKEN decimal strings; multiply back to
/// per-Mtok for display. `null` pricing (a rate-less audio model) renders `(unpriced)` — distinct from a
/// genuine `$0.00`.
fn pricing_label(m: &Value) -> String {
    let p = &m["pricing"];
    if p.is_null() {
        return "(unpriced)".into();
    }
    let per_mtok = |key: &str| -> Option<f64> {
        p[key]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .map(|v| v * 1_000_000.0)
    };
    match (per_mtok("prompt"), per_mtok("completion")) {
        (Some(i), Some(o)) => format!("${i:.2}/${o:.2} /Mtok"),
        _ => "(unpriced)".into(),
    }
}

/// The admin-view chain/health line, or `None` when the model carries no `endpoints` (public view, or a
/// catalog-only model with no chain). Provider names appear only here — only the admin view returns them.
fn internal_line(m: &Value, painter: &Painter) -> Option<String> {
    let endpoints = m.get("endpoints").and_then(Value::as_array)?;
    if endpoints.is_empty() {
        return None;
    }
    let providers: Vec<&str> = endpoints
        .iter()
        .filter_map(|e| e.get("provider").and_then(Value::as_str))
        .collect();
    let health: Vec<String> = endpoints
        .iter()
        .map(|e| {
            let p = e.get("provider").and_then(Value::as_str).unwrap_or("?");
            let down = e.get("down").and_then(Value::as_bool).unwrap_or(false);
            if down {
                format!("{p} {}", painter.red("down"))
            } else {
                format!("{p} {}", painter.green("up"))
            }
        })
        .collect();
    let line = format!(
        "↳ route: {}   health: {}",
        providers.join(" → "),
        health.join(", "),
    );
    Some(painter.dim(&line))
}

/// A dimmed line naming the sampling params this model REJECTS and any default values it pins, from the
/// catalog's `unsupported_parameters` / `default_parameters`. `None` when the model declares neither (no
/// restriction known) — so a reasoning model that 400s on `temperature` shows it, an ordinary one stays
/// quiet. Additive OpenRouter-extension fields; a model without them simply omits the line.
fn params_line(m: &Value, painter: &Painter) -> Option<String> {
    let rejects: Vec<String> = m
        .get("unsupported_parameters")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let defaults: Vec<String> = m
        .get("default_parameters")
        .and_then(Value::as_object)
        .map(|o| o.iter().map(|(k, v)| format!("{k}={v}")).collect())
        .unwrap_or_default();
    if rejects.is_empty() && defaults.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if !rejects.is_empty() {
        parts.push(format!("rejects: {}", rejects.join(", ")));
    }
    if !defaults.is_empty() {
        parts.push(format!("defaults: {}", defaults.join(", ")));
    }
    Some(painter.dim(&format!("↳ {}", parts.join("   "))))
}

/// Percent-encode the few characters that would break a query value, keeping the example free of a
/// urlencoding crate (mirrors `common::base64_encode`). Model ids and typical search terms are plain.
fn encode_query(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        // RFC 3986 unreserved set — everything else is percent-encoded.
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Optional-ANSI renderer (trimmed from probe.rs): colour only on a real TTY, unless `--no-color` or
/// `NO_COLOR` is set.
struct Painter {
    color: bool,
}

impl Painter {
    fn detect(no_color: bool) -> Self {
        let color =
            std::io::stdout().is_terminal() && !no_color && std::env::var_os("NO_COLOR").is_none();
        Painter { color }
    }

    fn paint(&self, s: &str, code: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn dim(&self, s: &str) -> String {
        self.paint(s, "2")
    }
    fn green(&self, s: &str) -> String {
        self.paint(s, "32")
    }
    fn red(&self, s: &str) -> String {
        self.paint(s, "31")
    }
}
