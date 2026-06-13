//! Example: submit an asynchronous batch to a stood-up llmleaf server, poll it, and print results.
//!
//! A *consumer* of the proxy's batch surface. It speaks llmleaf's inline batch dialect — an array of
//! OpenAI-shaped chat requests, each tagged with a `custom_id`:
//!
//! ```text
//! POST /v1/batches            { "requests": [ { "custom_id", "body": <chat request> }, … ] }
//! GET  /v1/batches/{id}       poll status
//! GET  /v1/batches/{id}/results   stream results as JSONL (one line per request)
//! ```
//!
//! The batch id it gets back is opaque — it encodes which upstream provider owns the job, so the poll
//! and results calls are routed entirely by the id (any llmleaf node can serve them; SOUL principle 9).
//! It reads connection settings from `llmleaf.toml` via the shared [`common`] helper.
//!
//! Run it (with the server up via `cargo run -p llmleaf`, pointed at a real batch-capable provider —
//! OpenAI, Anthropic, or Mistral):
//!
//! ```text
//! cargo run -p llmleaf --example batch              # default model = first route in llmleaf.toml
//! cargo run -p llmleaf --example batch -- smart     # pick a model by name
//! ```
//!
//! Real batches can take up to ~24h; this client polls a bounded number of times and then prints
//! whatever results are ready (often none yet for a fresh submission — that is expected).

#[path = "common/mod.rs"]
mod common;

use common::BoxError;
use serde_json::{json, Value};

/// The prompts to batch. Each becomes one request with its own `custom_id`.
const PROMPTS: &[(&str, &str)] = &[
    ("haiku", "Write a haiku about leaves."),
    ("fact", "State one surprising fact about octopuses."),
    ("math", "What is 17 * 23? Answer with just the number."),
];

/// How many times to poll, and how long to wait between polls. Batches are slow by design; this is a
/// demo, not a production poller.
const POLL_ATTEMPTS: usize = 10;
const POLL_INTERVAL_SECS: u64 = 6;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let conn = common::connect()?;
    let model = common::pick_model(&conn.config, std::env::args().nth(1))
        .ok_or("no model given and no [[routes]] in the config to default to")?;
    let client = reqwest::Client::new();

    println!("submitting {} requests to model '{model}' …", PROMPTS.len());

    // Build the inline batch: one OpenAI-shaped chat request per prompt.
    let requests: Vec<Value> = PROMPTS
        .iter()
        .map(|(id, prompt)| {
            json!({
                "custom_id": id,
                "body": {
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "max_tokens": 256
                }
            })
        })
        .collect();

    let created = post(
        &client,
        &conn,
        "/v1/batches",
        json!({ "requests": requests }),
    )
    .await?;
    let batch_id = created
        .get("id")
        .and_then(Value::as_str)
        .ok_or("create response had no batch id")?
        .to_string();
    println!(
        "  batch id: {batch_id}\n  status:   {}",
        status_of(&created)
    );

    // Poll until the job reaches a terminal state or we run out of attempts.
    let mut terminal = is_terminal(&created);
    for attempt in 1..=POLL_ATTEMPTS {
        if terminal {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(POLL_INTERVAL_SECS)).await;
        let handle = get(&client, &conn, &format!("/v1/batches/{batch_id}")).await?;
        let status = status_of(&handle);
        let counts = handle.get("request_counts").cloned().unwrap_or(Value::Null);
        println!("  poll {attempt}/{POLL_ATTEMPTS}: status={status} counts={counts}");
        terminal = is_terminal(&handle);
    }

    if !terminal {
        println!(
            "\nstill processing after {} polls — fetch results later with:\n  \
             curl -H 'Authorization: Bearer <key>' {}/v1/batches/{batch_id}/results",
            POLL_ATTEMPTS, conn.base_url
        );
        return Ok(());
    }

    // Fetch results — a JSONL body, one line per request.
    println!("\nresults:");
    let resp = client
        .get(format!("{}/v1/batches/{batch_id}/results", conn.base_url))
        .bearer_auth(&conn.bearer)
        .send()
        .await?;
    let body = resp.text().await?;
    for line in body.lines().filter(|l| !l.is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        let cid = v.get("custom_id").and_then(Value::as_str).unwrap_or("?");
        if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
            println!("  [{cid}] ERROR: {err}");
        } else {
            let text = v
                .get("response")
                .and_then(|r| r.get("body"))
                .and_then(|b| b.get("choices"))
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
                .unwrap_or("(no content)");
            println!("  [{cid}] {text}");
        }
    }
    Ok(())
}

fn status_of(v: &Value) -> &str {
    v.get("status").and_then(Value::as_str).unwrap_or("unknown")
}

/// A batch is done (one way or another) when it is no longer processing.
fn is_terminal(v: &Value) -> bool {
    matches!(
        status_of(v),
        "completed" | "failed" | "expired" | "cancelled" | "canceled"
    )
}

async fn post(
    client: &reqwest::Client,
    conn: &common::Conn,
    path: &str,
    body: Value,
) -> Result<Value, BoxError> {
    let resp = client
        .post(format!("{}{path}", conn.base_url))
        .bearer_auth(&conn.bearer)
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        return Err(format!("{status}: {}", common::error_detail(&text)).into());
    }
    Ok(serde_json::from_str(&text)?)
}

async fn get(client: &reqwest::Client, conn: &common::Conn, path: &str) -> Result<Value, BoxError> {
    let resp = client
        .get(format!("{}{path}", conn.base_url))
        .bearer_auth(&conn.bearer)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        return Err(format!("{status}: {}", common::error_detail(&text)).into());
    }
    Ok(serde_json::from_str(&text)?)
}
