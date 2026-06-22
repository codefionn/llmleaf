//! The `cargo test` allocation guard.
//!
//! Timing is machine-dependent and belongs in the `llmleaf-bench` binary; allocation *counts* are
//! deterministic and are the figure SOUL.md actually legislates ("justify every allocation"). So this
//! is where the constitution becomes a test: each per-request operation has an allocation ceiling, and
//! a change that smuggles in an extra clone trips it.
//!
//! Two tiers:
//!   - **Zero-allocation guarantees** — the per-request lookups principle 5 calls "a lookup, never
//!     arithmetic": unknown-key rejection, routing, health checks, pricing. These must allocate
//!     *nothing*. An assertion of `== 0` is the sharpest possible regression tripwire.
//!   - **Budgets** — the loops and mappings that legitimately allocate (collecting a stream, mapping a
//!     dialect, running the whole edge). Each has a `<=` ceiling a little above today's count: tight
//!     enough to catch a doubling, loose enough to tolerate stdlib noise.
//!
//! Run a single seed of the live numbers with `cargo run -p llmleaf-bench --release`.
//!
//! INVARIANT — this file must hold exactly ONE `#[test]`. Allocation counting is process-global
//! (`alloc::measure_allocs`), so it is only correct while a single thread allocates. `cargo test` runs
//! the `#[test]`s in a binary concurrently across threads; a second test here would allocate inside
//! this one's measurement window and silently corrupt the counts. Keep all measurement in the one test
//! below (or switch the target to a single-threaded custom runner) — do not add a sibling `#[test]`.

use llmleaf_bench::alloc::{self, CountingAllocator};

// The whole point — without it the counts are all zero and every assertion below passes vacuously.
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Operations that must allocate nothing at all (principle 5: a lookup and a comparison).
const ZERO_ALLOC: &[&str] = &[
    "keys/authorize_miss",
    "keys/authorize_denied", // warm identity, model gated out — a `contains`, no id to clone
    "router/resolve_hit",
    "router/resolve_miss",
    "health/is_down_clear",
    "health/clear_noop",
    "ratelimit/admit", // semaphore permit = an Arc clone; mutex + bucket arithmetic — never a heap alloc
    "pricing/price_known",
    "pricing/price_unknown",
    "speech/content_type", // `response_format` → a `&'static str`, never an allocation
];

/// Operations that allocate, with a per-`run` ceiling on the allocation count. Each current count is
/// noted; the ceiling sits a little above it.
const CEILINGS: &[(&str, u64)] = &[
    // Single, identifiable allocations — held tight so a +1 regression trips immediately.
    ("keys/authorize_hit", 1),        // the returned key-id string
    ("keys/authorize_restricted", 1), // ditto, after the allow-list check
    ("health/penalize", 1),           // the provider name inserted into the cooldown map
    ("events/emit_no_subscriber", 1), // the Arc<Envelope>
    ("events/emit_with_subscriber", 1),
    // The per-request loops — budgets with headroom over the observed counts.
    ("collect/small", 8),                    // ~5
    ("collect/chunks_sync", 8),              // ~5 (the sync fold, no stream machinery)
    ("collect/error", 4), // ~1 (mid-stream failure propagated straight out of the fold)
    ("collect/tools_multichoice", 20), // ~13 (tool-call + multi-choice fold)
    ("engine/run_collect", 48), // ~36
    ("hotpath/edge_nonstream", 53), // ~42 (map-out serialises a borrowed CompletionView, no Value)
    ("hotpath/edge_nonstream_payloads", 73), // ~58 (= edge_nonstream + request serialisation)
    ("hotpath/edge_stream", 48), // ~36 (encode_into writes borrowed structs into a reused buffer)
    // Edge mappings.
    ("openai/parse_small", 3), // ~2 (role borrowed out of the map, not copied)
    ("openai/parse_large", 88), // ~78 (role borrowed; the rest scales with message count)
    ("openai/encode_stream", 0), // 0 — per-frame encode allocates nothing (reused buffer)
    ("openai/response_collect", 6), // ~3 (borrowed CompletionView serialised to bytes — no Value tree)
    ("serde/request_to_value", 24), // ~16
    // The other three modalities + the batch envelope (the modalities memo). Edge mappings, the engine
    // middle through the shared dispatch, and the full per-request edge — held the same way as chat.
    ("embeddings/parse", 2), // ~0 (inputs moved; the input Vec reuses the source buffer)
    ("embeddings/parse_large", 2), // ~0 (inputs moved out of the array; the Vec reuses the source buffer)
    ("embeddings/response_float", 4), // ~2 (borrowed EmbeddingListView; the float slice serialises in place)
    ("embeddings/response_base64", 10), // ~6 (only the irreducible per-vector f32→base64 buffer + String)
    ("engine/embed", 40),               // ~29 (dispatch + price + emit batch tail)
    ("hotpath/embeddings", 40),         // ~32 (parse → engine → borrowed-view serialise)
    ("speech/parse", 2),                // ~0 (required fields moved out of the body)
    ("speech/collect", 4),              // ~1 (one BytesMut, grown in place as data frames append)
    ("hotpath/speech_stream", 40), // ~29 (parse → engine → drain the instrumented audio stream)
    ("transcription/build_request", 2), // ~0 (form fields moved; Bytes is refcounted)
    ("transcription/encode_json", 6), // ~3 (the minimal { text } object)
    ("transcription/encode_verbose", 12), // ~8 (task/language/duration/text)
    ("transcription/encode_text", 2), // ~1 (the transcript cloned verbatim, no JSON object)
    ("engine/transcribe", 36),     // ~27 (dispatch + price + emit batch tail)
    ("hotpath/transcription", 44), // ~32 (form → engine → encode_transcription)
    ("batch/parse_create", 24),    // ~17 (each item body parsed by the chat mapper)
    ("batch/handle_to_json", 26),  // ~19 (the job object + the request-counts superset)
    ("batch/result_line", 6), // ~3 (success arm reuses the borrowed CompletionView, serialised to bytes)
    ("batch/result_line_error", 4), // ~2 (error arm builds the { error: { code, message } } object)
    // A real provider's build-request + response-parse path over the fake transport (no network). These
    // are inherently allocation-heavy — the point is to MEASURE the edge, not to make it cheap: the chat
    // path builds the OpenAI-wire request as a Value tree and serialises it, runs the async-stream SSE
    // parser (a per-frame Value parse + per-chunk allocs), then folds the collected stream; embed builds
    // the request body, parses the JSON response, and rebuilds the canonical response. Ceilings sit a
    // little above the observed counts, same as every budget above.
    ("provider/openai_chat_sse", 160), // ~133 (request build + SSE parse + collect)
    ("provider/openai_embed", 64),     // ~51 (request build + JSON response parse)
];

#[test]
fn allocation_counts_stay_within_ceilings() {
    assert!(
        alloc::is_active(),
        "the counting allocator is not installed — this test would pass vacuously"
    );

    let suite = llmleaf_bench::run_alloc_only(None);
    let mut failures = Vec::new();

    for &name in ZERO_ALLOC {
        let Some(row) = suite.row(name) else {
            failures.push(format!("missing case `{name}` (renamed or dropped?)"));
            continue;
        };
        if row.alloc.allocs != 0 || row.alloc.reallocs != 0 {
            failures.push(format!(
                "`{name}` must allocate nothing (principle 5), but made {} alloc(s) + {} realloc(s)",
                row.alloc.allocs, row.alloc.reallocs
            ));
        }
    }

    for &(name, max) in CEILINGS {
        let Some(row) = suite.row(name) else {
            failures.push(format!("missing case `{name}` (renamed or dropped?)"));
            continue;
        };
        if row.alloc.allocs > max {
            failures.push(format!(
                "`{name}` allocated {} > ceiling {max} — justify the new allocation or raise the ceiling",
                row.alloc.allocs
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "allocation regressions:\n  {}\n\nfull table:\n{}",
        failures.join("\n  "),
        suite.table(),
    );
}
