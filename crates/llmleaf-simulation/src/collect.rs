//! Simulation of [`llmleaf_model::collect`] — the stream-folding that defines the non-streaming
//! response (SOUL.md principle 4: "a non-streaming response is a *collected* stream").
//!
//! A seed generates an arbitrary-but-valid chunk sequence — interleaved content and tool-call
//! fragments across several (possibly sparse) choice indices, with usage updates and per-choice
//! finishes scattered through. `collect` folds it; an independent reference fold ([`reference_fold`]),
//! written by a deliberately different method (repeated filtered scans over the same chunk list,
//! rather than `collect`'s incremental accumulators), computes what the result must be.

use llmleaf_model::{
    ChatResponse, Choice, FinishReason, ModelError, ResponseStream, StreamChunk, ToolCall,
    ToolCallDelta, Usage,
};

use crate::ensure;
use crate::rng::Rng;

/// Build a valid chunk sequence for a seed: exactly one `Start`, then a randomly interleaved body of
/// content/tool-call/usage/finish chunks across a handful of (possibly non-contiguous) choice indices.
fn generate_chunks(rng: &mut Rng) -> Vec<StreamChunk> {
    let mut chunks = vec![StreamChunk::Start {
        id: format!("resp-{}", rng.next_u64() % 10_000),
        model: format!("model-{}", rng.below(5)),
    }];

    // A small set of (possibly non-contiguous) choice indices, e.g. {0, 3, 4}.
    let nindices = rng.range(1, 3) as usize;
    let mut indices: Vec<u32> = Vec::new();
    while indices.len() < nindices {
        let candidate = rng.below(5) as u32;
        if !indices.contains(&candidate) {
            indices.push(candidate);
        }
    }

    // The "to-do" of fragments to emit; we drain it in a random order so the fold faces arbitrary
    // interleaving (content for one choice landing between tool fragments of another, etc.).
    let mut pending: Vec<StreamChunk> = Vec::new();
    for &index in &indices {
        for _ in 0..rng.below(4) {
            pending.push(StreamChunk::Content {
                index,
                delta: format!("c{}", rng.below(1000)),
            });
        }
        let ntools = rng.below(3) as u32;
        for call_index in 0..ntools {
            // Each tool call is built from several deltas: an opener carrying id+name, then argument
            // fragments. The fold must concatenate the arguments and keep the last id/name.
            pending.push(StreamChunk::ToolCall {
                index,
                call: ToolCallDelta {
                    index: call_index,
                    id: Some(format!("call-{index}-{call_index}")),
                    name: Some(format!("fn{}", rng.below(100))),
                    arguments: None,
                },
            });
            for _ in 0..rng.range(1, 3) {
                pending.push(StreamChunk::ToolCall {
                    index,
                    call: ToolCallDelta {
                        index: call_index,
                        id: None,
                        name: None,
                        arguments: Some(format!("a{}", rng.below(1000))),
                    },
                });
            }
        }
        // Usage updates sprinkled in (the fold keeps the last).
        if rng.chance(1, 2) {
            pending.push(StreamChunk::Usage(random_usage(rng)));
        }
        // Most choices finish; some are left open (finish_reason None); some finish more than once
        // (the fold keeps the last).
        for _ in 0..rng.below(3) {
            pending.push(StreamChunk::Finish {
                index,
                reason: random_finish(rng),
            });
        }
    }

    // Shuffle `pending` into the stream via Fisher–Yates with the seeded PRNG (deterministic).
    for i in (1..pending.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        pending.swap(i, j);
    }
    chunks.extend(pending);
    chunks
}

fn random_usage(rng: &mut Rng) -> Usage {
    let prompt = rng.below(1000);
    let completion = rng.below(1000);
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cost_usd: None,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
    }
}

fn random_finish(rng: &mut Rng) -> FinishReason {
    use FinishReason::*;
    *[Stop, Length, ToolCalls, ContentFilter, Error]
        .get(rng.below(5) as usize)
        .unwrap()
}

/// Independent reference fold — repeated filtered scans, structurally distinct from `collect`'s
/// incremental accumulators.
fn reference_fold(chunks: &[StreamChunk]) -> ChatResponse {
    // Identity: last Start wins (there is exactly one here).
    let (mut id, mut model) = (String::new(), String::new());
    for c in chunks {
        if let StreamChunk::Start { id: i, model: m } = c {
            id = i.clone();
            model = m.clone();
        }
    }

    // Usage: the last report, or default.
    let mut usage = Usage::default();
    for c in chunks {
        if let StreamChunk::Usage(u) = c {
            usage = *u;
        }
    }

    // A choice index exists iff it appears in any content, tool call, or finish.
    let mut indices: Vec<u32> = Vec::new();
    for c in chunks {
        let idx = match c {
            StreamChunk::Content { index, .. }
            | StreamChunk::ToolCall { index, .. }
            | StreamChunk::Finish { index, .. } => Some(*index),
            _ => None,
        };
        if let Some(idx) = idx {
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
    }
    indices.sort_unstable();

    let choices = indices
        .into_iter()
        .map(|index| {
            // Text: concatenate this index's content deltas in emission order.
            let mut text = String::new();
            for c in chunks {
                if let StreamChunk::Content { index: i, delta } = c {
                    if *i == index {
                        text.push_str(delta);
                    }
                }
            }

            // Tool calls: distinct call indices for this choice, in ascending order.
            let mut call_indices: Vec<u32> = Vec::new();
            for c in chunks {
                if let StreamChunk::ToolCall { index: i, call } = c {
                    if *i == index && !call_indices.contains(&call.index) {
                        call_indices.push(call.index);
                    }
                }
            }
            call_indices.sort_unstable();

            let tool_calls = call_indices
                .into_iter()
                .map(|ci| {
                    let (mut tid, mut tname, mut targs) =
                        (String::new(), String::new(), String::new());
                    for c in chunks {
                        if let StreamChunk::ToolCall { index: i, call } = c {
                            if *i == index && call.index == ci {
                                if let Some(x) = &call.id {
                                    tid = x.clone();
                                }
                                if let Some(x) = &call.name {
                                    tname = x.clone();
                                }
                                if let Some(x) = &call.arguments {
                                    targs.push_str(x);
                                }
                            }
                        }
                    }
                    ToolCall {
                        id: tid,
                        name: tname,
                        arguments: targs,
                    }
                })
                .collect();

            // Finish: the last finish for this index, if any.
            let mut finish_reason = None;
            for c in chunks {
                if let StreamChunk::Finish { index: i, reason } = c {
                    if *i == index {
                        finish_reason = Some(*reason);
                    }
                }
            }

            Choice {
                index,
                text,
                // The simulation collector exercises transport/usage accounting, not reasoning
                // fidelity; thinking blocks are not reconstructed here.
                thinking: Vec::new(),
                tool_calls,
                finish_reason,
            }
        })
        .collect();

    ChatResponse {
        id,
        model,
        choices,
        usage,
    }
}

fn stream_of(chunks: Vec<StreamChunk>) -> ResponseStream {
    let items: Vec<Result<StreamChunk, ModelError>> = chunks.into_iter().map(Ok).collect();
    Box::pin(futures::stream::iter(items))
}

/// Run one full `collect` folding scenario for `seed`.
pub fn run_scenario(seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    let chunks = generate_chunks(&mut rng);
    let expected = reference_fold(&chunks);
    let actual = futures::executor::block_on(llmleaf_model::collect(stream_of(chunks.clone())))
        .map_err(|e| format!("seed={seed}: collect returned an error: {e}"))?;
    ensure!(
        actual == expected,
        "seed={seed}: collect fold diverged from the reference\n  actual:   {actual:?}\n  expected: {expected:?}"
    );
    Ok(())
}
