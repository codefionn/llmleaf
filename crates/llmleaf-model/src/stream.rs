use std::collections::BTreeMap;
use std::pin::Pin;

use futures::{Stream, StreamExt};

use crate::{ChatResponse, Choice, FinishReason, ModelError, StreamChunk, ToolCall, Usage};

/// The canonical streaming response: a pinned, boxed, `Send` stream of [`StreamChunk`]s.
///
/// Everything inside the core moves this type. It is intentionally a trait object so providers
/// (compiled or WASM) and edge mappings can all produce/consume it without the core knowing their
/// concrete stream types.
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, ModelError>> + Send>>;

/// Collect a [`ResponseStream`] into a non-streaming [`ChatResponse`].
///
/// Principle 4 in code: non-streaming is *defined as* a collected stream. There is no separate
/// non-streaming path — output edges that need a whole object call this.
pub async fn collect(mut stream: ResponseStream) -> Result<ChatResponse, ModelError> {
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item?);
    }
    Ok(collect_chunks(chunks))
}

/// The synchronous core of [`collect`]: fold an already-materialized sequence of chunks into a
/// [`ChatResponse`]. Used when the chunks arrive all at once rather than streamed — e.g. a batch
/// result line that is itself a whole upstream response, mapped to chunks then folded here without
/// spinning up a stream (the JSONL line mapper is synchronous).
pub fn collect_chunks<I: IntoIterator<Item = StreamChunk>>(chunks: I) -> ChatResponse {
    let mut id = String::new();
    let mut model = String::new();
    let mut usage = Usage::default();
    // index -> accumulating choice
    let mut texts: BTreeMap<u32, String> = BTreeMap::new();
    let mut finishes: BTreeMap<u32, FinishReason> = BTreeMap::new();
    // index -> (call index -> partial tool call)
    let mut tools: BTreeMap<u32, BTreeMap<u32, PartialToolCall>> = BTreeMap::new();

    for item in chunks {
        match item {
            StreamChunk::Start { id: i, model: m } => {
                id = i;
                model = m;
            }
            StreamChunk::Content { index, delta } => {
                texts.entry(index).or_default().push_str(&delta);
            }
            StreamChunk::ToolCall { index, call } => {
                let entry = tools
                    .entry(index)
                    .or_default()
                    .entry(call.index)
                    .or_default();
                if let Some(id) = call.id {
                    entry.id = id;
                }
                if let Some(name) = call.name {
                    entry.name = name;
                }
                if let Some(args) = call.arguments {
                    entry.arguments.push_str(&args);
                }
            }
            StreamChunk::Usage(u) => usage = u,
            StreamChunk::Finish { index, reason } => {
                finishes.insert(index, reason);
            }
        }
    }

    // Fold per-choice accumulators into ordered choices.
    let mut indices: Vec<u32> = texts
        .keys()
        .chain(finishes.keys())
        .chain(tools.keys())
        .copied()
        .collect();
    indices.sort_unstable();
    indices.dedup();

    let choices = indices
        .into_iter()
        .map(|index| Choice {
            index,
            text: texts.remove(&index).unwrap_or_default(),
            tool_calls: tools
                .remove(&index)
                .map(|m| m.into_values().map(PartialToolCall::into_call).collect())
                .unwrap_or_default(),
            finish_reason: finishes.remove(&index),
        })
        .collect();

    ChatResponse {
        id,
        model,
        choices,
        usage,
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl PartialToolCall {
    fn into_call(self) -> ToolCall {
        ToolCall {
            id: self.id,
            name: self.name,
            arguments: self.arguments,
        }
    }
}
