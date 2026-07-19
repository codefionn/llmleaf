//! Hand-rolled line/frame parsing over `reqwest::Response::bytes_stream()`.
//!
//! Three stream shapes share the same byte-accumulation core:
//!
//! * **SSE** (streaming chat): events are `data: <json>` lines terminated by a blank
//!   line; the stream ends with the sentinel `data: [DONE]`, which we stop on without
//!   parsing (SPEC.md). We only consume `data:` lines — comments (`:`) and other SSE
//!   fields are ignored, which is all this surface needs.
//! * **Typed SSE** (streaming responses): the same `data: <json>` framing, but the JSON
//!   is self-describing via its `type` field and there is **no** `[DONE]` sentinel — the
//!   stream ends after the terminal `response.completed` / `.incomplete` / `.failed`
//!   event. Unrecognised event types are skipped and the `"error"` event surfaces as an
//!   `Err`, mirroring how the chat stream surfaces a mid-stream failure.
//! * **NDJSON** (batch results): one JSON object per line.
//!
//! No SSE crate is pulled in; this keeps the dependency surface lean as instructed.

use crate::error::{Error, Result};
use crate::types::{BatchResultLine, ChatCompletionChunk, ResponsesStreamEvent};
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use serde::de::DeserializeOwned;

/// SSE sentinel line body that terminates a streaming chat response.
const DONE: &str = "[DONE]";

/// Turn a byte stream of SSE frames into a stream of decoded `ChatCompletionChunk`s,
/// stopping at the `data: [DONE]` sentinel.
pub(crate) fn sse_chunks<S>(byte_stream: S) -> impl Stream<Item = Result<ChatCompletionChunk>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    let mut buf = BytesMut::new();
    let mut byte_stream = byte_stream;
    let mut done = false;

    futures::stream::poll_fn(move |cx| {
        use std::task::Poll;
        loop {
            if done {
                return Poll::Ready(None);
            }

            // Emit any complete events already buffered.
            if let Some(line) = take_line(&mut buf) {
                match parse_sse_data_line(&line) {
                    SseLine::Done => {
                        done = true;
                        return Poll::Ready(None);
                    }
                    SseLine::Skip => continue,
                    SseLine::Json(json) => {
                        let parsed =
                            serde_json::from_str::<ChatCompletionChunk>(&json).map_err(Error::from);
                        return Poll::Ready(Some(parsed));
                    }
                }
            }

            // Need more bytes.
            match byte_stream.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => {
                    buf.extend_from_slice(&chunk);
                }
                Poll::Ready(Some(Err(e))) => {
                    done = true;
                    return Poll::Ready(Some(Err(Error::Http(e))));
                }
                Poll::Ready(None) => {
                    // Flush a trailing line with no terminating newline.
                    if let Some(line) = take_remaining(&mut buf) {
                        match parse_sse_data_line(&line) {
                            SseLine::Done => {}
                            SseLine::Skip => {}
                            SseLine::Json(json) => {
                                done = true;
                                let parsed = serde_json::from_str::<ChatCompletionChunk>(&json)
                                    .map_err(Error::from);
                                return Poll::Ready(Some(parsed));
                            }
                        }
                    }
                    done = true;
                    return Poll::Ready(None);
                }
            }
        }
    })
}

/// Turn a byte stream of typed Responses SSE frames into a stream of decoded
/// [`ResponsesStreamEvent`]s. There is no `[DONE]` sentinel: the stream ends after the
/// terminal `response.completed` / `.incomplete` / `.failed` event (or when the connection
/// closes). Unrecognised event types are skipped; the `"error"` event becomes an `Err`.
pub(crate) fn sse_responses<S>(byte_stream: S) -> impl Stream<Item = Result<ResponsesStreamEvent>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    let mut buf = BytesMut::new();
    let mut byte_stream = byte_stream;
    let mut done = false;

    futures::stream::poll_fn(move |cx| {
        use std::task::Poll;
        loop {
            if done {
                return Poll::Ready(None);
            }

            if let Some(line) = take_line(&mut buf) {
                match parse_responses_frame(&line) {
                    ResponsesFrame::Skip => continue,
                    ResponsesFrame::Decode(err) => return Poll::Ready(Some(Err(err))),
                    ResponsesFrame::Error(err) => {
                        done = true;
                        return Poll::Ready(Some(Err(err)));
                    }
                    ResponsesFrame::Event(ev) => {
                        // Terminal events are yielded, then the stream ends (no sentinel).
                        if ev.is_terminal() {
                            done = true;
                        }
                        return Poll::Ready(Some(Ok(ev)));
                    }
                }
            }

            match byte_stream.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => buf.extend_from_slice(&chunk),
                Poll::Ready(Some(Err(e))) => {
                    done = true;
                    return Poll::Ready(Some(Err(Error::Http(e))));
                }
                Poll::Ready(None) => {
                    // Flush a trailing frame with no terminating newline.
                    if let Some(line) = take_remaining(&mut buf) {
                        match parse_responses_frame(&line) {
                            ResponsesFrame::Event(ev) => {
                                done = true;
                                return Poll::Ready(Some(Ok(ev)));
                            }
                            ResponsesFrame::Error(err) => {
                                done = true;
                                return Poll::Ready(Some(Err(err)));
                            }
                            ResponsesFrame::Decode(err) => {
                                done = true;
                                return Poll::Ready(Some(Err(err)));
                            }
                            ResponsesFrame::Skip => {}
                        }
                    }
                    done = true;
                    return Poll::Ready(None);
                }
            }
        }
    })
}

/// Turn a byte stream of NDJSON (`application/x-ndjson`) into a stream of decoded
/// `BatchResultLine`s — one JSON object per line.
pub(crate) fn ndjson_lines<S>(byte_stream: S) -> impl Stream<Item = Result<BatchResultLine>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    ndjson::<S, BatchResultLine>(byte_stream)
}

/// Generic NDJSON decoder.
fn ndjson<S, T>(byte_stream: S) -> impl Stream<Item = Result<T>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
    T: DeserializeOwned,
{
    let mut buf = BytesMut::new();
    let mut byte_stream = byte_stream;
    let mut done = false;

    futures::stream::poll_fn(move |cx| {
        use std::task::Poll;
        loop {
            if done {
                return Poll::Ready(None);
            }

            if let Some(line) = take_line(&mut buf) {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                return Poll::Ready(Some(
                    serde_json::from_str::<T>(trimmed).map_err(Error::from),
                ));
            }

            match byte_stream.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => buf.extend_from_slice(&chunk),
                Poll::Ready(Some(Err(e))) => {
                    done = true;
                    return Poll::Ready(Some(Err(Error::Http(e))));
                }
                Poll::Ready(None) => {
                    if let Some(line) = take_remaining(&mut buf) {
                        let trimmed = line.trim().to_string();
                        if !trimmed.is_empty() {
                            done = true;
                            return Poll::Ready(Some(
                                serde_json::from_str::<T>(&trimmed).map_err(Error::from),
                            ));
                        }
                    }
                    done = true;
                    return Poll::Ready(None);
                }
            }
        }
    })
}

enum SseLine {
    Json(String),
    Done,
    Skip,
}

/// Interpret one raw line of an SSE stream. We only care about `data:` lines.
fn parse_sse_data_line(line: &str) -> SseLine {
    let line = line.trim_end_matches(['\r', '\n']);
    let Some(rest) = line.strip_prefix("data:") else {
        // Comments (`:`), `event:`, `id:`, blank separators — all irrelevant here.
        return SseLine::Skip;
    };
    let body = rest.trim();
    if body.is_empty() {
        return SseLine::Skip;
    }
    if body == DONE {
        return SseLine::Done;
    }
    SseLine::Json(body.to_string())
}

/// The outcome of interpreting one raw line of a Responses SSE stream. The `Event` variant
/// is intentionally large: it is a transient dispatch result, constructed and immediately
/// matched per frame, so boxing it would only add a heap allocation on the read path (the
/// event is already yielded to the caller by value).
#[allow(clippy::large_enum_variant)]
enum ResponsesFrame {
    /// A recognised, non-error event to yield.
    Event(ResponsesStreamEvent),
    /// The `"error"` event — surfaced as a typed [`Error::Api`], terminating the stream.
    Error(Error),
    /// A `data:` frame that failed to JSON-decode.
    Decode(Error),
    /// A non-`data:` line, an empty body, or an unrecognised event type — ignored.
    Skip,
}

/// Interpret one raw SSE line for the Responses stream. There is no `[DONE]` sentinel; we
/// decode `data:` frames via their `type` field, skip unrecognised types, and turn the
/// `"error"` event into a typed error.
fn parse_responses_frame(line: &str) -> ResponsesFrame {
    let line = line.trim_end_matches(['\r', '\n']);
    let Some(rest) = line.strip_prefix("data:") else {
        // The redundant `event:` line, comments (`:`), `id:`, blank separators — ignored.
        return ResponsesFrame::Skip;
    };
    let body = rest.trim();
    if body.is_empty() {
        return ResponsesFrame::Skip;
    }
    let event: ResponsesStreamEvent = match serde_json::from_str(body) {
        Ok(ev) => ev,
        Err(e) => return ResponsesFrame::Decode(Error::from(e)),
    };
    if !event.is_recognised() {
        // Forward compatibility: SDKs ignore event types they don't recognise (SPEC.md).
        return ResponsesFrame::Skip;
    }
    if event.is_error() {
        // A mid-stream failure. 502 = upstream failure per SPEC.md's status table.
        return ResponsesFrame::Error(Error::Api {
            status: 502,
            message: event.error_message(),
        });
    }
    ResponsesFrame::Event(event)
}

/// Pop one complete `\n`-terminated line (without the trailing newline) from the buffer,
/// or `None` if no full line is buffered yet.
fn take_line(buf: &mut BytesMut) -> Option<String> {
    let pos = buf.iter().position(|&b| b == b'\n')?;
    let line = buf.split_to(pos + 1);
    // Drop the trailing '\n'; keep CR handling to the parser.
    let s = String::from_utf8_lossy(&line[..line.len() - 1]).into_owned();
    Some(s)
}

/// Take whatever remains in the buffer at end-of-stream (no trailing newline).
fn take_remaining(buf: &mut BytesMut) -> Option<String> {
    if buf.is_empty() {
        return None;
    }
    let rest = buf.split();
    Some(String::from_utf8_lossy(&rest).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn bytes_stream(parts: Vec<&'static str>) -> impl Stream<Item = reqwest::Result<Bytes>> {
        stream::iter(
            parts
                .into_iter()
                .map(|p| Ok(Bytes::from_static(p.as_bytes()))),
        )
    }

    #[tokio::test]
    async fn sse_parses_chunks_and_stops_on_done() {
        let frames = vec![
            "data: {\"id\":\"a\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"He\"}}]}\n",
            "\n",
            "data: {\"id\":\"a\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\"}}]}\n\n",
            "data: [DONE]\n\n",
            // Anything after DONE must be ignored.
            "data: {\"id\":\"z\"}\n",
        ];
        let s = sse_chunks(bytes_stream(frames));
        let out: Vec<_> = s.collect().await;
        assert_eq!(out.len(), 2);
        let text: String = out
            .into_iter()
            .map(|c| c.unwrap().first_delta_text().unwrap_or("").to_string())
            .collect();
        assert_eq!(text, "Hello");
    }

    #[tokio::test]
    async fn sse_preserves_split_tool_call_deltas() {
        let frames = vec![
            r#"data: {"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Par"}}]}}]}

"#,
            r#"data: {"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"is\"}"}}]}}]}

"#,
            r#"data: {"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

"#,
        ];

        let chunks: Vec<ChatCompletionChunk> = sse_chunks(bytes_stream(frames))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|chunk| chunk.unwrap())
            .collect();

        let calls: Vec<_> = chunks
            .iter()
            .flat_map(|chunk| &chunk.choices)
            .flat_map(|choice| &choice.delta.tool_calls)
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].kind.as_deref(), Some("function"));
        assert_eq!(
            calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        let arguments: String = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(arguments, r#"{"city":"Paris"}"#);
        assert_eq!(
            chunks[2].choices[0].finish_reason,
            Some(crate::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn sse_handles_split_across_byte_chunks() {
        // A single SSE line delivered in two arbitrary byte slices.
        let frames = vec![
            "data: {\"id\":\"a\",\"object\":\"chat.completion.chunk\",\"created\":1,\"mod",
            "el\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n",
        ];
        let s = sse_chunks(bytes_stream(frames));
        let out: Vec<_> = s.collect().await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().first_delta_text(), Some("hi"));
    }

    #[tokio::test]
    async fn sse_responses_parses_typed_events_and_stops_on_completed() {
        // A full typed-event turn: created -> output_item.added(function_call) ->
        // function_call_arguments.delta -> output_text.delta x2 -> completed. An
        // unrecognised event type sits in the middle and a stray frame trails the
        // terminal event; both must be dropped. `event:` lines are redundant noise.
        let frames = vec![
            r#"event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","status":"in_progress","model":"m"}}

"#,
            r#"event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":""}}

"#,
            r#"event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","output_index":0,"delta":"arg-chunk"}

"#,
            r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":3,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hi"}

"#,
            r#"data: {"type":"response.output_text.delta","sequence_number":4,"delta":"!"}

"#,
            // Unrecognised event type (outside the `response.*`/`error` namespace) — skipped.
            r#"event: queue.status
data: {"type":"queue.status","sequence_number":5}

"#,
            r#"event: response.completed
data: {"type":"response.completed","sequence_number":6,"response":{"id":"resp_1","object":"response","status":"completed","model":"m","store":false,"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi!","annotations":[]}]}],"usage":{"input_tokens":11,"output_tokens":2,"total_tokens":13}}}

"#,
            // Anything after the terminal event must be ignored.
            r#"data: {"type":"response.output_text.delta","sequence_number":7,"delta":"ignored"}

"#,
        ];

        let events: Vec<ResponsesStreamEvent> = sse_responses(bytes_stream(frames))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.completed",
            ]
        );

        // Accumulated assistant text from the output_text deltas.
        let text: String = events.iter().filter_map(|e| e.output_text_delta()).collect();
        assert_eq!(text, "Hi!");

        // Terminal snapshot carries the full output and usage.
        let terminal = events.last().unwrap();
        assert!(terminal.is_terminal());
        let snapshot = terminal.terminal_response().unwrap();
        assert_eq!(snapshot.output_text(), "Hi!");
        assert_eq!(snapshot.usage.as_ref().unwrap().input_tokens, 11);
        assert_eq!(snapshot.store, Some(false));
    }

    #[tokio::test]
    async fn sse_responses_surfaces_error_event_as_api_error() {
        let frames = vec![
            r#"event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"r","object":"response","status":"in_progress","model":"m"}}

"#,
            r#"event: error
data: {"type":"error","sequence_number":1,"code":"server_error","message":"upstream exploded"}

"#,
            // Never reached: the error event terminates the stream.
            r#"data: {"type":"response.output_text.delta","sequence_number":2,"delta":"nope"}

"#,
        ];

        let out: Vec<Result<ResponsesStreamEvent>> =
            sse_responses(bytes_stream(frames)).collect().await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].as_ref().unwrap().kind, "response.created");
        match out[1].as_ref().unwrap_err() {
            Error::Api { status, message } => {
                assert_eq!(*status, 502);
                assert_eq!(message, "upstream exploded");
            }
            other => panic!("expected Error::Api, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn ndjson_parses_lines() {
        let lines = vec![
            "{\"custom_id\":\"a\",\"response\":{\"status_code\":200,\"body\":{\"id\":\"x\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"m\",\"choices\":[]}}}\n",
            "{\"custom_id\":\"b\",\"error\":{\"code\":\"e\",\"message\":\"boom\"}}\n",
        ];
        let s = ndjson_lines(bytes_stream(lines));
        let out: Vec<_> = s.collect().await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].as_ref().unwrap().custom_id, "a");
        assert_eq!(
            out[1].as_ref().unwrap().error.as_ref().unwrap().message,
            "boom"
        );
    }
}
