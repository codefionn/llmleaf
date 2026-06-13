//! Hand-rolled line/frame parsing over `reqwest::Response::bytes_stream()`.
//!
//! Two stream shapes share the same byte-accumulation core:
//!
//! * **SSE** (streaming chat): events are `data: <json>` lines terminated by a blank
//!   line; the stream ends with the sentinel `data: [DONE]`, which we stop on without
//!   parsing (SPEC.md). We only consume `data:` lines — comments (`:`) and other SSE
//!   fields are ignored, which is all this surface needs.
//! * **NDJSON** (batch results): one JSON object per line.
//!
//! No SSE crate is pulled in; this keeps the dependency surface lean as instructed.

use crate::error::{Error, Result};
use crate::types::{BatchResultLine, ChatCompletionChunk};
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
