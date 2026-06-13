//! Shared batch plumbing for first-party providers.
//!
//! Two pieces every batch dialect needs, kept here so each provider only writes its own wire mapping:
//!   - [`build_jsonl`] — serialize per-request lines into a JSONL body for the file-based dialects.
//!   - [`jsonl_result_stream`] — turn a result-file response into a canonical [`BatchResultStream`],
//!     parsed line-by-line (principle 4: never buffered whole). The per-line mapping is the caller's,
//!     because a result line is provider-specific; the framing (split on `\n`, skip blanks, map
//!     transport/parse errors to [`ModelError`]) is identical everywhere.

use futures::StreamExt;
use llmleaf_model::{BatchResult, BatchResultStream, ModelError};
use serde_json::Value;

use crate::transport::BytesStream;

/// Serialize an iterator of per-request JSON values into a newline-delimited JSONL body.
pub fn build_jsonl<I: IntoIterator<Item = Value>>(lines: I) -> Vec<u8> {
    let mut out = Vec::new();
    for v in lines {
        // A `serde_json::Value` always serializes; the only failure mode is the writer, and a `Vec`
        // never fails to write.
        serde_json::to_writer(&mut out, &v).expect("Value serializes into a Vec");
        out.push(b'\n');
    }
    out
}

/// Stream a result-file response as canonical [`BatchResult`]s, applying `parse` to each non-empty
/// line's JSON. `parse` returning `None` skips that line (a shape we choose not to surface) without
/// failing the stream; a malformed line or a transport error ends the stream with a [`ModelError`].
pub fn jsonl_result_stream<F>(body: BytesStream, parse: F) -> BatchResultStream
where
    F: Fn(Value) -> Option<BatchResult> + Send + 'static,
{
    Box::pin(async_stream::stream! {
        let mut bytes = body;
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        while let Some(item) = bytes.next().await {
            match item {
                Ok(chunk) => buf.extend_from_slice(&chunk),
                Err(e) => {
                    // The transport already mapped this to the canonical taxonomy.
                    yield Err(e);
                    return;
                }
            }
            // Emit every complete line currently buffered.
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                match parse_line(&line[..line.len() - 1], &parse) {
                    Ok(Some(result)) => yield Ok(result),
                    Ok(None) => {}
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
        }
        // A final line not terminated by a newline.
        match parse_line(&buf, &parse) {
            Ok(Some(result)) => yield Ok(result),
            Ok(None) => {}
            Err(e) => yield Err(e),
        }
    })
}

/// Parse one raw (newline-stripped) line: trim surrounding ASCII whitespace (incl. a trailing `\r`),
/// skip a blank line, else JSON-parse and hand to the caller's mapper.
fn parse_line<F>(raw: &[u8], parse: &F) -> Result<Option<BatchResult>, ModelError>
where
    F: Fn(Value) -> Option<BatchResult>,
{
    let line = trim_ascii(raw);
    if line.is_empty() {
        return Ok(None);
    }
    let value: Value =
        serde_json::from_slice(line).map_err(|e| ModelError::Mapping(e.to_string()))?;
    Ok(parse(value))
}

fn trim_ascii(mut b: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = b {
        if first.is_ascii_whitespace() {
            b = rest;
        } else {
            break;
        }
    }
    while let [rest @ .., last] = b {
        if last.is_ascii_whitespace() {
            b = rest;
        } else {
            break;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::BatchOutcome;
    use serde_json::json;

    #[test]
    fn build_jsonl_appends_newlines() {
        let body = build_jsonl([json!({ "a": 1 }), json!({ "b": 2 })]);
        assert_eq!(body, b"{\"a\":1}\n{\"b\":2}\n");
    }

    #[test]
    fn trims_cr_and_blanks() {
        assert_eq!(trim_ascii(b"  hi \r"), b"hi");
        assert!(trim_ascii(b"   \r\n").is_empty());
    }

    #[test]
    fn parse_line_skips_blanks_and_maps_json() {
        // A reqwest::Response is awkward to fabricate, so exercise the per-line mapping directly;
        // `jsonl_result_stream`'s framing is covered end-to-end by the core's batch e2e test.
        let parse = |v: Value| {
            v.get("custom_id")
                .and_then(Value::as_str)
                .map(|id| BatchResult {
                    custom_id: id.to_string(),
                    outcome: BatchOutcome::Canceled,
                })
        };
        let mut results = Vec::new();
        for raw in [
            &b"{\"custom_id\":\"a\"}"[..],
            b"  ",
            b"{\"custom_id\":\"b\"}\r",
        ] {
            if let Some(r) = parse_line(raw, &parse).unwrap() {
                results.push(r.custom_id);
            }
        }
        assert_eq!(results, vec!["a", "b"]);
        assert!(parse_line(b"not json", &parse).is_err());
    }
}
