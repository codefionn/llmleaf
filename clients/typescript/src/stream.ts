// Streaming helpers: decode the response body's ReadableStream into lines, then
// surface SSE `data:` frames (chat streaming) or NDJSON objects (batch results).
//
// These are runtime-agnostic: they only use the WHATWG ReadableStream + TextDecoder
// that Node 20+, Deno, Bun and browsers all provide.

/** Yield decoded text lines from a byte stream, handling \n / \r\n splits across chunks. */
async function* iterLines(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        let line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        yield line;
      }
    }
    buffer += decoder.decode();
    if (buffer.endsWith("\r")) buffer = buffer.slice(0, -1);
    if (buffer.length > 0) yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/**
 * Parse a `text/event-stream` body into the raw JSON payload of each `data:` frame.
 *
 * Stops (returns) on the sentinel line `data: [DONE]` WITHOUT yielding it — callers
 * must not JSON-parse the sentinel (SPEC.md). Blank lines (frame separators) and any
 * non-`data:` SSE fields (`event:`, `id:`, comments) are ignored. Multi-line `data:`
 * frames are concatenated with newlines per the SSE spec.
 */
export async function* parseSseData(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, unknown> {
  let dataLines: string[] = [];
  for await (const line of iterLines(body)) {
    if (line === "") {
      // Dispatch the accumulated event.
      if (dataLines.length > 0) {
        yield dataLines.join("\n");
        dataLines = [];
      }
      continue;
    }
    if (line.startsWith(":")) continue; // comment
    if (!line.startsWith("data:")) continue; // ignore event:/id:/retry:
    // Strip "data:" and a single optional leading space.
    let value = line.slice(5);
    if (value.startsWith(" ")) value = value.slice(1);
    if (value === "[DONE]") return;
    dataLines.push(value);
  }
  // Flush a trailing event that wasn't followed by a blank line.
  if (dataLines.length > 0) {
    const payload = dataLines.join("\n");
    if (payload !== "[DONE]") yield payload;
  }
}

/**
 * Parse an `application/x-ndjson` body into one parsed JSON value per non-empty line.
 */
export async function* parseNdjson(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<unknown, void, unknown> {
  for await (const line of iterLines(body)) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    yield JSON.parse(trimmed);
  }
}
