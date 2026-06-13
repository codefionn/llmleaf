// Streaming helpers: turn a response body Stream into decoded text lines, then surface SSE `data:`
// frames (chat streaming) or NDJSON objects (batch results) as IAsyncEnumerable. Cancellation
// flows through to the underlying read; the stream is always disposed.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Llmleaf.Client.Wire;

internal static class LineReader
{
    /// <summary>Yield decoded UTF-8 text lines from a byte stream, splitting on \n and stripping a trailing \r.</summary>
    private static async IAsyncEnumerable<string> ReadLinesAsync(
        Stream stream,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var decoder = Encoding.UTF8.GetDecoder();
        var bytes = new byte[8192];
        var chars = new char[8192];
        var buffer = new StringBuilder();

        while (true)
        {
            var read = await stream.ReadAsync(bytes.AsMemory(), ct).ConfigureAwait(false);
            if (read == 0)
            {
                break;
            }

            var maxChars = decoder.GetCharCount(bytes, 0, read);
            if (maxChars > chars.Length)
            {
                chars = new char[maxChars];
            }
            var charCount = decoder.GetChars(bytes, 0, read, chars, 0);
            buffer.Append(chars, 0, charCount);

            int nl;
            while ((nl = IndexOf(buffer, '\n')) >= 0)
            {
                var end = nl;
                if (end > 0 && buffer[end - 1] == '\r')
                {
                    end--;
                }
                yield return buffer.ToString(0, end);
                buffer.Remove(0, nl + 1);
            }
        }

        if (buffer.Length > 0)
        {
            var end = buffer.Length;
            if (buffer[end - 1] == '\r')
            {
                end--;
            }
            yield return buffer.ToString(0, end);
        }
    }

    private static int IndexOf(StringBuilder sb, char c)
    {
        for (var i = 0; i < sb.Length; i++)
        {
            if (sb[i] == c)
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>
    /// Parse a <c>text/event-stream</c> body into the raw JSON payload of each <c>data:</c> frame.
    /// Stops (returns) on the sentinel line <c>data: [DONE]</c> WITHOUT yielding it — callers must
    /// not JSON-parse the sentinel (SPEC.md). Blank lines separate events; multi-line <c>data:</c>
    /// frames are concatenated with newlines; <c>event:</c>/<c>id:</c>/comments are ignored.
    /// </summary>
    public static async IAsyncEnumerable<string> ParseSseDataAsync(
        Stream stream,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var dataLines = new List<string>();
        await foreach (var line in ReadLinesAsync(stream, ct).ConfigureAwait(false))
        {
            if (line.Length == 0)
            {
                if (dataLines.Count > 0)
                {
                    var payload = string.Join("\n", dataLines);
                    dataLines.Clear();
                    if (payload == "[DONE]")
                    {
                        yield break;
                    }
                    yield return payload;
                }
                continue;
            }

            if (line[0] == ':')
            {
                continue; // comment
            }
            if (!line.StartsWith("data:", System.StringComparison.Ordinal))
            {
                continue; // ignore event:/id:/retry:
            }

            var value = line.Substring(5);
            if (value.Length > 0 && value[0] == ' ')
            {
                value = value.Substring(1);
            }
            if (value == "[DONE]")
            {
                yield break;
            }
            dataLines.Add(value);
        }

        // Flush a trailing event not followed by a blank line.
        if (dataLines.Count > 0)
        {
            var payload = string.Join("\n", dataLines);
            if (payload != "[DONE]")
            {
                yield return payload;
            }
        }
    }

    /// <summary>Parse an <c>application/x-ndjson</c> body into one raw JSON string per non-empty line.</summary>
    public static async IAsyncEnumerable<string> ParseNdjsonAsync(
        Stream stream,
        [EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var line in ReadLinesAsync(stream, ct).ConfigureAwait(false))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0)
            {
                continue;
            }
            yield return trimmed;
        }
    }
}
