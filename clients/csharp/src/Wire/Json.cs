// Shared System.Text.Json plumbing for the wire layer: the serializer options, free-form-JSON
// splicing (raw passthrough — never double-encoded as a string), and `extra` top-level merging.

using System;
using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Llmleaf.Client.Wire;

internal static class Json
{
    /// <summary>
    /// The one options instance the whole transport uses. snake_case keys are written explicitly via
    /// [JsonPropertyName] on every DTO field; null fields are dropped (omitempty); numbers are written
    /// as numbers (no quoting).
    /// </summary>
    internal static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.General)
    {
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        // The wire is ASCII-safe JSON; keep non-ASCII unescaped so prompts round-trip verbatim.
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>
    /// Parse a free-form JSON string (e.g. ChatRequest.Extra, FunctionDef.Parameters) into a
    /// <see cref="JsonNode"/> so it is spliced into the body verbatim as a JSON value, never
    /// double-encoded as a string. An empty/null input yields null (field omitted). Invalid JSON
    /// throws.
    /// </summary>
    internal static JsonNode? RawValue(string? s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return null;
        }
        try
        {
            return JsonNode.Parse(s);
        }
        catch (JsonException ex)
        {
            throw new ArgumentException($"llmleaf: invalid free-form JSON: {s}", ex);
        }
    }

    /// <summary>
    /// Capture a decoded JSON sub-value back into a raw-JSON string for the public model. A null
    /// node (or JSON null) yields null.
    /// </summary>
    internal static string? RawString(JsonNode? node)
        => node is null ? null : node.ToJsonString(Options);

    internal static string? RawString(JsonElement element)
        => element.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null
            ? null
            : element.GetRawText();

    /// <summary>
    /// Serialise <paramref name="value"/> to a UTF-8 JSON object, then splice the keys of
    /// <paramref name="extra"/> (a JSON object) into the top level WITHOUT overwriting keys the value
    /// already set (explicit fields win — SPEC.md). Returns the merged UTF-8 bytes.
    /// </summary>
    internal static byte[] MergeExtra<T>(T value, JsonNode? extra)
    {
        var baseBytes = JsonSerializer.SerializeToUtf8Bytes(value, Options);
        if (extra is null)
        {
            return baseBytes;
        }
        if (extra is not JsonObject extraObj)
        {
            throw new ArgumentException("llmleaf: `extra` must be a JSON object");
        }

        var merged = JsonNode.Parse(baseBytes) as JsonObject
                     ?? throw new InvalidOperationException("llmleaf: request body did not serialise to a JSON object");

        foreach (var kvp in extraObj)
        {
            if (!merged.ContainsKey(kvp.Key))
            {
                // Detach the child from extraObj before re-parenting.
                merged[kvp.Key] = kvp.Value?.DeepClone();
            }
        }

        return Encoding.UTF8.GetBytes(merged.ToJsonString(Options));
    }
}
