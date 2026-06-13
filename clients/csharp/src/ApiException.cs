// Typed error surface. Any non-2xx response carries the envelope
//   {"error":{"message":"...", "type"?:"...", "code"?:"..."}}
// (SPEC.md "Errors"). We parse it into ApiException and throw.

using System;
using System.Text.Json;

namespace Llmleaf.Client;

/// <summary>
/// Thrown for any non-2xx HTTP response from the gateway.
/// </summary>
/// <remarks>
/// Status codes (SPEC.md): 400 bad request · 401 missing/invalid key · 403 blocked or
/// model-not-allowed · 404 no route for model · 429 key suspended (limiter) · 502 all upstreams
/// failed.
/// </remarks>
public sealed class ApiException : Exception
{
    /// <summary>The HTTP status code.</summary>
    public int Status { get; }

    /// <summary>Present on some dialects; absent on the llmleaf core envelope.</summary>
    public string? Type { get; }

    /// <summary>Present on some dialects; absent on the llmleaf core envelope.</summary>
    public string? Code { get; }

    public ApiException(int status, string message, string? type = null, string? code = null)
        : base(message)
    {
        Status = status;
        Type = type;
        Code = code;
    }

    public override string ToString() => $"llmleaf: HTTP {Status}: {Message}";

    /// <summary>
    /// Build an <see cref="ApiException"/> from a non-2xx response body, preferring the
    /// <c>{"error":{"message":...}}</c> envelope and falling back to the raw body / status text.
    /// </summary>
    internal static ApiException FromBody(int status, string fallback, string body)
    {
        var trimmed = body.Trim();
        if (trimmed.Length == 0)
        {
            return new ApiException(status, fallback);
        }

        try
        {
            using var doc = JsonDocument.Parse(trimmed);
            var root = doc.RootElement;
            // Accept either {"error":{...}} or a bare {message,type,code} object.
            var errEl = root;
            if (root.ValueKind == JsonValueKind.Object
                && root.TryGetProperty("error", out var nested)
                && nested.ValueKind == JsonValueKind.Object)
            {
                errEl = nested;
            }

            if (errEl.ValueKind == JsonValueKind.Object)
            {
                var message = errEl.TryGetProperty("message", out var m) && m.ValueKind == JsonValueKind.String
                    ? m.GetString()
                    : null;
                var type = errEl.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String
                    ? t.GetString()
                    : null;
                var code = errEl.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                    ? c.GetString()
                    : null;
                return new ApiException(
                    status,
                    string.IsNullOrEmpty(message) ? fallback : message!,
                    type,
                    code);
            }
        }
        catch (JsonException)
        {
            // Not JSON — fall through and surface the raw text.
        }

        // Cap the surfaced body so a huge HTML error page doesn't blow up the message.
        return new ApiException(status, trimmed.Length > 2048 ? trimmed[..2048] : trimmed);
    }
}
