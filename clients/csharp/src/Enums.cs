// Enum <-> wire-token mapping (SPEC.md "Enum ⇄ wire mapping").
//
// Every closed-set enum maps to its wire token by LOWERCASING the proto value name:
//   TOOL_CALLS -> "tool_calls", ASSISTANT -> "assistant", IN_PROGRESS -> "in_progress".
// The `*_UNSPECIFIED` zero value <-> the field is absent on the wire (null here).
//
// The generated Google.Protobuf enums (Llmleaf.V1.Role / FinishReason / BatchStatus) carry
// the proto value name in a `[pbr::OriginalName("...")]` attribute on every member; that is
// the schema proof. These public enums mirror them member-for-member and carry the SAME proto
// value name in a [WireName] attribute, so one generic helper pair (EnumWire.ToWire /
// EnumWire.FromWire) derives the wire token mechanically from the attribute — no per-enum hand
// mapping of tokens (SPEC.md). A debug-only check asserts the two enum sets stay in lockstep.

using System;
using System.Collections.Concurrent;
using System.Reflection;

namespace Llmleaf.Client;

/// <summary>
/// Carries the canonical proto value name (e.g. "TOOL_CALLS") for an enum member. The wire
/// token is this name lowercased.
/// </summary>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
public sealed class WireNameAttribute(string protoName) : Attribute
{
    public string ProtoName { get; } = protoName;
}

/// <summary>Message author role. Maps to the OpenAI <c>role</c> token.</summary>
public enum Role
{
    [WireName("ROLE_UNSPECIFIED")] Unspecified = 0,
    [WireName("SYSTEM")] System = 1,
    [WireName("USER")] User = 2,
    [WireName("ASSISTANT")] Assistant = 3,
    [WireName("TOOL")] Tool = 4,
}

/// <summary>Why a completion stopped. Maps to the OpenAI <c>finish_reason</c> token.</summary>
public enum FinishReason
{
    [WireName("FINISH_REASON_UNSPECIFIED")] Unspecified = 0,
    [WireName("STOP")] Stop = 1,
    [WireName("LENGTH")] Length = 2,
    [WireName("TOOL_CALLS")] ToolCalls = 3,
    [WireName("CONTENT_FILTER")] ContentFilter = 4,
}

/// <summary>Lifecycle state of a batch. Maps to the OpenAI <c>status</c> token.</summary>
public enum BatchStatus
{
    [WireName("BATCH_STATUS_UNSPECIFIED")] Unspecified = 0,
    [WireName("VALIDATING")] Validating = 1,
    [WireName("IN_PROGRESS")] InProgress = 2,
    [WireName("FINALIZING")] Finalizing = 3,
    [WireName("COMPLETED")] Completed = 4,
    [WireName("FAILED")] Failed = 5,
    [WireName("EXPIRED")] Expired = 6,
    [WireName("CANCELING")] Canceling = 7,
    [WireName("CANCELED")] Canceled = 8,
}

/// <summary>
/// One generic helper pair mapping any closed-set enum to/from its wire token. The token is the
/// proto value name (from <see cref="WireNameAttribute"/>) lowercased; the unspecified zero value
/// means "field absent" and maps to/from <c>null</c>.
/// </summary>
internal static class EnumWire
{
    // Cache per enum type: value -> lowercased wire token, and token -> value.
    private static readonly ConcurrentDictionary<Type, (string?[] toWire, System.Collections.Generic.Dictionary<string, int> fromWire)> Cache = new();

    private static (string?[] toWire, System.Collections.Generic.Dictionary<string, int> fromWire) TableFor(Type enumType)
    {
        return Cache.GetOrAdd(enumType, static t =>
        {
            var values = (int[])Enum.GetValuesAsUnderlyingType(t)!;
            int max = 0;
            foreach (var v in values)
            {
                if (v > max) max = v;
            }

            var toWire = new string?[max + 1];
            var fromWire = new System.Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var name in Enum.GetNames(t))
            {
                var field = t.GetField(name)!;
                var value = (int)field.GetRawConstantValue()!;
                var proto = field.GetCustomAttribute<WireNameAttribute>()?.ProtoName ?? name;
                if (proto.EndsWith("_UNSPECIFIED", StringComparison.Ordinal))
                {
                    continue; // zero value: absent on the wire
                }
                var token = proto.ToLowerInvariant();
                toWire[value] = token;
                fromWire[token] = value;
            }
            return (toWire, fromWire);
        });
    }

    /// <summary>
    /// The wire token for <paramref name="value"/>, or <c>null</c> for the unspecified zero value
    /// (which means the field is omitted on the wire).
    /// </summary>
    public static string? ToWire<TEnum>(TEnum value) where TEnum : struct, Enum
    {
        var (toWire, _) = TableFor(typeof(TEnum));
        var idx = Convert.ToInt32(value);
        return idx >= 0 && idx < toWire.Length ? toWire[idx] : null;
    }

    /// <summary>
    /// The wire token for a nullable enum, or <c>null</c> when the value is null or unspecified.
    /// </summary>
    public static string? ToWire<TEnum>(TEnum? value) where TEnum : struct, Enum
        => value.HasValue ? ToWire(value.Value) : null;

    /// <summary>
    /// The enum value for a wire token (case-insensitively). A null/empty/unknown token maps to
    /// the unspecified zero value.
    /// </summary>
    public static TEnum FromWire<TEnum>(string? token) where TEnum : struct, Enum
    {
        if (string.IsNullOrEmpty(token))
        {
            return default;
        }
        var (_, fromWire) = TableFor(typeof(TEnum));
        return fromWire.TryGetValue(token.ToLowerInvariant(), out var v) ? (TEnum)(object)v : default;
    }

    /// <summary>
    /// As <see cref="FromWire{TEnum}(string?)"/>, but returns <c>null</c> (field absent) rather than
    /// the zero value when the token is null/empty. Use for optional response fields.
    /// </summary>
    public static TEnum? FromWireOptional<TEnum>(string? token) where TEnum : struct, Enum
        => string.IsNullOrEmpty(token) ? null : FromWire<TEnum>(token);
}
