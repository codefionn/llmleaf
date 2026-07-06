// System.Text.Json converter for a single Responses item. On the wire the item is discriminated by
// its "type":
//   {"role":...,"content":...}                                 -> message (role-keyed, NO "type")
//   {"type":"function_call","call_id","name","arguments"}      -> function_call
//   {"type":"function_call_output","call_id","output"}         -> function_call_output
//   {"type":"reasoning","summary":[...],"content":[...]}       -> reasoning
// A message item is written role-keyed WITHOUT a "type" (SPEC.md); the others carry theirs. Reasoning
// summary[] entries serialise as {"type":"summary_text","text"}, content[] entries as
// {"type":"reasoning_text","text"} — the list decides the token. Unknown item types decode leniently
// (as a message when a role is present, else an empty item the mapper drops).

using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace Llmleaf.Client.Wire;

internal sealed class ResponseItemConverter : JsonConverter<WireResponseItem>
{
    public override WireResponseItem Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        var item = new WireResponseItem();

        var type = root.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String
            ? t.GetString()
            : null;
        // A message may arrive as a bare role-keyed object with no "type".
        if (type is null && root.TryGetProperty("role", out _))
        {
            type = "message";
        }

        switch (type)
        {
            case "message":
                item.Message = ReadMessage(root);
                break;
            case "function_call":
                item.FunctionCall = new WireResponseFunctionCallItem
                {
                    Id = Str(root, "id"),
                    CallId = Str(root, "call_id") ?? "",
                    Name = Str(root, "name") ?? "",
                    Arguments = Str(root, "arguments") ?? "",
                    Status = Str(root, "status"),
                };
                break;
            case "function_call_output":
                item.FunctionCallOutput = new WireResponseFunctionCallOutputItem
                {
                    Id = Str(root, "id"),
                    CallId = Str(root, "call_id") ?? "",
                    Output = Str(root, "output") ?? "",
                };
                break;
            case "reasoning":
                item.Reasoning = new WireResponseReasoningItem
                {
                    Id = Str(root, "id"),
                    Summary = ReadReasoningTexts(root, "summary"),
                    Content = ReadReasoningTexts(root, "content"),
                    EncryptedContent = Str(root, "encrypted_content"),
                };
                break;
            default:
                // Unknown item type: treat as a message when a role is present, else leave empty.
                if (root.TryGetProperty("role", out _))
                {
                    item.Message = ReadMessage(root);
                }
                break;
        }

        return item;
    }

    private static WireResponseMessageItem ReadMessage(JsonElement root) => new()
    {
        Id = Str(root, "id"),
        Role = Str(root, "role") ?? "",
        Content = root.TryGetProperty("content", out var c) && c.ValueKind is not JsonValueKind.Null and not JsonValueKind.Undefined
            ? JsonNode.Parse(c.GetRawText())
            : null,
        Status = Str(root, "status"),
    };

    private static string? Str(JsonElement root, string name)
        => root.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    // Read a reasoning list ("summary" | "content"): each entry is {"type":...,"text":...}; collect texts.
    private static List<string>? ReadReasoningTexts(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var arr) || arr.ValueKind != JsonValueKind.Array)
        {
            return null;
        }
        var texts = new List<string>(arr.GetArrayLength());
        foreach (var entry in arr.EnumerateArray())
        {
            if (entry.ValueKind == JsonValueKind.Object
                && entry.TryGetProperty("text", out var txt)
                && txt.ValueKind == JsonValueKind.String)
            {
                texts.Add(txt.GetString()!);
            }
        }
        return texts;
    }

    public override void Write(Utf8JsonWriter writer, WireResponseItem value, JsonSerializerOptions options)
    {
        if (value.Message is { } msg)
        {
            WriteMessage(writer, msg, options);
        }
        else if (value.FunctionCall is { } fc)
        {
            writer.WriteStartObject();
            writer.WriteString("type", "function_call");
            WriteOptString(writer, "id", fc.Id);
            writer.WriteString("call_id", fc.CallId);
            writer.WriteString("name", fc.Name);
            writer.WriteString("arguments", fc.Arguments);
            WriteOptString(writer, "status", fc.Status);
            writer.WriteEndObject();
        }
        else if (value.FunctionCallOutput is { } fco)
        {
            writer.WriteStartObject();
            writer.WriteString("type", "function_call_output");
            WriteOptString(writer, "id", fco.Id);
            writer.WriteString("call_id", fco.CallId);
            writer.WriteString("output", fco.Output);
            writer.WriteEndObject();
        }
        else if (value.Reasoning is { } r)
        {
            writer.WriteStartObject();
            writer.WriteString("type", "reasoning");
            WriteOptString(writer, "id", r.Id);
            WriteReasoningTexts(writer, "summary", "summary_text", r.Summary);
            WriteReasoningTexts(writer, "content", "reasoning_text", r.Content);
            WriteOptString(writer, "encrypted_content", r.EncryptedContent);
            writer.WriteEndObject();
        }
        else
        {
            // Empty item: emit an empty object rather than throwing.
            writer.WriteStartObject();
            writer.WriteEndObject();
        }
    }

    // A message item is written role-keyed WITHOUT a "type" (SPEC.md).
    private static void WriteMessage(Utf8JsonWriter writer, WireResponseMessageItem msg, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        WriteOptString(writer, "id", msg.Id);
        writer.WriteString("role", msg.Role);
        if (msg.Content is { } content)
        {
            writer.WritePropertyName("content");
            content.WriteTo(writer, options);
        }
        WriteOptString(writer, "status", msg.Status);
        writer.WriteEndObject();
    }

    // Emit a reasoning list: [{"type":<wireType>,"text":...}, ...]. Null/empty -> field omitted.
    private static void WriteReasoningTexts(Utf8JsonWriter writer, string field, string wireType, List<string>? texts)
    {
        if (texts is null || texts.Count == 0)
        {
            return;
        }
        writer.WritePropertyName(field);
        writer.WriteStartArray();
        foreach (var text in texts)
        {
            writer.WriteStartObject();
            writer.WriteString("type", wireType);
            writer.WriteString("text", text);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static void WriteOptString(Utf8JsonWriter writer, string name, string? value)
    {
        if (value is not null)
        {
            writer.WriteString(name, value);
        }
    }
}
