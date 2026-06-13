// System.Text.Json converter for a single content part. On the wire a part is one of:
//   {"type":"text","text":"..."}
//   {"type":"image_url","image_url":{"url":"...","detail":"auto"}}
// An unknown/absent type is treated as text (lenient decode), matching the Go client.

using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Llmleaf.Client.Wire;

internal sealed class ContentPartConverter : JsonConverter<WireContentPart>
{
    public override WireContentPart Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        var part = new WireContentPart();

        var type = root.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String
            ? t.GetString()
            : null;

        if (type == "image_url" && root.TryGetProperty("image_url", out var iu) && iu.ValueKind == JsonValueKind.Object)
        {
            part.ImageUrl = new WireImageUrl
            {
                Url = iu.TryGetProperty("url", out var url) && url.ValueKind == JsonValueKind.String ? url.GetString()! : "",
                Detail = iu.TryGetProperty("detail", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString() : null,
            };
        }
        else
        {
            // "text" or unknown -> treat as text.
            part.Text = root.TryGetProperty("text", out var txt) && txt.ValueKind == JsonValueKind.String
                ? txt.GetString()!
                : "";
        }

        return part;
    }

    public override void Write(Utf8JsonWriter writer, WireContentPart value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        if (value.ImageUrl is { } image)
        {
            writer.WriteString("type", "image_url");
            writer.WritePropertyName("image_url");
            writer.WriteStartObject();
            writer.WriteString("url", image.Url);
            if (image.Detail is not null)
            {
                writer.WriteString("detail", image.Detail);
            }
            writer.WriteEndObject();
        }
        else
        {
            writer.WriteString("type", "text");
            writer.WriteString("text", value.Text ?? "");
        }
        writer.WriteEndObject();
    }
}
