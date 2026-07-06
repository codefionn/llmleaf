// System.Text.Json converter for a single Responses content part. On the wire a part is one of:
//   {"type":"input_text","text":"..."}
//   {"type":"input_image","image_url":"<url>","detail":"auto"}   (image_url is a STRING here)
//   {"type":"output_text","text":"...","annotations":[]}
// A constructed output_text part always emits "annotations":[] (SPEC.md). An unknown/absent type is
// decoded leniently (image_url -> input_image, else text -> output_text, else empty input_text).

using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Llmleaf.Client.Wire;

internal sealed class ResponseContentPartConverter : JsonConverter<WireResponseContentPart>
{
    public override WireResponseContentPart Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        var part = new WireResponseContentPart();

        var type = root.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String
            ? t.GetString()
            : null;

        var hasImage = root.TryGetProperty("image_url", out var iu) && iu.ValueKind == JsonValueKind.String;
        var text = root.TryGetProperty("text", out var txt) && txt.ValueKind == JsonValueKind.String ? txt.GetString()! : "";

        switch (type)
        {
            case "input_image":
                part.InputImage = ReadImage(root, iu);
                break;
            case "input_text":
                part.InputText = text;
                break;
            case "output_text":
                part.OutputText = text;
                break;
            default:
                // Lenient decode of an unknown/absent type.
                if (hasImage)
                {
                    part.InputImage = ReadImage(root, iu);
                }
                else
                {
                    // Prefer output_text when a bare text is present (the common decode case).
                    part.OutputText = text;
                }
                break;
        }

        return part;
    }

    private static WireResponseInputImage ReadImage(JsonElement root, JsonElement imageUrl)
        => new()
        {
            ImageUrl = imageUrl.ValueKind == JsonValueKind.String ? imageUrl.GetString()! : "",
            Detail = root.TryGetProperty("detail", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString() : null,
        };

    public override void Write(Utf8JsonWriter writer, WireResponseContentPart value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        if (value.InputImage is { } image)
        {
            writer.WriteString("type", "input_image");
            writer.WriteString("image_url", image.ImageUrl);
            if (image.Detail is not null)
            {
                writer.WriteString("detail", image.Detail);
            }
        }
        else if (value.OutputText is { } outText)
        {
            writer.WriteString("type", "output_text");
            writer.WriteString("text", outText);
            // Constructed output_text parts carry an (empty) annotations array (SPEC.md).
            writer.WritePropertyName("annotations");
            writer.WriteStartArray();
            writer.WriteEndArray();
        }
        else
        {
            writer.WriteString("type", "input_text");
            writer.WriteString("text", value.InputText ?? "");
        }
        writer.WriteEndObject();
    }
}
