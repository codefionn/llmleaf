// The deliberate JSON serialization layer: maps the public records (Models.cs) to/from the
// internal wire DTOs (WireDtos.cs), then produces the final UTF-8 request bytes. Enum tokens go
// through EnumWire (lowercased proto value names); free-form JSON is spliced raw; `content`,
// `stop`, `input` are emitted as string-or-array; embeddings are decoded (incl. base64) to floats.

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Llmleaf.Client.Wire;

internal static class Mapper
{
    // ---- chat request (encode) ------------------------------------------

    /// <summary>
    /// Encode a <see cref="ChatRequest"/> into the final UTF-8 request body, merging <c>extra</c> at
    /// the top level. <paramref name="streamOverride"/>, when non-null, sets the wire stream flag
    /// without mutating the caller's request.
    /// </summary>
    internal static byte[] EncodeChatRequest(ChatRequest req, bool? streamOverride)
        => Json.MergeExtra(ChatRequestToWire(req, streamOverride), Json.RawValue(req.Extra));

    /// <summary>Encode a chat request into a spliceable <see cref="JsonNode"/> (with extra merged) for batch bodies.</summary>
    internal static JsonNode EncodeChatRequestNode(ChatRequest req)
    {
        var bytes = EncodeChatRequest(req, null);
        return JsonNode.Parse(bytes)!;
    }

    private static WireChatRequest ChatRequestToWire(ChatRequest req, bool? streamOverride)
    {
        var w = new WireChatRequest
        {
            Model = req.Model,
            Stream = streamOverride ?? req.Stream,
            Temperature = req.Temperature,
            TopP = req.TopP,
            MaxTokens = req.MaxTokens,
            MaxCompletionTokens = req.MaxCompletionTokens,
            Stop = EncodeStringOrArray(req.Stop),
            N = req.N,
            Seed = req.Seed,
            FrequencyPenalty = req.FrequencyPenalty,
            PresencePenalty = req.PresencePenalty,
            ToolChoice = ToolChoiceToWire(req.ToolChoice),
            ResponseFormat = ResponseFormatToWire(req.ResponseFormat),
            ReasoningEffort = req.ReasoningEffort,
        };

        w.Messages = req.Messages.Select(ChatMessageToWire).ToList();

        if (req.Tools is { Count: > 0 })
        {
            w.Tools = req.Tools.Select(ToolDefToWire).ToList();
        }

        return w;
    }

    /// <summary>A bare string for one element, an array otherwise, null for empty/null.</summary>
    private static JsonNode? EncodeStringOrArray(IReadOnlyList<string>? items)
    {
        if (items is null || items.Count == 0)
        {
            return null;
        }
        if (items.Count == 1)
        {
            return JsonValue.Create(items[0]);
        }
        var arr = new JsonArray();
        foreach (var s in items)
        {
            arr.Add(JsonValue.Create(s));
        }
        return arr;
    }

    private static WireChatMessage ChatMessageToWire(ChatMessage m)
    {
        var wm = new WireChatMessage
        {
            Role = EnumWire.ToWire(m.Role),
            Name = m.Name,
            ToolCallId = m.ToolCallId,
            Reasoning = m.Reasoning,
        };

        if (m.Content is { } content)
        {
            if (content.Parts is { } parts)
            {
                var arr = new JsonArray();
                foreach (var p in parts)
                {
                    arr.Add(JsonSerializer.SerializeToNode(ContentPartToWire(p), Json.Options));
                }
                wm.Content = arr;
            }
            else if (content.Text is { } text)
            {
                wm.Content = JsonValue.Create(text);
            }
        }

        if (m.ToolCalls is { Count: > 0 })
        {
            wm.ToolCalls = m.ToolCalls.Select(ToolCallToWire).ToList();
        }

        if (m.ReasoningDetails is { Count: > 0 })
        {
            wm.ReasoningDetails = m.ReasoningDetails.Select(ReasoningDetailToWire).ToList();
        }

        return wm;
    }

    private static WireReasoningDetail ReasoningDetailToWire(ReasoningDetail d) => new()
    {
        Type = d.Type,
        Text = d.Text,
        Summary = d.Summary,
        Data = d.Data,
        Signature = d.Signature,
        Id = d.Id,
        Format = d.Format,
        Index = d.Index,
    };

    private static ReasoningDetail ReasoningDetailFromWire(WireReasoningDetail d) => new()
    {
        Type = d.Type,
        Text = d.Text,
        Summary = d.Summary,
        Data = d.Data,
        Signature = d.Signature,
        Id = d.Id,
        Format = d.Format,
        Index = d.Index,
    };

    private static WireContentPart ContentPartToWire(ContentPart p) => p switch
    {
        ImageUrlPart img => new WireContentPart { ImageUrl = new WireImageUrl { Url = img.Url, Detail = img.Detail } },
        TextPart txt => new WireContentPart { Text = txt.Text },
        _ => new WireContentPart { Text = "" },
    };

    private static WireToolCall ToolCallToWire(ToolCall tc) => new()
    {
        Id = tc.Id,
        Type = tc.Type,
        Function = new WireFunctionCall { Name = tc.Function.Name, Arguments = tc.Function.Arguments },
    };

    private static WireToolDef ToolDefToWire(ToolDef t) => new()
    {
        Type = t.Type,
        Function = new WireFunctionDef
        {
            Name = t.Function.Name,
            Description = t.Function.Description,
            Parameters = Json.RawValue(t.Function.Parameters),
        },
    };

    private static JsonNode? ToolChoiceToWire(ToolChoice? tc)
    {
        if (tc is null)
        {
            return null;
        }
        if (tc.FunctionName is { } fn)
        {
            return new JsonObject
            {
                ["type"] = "function",
                ["function"] = new JsonObject { ["name"] = fn },
            };
        }
        return tc.Mode is { } mode ? JsonValue.Create(mode) : null;
    }

    private static WireResponseFormat? ResponseFormatToWire(ResponseFormat? rf)
        => rf is null ? null : new WireResponseFormat { Type = rf.Type, JsonSchema = Json.RawValue(rf.JsonSchema) };

    // ---- chat response (decode) -----------------------------------------

    internal static ChatResponse ChatResponseFromWire(WireChatResponse w) => new(
        w.Id,
        w.Object,
        w.Created,
        w.Model,
        w.Choices.Select(c => new Choice(
            c.Index,
            ChatMessageFromWire(c.Message),
            EnumWire.FromWireOptional<FinishReason>(c.FinishReason))).ToList(),
        UsageFromWire(w.Usage));

    private static ChatMessage ChatMessageFromWire(WireChatMessage m) => new()
    {
        Role = EnumWire.FromWire<Role>(m.Role),
        Content = ContentFromWire(m.Content),
        Name = m.Name,
        ToolCalls = m.ToolCalls?.Select(ToolCallFromWire).ToList(),
        ToolCallId = m.ToolCallId,
        Reasoning = m.Reasoning,
        ReasoningDetails = m.ReasoningDetails?.Select(ReasoningDetailFromWire).ToList(),
    };

    private static MessageContent? ContentFromWire(JsonNode? content)
    {
        switch (content)
        {
            case null:
                return null;
            case JsonValue v when v.TryGetValue<string>(out var s):
                return MessageContent.FromText(s);
            case JsonArray arr:
            {
                var parts = new List<ContentPart>(arr.Count);
                foreach (var el in arr)
                {
                    if (el is null)
                    {
                        continue;
                    }
                    var wp = el.Deserialize<WireContentPart>(Json.Options);
                    if (wp is null)
                    {
                        continue;
                    }
                    parts.Add(wp.ImageUrl is { } iu
                        ? new ImageUrlPart(iu.Url, iu.Detail)
                        : new TextPart(wp.Text ?? ""));
                }
                return MessageContent.FromParts(parts);
            }
            default:
                return null;
        }
    }

    private static ToolCall ToolCallFromWire(WireToolCall tc)
        => new(tc.Id, tc.Type, new FunctionCall(tc.Function.Name, tc.Function.Arguments));

    private static Usage? UsageFromWire(WireUsage? u)
        => u is null
            ? null
            : new Usage(
                u.PromptTokens,
                u.CompletionTokens,
                u.TotalTokens,
                u.CostUsd,
                u.PromptTokensDetails is null ? null : new PromptTokensDetails(u.PromptTokensDetails.CachedTokens),
                u.CacheCreationTokens);

    // ---- streaming chunk (decode) ---------------------------------------

    internal static ChatCompletionChunk ChunkFromWire(WireChunk w) => new(
        w.Id,
        w.Object,
        w.Created,
        w.Model,
        w.Choices.Select(c => new ChunkChoice(
            c.Index,
            new Delta(
                EnumWire.FromWireOptional<Role>(c.Delta.Role),
                c.Delta.Content,
                c.Delta.ToolCalls?.Select(ToolCallDeltaFromWire).ToList(),
                c.Delta.Reasoning,
                c.Delta.ReasoningDetails?.Select(ReasoningDetailFromWire).ToList()),
            EnumWire.FromWireOptional<FinishReason>(c.FinishReason))).ToList(),
        UsageFromWire(w.Usage));

    private static ToolCallDelta ToolCallDeltaFromWire(WireToolCallDelta d)
        => new(d.Index, d.Id, d.Type, d.Function is null ? null : new FunctionCallDelta(d.Function.Name, d.Function.Arguments));

    // ---- responses request (encode) -------------------------------------

    /// <summary>
    /// Encode a <see cref="ResponsesRequest"/> into the final UTF-8 request body, merging <c>extra</c>
    /// at the top level. <paramref name="streamOverride"/>, when non-null, sets the wire stream flag
    /// without mutating the caller's request.
    /// </summary>
    internal static byte[] EncodeResponsesRequest(ResponsesRequest req, bool? streamOverride)
        => Json.MergeExtra(ResponsesRequestToWire(req, streamOverride), Json.RawValue(req.Extra));

    private static WireResponsesRequest ResponsesRequestToWire(ResponsesRequest req, bool? streamOverride)
    {
        var w = new WireResponsesRequest
        {
            Model = req.Model,
            Input = ResponsesInputToWire(req.Input),
            Instructions = req.Instructions,
            Stream = streamOverride ?? req.Stream,
            Temperature = req.Temperature,
            TopP = req.TopP,
            MaxOutputTokens = req.MaxOutputTokens,
            ToolChoice = ResponsesToolChoiceToWire(req.ToolChoice),
            Reasoning = req.Reasoning is null
                ? null
                : new WireResponsesReasoning { Effort = req.Reasoning.Effort, Summary = req.Reasoning.Summary },
            Store = req.Store,
        };

        if (req.Tools is { Count: > 0 })
        {
            w.Tools = req.Tools.Select(ResponsesToolDefToWire).ToList();
        }

        return w;
    }

    /// <summary>Encode the request <c>input</c>: a bare string for one user message, else an item array.</summary>
    private static JsonNode? ResponsesInputToWire(ResponsesInput input)
    {
        if (input.Items is { } items)
        {
            var arr = new JsonArray();
            foreach (var item in items)
            {
                arr.Add(JsonSerializer.SerializeToNode(ResponseItemToWire(item), Json.Options));
            }
            return arr;
        }
        return input.Text is { } text ? JsonValue.Create(text) : null;
    }

    private static WireResponseItem ResponseItemToWire(ResponseItem item) => item switch
    {
        ResponseMessageItem m => new WireResponseItem { Message = ResponseMessageItemToWire(m) },
        ResponseFunctionCallItem fc => new WireResponseItem
        {
            FunctionCall = new WireResponseFunctionCallItem
            {
                Id = fc.Id, CallId = fc.CallId, Name = fc.Name, Arguments = fc.Arguments, Status = fc.Status,
            },
        },
        ResponseFunctionCallOutputItem fco => new WireResponseItem
        {
            FunctionCallOutput = new WireResponseFunctionCallOutputItem { Id = fco.Id, CallId = fco.CallId, Output = fco.Output },
        },
        ResponseReasoningItem r => new WireResponseItem
        {
            Reasoning = new WireResponseReasoningItem
            {
                Id = r.Id,
                Summary = r.Summary?.Select(x => x.Text).ToList(),
                Content = r.Content?.Select(x => x.Text).ToList(),
                EncryptedContent = r.EncryptedContent,
            },
        },
        _ => new WireResponseItem(),
    };

    private static WireResponseMessageItem ResponseMessageItemToWire(ResponseMessageItem m)
    {
        var wm = new WireResponseMessageItem { Id = m.Id, Role = m.Role, Status = m.Status };

        if (m.Content is { } content)
        {
            if (content.Parts is { } parts)
            {
                var arr = new JsonArray();
                foreach (var p in parts)
                {
                    arr.Add(JsonSerializer.SerializeToNode(ResponseContentPartToWire(p), Json.Options));
                }
                wm.Content = arr;
            }
            else if (content.Text is { } text)
            {
                wm.Content = JsonValue.Create(text);
            }
        }

        return wm;
    }

    private static WireResponseContentPart ResponseContentPartToWire(ResponseContentPart p) => p switch
    {
        InputImagePart img => new WireResponseContentPart { InputImage = new WireResponseInputImage { ImageUrl = img.ImageUrl, Detail = img.Detail } },
        OutputTextPart o => new WireResponseContentPart { OutputText = o.Text },
        InputTextPart txt => new WireResponseContentPart { InputText = txt.Text },
        _ => new WireResponseContentPart { InputText = "" },
    };

    private static WireResponsesToolDef ResponsesToolDefToWire(ResponsesToolDef t) => new()
    {
        Type = t.Type,
        Name = t.Name,
        Description = t.Description,
        Parameters = Json.RawValue(t.Parameters),
        Strict = t.Strict,
    };

    // FLAT named object {"type":"function","name":...}, or a bare mode string.
    private static JsonNode? ResponsesToolChoiceToWire(ResponsesToolChoice? tc)
    {
        if (tc is null)
        {
            return null;
        }
        if (tc.FunctionName is { } fn)
        {
            return new JsonObject { ["type"] = "function", ["name"] = fn };
        }
        return tc.Mode is { } mode ? JsonValue.Create(mode) : null;
    }

    // ---- responses response (decode) ------------------------------------

    internal static ResponsesResponse ResponsesResponseFromWire(WireResponsesResponse w) => new()
    {
        Id = w.Id,
        Object = w.Object,
        CreatedAt = w.CreatedAt,
        Status = w.Status,
        IncompleteDetails = w.IncompleteDetails is null ? null : new ResponsesIncompleteDetails(w.IncompleteDetails.Reason),
        Error = ErrorBodyFromWire(w.Error),
        Model = w.Model,
        Output = ResponseItemsFromWire(w.Output),
        Usage = ResponsesUsageFromWire(w.Usage),
        Store = w.Store,
        Instructions = w.Instructions,
        MaxOutputTokens = w.MaxOutputTokens,
        Temperature = w.Temperature,
        TopP = w.TopP,
        Reasoning = w.Reasoning is null ? null : new ResponsesReasoning(w.Reasoning.Effort, w.Reasoning.Summary),
    };

    private static IReadOnlyList<ResponseItem> ResponseItemsFromWire(List<WireResponseItem>? items)
    {
        if (items is null || items.Count == 0)
        {
            return [];
        }
        var list = new List<ResponseItem>(items.Count);
        foreach (var w in items)
        {
            if (ResponseItemFromWire(w) is { } item)
            {
                list.Add(item);
            }
        }
        return list;
    }

    private static ResponseItem? ResponseItemFromWire(WireResponseItem w)
    {
        if (w.Message is { } m)
        {
            return new ResponseMessageItem { Id = m.Id, Role = m.Role, Content = ResponseContentFromWire(m.Content), Status = m.Status };
        }
        if (w.FunctionCall is { } fc)
        {
            return new ResponseFunctionCallItem { Id = fc.Id, CallId = fc.CallId, Name = fc.Name, Arguments = fc.Arguments, Status = fc.Status };
        }
        if (w.FunctionCallOutput is { } fco)
        {
            return new ResponseFunctionCallOutputItem { Id = fco.Id, CallId = fco.CallId, Output = fco.Output };
        }
        if (w.Reasoning is { } r)
        {
            return new ResponseReasoningItem
            {
                Id = r.Id,
                Summary = r.Summary?.Select(x => new ResponseReasoningText(x)).ToList(),
                Content = r.Content?.Select(x => new ResponseReasoningText(x)).ToList(),
                EncryptedContent = r.EncryptedContent,
            };
        }
        return null;
    }

    private static ResponseContent? ResponseContentFromWire(JsonNode? content)
    {
        switch (content)
        {
            case null:
                return null;
            case JsonValue v when v.TryGetValue<string>(out var s):
                return ResponseContent.FromText(s);
            case JsonArray arr:
            {
                var parts = new List<ResponseContentPart>(arr.Count);
                foreach (var el in arr)
                {
                    if (el is null)
                    {
                        continue;
                    }
                    var wp = el.Deserialize<WireResponseContentPart>(Json.Options);
                    if (wp is null)
                    {
                        continue;
                    }
                    parts.Add(ResponseContentPartFromWire(wp));
                }
                return ResponseContent.FromParts(parts);
            }
            default:
                return null;
        }
    }

    private static ResponseContentPart ResponseContentPartFromWire(WireResponseContentPart wp)
    {
        if (wp.InputImage is { } img)
        {
            return new InputImagePart(img.ImageUrl, img.Detail);
        }
        if (wp.OutputText is { } outText)
        {
            return new OutputTextPart(outText);
        }
        return new InputTextPart(wp.InputText ?? "");
    }

    private static ErrorBody? ErrorBodyFromWire(WireErrorBody? e)
        => e is null ? null : new ErrorBody(e.Message, e.Type, e.Code);

    private static ResponsesUsage? ResponsesUsageFromWire(WireResponsesUsage? u)
        => u is null
            ? null
            : new ResponsesUsage(
                u.InputTokens,
                u.OutputTokens,
                u.TotalTokens,
                u.InputTokensDetails is null ? null : new ResponsesInputTokensDetails(u.InputTokensDetails.CachedTokens),
                u.OutputTokensDetails is null ? null : new ResponsesOutputTokensDetails(u.OutputTokensDetails.ReasoningTokens));

    // ---- responses stream event (decode) --------------------------------

    internal static ResponsesStreamEvent ResponsesStreamEventFromWire(WireResponsesStreamEvent w) => new()
    {
        Type = w.Type,
        SequenceNumber = w.SequenceNumber,
        Response = w.Response is null ? null : ResponsesResponseFromWire(w.Response),
        OutputIndex = w.OutputIndex,
        ItemId = w.ItemId,
        ContentIndex = w.ContentIndex,
        Item = w.Item is null ? null : ResponseItemFromWire(w.Item),
        Part = w.Part is null ? null : ResponseContentPartFromWire(w.Part),
        Delta = w.Delta,
        Text = w.Text,
        Arguments = w.Arguments,
        Message = w.Message,
    };

    // Recognised Responses stream event types. SDKs MUST ignore types they don't recognise (the dialect
    // grows by adding types) — an unrecognised type is skipped rather than surfaced (SPEC.md).
    private static readonly HashSet<string> RecognisedResponseEvents = new(StringComparer.Ordinal)
    {
        "response.created",
        "response.in_progress",
        "response.completed",
        "response.incomplete",
        "response.failed",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "error",
    };

    /// <summary>Whether a Responses stream event <c>type</c> is one this SDK understands; unknown types
    /// are skipped by the stream reader (forward-compatible with dialect growth).</summary>
    internal static bool IsRecognisedResponseEvent(string type) => RecognisedResponseEvents.Contains(type);

    // ---- embeddings -----------------------------------------------------

    internal static byte[] EncodeEmbeddingRequest(EmbeddingRequest req)
    {
        var w = new WireEmbeddingRequest
        {
            Model = req.Model,
            Input = EncodeStringOrArray(req.Input) ?? new JsonArray(),
            Dimensions = req.Dimensions,
            EncodingFormat = req.EncodingFormat,
        };
        return Json.MergeExtra(w, Json.RawValue(req.Extra));
    }

    internal static EmbeddingResponse EmbeddingResponseFromWire(WireEmbeddingResponse w) => new(
        w.Object,
        w.Data.Select(e => new Embedding(e.Object, e.Index, DecodeEmbeddingVector(e.Embedding))).ToList(),
        w.Model,
        UsageFromWire(w.Usage));

    /// <summary>
    /// Decode the wire <c>embedding</c> value — a JSON float array (encoding_format "float") or a
    /// base64 string of little-endian f32 bytes (encoding_format "base64") — into a float vector.
    /// </summary>
    internal static IReadOnlyList<float> DecodeEmbeddingVector(JsonNode? node)
    {
        switch (node)
        {
            case null:
                return [];
            case JsonArray arr:
            {
                var vec = new float[arr.Count];
                for (var i = 0; i < arr.Count; i++)
                {
                    vec[i] = arr[i]!.GetValue<float>();
                }
                return vec;
            }
            case JsonValue v when v.TryGetValue<string>(out var b64):
                return DecodeBase64F32(b64);
            default:
                throw new FormatException($"llmleaf: unexpected embedding encoding: {node.ToJsonString()}");
        }
    }

    private static float[] DecodeBase64F32(string s)
    {
        var data = Convert.FromBase64String(s);
        if (data.Length % 4 != 0)
        {
            throw new FormatException($"llmleaf: base64 embedding byte length {data.Length} is not a multiple of 4");
        }
        var vec = new float[data.Length / 4];
        for (var i = 0; i < vec.Length; i++)
        {
            vec[i] = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(i * 4, 4));
        }
        return vec;
    }

    // ---- speech / voices ------------------------------------------------

    internal static byte[] EncodeSpeechRequest(SpeechRequest req)
    {
        var w = new WireSpeechRequest
        {
            Model = req.Model,
            Input = req.Input,
            Voice = req.Voice,
            ResponseFormat = req.ResponseFormat,
            Speed = req.Speed,
        };
        return Json.MergeExtra(w, Json.RawValue(req.Extra));
    }

    internal static VoicesResponse VoicesResponseFromWire(WireVoicesResponse w) => new(
        w.Model,
        w.Voices.Select(v => new Voice(v.Id, v.Name, v.Languages)).ToList());

    // ---- transcription --------------------------------------------------

    internal static TranscriptionResponse TranscriptionResponseFromWire(WireTranscriptionResponse w)
        => new(w.Text, w.Task, w.Language, w.Duration, UsageFromWire(w.Usage));

    // ---- models catalog -------------------------------------------------

    internal static ListModelsResponse ListModelsResponseFromWire(WireListModelsResponse w)
        => new(w.Data.Select(ModelEntryFromWire).ToList());

    private static ModelEntry ModelEntryFromWire(WireModelEntry m) => new()
    {
        Id = m.Id,
        CanonicalSlug = m.CanonicalSlug,
        Name = m.Name,
        Created = m.Created,
        Description = m.Description,
        ContextLength = m.ContextLength,
        Architecture = m.Architecture is null
            ? null
            : new Architecture(
                m.Architecture.InputModalities ?? [],
                m.Architecture.OutputModalities ?? [],
                m.Architecture.Tokenizer,
                m.Architecture.Modality,
                m.Architecture.InstructType),
        Pricing = m.Pricing is null ? null : new Pricing(m.Pricing.Prompt, m.Pricing.Completion),
        TopProvider = m.TopProvider is null
            ? null
            : new TopProvider(
                m.TopProvider.IsModerated,
                m.TopProvider.ContextLength,
                m.TopProvider.MaxCompletionTokens,
                m.TopProvider.MaxThinkingTokens),
        SupportedParameters = m.SupportedParameters ?? [],
        UnsupportedParameters = m.UnsupportedParameters ?? [],
        DefaultParameters = Json.RawString(m.DefaultParameters),
        Endpoints = m.Endpoints?.Select(e => new ModelEndpoint(e.Provider, e.Model, e.Down, e.Source)).ToList() ?? [],
    };

    // ---- batches --------------------------------------------------------

    internal static byte[] EncodeBatchCreateRequest(BatchCreateRequest req)
    {
        var w = new WireBatchCreateRequest
        {
            Requests = req.Requests
                .Select(item => new WireBatchRequestItem
                {
                    CustomId = item.CustomId,
                    Body = EncodeChatRequestNode(item.Body),
                })
                .ToList(),
        };
        return JsonSerializer.SerializeToUtf8Bytes(w, Json.Options);
    }

    internal static BatchHandle BatchHandleFromWire(WireBatchHandle w) => new(
        w.Id,
        EnumWire.FromWire<BatchStatus>(w.Status),
        w.Counts is null
            ? null
            : new BatchCounts(w.Counts.Total, w.Counts.Processing, w.Counts.Succeeded, w.Counts.Errored, w.Counts.Canceled, w.Counts.Expired),
        w.CreatedAt,
        w.ExpiresAt,
        w.EndedAt,
        w.Endpoint);

    internal static BatchResultLine BatchResultLineFromWire(WireBatchResultLine w) => new(
        w.CustomId,
        w.Response is null ? null : new BatchResponse(w.Response.StatusCode, ChatResponseFromWire(w.Response.Body)),
        w.Error is null ? null : new BatchError(w.Error.Code, w.Error.Message));
}
