using System.Text.Json;
using Xunit;

namespace Llmleaf.Client.Tests;

// End-to-end tests of the HttpClient + System.Text.Json transport against an in-process server.
// They assert both the exact request bytes the SDK emits AND the decode of canned responses,
// proving the SPEC.md wire mapping (snake_case keys, lowercase enum tokens, string-or-array
// content/stop/input, raw free-form JSON splicing, base64 embedding decode, SSE/NDJSON streaming,
// the error envelope).
public sealed class WireTests
{
    private static LlmleafClient Client(TestServer server, LlmleafClientOptions? opts = null)
        => new(server.BaseUrl, "test-key", opts);

    private static JsonElement Parse(string json) => JsonDocument.Parse(json).RootElement;

    // ---- chat: request encoding ----------------------------------------

    [Fact]
    public async Task ChatRequest_EmitsSnakeCaseAndLowercaseEnums()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[]}"""));
        using var client = Client(server);

        await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "gpt-4o-mini",
            Messages = [ChatMessage.Text(Role.User, "hi")],
            MaxCompletionTokens = 64,
            Temperature = 0.5f,
        });

        var body = Parse(server.LastRequest!.Body);
        Assert.Equal("/v1/chat/completions", server.LastRequest.Path);
        Assert.Equal("Bearer test-key", server.LastRequest.Headers["Authorization"]);
        Assert.Equal("gpt-4o-mini", body.GetProperty("model").GetString());
        Assert.False(body.GetProperty("stream").GetBoolean()); // forced false for non-streaming
        Assert.Equal(64u, body.GetProperty("max_completion_tokens").GetUInt32());
        var msg = body.GetProperty("messages")[0];
        Assert.Equal("user", msg.GetProperty("role").GetString());     // lowercase token
        Assert.Equal("hi", msg.GetProperty("content").GetString());    // bare string content
    }

    [Fact]
    public async Task ChatRequest_StopOneElementIsBareString_MultipleIsArray()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[]}"""));
        using var client = Client(server);

        await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")], Stop = ["\n"],
        });
        Assert.Equal(JsonValueKind.String, Parse(server.LastRequest!.Body).GetProperty("stop").ValueKind);

        await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")], Stop = ["a", "b"],
        });
        Assert.Equal(JsonValueKind.Array, Parse(server.LastRequest!.Body).GetProperty("stop").ValueKind);
    }

    [Fact]
    public async Task ChatRequest_ExtraIsMergedAtTopLevelAsRawJson()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[]}"""));
        using var client = Client(server);

        await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m",
            Messages = [ChatMessage.Text(Role.User, "hi")],
            Extra = """{"provider":{"order":["a","b"]},"logprobs":true}""",
        });

        var body = Parse(server.LastRequest!.Body);
        // Spliced as a JSON value, not double-encoded as a string.
        Assert.Equal(JsonValueKind.Object, body.GetProperty("provider").ValueKind);
        Assert.Equal("a", body.GetProperty("provider").GetProperty("order")[0].GetString());
        Assert.True(body.GetProperty("logprobs").GetBoolean());
    }

    [Fact]
    public async Task ChatRequest_ToolsAndToolChoiceAndResponseFormat_RawSchemas()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[]}"""));
        using var client = Client(server);

        await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m",
            Messages = [ChatMessage.Text(Role.User, "hi")],
            Tools = [new ToolDef("function", new FunctionDef("get_weather", "Get weather", """{"type":"object","properties":{"city":{"type":"string"}}}"""))],
            ToolChoice = ToolChoice.Named("get_weather"),
            ResponseFormat = new ResponseFormat("json_schema", """{"name":"out","schema":{"type":"object"}}"""),
        });

        var body = Parse(server.LastRequest!.Body);
        var fn = body.GetProperty("tools")[0].GetProperty("function");
        Assert.Equal("get_weather", fn.GetProperty("name").GetString());
        Assert.Equal(JsonValueKind.Object, fn.GetProperty("parameters").ValueKind); // raw schema, not a string
        var tc = body.GetProperty("tool_choice");
        Assert.Equal("function", tc.GetProperty("type").GetString());
        Assert.Equal("get_weather", tc.GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("json_schema", body.GetProperty("response_format").GetProperty("type").GetString());
        Assert.Equal(JsonValueKind.Object, body.GetProperty("response_format").GetProperty("json_schema").ValueKind);
    }

    [Fact]
    public async Task ChatRequest_MultimodalContentIsArray()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[]}"""));
        using var client = Client(server);

        await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m",
            Messages =
            [
                new ChatMessage
                {
                    Role = Role.User,
                    Content = MessageContent.FromParts(
                    [
                        new TextPart("look:"),
                        new ImageUrlPart("https://x/y.png", "high"),
                    ]),
                },
            ],
        });

        var content = Parse(server.LastRequest!.Body).GetProperty("messages")[0].GetProperty("content");
        Assert.Equal(JsonValueKind.Array, content.ValueKind);
        Assert.Equal("text", content[0].GetProperty("type").GetString());
        Assert.Equal("image_url", content[1].GetProperty("type").GetString());
        Assert.Equal("https://x/y.png", content[1].GetProperty("image_url").GetProperty("url").GetString());
        Assert.Equal("high", content[1].GetProperty("image_url").GetProperty("detail").GetString());
    }

    // ---- chat: response decoding ---------------------------------------

    [Fact]
    public async Task ChatResponse_DecodesChoicesEnumAndUsage()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """
            {"id":"chatcmpl-1","object":"chat.completion","created":123,"model":"m",
             "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
             "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4,"cost_usd":0.0001}}
            """));
        using var client = Client(server);

        var resp = await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")],
        });

        Assert.Equal("chatcmpl-1", resp.Id);
        var choice = resp.Choices[0];
        Assert.Equal(Role.Assistant, choice.Message.Role);
        Assert.Equal("hello", choice.Message.Content!.Text);
        Assert.Equal(FinishReason.Stop, choice.FinishReason);
        Assert.Equal(4u, resp.Usage!.TotalTokens);
        Assert.Equal(0.0001, resp.Usage.CostUsd!.Value, 6);
    }

    [Fact]
    public async Task ChatResponse_DecodesCacheMetadataAndReasoning()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """
            {"id":"r","object":"chat.completion","created":1,"model":"m",
             "choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning":"let me think",
               "reasoning_details":[
                 {"type":"reasoning.text","text":"step","signature":"sig-1","index":0},
                 {"type":"reasoning.encrypted","data":"opaque-blob"}]},"finish_reason":"stop"}],
             "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,
               "prompt_tokens_details":{"cached_tokens":6},"cache_creation_tokens":4}}
            """));
        using var client = Client(server);

        var resp = await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")],
        });

        // Cache accounting.
        Assert.Equal(6u, resp.Usage!.PromptTokensDetails!.CachedTokens);
        Assert.Equal(4u, resp.Usage.CacheCreationTokens);

        // Reasoning: flat + structured (open text block, hidden encrypted block).
        var msg = resp.Choices[0].Message;
        Assert.Equal("let me think", msg.Reasoning);
        Assert.Equal(2, msg.ReasoningDetails!.Count);
        var open = msg.ReasoningDetails[0];
        Assert.Equal("reasoning.text", open.Type);
        Assert.Equal("step", open.Text);
        Assert.Equal("sig-1", open.Signature);
        Assert.False(open.IsHidden);
        Assert.Equal("step", open.OpenText);
        var hidden = msg.ReasoningDetails[1];
        Assert.Equal("opaque-blob", hidden.Data);
        Assert.True(hidden.IsHidden);
        Assert.Null(hidden.OpenText);
    }

    [Fact]
    public async Task ChatResponse_OmitsCacheMetadataWhenAbsent()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """
            {"id":"r","object":"chat.completion","created":1,"model":"m",
             "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
             "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}
            """));
        using var client = Client(server);

        var resp = await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")],
        });

        Assert.Null(resp.Usage!.PromptTokensDetails);
        Assert.Null(resp.Usage.CacheCreationTokens);
        Assert.Null(resp.Choices[0].Message.Reasoning);
        Assert.Null(resp.Choices[0].Message.ReasoningDetails);
    }

    [Fact]
    public async Task ChatRequest_EchoesReasoningDetailsVerbatim()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[]}"""));
        using var client = Client(server);

        await client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m",
            Messages =
            [
                new ChatMessage
                {
                    Role = Role.Assistant,
                    Content = "ok",
                    Reasoning = "thought",
                    ReasoningDetails =
                    [
                        new ReasoningDetail { Type = "reasoning.text", Text = "step", Signature = "sig-1", Index = 0 },
                        new ReasoningDetail { Type = "reasoning.encrypted", Data = "opaque-blob" },
                    ],
                },
            ],
        });

        var msg = Parse(server.LastRequest!.Body).GetProperty("messages")[0];
        Assert.Equal("thought", msg.GetProperty("reasoning").GetString());
        var details = msg.GetProperty("reasoning_details");
        Assert.Equal(JsonValueKind.Array, details.ValueKind);
        Assert.Equal("reasoning.text", details[0].GetProperty("type").GetString());
        Assert.Equal("step", details[0].GetProperty("text").GetString());
        Assert.Equal("sig-1", details[0].GetProperty("signature").GetString());
        Assert.Equal(0u, details[0].GetProperty("index").GetUInt32());
        Assert.Equal("reasoning.encrypted", details[1].GetProperty("type").GetString());
        Assert.Equal("opaque-blob", details[1].GetProperty("data").GetString());
        // Unset opaque fields are omitted, not emitted as null.
        Assert.False(details[1].TryGetProperty("text", out _));
        Assert.False(details[1].TryGetProperty("signature", out _));
    }

    [Fact]
    public async Task ChatStream_DecodesIncrementalReasoning()
    {
        const string sse =
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"thin\",\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"thin\"}]}}]}\n\n" +
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"king\"}}]}\n\n" +
            "data: [DONE]\n\n";
        using var server = new TestServer(_ => CannedResponse.Sse(sse));
        using var client = Client(server);

        var reasoning = "";
        ReasoningDetail? firstDetail = null;
        await foreach (var chunk in client.CreateChatCompletionStreamAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")],
        }))
        {
            var delta = chunk.Choices.Count > 0 ? chunk.Choices[0].Delta : null;
            if (delta?.Reasoning is { } r) reasoning += r;
            if (delta?.ReasoningDetails is { Count: > 0 } d) firstDetail ??= d[0];
        }

        Assert.Equal("thinking", reasoning);
        Assert.NotNull(firstDetail);
        Assert.Equal("reasoning.text", firstDetail!.Type);
        Assert.Equal("thin", firstDetail.Text);
    }

    // ---- chat: streaming -----------------------------------------------

    [Fact]
    public async Task ChatStream_ParsesSseStopsOnDoneAndAccumulates()
    {
        const string sse =
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"He\"}}]}\n\n" +
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\"}}]}\n\n" +
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n" +
            "data: [DONE]\n\n";
        using var server = new TestServer(_ => CannedResponse.Sse(sse));
        using var client = Client(server);

        var text = "";
        Usage? usage = null;
        var count = 0;
        await foreach (var chunk in client.CreateChatCompletionStreamAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")],
        }))
        {
            count++;
            var delta = chunk.Choices.Count > 0 ? chunk.Choices[0].Delta.Content : null;
            if (delta is not null) text += delta;
            if (chunk.Usage is not null) usage = chunk.Usage;
        }

        Assert.Equal(3, count); // [DONE] not yielded
        Assert.Equal("Hello", text);
        Assert.Equal(3u, usage!.TotalTokens);
        // Confirm we sent stream:true on the wire.
        Assert.True(Parse(server.LastRequest!.Body).GetProperty("stream").GetBoolean());
    }

    // ---- embeddings ----------------------------------------------------

    [Fact]
    public async Task Embeddings_DecodesBase64LittleEndianF32()
    {
        // [1.0f, 2.0f] little-endian f32 -> base64.
        var bytes = new byte[8];
        BitConverter.GetBytes(1.0f).CopyTo(bytes, 0);
        BitConverter.GetBytes(2.0f).CopyTo(bytes, 4);
        var b64 = Convert.ToBase64String(bytes);

        var json = "{\"object\":\"list\",\"model\":\"e\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":\""
                   + b64
                   + "\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":0,\"total_tokens\":1}}";
        using var server = new TestServer(req => CannedResponse.Json(json));
        using var client = Client(server);

        var resp = await client.CreateEmbeddingAsync(new EmbeddingRequest
        {
            Model = "e", Input = ["hello"], EncodingFormat = "base64",
        });

        Assert.Equal("base64", Parse(server.LastRequest!.Body).GetProperty("encoding_format").GetString());
        Assert.Equal(JsonValueKind.String, Parse(server.LastRequest!.Body).GetProperty("input").ValueKind); // single -> bare string
        var vec = resp.Data[0].Vector;
        Assert.Equal(2, vec.Count);
        Assert.Equal(1.0f, vec[0]);
        Assert.Equal(2.0f, vec[1]);
    }

    [Fact]
    public async Task Embeddings_DecodesFloatArray()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"object":"list","model":"e","data":[{"object":"embedding","index":0,"embedding":[0.5,-0.25]}]}"""));
        using var client = Client(server);

        var resp = await client.CreateEmbeddingAsync(new EmbeddingRequest { Model = "e", Input = ["a", "b"] });
        Assert.Equal(JsonValueKind.Array, Parse(server.LastRequest!.Body).GetProperty("input").ValueKind); // multi -> array
        Assert.Equal([0.5f, -0.25f], resp.Data[0].Vector);
    }

    // ---- rerank --------------------------------------------------------

    [Fact]
    public async Task Rerank_EncodesRequestAndDecodesResultsAndUsage()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """
            {"object":"list","model":"rerank-1",
             "results":[
               {"index":2,"relevance_score":0.98,"document":"the third doc"},
               {"index":0,"relevance_score":0.12}],
             "usage":{"total_tokens":15,"cost_usd":0.0002}}
            """));
        using var client = Client(server);

        var resp = await client.CreateRerankAsync(new RerankRequest
        {
            Model = "rerank-1",
            Query = "find it",
            Documents = ["a", "b", "the third doc"],
            TopN = 2,
            ReturnDocuments = true,
        });

        // Request: flat query + always-array documents + the optional flags (no vector encoding).
        var body = Parse(server.LastRequest!.Body);
        Assert.Equal("/v1/rerank", server.LastRequest.Path);
        Assert.Equal("rerank-1", body.GetProperty("model").GetString());
        Assert.Equal("find it", body.GetProperty("query").GetString());
        Assert.Equal(JsonValueKind.Array, body.GetProperty("documents").ValueKind);
        Assert.Equal("the third doc", body.GetProperty("documents")[2].GetString());
        Assert.Equal(2u, body.GetProperty("top_n").GetUInt32());
        Assert.True(body.GetProperty("return_documents").GetBoolean());

        // Response: results in returned order, with the echoed document (string) and usage.
        Assert.Equal("rerank-1", resp.Model);
        Assert.Equal(2, resp.Results.Count);
        Assert.Equal(2u, resp.Results[0].Index);
        Assert.Equal(0.98f, resp.Results[0].RelevanceScore);
        Assert.Equal("the third doc", resp.Results[0].Document!.Value.GetString());
        Assert.Null(resp.Results[1].Document); // absent when not echoed
        Assert.Equal(15u, resp.Usage!.TotalTokens);
        Assert.Equal(0.0002, resp.Usage.CostUsd!.Value, 6);
    }

    // ---- models --------------------------------------------------------

    [Fact]
    public async Task ListModels_SendsQueryAndAdminTokenOnlyWhenOptedIn()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"data":[{"id":"gpt-4o-mini","canonical_slug":"openai/gpt-4o-mini","name":"GPT","created":1,"description":"d","supported_parameters":["temperature"]}]}"""));
        using var client = Client(server, new LlmleafClientOptions { AdminToken = "admin-secret" });

        var resp = await client.ListModelsAsync(new ListModelsOptions { Type = ModelType.Llm, Search = "gpt", Admin = true });
        Assert.Contains("type=llm", server.LastRequest!.Query);
        Assert.Contains("search=gpt", server.LastRequest.Query);
        Assert.Equal("admin-secret", server.LastRequest.Headers["x-admin-token"]);
        Assert.Equal("gpt-4o-mini", resp.Data[0].Id);
        Assert.Equal("temperature", resp.Data[0].SupportedParameters[0]);

        // Without Admin=true the token must not be sent.
        await client.ListModelsAsync(new ListModelsOptions { Type = ModelType.All });
        Assert.False(server.LastRequest!.Headers.ContainsKey("x-admin-token"));
    }

    // ---- audio: speech / voices ----------------------------------------

    [Fact]
    public async Task Speech_ReturnsBytesAndContentType()
    {
        using var server = new TestServer(_ => new CannedResponse(200, "audio/mpeg", "ID3audio-bytes"));
        using var client = Client(server);

        var result = await client.CreateSpeechAsync(new SpeechRequest
        {
            Model = "tts-1", Input = "hello", Voice = "alloy", ResponseFormat = "mp3",
        });

        Assert.Equal("audio/mpeg", result.ContentType);
        Assert.Equal("ID3audio-bytes", System.Text.Encoding.UTF8.GetString(result.Bytes));
        var body = Parse(server.LastRequest!.Body);
        Assert.Equal("alloy", body.GetProperty("voice").GetString());
        Assert.Equal("mp3", body.GetProperty("response_format").GetString());
    }

    [Fact]
    public async Task Voices_DecodesList()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"model":"tts-1","voices":[{"id":"alloy","name":"Alloy","languages":["en-US"]}]}"""));
        using var client = Client(server);

        var resp = await client.ListVoicesAsync("tts-1");
        Assert.Contains("model=tts-1", server.LastRequest!.Query);
        Assert.Equal("alloy", resp.Voices[0].Id);
        Assert.Equal("en-US", resp.Voices[0].Languages![0]);
    }

    // ---- audio: transcription ------------------------------------------

    [Fact]
    public async Task Transcription_SendsMultipartAndDecodesJson()
    {
        using var server = new TestServer(_ => CannedResponse.Json("""{"text":"hello world","language":"en"}"""));
        using var client = Client(server);

        var resp = await client.CreateTranscriptionAsync(
            new TranscriptionFile([1, 2, 3, 4], "audio.wav", "audio/wav"),
            new TranscriptionRequest { Model = "whisper-1", ResponseFormat = "json", Language = "en" });

        Assert.StartsWith("multipart/form-data", server.LastRequest!.ContentType);
        Assert.Contains("name=\"file\"", server.LastRequest.Body);
        Assert.Contains("filename=\"audio.wav\"", server.LastRequest.Body);
        Assert.Contains("name=\"model\"", server.LastRequest.Body);
        Assert.Equal("hello world", resp.Text);
        Assert.Equal("en", resp.Language);
    }

    [Fact]
    public async Task Transcription_PlainTextBodyReturnedVerbatim()
    {
        using var server = new TestServer(_ => new CannedResponse(200, "text/plain", "just the words"));
        using var client = Client(server);

        var resp = await client.CreateTranscriptionAsync(
            new TranscriptionFile([1, 2], "a.mp3"),
            new TranscriptionRequest { Model = "whisper-1", ResponseFormat = "text" });

        Assert.Equal("just the words", resp.Text);
        Assert.Null(resp.Language);
    }

    // ---- batches -------------------------------------------------------

    [Fact]
    public async Task Batch_CreateEncodesBodiesAndDecodesHandle()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"id":"batch_1","status":"in_progress","counts":{"total":2,"processing":2,"succeeded":0,"errored":0,"canceled":0,"expired":0}}"""));
        using var client = Client(server);

        var handle = await client.CreateBatchAsync(new BatchCreateRequest(
        [
            new BatchRequestItem("a", new ChatRequest { Model = "m", Messages = [ChatMessage.Text(Role.User, "1")] }),
            new BatchRequestItem("b", new ChatRequest { Model = "m", Messages = [ChatMessage.Text(Role.User, "2")] }),
        ]));

        var body = Parse(server.LastRequest!.Body);
        Assert.Equal("a", body.GetProperty("requests")[0].GetProperty("custom_id").GetString());
        Assert.Equal("m", body.GetProperty("requests")[0].GetProperty("body").GetProperty("model").GetString());
        Assert.Equal(BatchStatus.InProgress, handle.Status); // in_progress -> InProgress
        Assert.Equal(2ul, handle.Counts!.Total);
    }

    [Fact]
    public async Task Batch_ResultsStreamAsNdjson()
    {
        const string ndjson =
            "{\"custom_id\":\"a\",\"response\":{\"status_code\":200,\"body\":{\"id\":\"r1\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}}}\n" +
            "{\"custom_id\":\"b\",\"error\":{\"code\":\"rate_limited\",\"message\":\"slow down\"}}\n";
        using var server = new TestServer(_ => CannedResponse.Ndjson(ndjson));
        using var client = Client(server);

        var lines = new List<BatchResultLine>();
        await foreach (var line in client.GetBatchResultsAsync("batch_1"))
        {
            lines.Add(line);
        }

        Assert.Equal(2, lines.Count);
        Assert.Equal("ok", lines[0].Response!.Body.Choices[0].Message.Content!.Text);
        Assert.Equal(200u, lines[0].Response!.StatusCode);
        Assert.Equal("rate_limited", lines[1].Error!.Code);
    }

    // ---- errors --------------------------------------------------------

    [Fact]
    public async Task NonSuccess_ThrowsTypedApiExceptionFromEnvelope()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"error":{"message":"key suspended"}}""", status: 429));
        using var client = Client(server);

        var ex = await Assert.ThrowsAsync<ApiException>(() => client.CreateChatCompletionAsync(new ChatRequest
        {
            Model = "m", Messages = [ChatMessage.Text(Role.User, "hi")],
        }));
        Assert.Equal(429, ex.Status);
        Assert.Equal("key suspended", ex.Message);
    }
}
