using System.Text.Json;
using Xunit;

namespace Llmleaf.Client.Tests;

// End-to-end tests of the Responses dialect (POST /v1/responses) transport against an in-process
// server. They assert the exact request bytes (flat tools/tool_choice, the type-discriminated item
// array with function_call / function_call_output / reasoning replay, input_image.image_url as a
// plain string, output_text annotations:[]), the decode of a canned response (output items, usage
// with cached_tokens, store:false), typed-SSE streaming (no [DONE]; unknown events skipped; stop on
// the terminal event), the mid-stream "error" event, and the error envelope -> ApiException.
public sealed class ResponsesWireTests
{
    private static LlmleafClient Client(TestServer server, LlmleafClientOptions? opts = null)
        => new(server.BaseUrl, "test-key", opts);

    private static JsonElement Parse(string json) => JsonDocument.Parse(json).RootElement;

    // A minimal completed response the server can echo when a test only cares about the request.
    private const string MinimalResponse =
        """{"id":"resp_0","object":"response","created_at":1,"status":"completed","model":"m","store":false}""";

    // ---- request encoding ----------------------------------------------

    [Fact]
    public async Task ResponsesRequest_EncodesFlatToolsAndTypedItemArray()
    {
        // A rich response so the round trip also proves decoding (asserted below).
        const string response =
            """
            {"id":"resp_1","object":"response","created_at":1720000000,"status":"completed","model":"gpt-4o",
             "output":[
               {"type":"message","id":"msg_1","role":"assistant","status":"completed",
                "content":[{"type":"output_text","text":"It is sunny in Paris.","annotations":[]}]},
               {"type":"function_call","id":"fc_2","call_id":"call_2","name":"get_weather",
                "arguments":"{\"city\":\"Paris\"}","status":"completed"}],
             "usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":10},
                      "output_tokens":8,"output_tokens_details":{"reasoning_tokens":3},"total_tokens":50},
             "store":false}
            """;
        using var server = new TestServer(_ => CannedResponse.Json(response));
        using var client = Client(server);

        var resp = await client.CreateResponseAsync(new ResponsesRequest
        {
            Model = "gpt-4o",
            Instructions = "Be terse.",
            MaxOutputTokens = 256,
            Reasoning = new ResponsesReasoning("medium", "auto"),
            Tools = [new ResponsesToolDef("function", "get_weather", "Get weather", """{"type":"object","properties":{"city":{"type":"string"}}}""", Strict: false)],
            ToolChoice = ResponsesToolChoice.Named("get_weather"),
            Store = false,
            Input = ResponsesInput.FromItems(
            [
                new ResponseMessageItem
                {
                    Role = "user",
                    Content = ResponseContent.FromParts(
                    [
                        new InputTextPart("look:"),
                        new InputImagePart("data:image/png;base64,AAAA", "high"),
                    ]),
                },
                new ResponseMessageItem
                {
                    Role = "assistant",
                    Content = ResponseContent.FromParts([new OutputTextPart("earlier reply")]),
                },
                new ResponseFunctionCallItem { Id = "fc_1", CallId = "call_1", Name = "get_weather", Arguments = """{"city":"Paris"}""" },
                new ResponseFunctionCallOutputItem { CallId = "call_1", Output = "sunny" },
                new ResponseReasoningItem
                {
                    Id = "rs_1",
                    Summary = [new ResponseReasoningText("thinking summary")],
                    Content = [new ResponseReasoningText("full thought")],
                    EncryptedContent = "opaque==",
                },
            ]),
        });

        var body = Parse(server.LastRequest!.Body);
        Assert.Equal("/v1/responses", server.LastRequest.Path);
        Assert.Equal("Bearer test-key", server.LastRequest.Headers["Authorization"]);
        Assert.False(body.GetProperty("stream").GetBoolean()); // forced false for non-streaming
        Assert.Equal("Be terse.", body.GetProperty("instructions").GetString());
        Assert.Equal(256u, body.GetProperty("max_output_tokens").GetUInt32());

        // reasoning config (distinct from the reasoning item inside `input`).
        Assert.Equal("medium", body.GetProperty("reasoning").GetProperty("effort").GetString());
        Assert.Equal("auto", body.GetProperty("reasoning").GetProperty("summary").GetString());

        // Tools are FLAT: type/name/parameters at the top level, no nested `function` object.
        var tool = body.GetProperty("tools")[0];
        Assert.Equal("function", tool.GetProperty("type").GetString());
        Assert.Equal("get_weather", tool.GetProperty("name").GetString());
        Assert.False(tool.TryGetProperty("function", out _));
        Assert.Equal(JsonValueKind.Object, tool.GetProperty("parameters").ValueKind); // raw schema, not a string
        Assert.False(tool.GetProperty("strict").GetBoolean());

        // tool_choice is the FLAT named object.
        var tc = body.GetProperty("tool_choice");
        Assert.Equal("function", tc.GetProperty("type").GetString());
        Assert.Equal("get_weather", tc.GetProperty("name").GetString());
        Assert.False(tc.TryGetProperty("function", out _));

        var input = body.GetProperty("input");
        Assert.Equal(JsonValueKind.Array, input.ValueKind);

        // [0] user message: role-keyed with NO "type"; input_image.image_url is a PLAIN STRING.
        var userMsg = input[0];
        Assert.Equal("user", userMsg.GetProperty("role").GetString());
        Assert.False(userMsg.TryGetProperty("type", out _));
        var userParts = userMsg.GetProperty("content");
        Assert.Equal("input_text", userParts[0].GetProperty("type").GetString());
        Assert.Equal("input_image", userParts[1].GetProperty("type").GetString());
        Assert.Equal(JsonValueKind.String, userParts[1].GetProperty("image_url").ValueKind);
        Assert.Equal("data:image/png;base64,AAAA", userParts[1].GetProperty("image_url").GetString());
        Assert.Equal("high", userParts[1].GetProperty("detail").GetString());

        // [1] assistant message: a constructed output_text part carries "annotations":[].
        var outPart = input[1].GetProperty("content")[0];
        Assert.Equal("output_text", outPart.GetProperty("type").GetString());
        Assert.Equal(JsonValueKind.Array, outPart.GetProperty("annotations").ValueKind);
        Assert.Equal(0, outPart.GetProperty("annotations").GetArrayLength());

        // [2] function_call replay: typed; arguments is a raw JSON STRING.
        var fc = input[2];
        Assert.Equal("function_call", fc.GetProperty("type").GetString());
        Assert.Equal("call_1", fc.GetProperty("call_id").GetString());
        Assert.Equal("get_weather", fc.GetProperty("name").GetString());
        Assert.Equal("""{"city":"Paris"}""", fc.GetProperty("arguments").GetString());

        // [3] function_call_output replay.
        var fco = input[3];
        Assert.Equal("function_call_output", fco.GetProperty("type").GetString());
        Assert.Equal("call_1", fco.GetProperty("call_id").GetString());
        Assert.Equal("sunny", fco.GetProperty("output").GetString());

        // [4] reasoning replay: summary[] -> summary_text, content[] -> reasoning_text; encrypted echoed.
        var reasoning = input[4];
        Assert.Equal("reasoning", reasoning.GetProperty("type").GetString());
        Assert.Equal("summary_text", reasoning.GetProperty("summary")[0].GetProperty("type").GetString());
        Assert.Equal("thinking summary", reasoning.GetProperty("summary")[0].GetProperty("text").GetString());
        Assert.Equal("reasoning_text", reasoning.GetProperty("content")[0].GetProperty("type").GetString());
        Assert.Equal("full thought", reasoning.GetProperty("content")[0].GetProperty("text").GetString());
        Assert.Equal("opaque==", reasoning.GetProperty("encrypted_content").GetString());

        // ---- decode side of the round trip ----
        Assert.Equal("resp_1", resp.Id);
        Assert.Equal("response", resp.Object);
        Assert.Equal(1720000000L, resp.CreatedAt);
        Assert.Equal("completed", resp.Status);
        Assert.False(resp.Store!.Value); // llmleaf always reports store:false

        Assert.Equal(2, resp.Output.Count);
        var respMsg = Assert.IsType<ResponseMessageItem>(resp.Output[0]);
        Assert.Equal("assistant", respMsg.Role);
        Assert.Equal("completed", respMsg.Status);
        var respOut = Assert.IsType<OutputTextPart>(respMsg.Content!.Parts![0]);
        Assert.Equal("It is sunny in Paris.", respOut.Text);
        var respCall = Assert.IsType<ResponseFunctionCallItem>(resp.Output[1]);
        Assert.Equal("call_2", respCall.CallId);
        Assert.Equal("get_weather", respCall.Name);

        Assert.Equal(42u, resp.Usage!.InputTokens);
        Assert.Equal(8u, resp.Usage.OutputTokens);
        Assert.Equal(50u, resp.Usage.TotalTokens);
        Assert.Equal(10u, resp.Usage.InputTokensDetails!.CachedTokens);
        Assert.Equal(3u, resp.Usage.OutputTokensDetails!.ReasoningTokens);
    }

    [Fact]
    public async Task ResponsesRequest_BareStringInputSerialisesAsString()
    {
        using var server = new TestServer(_ => CannedResponse.Json(MinimalResponse));
        using var client = Client(server);

        await client.CreateResponseAsync(new ResponsesRequest { Model = "m", Input = "hello" });

        var input = Parse(server.LastRequest!.Body).GetProperty("input");
        Assert.Equal(JsonValueKind.String, input.ValueKind);
        Assert.Equal("hello", input.GetString());
    }

    [Fact]
    public async Task ResponsesRequest_ExtraMergedAtTopLevelAsRawJson()
    {
        using var server = new TestServer(_ => CannedResponse.Json(MinimalResponse));
        using var client = Client(server);

        await client.CreateResponseAsync(new ResponsesRequest
        {
            Model = "m",
            Input = "hi",
            Extra = """{"metadata":{"trace":"abc"},"safety_identifier":"u1"}""",
        });

        var body = Parse(server.LastRequest!.Body);
        Assert.Equal("abc", body.GetProperty("metadata").GetProperty("trace").GetString());
        Assert.Equal("u1", body.GetProperty("safety_identifier").GetString());
    }

    // ---- streaming -----------------------------------------------------

    private static string Frame(string type, string data) => $"event: {type}\ndata: {data}\n\n";

    [Fact]
    public async Task ResponsesStream_DecodesTypedEventsAccumulatesTextAndStops()
    {
        var sse =
            Frame("response.created",
                """{"type":"response.created","sequence_number":0,"response":{"id":"resp","object":"response","created_at":1,"status":"in_progress","model":"m"}}""") +
            Frame("response.output_item.added",
                """{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc","call_id":"call_1","name":"get_weather","arguments":""}}""") +
            Frame("response.function_call_arguments.delta",
                """{"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc","delta":"{\"city\":"}""") +
            // An unknown event type — must be skipped (forward compatibility).
            Frame("response.custom_ping",
                """{"type":"response.custom_ping","sequence_number":3}""") +
            Frame("response.output_text.delta",
                """{"type":"response.output_text.delta","sequence_number":4,"delta":"Hel"}""") +
            Frame("response.output_text.delta",
                """{"type":"response.output_text.delta","sequence_number":5,"delta":"lo"}""") +
            Frame("response.completed",
                """{"type":"response.completed","sequence_number":6,"response":{"id":"resp","object":"response","created_at":1,"status":"completed","model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello","annotations":[]}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}""");
        using var server = new TestServer(_ => CannedResponse.Sse(sse));
        using var client = Client(server);

        var events = new List<ResponsesStreamEvent>();
        await foreach (var e in client.CreateResponseStreamAsync(new ResponsesRequest { Model = "m", Input = "hi" }))
        {
            events.Add(e);
        }

        // Event order, with the unknown "response.custom_ping" skipped.
        Assert.Equal(
            new[]
            {
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.completed",
            },
            events.Select(e => e.Type).ToArray());
        Assert.DoesNotContain("response.custom_ping", events.Select(e => e.Type));

        // sequence_number strictly increasing across the surfaced events.
        var seqs = events.Select(e => e.SequenceNumber).ToArray();
        for (var i = 1; i < seqs.Length; i++)
        {
            Assert.True(seqs[i] > seqs[i - 1]);
        }

        // The output_item.added event carries a decoded function_call item.
        var added = events.Single(e => e.Type == "response.output_item.added");
        var call = Assert.IsType<ResponseFunctionCallItem>(added.Item);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("call_1", call.CallId);

        // Accumulated text from the output_text deltas.
        var text = string.Concat(events.Where(e => e.Type == "response.output_text.delta").Select(e => e.Delta));
        Assert.Equal("Hello", text);

        // Terminal event carries the full snapshot with usage.
        var terminal = events[^1];
        Assert.True(terminal.IsTerminal);
        Assert.Equal("completed", terminal.Response!.Status);
        Assert.Equal(7u, terminal.Response.Usage!.TotalTokens);

        // Confirm we sent stream:true on the wire.
        Assert.True(Parse(server.LastRequest!.Body).GetProperty("stream").GetBoolean());
    }

    [Fact]
    public async Task ResponsesStream_SurfacesErrorEventAsFrame()
    {
        var sse = Frame("error", """{"type":"error","sequence_number":0,"message":"upstream exploded"}""");
        using var server = new TestServer(_ => CannedResponse.Sse(sse));
        using var client = Client(server);

        var events = new List<ResponsesStreamEvent>();
        await foreach (var e in client.CreateResponseStreamAsync(new ResponsesRequest { Model = "m", Input = "hi" }))
        {
            events.Add(e);
        }

        var err = Assert.Single(events);
        Assert.Equal("error", err.Type);
        Assert.Equal("upstream exploded", err.Message);
        Assert.False(err.IsTerminal); // the connection closing ends the stream, not a terminal event
    }

    // ---- errors --------------------------------------------------------

    [Fact]
    public async Task Responses_NonSuccessThrowsTypedApiExceptionFromEnvelope()
    {
        using var server = new TestServer(_ => CannedResponse.Json(
            """{"error":{"message":"previous_response_id is not supported"}}""", status: 400));
        using var client = Client(server);

        var ex = await Assert.ThrowsAsync<ApiException>(() => client.CreateResponseAsync(new ResponsesRequest
        {
            Model = "m", Input = "hi",
        }));
        Assert.Equal(400, ex.Status);
        Assert.Equal("previous_response_id is not supported", ex.Message);
    }
}
