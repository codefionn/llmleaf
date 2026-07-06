// Runnable example for the llmleaf C# client: non-streaming chat, streaming chat, and listing
// models. Reads the gateway base URL + API key from the environment:
//
//   LLMLEAF_BASE_URL   e.g. https://gateway.example.com
//   LLMLEAF_API_KEY    your virtual API key
//   LLMLEAF_MODEL      (optional) model id; defaults to "gpt-4o-mini"
//
// Run it with:  dotnet run --project examples/Basic

using Llmleaf.Client;

var baseUrl = Environment.GetEnvironmentVariable("LLMLEAF_BASE_URL");
var apiKey = Environment.GetEnvironmentVariable("LLMLEAF_API_KEY");
var model = Environment.GetEnvironmentVariable("LLMLEAF_MODEL") ?? "gpt-4o-mini";

if (string.IsNullOrEmpty(baseUrl) || string.IsNullOrEmpty(apiKey))
{
    Console.Error.WriteLine("Set LLMLEAF_BASE_URL and LLMLEAF_API_KEY in the environment.");
    return 1;
}

// Ctrl-C cancels in-flight calls cleanly.
using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cts.Cancel();
};

using var client = new LlmleafClient(baseUrl, apiKey, new LlmleafClientOptions
{
    Timeout = TimeSpan.FromSeconds(60),
});

try
{
    // 1) Non-streaming chat: print the assembled text.
    Console.WriteLine("== non-streaming chat ==");
    var chat = await client.CreateChatCompletionAsync(new ChatRequest
    {
        Model = model,
        Messages = [ChatMessage.Text(Role.User, "In one sentence, what is an LLM proxy?")],
    }, cts.Token);

    var first = chat.Choices.Count > 0 ? chat.Choices[0] : null;
    Console.WriteLine(first?.Message.Content?.Text ?? "(no content)");
    if (chat.Usage is { } u)
    {
        var cost = u.CostUsd is { } c ? $", ${c:0.######}" : "";
        Console.WriteLine($"[tokens: prompt={u.PromptTokens} completion={u.CompletionTokens} total={u.TotalTokens}{cost}]");
    }

    // 2) Streaming chat: await foreach over the deltas, print as they arrive.
    Console.WriteLine();
    Console.WriteLine("== streaming chat ==");
    await foreach (var chunk in client.CreateChatCompletionStreamAsync(new ChatRequest
    {
        Model = model,
        Messages = [ChatMessage.Text(Role.User, "Count from one to five, words only.")],
    }, cts.Token))
    {
        var delta = chunk.Choices.Count > 0 ? chunk.Choices[0].Delta.Content : null;
        if (!string.IsNullOrEmpty(delta))
        {
            Console.Write(delta);
        }
    }
    Console.WriteLine();

    // 3) Non-streaming responses (the OpenAI Responses dialect): print the assembled output text.
    Console.WriteLine();
    Console.WriteLine("== non-streaming responses ==");
    var response = await client.CreateResponseAsync(new ResponsesRequest
    {
        Model = model,
        Input = "In one sentence, what is an LLM proxy?",
    }, cts.Token);

    Console.WriteLine(ResponsesText(response) is { Length: > 0 } outText ? outText : "(no content)");
    if (response.Usage is { } ru)
    {
        Console.WriteLine($"[tokens: input={ru.InputTokens} output={ru.OutputTokens} total={ru.TotalTokens}]");
    }

    // 4) Streaming responses: accumulate the typed response.output_text.delta events (no [DONE] sentinel).
    Console.WriteLine();
    Console.WriteLine("== streaming responses ==");
    await foreach (var evt in client.CreateResponseStreamAsync(new ResponsesRequest
    {
        Model = model,
        Input = "Count from one to five, words only.",
    }, cts.Token))
    {
        if (evt.Type == "response.output_text.delta" && evt.Delta is { } delta)
        {
            Console.Write(delta);
        }
    }
    Console.WriteLine();

    // 5) List models: print the first few ids.
    Console.WriteLine();
    Console.WriteLine("== models ==");
    var models = await client.ListModelsAsync(new ListModelsOptions { Type = ModelType.Llm }, cts.Token);
    foreach (var m in models.Data.Take(10))
    {
        Console.WriteLine($"- {m.Id}");
    }
    Console.WriteLine($"({models.Data.Count} models total)");

    return 0;
}
catch (ApiException ex)
{
    Console.Error.WriteLine($"API error (HTTP {ex.Status}): {ex.Message}");
    return 1;
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("canceled");
    return 130;
}

// Assemble the visible text from a Responses result: every output_text part of every message item.
static string ResponsesText(ResponsesResponse response)
{
    var sb = new System.Text.StringBuilder();
    foreach (var item in response.Output)
    {
        if (item is ResponseMessageItem { Content.Parts: { } parts })
        {
            foreach (var part in parts)
            {
                if (part is OutputTextPart text)
                {
                    sb.Append(text.Text);
                }
            }
        }
    }
    return sb.ToString();
}
