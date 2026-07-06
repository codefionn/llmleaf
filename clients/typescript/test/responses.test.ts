// Responses dialect (POST /v1/responses) (de)serialization + transport tests.
// Run with `npm test` (tsx + node:test).
//
// Focus: the flat tools / flat tool_choice shape, the item-array input (incl.
// function_call / function_call_output / reasoning replay), the list-decides-the-token
// reasoning encoding, the stateless `store:false` echo + cached-tokens usage, and the
// typed streaming events (no `[DONE]` sentinel, unknown types skipped, terminal stop).

import { test } from "node:test";
import assert from "node:assert/strict";

import { encodeResponsesRequest, decodeResponsesResponse } from "../src/wire.js";
import { LlmleafClient, ApiError } from "../src/index.js";
import type {
  ResponsesRequest,
  ResponseContentPart,
  ResponsesStreamEvent,
  FetchLike,
} from "../src/index.js";

test("request encodes an item array with flat tools + reasoning replay", () => {
  const req: ResponsesRequest = {
    model: "gpt-4o-mini",
    input: [
      { type: "message", role: "user", content: "What's the weather in Paris?" },
      {
        type: "function_call",
        callId: "call_1",
        name: "get_weather",
        arguments: '{"city":"Paris"}',
      },
      { type: "function_call_output", callId: "call_1", output: "sunny, 21C" },
      {
        type: "reasoning",
        id: "rs_1",
        summary: [{ text: "user wants weather" }],
        content: [{ text: "call get_weather" }],
        encryptedContent: "ENC==",
      },
    ],
    tools: [
      {
        type: "function",
        name: "get_weather",
        description: "Get the weather for a city",
        parameters: JSON.stringify({
          type: "object",
          properties: { city: { type: "string" } },
        }),
      },
    ],
    toolChoice: { type: "function", name: "get_weather" },
    temperature: 0.7,
    store: false,
  };

  const body = encodeResponsesRequest(req, false);

  // Exact request body: input serialises as an item array — message items are bare
  // role-keyed objects (NO "type"), typed items carry "type"; tools/tool_choice are flat;
  // reasoning summary/content entries take their token from the list they live in.
  assert.deepEqual(body, {
    model: "gpt-4o-mini",
    input: [
      { role: "user", content: "What's the weather in Paris?" },
      {
        type: "function_call",
        call_id: "call_1",
        name: "get_weather",
        arguments: '{"city":"Paris"}',
      },
      { type: "function_call_output", call_id: "call_1", output: "sunny, 21C" },
      {
        type: "reasoning",
        id: "rs_1",
        summary: [{ type: "summary_text", text: "user wants weather" }],
        content: [{ type: "reasoning_text", text: "call get_weather" }],
        encrypted_content: "ENC==",
      },
    ],
    stream: false,
    temperature: 0.7,
    tools: [
      {
        type: "function",
        name: "get_weather",
        description: "Get the weather for a city",
        parameters: { type: "object", properties: { city: { type: "string" } } },
      },
    ],
    tool_choice: { type: "function", name: "get_weather" },
    store: false,
  });
});

test("request encodes a bare-string input + an input_image part", () => {
  const bare = encodeResponsesRequest({ model: "m", input: "hi" }, false);
  assert.equal(bare["input"], "hi");

  const withImage = encodeResponsesRequest(
    {
      model: "m",
      input: [
        {
          type: "message",
          role: "user",
          content: [
            { type: "input_text", text: "describe this" },
            { type: "input_image", imageUrl: "https://x/y.png", detail: "high" },
          ],
        },
      ],
    },
    false,
  );
  const items = withImage["input"] as Array<Record<string, unknown>>;
  const parts = items[0]!["content"] as Array<Record<string, unknown>>;
  // input_image.image_url is a plain STRING, not the chat dialect's nested {url} object.
  assert.deepEqual(parts[1], {
    type: "input_image",
    image_url: "https://x/y.png",
    detail: "high",
  });
});

test("response decodes output + store:false + cached-tokens usage", () => {
  const wire = {
    id: "resp_1",
    object: "response",
    created_at: 1234567890,
    status: "completed",
    model: "gpt-4o-mini",
    output: [
      {
        type: "message",
        id: "msg_1",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "It's sunny in Paris.", annotations: [] }],
      },
    ],
    usage: {
      input_tokens: 40,
      input_tokens_details: { cached_tokens: 24 },
      output_tokens: 8,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 48,
    },
    store: false,
  };

  const resp = decodeResponsesResponse(wire);
  assert.equal(resp.id, "resp_1");
  assert.equal(resp.object, "response");
  assert.equal(resp.status, "completed");
  // llmleaf is stateless — the store echo is always false.
  assert.equal(resp.store, false);

  assert.equal(resp.output.length, 1);
  const item = resp.output[0]!;
  assert.equal(item.type, "message");
  if (item.type === "message") {
    assert.equal(item.role, "assistant");
    assert.equal(item.status, "completed");
    const parts = item.content as ResponseContentPart[];
    const first = parts[0]!;
    assert.equal(first.type, "output_text");
    if (first.type === "output_text") assert.equal(first.text, "It's sunny in Paris.");
  }

  assert.ok(resp.usage);
  assert.equal(resp.usage!.inputTokens, 40);
  assert.equal(resp.usage!.inputTokensDetails!.cachedTokens, 24);
  assert.equal(resp.usage!.outputTokens, 8);
  assert.equal(resp.usage!.totalTokens, 48);
});

// --- Streaming ------------------------------------------------------------

function frame(type: string, payload: Record<string, unknown>): string {
  // Typed SSE frame: an `event:` line (redundant) + a self-describing `data:` line.
  return `event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function sseResponse(body: string): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

test("responsesStream yields typed events, accumulates text, stops on terminal, skips unknown", async () => {
  const sse =
    frame("response.created", {
      type: "response.created",
      sequence_number: 0,
      response: {
        id: "resp_1",
        object: "response",
        created_at: 1,
        status: "in_progress",
        model: "m",
        output: [],
      },
    }) +
    frame("response.output_item.added", {
      type: "response.output_item.added",
      sequence_number: 1,
      output_index: 0,
      item: { type: "function_call", id: "fc_1", call_id: "call_1", name: "get_weather", arguments: "" },
    }) +
    frame("response.function_call_arguments.delta", {
      type: "response.function_call_arguments.delta",
      sequence_number: 2,
      item_id: "fc_1",
      output_index: 0,
      delta: '{"city":"Paris"}',
    }) +
    // Unknown event type — the SDK must skip it (the dialect grows by adding types).
    frame("response.some_future_event", {
      type: "response.some_future_event",
      sequence_number: 3,
      whatever: true,
    }) +
    frame("response.output_text.delta", {
      type: "response.output_text.delta",
      sequence_number: 4,
      item_id: "msg_1",
      output_index: 1,
      content_index: 0,
      delta: "Hello",
    }) +
    frame("response.output_text.delta", {
      type: "response.output_text.delta",
      sequence_number: 5,
      item_id: "msg_1",
      output_index: 1,
      content_index: 0,
      delta: ", world",
    }) +
    frame("response.completed", {
      type: "response.completed",
      sequence_number: 6,
      response: {
        id: "resp_1",
        object: "response",
        created_at: 1,
        status: "completed",
        model: "m",
        output: [
          {
            type: "message",
            id: "msg_1",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "Hello, world", annotations: [] }],
          },
        ],
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
      },
    }) +
    // Anything after the terminal event must never be surfaced.
    frame("response.output_text.delta", {
      type: "response.output_text.delta",
      sequence_number: 7,
      delta: "SHOULD NOT APPEAR",
    });

  const fetchMock: FetchLike = async () => sseResponse(sse);
  const client = new LlmleafClient({ baseUrl: "http://x", apiKey: "k", fetch: fetchMock });

  const events: ResponsesStreamEvent[] = [];
  let text = "";
  for await (const ev of client.responsesStream({ model: "m", input: "weather?" })) {
    events.push(ev);
    if (ev.type === "response.output_text.delta") text += ev.delta ?? "";
  }

  assert.deepEqual(
    events.map((e) => e.type),
    [
      "response.created",
      "response.output_item.added",
      "response.function_call_arguments.delta",
      "response.output_text.delta",
      "response.output_text.delta",
      "response.completed",
    ],
  );

  // The unknown event type was skipped, and nothing after the terminal event leaked.
  assert.ok(!events.some((e) => e.type === "response.some_future_event"));
  assert.ok(!events.some((e) => e.delta === "SHOULD NOT APPEAR"));

  // Accumulated output text and terminal usage from the completed snapshot.
  assert.equal(text, "Hello, world");
  const terminal = events[events.length - 1]!;
  assert.equal(terminal.type, "response.completed");
  assert.equal(terminal.response!.status, "completed");
  assert.equal(terminal.response!.usage!.inputTokens, 10);
  assert.equal(terminal.response!.usage!.totalTokens, 15);

  // A typed item survives the round trip through the added event.
  const added = events[1]!;
  assert.equal(added.item!.type, "function_call");
});

test("responses surfaces a non-2xx error envelope as a typed ApiError", async () => {
  const fetchMock: FetchLike = async () =>
    new Response(JSON.stringify({ error: { message: "model not allowed" } }), {
      status: 403,
      headers: { "content-type": "application/json" },
    });
  const client = new LlmleafClient({ baseUrl: "http://x", apiKey: "k", fetch: fetchMock });

  await assert.rejects(
    () => client.responses({ model: "m", input: "hi" }),
    (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 403);
      assert.equal(err.message, "model not allowed");
      return true;
    },
  );
});
