import { test } from "node:test";
import assert from "node:assert/strict";

import { FinishReason, LlmleafClient, Role } from "../src/index.js";
import type { FetchLike, ToolCallDelta } from "../src/index.js";

test("chatStream preserves split tool-call deltas", async () => {
  const frames = [
    {
      id: "c1",
      object: "chat.completion.chunk",
      created: 1,
      model: "m",
      choices: [{
        index: 0,
        delta: {
          tool_calls: [{
            index: 0,
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: '{"city":"Par' },
          }],
        },
      }],
    },
    {
      id: "c1",
      object: "chat.completion.chunk",
      created: 1,
      model: "m",
      choices: [{
        index: 0,
        delta: {
          tool_calls: [{ index: 0, function: { arguments: 'is"}' } }],
        },
      }],
    },
    {
      id: "c1",
      object: "chat.completion.chunk",
      created: 1,
      model: "m",
      choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    },
  ];
  const sse = frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("")
    + "data: [DONE]\n\n";

  let requestBody: Record<string, unknown> | undefined;
  const fetch: FetchLike = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body));
    return new Response(sse, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  };
  const client = new LlmleafClient({ baseUrl: "https://example.test", apiKey: "test", fetch });

  const deltas: ToolCallDelta[] = [];
  let finishReason: FinishReason | undefined;
  for await (const chunk of client.chatStream({
    model: "m",
    messages: [{ role: Role.USER, content: "weather?" }],
  })) {
    for (const choice of chunk.choices) {
      deltas.push(...(choice.delta.toolCalls ?? []));
      finishReason = choice.finishReason ?? finishReason;
    }
  }

  assert.equal(requestBody?.["stream"], true);
  assert.equal(deltas.length, 2);
  assert.deepEqual(deltas[0], {
    index: 0,
    id: "call_1",
    type: "function",
    function: { name: "get_weather", arguments: '{"city":"Par' },
  });
  assert.deepEqual(deltas[1], {
    index: 0,
    function: { arguments: 'is"}' },
  });
  assert.equal(deltas.map((d) => d.function?.arguments ?? "").join(""), '{"city":"Paris"}');
  assert.equal(finishReason, FinishReason.TOOL_CALLS);
});
