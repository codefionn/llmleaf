import assert from "node:assert/strict";
import { test } from "node:test";

import { ApiError, LlmleafClient } from "../src/index.js";
import { decodeDecisionsResponse, encodeDecisionsRequest } from "../src/wire.js";
import type { DecisionsRequest, FetchLike } from "../src/index.js";

test("decisions request keeps structured JSON and protects canonical fields", () => {
  const request: DecisionsRequest = {
    model: "jev",
    state: { turn: 2, context: ["a", "b"] },
    questions: { route: { type: "choice", instructions: "Which route?", criteria: { left: null, right: null } } },
    extra: {
      metadata: { trace: true },
      model: "wrong",
      state: "wrong",
      questions: {},
    },
  };
  assert.deepEqual(encodeDecisionsRequest(request), {
    metadata: { trace: true },
    model: "jev",
    state: { turn: 2, context: ["a", "b"] },
    questions: { route: { type: "choice", instructions: "Which route?", criteria: { left: null, right: null } } },
  });
});

test("decisions response preserves answers, unknown metadata, and zero cost", () => {
  const response = decodeDecisionsResponse({
    id: "dec_1",
    model: "jev",
    provider: "openrouter",
    answers: { route: { type: "choice", choice: "left", probabilities: { left: 0.7, right: 0.3 } } },
    usage: { input_tokens: 12, output_tokens: 4, cost: 0, cached: { tokens: 8 } },
    trace_id: "tr_1",
  });
  assert.equal(response.id, "dec_1");
  assert.equal(response.provider, "openrouter");
  assert.deepEqual(response.answers, { route: { type: "choice", choice: "left", probabilities: { left: 0.7, right: 0.3 } } });
  assert.equal(response.usage.cost, 0);
  assert.deepEqual(response.usage.extra, { cached: { tokens: 8 } });
  assert.deepEqual(response.extra, { trace_id: "tr_1" });
});

test("decisions keeps __proto__ metadata as data", () => {
  const response = decodeDecisionsResponse(JSON.parse('{"model":"m","answers":{},"usage":{"input_tokens":0,"output_tokens":0,"__proto__":{"cached":true}},"__proto__":{"trace":true}}'));
  assert.deepEqual(response.usage.extra, JSON.parse('{"__proto__":{"cached":true}}'));
  assert.deepEqual(response.extra, JSON.parse('{"__proto__":{"trace":true}}'));
});

test("decisions posts auth and /v1/decisions", async () => {
  let url = "";
  let authorization: string | null = null;
  let body: unknown;
  const fetchMock: FetchLike = async (input, init) => {
    url = String(input);
    authorization = new Headers(init?.headers).get("authorization");
    body = JSON.parse(String(init?.body));
    return new Response(JSON.stringify({ model: "m", answers: {}, usage: { input_tokens: 1, output_tokens: 2, cost: 0 } }), { status: 200 });
  };
  const client = new LlmleafClient({ baseUrl: "http://x", apiKey: "sk-test", fetch: fetchMock });
  const response = await client.decisions({ model: "m", state: { active: true }, questions: { next: { type: "noul", instructions: "Should we continue?" } } });
  assert.equal(url, "http://x/v1/decisions");
  assert.equal(authorization, "Bearer sk-test");
  assert.deepEqual(body, { model: "m", state: { active: true }, questions: { next: { type: "noul", instructions: "Should we continue?" } } });
  assert.equal(response.usage.cost, 0);
});

test("decisions surfaces a non-2xx error envelope as ApiError", async () => {
  const fetchMock: FetchLike = async () => new Response(JSON.stringify({ error: { message: "model not allowed" } }), { status: 403 });
  const client = new LlmleafClient({ baseUrl: "http://x", apiKey: "k", fetch: fetchMock });
  await assert.rejects(
    () => client.decisions({ model: "m", state: {}, questions: {} }),
    (error: unknown) => error instanceof ApiError && error.status === 403 && error.message === "model not allowed",
  );
});
