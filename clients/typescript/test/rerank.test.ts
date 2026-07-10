// Rerank (POST /v1/rerank) (de)serialization + transport tests.
// Run with `npm test` (tsx + node:test).
//
// Focus: the string-or-object `documents` encoding (top_n / return_documents), and the
// decoded results (index / relevance_score, optional echoed document) + usage. Unlike
// embeddings, rerank results are plain JSON — there is NO base64 vector to decode.

import { test } from "node:test";
import assert from "node:assert/strict";

import { encodeRerankRequest, decodeRerankResponse } from "../src/wire.js";
import { LlmleafClient, ApiError } from "../src/index.js";
import type { RerankRequest, RerankResponse, FetchLike } from "../src/index.js";

test("request encodes documents (string + object) with top_n + return_documents", () => {
  const req: RerankRequest = {
    model: "rerank-1",
    query: "capital of France",
    documents: [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      { text: "The Eiffel Tower is in Paris.", image: "https://x/y.png" },
    ],
    topN: 2,
    returnDocuments: true,
  };

  const body = encodeRerankRequest(req);

  // Exact request body: documents (strings or structured objects) are spliced verbatim;
  // top_n / return_documents map to snake_case.
  assert.deepEqual(body, {
    model: "rerank-1",
    query: "capital of France",
    documents: [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      { text: "The Eiffel Tower is in Paris.", image: "https://x/y.png" },
    ],
    top_n: 2,
    return_documents: true,
  });
});

test("request omits absent optionals and merges extra at the top level", () => {
  const body = encodeRerankRequest({
    model: "rerank-1",
    query: "q",
    documents: ["a", "b"],
    extra: JSON.stringify({ truncation: "END" }),
  });

  assert.deepEqual(body, {
    model: "rerank-1",
    query: "q",
    documents: ["a", "b"],
    truncation: "END",
  });
  // top_n / return_documents were absent, so they must not appear on the wire.
  assert.ok(!("top_n" in body));
  assert.ok(!("return_documents" in body));
});

test("response decodes results (index / relevance_score, echoed document) + usage", () => {
  const wire = {
    object: "list",
    model: "rerank-1",
    results: [
      { index: 0, relevance_score: 0.98, document: "Paris is the capital of France." },
      { index: 2, relevance_score: 0.41, document: { text: "The Eiffel Tower is in Paris." } },
    ],
    usage: { total_tokens: 24, cost_usd: 0.0001 },
  };

  const resp = decodeRerankResponse(wire);
  assert.equal(resp.object, "list");
  assert.equal(resp.model, "rerank-1");

  assert.equal(resp.results.length, 2);
  const first = resp.results[0]!;
  assert.equal(first.index, 0);
  assert.equal(first.relevanceScore, 0.98);
  assert.equal(first.document, "Paris is the capital of France.");

  const second = resp.results[1]!;
  assert.equal(second.index, 2);
  assert.equal(second.relevanceScore, 0.41);
  assert.deepEqual(second.document, { text: "The Eiffel Tower is in Paris." });

  assert.ok(resp.usage);
  assert.equal(resp.usage!.totalTokens, 24);
  assert.equal(resp.usage!.costUsd, 0.0001);
});

test("response omits document when return_documents was not set", () => {
  const resp = decodeRerankResponse({
    object: "list",
    model: "rerank-1",
    results: [{ index: 1, relevance_score: 0.7 }],
    usage: { total_tokens: 10 },
  });
  assert.equal(resp.results[0]!.document, undefined);
});

test("rerank() POSTs /v1/rerank and returns decoded results + usage", async () => {
  let seenUrl = "";
  let seenBody: unknown;
  const fetchMock: FetchLike = async (url, init) => {
    seenUrl = String(url);
    seenBody = JSON.parse(String(init!.body));
    return new Response(
      JSON.stringify({
        object: "list",
        model: "rerank-1",
        results: [
          { index: 1, relevance_score: 0.91 },
          { index: 0, relevance_score: 0.32 },
        ],
        usage: { total_tokens: 18 },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };
  const client = new LlmleafClient({ baseUrl: "http://x", apiKey: "k", fetch: fetchMock });

  const resp: RerankResponse = await client.rerank({
    model: "rerank-1",
    query: "which is relevant?",
    documents: ["doc a", "doc b"],
    topN: 2,
  });

  assert.equal(seenUrl, "http://x/v1/rerank");
  assert.deepEqual(seenBody, {
    model: "rerank-1",
    query: "which is relevant?",
    documents: ["doc a", "doc b"],
    top_n: 2,
  });

  assert.equal(resp.object, "list");
  assert.equal(resp.results.length, 2);
  assert.equal(resp.results[0]!.index, 1);
  assert.equal(resp.results[0]!.relevanceScore, 0.91);
  assert.ok(resp.usage);
  assert.equal(resp.usage!.totalTokens, 18);
});

test("rerank surfaces a non-2xx error envelope as a typed ApiError", async () => {
  const fetchMock: FetchLike = async () =>
    new Response(JSON.stringify({ error: { message: "model not allowed" } }), {
      status: 403,
      headers: { "content-type": "application/json" },
    });
  const client = new LlmleafClient({ baseUrl: "http://x", apiKey: "k", fetch: fetchMock });

  await assert.rejects(
    () => client.rerank({ model: "m", query: "q", documents: ["a"] }),
    (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 403);
      assert.equal(err.message, "model not allowed");
      return true;
    },
  );
});
