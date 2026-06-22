// Wire (de)serialization round-trip tests. Run with `npm test` (tsx + node:test).
//
// Focus: the OpenRouter reasoning surface (reasoning / reasoning_details, including the
// opaque `signature` that must round-trip verbatim) and the prompt-cache usage metadata
// (prompt_tokens_details.cached_tokens, cache_creation_tokens).

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  encodeChatRequest,
  decodeChatResponse,
  decodeChatCompletionChunk,
  decodeUsage,
} from "../src/wire.js";
import { cachedTokens } from "../src/types.js";
import type { ChatRequest } from "../src/types.js";
import { Role } from "../src/enums.js";

test("request encodes reasoning + reasoning_details (signed, verbatim)", () => {
  const req: ChatRequest = {
    model: "some-model",
    messages: [
      {
        role: Role.ASSISTANT,
        content: "the answer is 42",
        reasoning: "let me think about this",
        reasoningDetails: [
          {
            type: "reasoning.text",
            text: "let me think about this",
            signature: "sig-abc-123==",
            id: "rd_0",
            format: "anthropic-claude-v1",
            index: 0,
          },
          {
            type: "reasoning.encrypted",
            data: "ENCRYPTED_BLOB_xyz==",
            index: 1,
          },
        ],
      },
    ],
  };

  const body = encodeChatRequest(req) as Record<string, unknown>;
  const messages = body["messages"] as Array<Record<string, unknown>>;
  const msg = messages[0]!;

  assert.equal(msg["reasoning"], "let me think about this");

  const details = msg["reasoning_details"] as Array<Record<string, unknown>>;
  assert.equal(details.length, 2);

  // Open, signed block: text + opaque signature must be present and verbatim.
  assert.deepEqual(details[0], {
    type: "reasoning.text",
    text: "let me think about this",
    signature: "sig-abc-123==",
    id: "rd_0",
    format: "anthropic-claude-v1",
    index: 0,
  });

  // Hidden block: opaque `data` blob, no `text`.
  assert.deepEqual(details[1], {
    type: "reasoning.encrypted",
    data: "ENCRYPTED_BLOB_xyz==",
    index: 1,
  });
});

test("response decodes reasoning_details with signature verbatim", () => {
  const wire = {
    id: "cmpl-1",
    object: "chat.completion",
    created: 1,
    model: "some-model",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: "the answer is 42",
          reasoning: "step by step",
          reasoning_details: [
            {
              type: "reasoning.text",
              text: "step by step",
              signature: "SIGNATURE_BLOB==",
              index: 0,
            },
            {
              type: "reasoning.encrypted",
              data: "REDACTED==",
              index: 1,
            },
          ],
        },
        finish_reason: "stop",
      },
    ],
  };

  const resp = decodeChatResponse(wire);
  const msg = resp.choices[0]!.message;

  assert.equal(msg.reasoning, "step by step");
  assert.ok(msg.reasoningDetails);
  assert.equal(msg.reasoningDetails!.length, 2);

  const open = msg.reasoningDetails![0]!;
  assert.equal(open.type, "reasoning.text");
  assert.equal(open.text, "step by step");
  // Opaque signature preserved exactly.
  assert.equal(open.signature, "SIGNATURE_BLOB==");
  assert.equal(open.index, 0);

  const hidden = msg.reasoningDetails![1]!;
  assert.equal(hidden.type, "reasoning.encrypted");
  assert.equal(hidden.data, "REDACTED==");
  assert.equal(hidden.text, undefined);
});

test("encode → decode round-trip preserves a signed reasoning block byte-for-byte", () => {
  const req: ChatRequest = {
    model: "m",
    messages: [
      {
        role: Role.ASSISTANT,
        content: "ok",
        reasoningDetails: [
          {
            type: "reasoning.text",
            text: "thought",
            signature: "OPAQUE_SIG_/+==",
          },
        ],
      },
    ],
  };

  const body = encodeChatRequest(req);
  // Feed the encoded message straight back through the response decoder.
  const messages = (body as Record<string, unknown>)["messages"] as unknown[];
  const decoded = decodeChatResponse({
    id: "x",
    object: "chat.completion",
    created: 0,
    model: "m",
    choices: [{ index: 0, message: messages[0], finish_reason: "stop" }],
  });

  const rd = decoded.choices[0]!.message.reasoningDetails![0]!;
  assert.equal(rd.type, "reasoning.text");
  assert.equal(rd.text, "thought");
  assert.equal(rd.signature, "OPAQUE_SIG_/+==");
});

test("usage decodes prompt_tokens_details.cached_tokens + cache_creation_tokens", () => {
  const usage = decodeUsage({
    prompt_tokens: 100,
    completion_tokens: 20,
    total_tokens: 120,
    prompt_tokens_details: { cached_tokens: 80 },
    cache_creation_tokens: 15,
  });

  assert.ok(usage);
  assert.equal(usage!.promptTokens, 100);
  assert.ok(usage!.promptTokensDetails);
  assert.equal(usage!.promptTokensDetails!.cachedTokens, 80);
  assert.equal(usage!.cacheCreationTokens, 15);

  // Ergonomic accessor flattens the nested detail to a plain count.
  assert.equal(cachedTokens(usage!), 80);
});

test("usage without cache metadata omits the fields and cachedTokens is 0", () => {
  const usage = decodeUsage({
    prompt_tokens: 10,
    completion_tokens: 5,
    total_tokens: 15,
  });

  assert.ok(usage);
  assert.equal(usage!.promptTokensDetails, undefined);
  assert.equal(usage!.cacheCreationTokens, undefined);
  assert.equal(cachedTokens(usage!), 0);
});

test("streaming delta decodes incremental reasoning + reasoning_details", () => {
  const chunk = decodeChatCompletionChunk({
    id: "chunk-1",
    object: "chat.completion.chunk",
    created: 1,
    model: "m",
    choices: [
      {
        index: 0,
        delta: {
          reasoning: "partial thought",
          reasoning_details: [
            { type: "reasoning.text", text: "partial thought", index: 0 },
          ],
        },
      },
    ],
  });

  const delta = chunk.choices[0]!.delta;
  assert.equal(delta.reasoning, "partial thought");
  assert.ok(delta.reasoningDetails);
  assert.equal(delta.reasoningDetails!.length, 1);
  assert.equal(delta.reasoningDetails![0]!.type, "reasoning.text");
  assert.equal(delta.reasoningDetails![0]!.text, "partial thought");
});
