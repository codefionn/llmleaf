// Runnable example for @codefionn/llmleaf-client.
//
//   LLMLEAF_BASE_URL=https://gateway.example.com \
//   LLMLEAF_API_KEY=sk-... \
//   npx tsx examples/basic.ts
//
// Without a reachable gateway it will fail to connect — that's expected; it still
// type-checks and loads. Set LLMLEAF_MODEL to override the model (default gpt-4o-mini).

import { LlmleafClient, ApiError, Role } from "../src/index.js";

const baseUrl = process.env["LLMLEAF_BASE_URL"] ?? "http://localhost:8080";
const apiKey = process.env["LLMLEAF_API_KEY"] ?? "sk-no-key";
const model = process.env["LLMLEAF_MODEL"] ?? "gpt-4o-mini";

async function main(): Promise<void> {
  const client = new LlmleafClient({ baseUrl, apiKey, timeoutMs: 30_000 });

  // 1. Non-streaming chat — print the assembled text.
  console.log("== non-streaming chat ==");
  const res = await client.chat({
    model,
    messages: [{ role: Role.USER, content: "Say hello in one short sentence." }],
  });
  const first = res.choices[0];
  const text =
    first && typeof first.message.content === "string" ? first.message.content : "";
  console.log(text);
  if (res.usage) {
    console.log(
      `tokens: prompt=${res.usage.promptTokens} completion=${res.usage.completionTokens}` +
        (res.usage.costUsd !== undefined ? ` cost=$${res.usage.costUsd}` : ""),
    );
  }

  // 2. Streaming chat — for-await over the deltas, print them as they arrive.
  console.log("\n== streaming chat ==");
  process.stdout.write("");
  for await (const chunk of client.chatStream({
    model,
    messages: [{ role: Role.USER, content: "Count from 1 to 5." }],
  })) {
    const delta = chunk.choices[0]?.delta.content;
    if (delta) process.stdout.write(delta);
  }
  process.stdout.write("\n");

  // 3. List models.
  console.log("\n== models ==");
  const models = await client.listModels({ type: "llm" });
  for (const m of models.data.slice(0, 10)) {
    console.log(`- ${m.id}${m.contextLength ? ` (ctx ${m.contextLength})` : ""}`);
  }
}

main().catch((err: unknown) => {
  if (err instanceof ApiError) {
    console.error(`ApiError ${err.status}: ${err.message}`);
  } else {
    console.error(err);
  }
  process.exitCode = 1;
});
