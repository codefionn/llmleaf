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

  // 3. Non-streaming responses (OpenAI Responses dialect) — print the output text.
  console.log("\n== non-streaming responses ==");
  const resp = await client.responses({
    model,
    input: [
      { type: "message", role: "user", content: "Say hello in one short sentence." },
    ],
  });
  for (const item of resp.output) {
    if (item.type === "message" && Array.isArray(item.content)) {
      for (const part of item.content) {
        if (part.type === "output_text") process.stdout.write(part.text);
      }
    }
  }
  process.stdout.write("\n");
  if (resp.usage) {
    console.log(
      `tokens: input=${resp.usage.inputTokens} output=${resp.usage.outputTokens}`,
    );
  }

  // 4. Streaming responses — typed events, no `[DONE]`; accumulate output_text deltas.
  console.log("\n== streaming responses ==");
  for await (const event of client.responsesStream({
    model,
    input: "Count from 1 to 5.",
  })) {
    if (event.type === "response.output_text.delta") {
      process.stdout.write(event.delta ?? "");
    }
  }
  process.stdout.write("\n");

  // 5. List models.
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
