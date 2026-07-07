# e2e-tests

A **self-contained** stack that stands up the **llmleaf** proxy and runs the capability
**probe** against real providers — chat, tool calls, text-to-speech, and speech-to-text —
using a cheap pre-configured model per provider.

Everything runs on a dedicated, isolated bridge network (`llmleaf-e2e`). The probe reaches
the proxy only by its name on that network; **nothing is published to the host**. The
network is a normal bridge (not `internal`), so the proxy can still reach the upstream
provider APIs over the internet.

## Run it

```sh
cd e2e-tests
cp .env.example .env        # fill in the provider API keys you have (the rest are skipped)
./run.sh
```

`run.sh` works with **docker or podman**: it uses a Compose plugin if you have one
(`docker compose` / `podman compose` / `podman-compose` / `docker-compose`), and otherwise
drives the two containers directly on the `llmleaf-e2e` network — same images, same result.

With the docker compose plugin you can also run it directly:

```sh
docker compose up --build --abort-on-container-exit --exit-code-from probe
```

The `probe` container's exit code is the result: **0 = all green**, **N = N failed probe
runs**. Both images are built from the repo's multi-stage `Dockerfile` (targets `runtime`
and `probe-runtime`), so they share the build cache with the main project.

## What it probes

The suite (`probe-suite.sh`) only probes a provider when its key is set in `.env`:

| Group            | Providers (key-gated)                                                                 | Capabilities   |
| ---------------- | ------------------------------------------------------------------------------------ | -------------- |
| chat + tools     | openai, anthropic, gemini, cohere, groq, openrouter, requesty, deepseek, xai, mistral, together, fireworks, cerebras, zai, moonshot, minimax, zai-coding, kimi-coding | `chat`, `tools` |
| chat only        | perplexity                                                                            | `chat`         |
| audio (tts→stt)  | openai, groq, mistral                                                                 | `tts`, `stt`   |

STT always runs paired with a TTS in the same probe invocation, so the transcription gets
**real synthesized audio** (a lone STT run would upload a non-audio stub and fail).

With **no keys at all**, the suite falls back to the offline `echo` route as a smoke check
(chat/tts/stt PASS, tools FAIL — `echo` never emits a tool call).

## Files

| File                 | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `run.sh`             | portable launcher (docker/podman, compose or raw) on the isolated network |
| `docker-compose.yml` | the two-service stack (`llmleaf` + `probe`) on the `llmleaf-e2e` network |
| `llmleaf.e2e.toml`   | proxy config: every provider kind + one cheap model per (provider, modality) |
| `probe-suite.sh`     | the key-gated probe runner (mounted into the probe container)           |
| `.env.example`       | tracked template of every provider variable                            |
| `.env`               | your real keys — **gitignored**                                          |

## Tuning

- **Model names** live in `llmleaf.e2e.toml` (one place per vendor) — bump them if a vendor
  renames or retires a cheap model.
- **Local backends** (Ollama, LM Studio): first-class native providers — uncomment the provider +
  a route in `llmleaf.e2e.toml` and point `endpoint` at your runtime (the bare host for Ollama's
  `/api/*`, the `/api/v0` base for LM Studio — not the OpenAI-compat `/v1` shims).
- **Manual poking**: the stack is self-contained — the proxy isn't on the host by default.
  To reach it from your machine, uncomment the `ports:` block in `docker-compose.yml` and
  run `docker compose up`, then authenticate with the published local-only key:
  `Authorization: Bearer ZXhhbXBsZS1jbGk6bGxtbGVhZi1kZXY=` (`base64("example-cli:llmleaf-dev")`).
  Or attach an ephemeral container to the network: `docker run --rm -it --network llmleaf-e2e curlimages/curl ...`.
