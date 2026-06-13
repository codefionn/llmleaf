#!/usr/bin/env bash
#
# e2e probe suite. Runs the llmleaf capability probe against every provider whose API
# key is present in the environment (from the compose `.env`), exercising:
#   - chat + tool calls  (tool-capable vendors)
#   - chat only          (vendors without a tool-calling surface, e.g. Perplexity)
#   - text-to-speech     (vendors with /audio/speech)
#   - speech-to-text     (vendors with /audio/transcriptions)
#
# STT is always run together with a TTS in the SAME probe invocation so the probe
# feeds real synthesized audio into the transcription (a lone STT run would upload a
# non-audio stub and fail). Each audio vendor runs a native TTS->STT round-trip on its
# own models; a vendor's TTS voices are its own catalog, not OpenAI's "alloy".
#
# Exit code = number of failed probe runs (0 = all green), so it drops into CI via
#   docker compose up --build --abort-on-container-exit --exit-code-from probe

set -uo pipefail

PROBE="${PROBE_BIN:-probe}"
HOST="${LLMLEAF_HOST:-llmleaf}"
PORT="${LLMLEAF_PORT:-8080}"

# --- wait for the server to accept connections (distroless server has no shell to
#     run a healthcheck in, so we poll the TCP port from here) ---
echo "waiting for ${HOST}:${PORT} ..."
ready=0
for _ in $(seq 1 60); do
  if (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null; then
    exec 3>&- 3<&-
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "!! ${HOST}:${PORT} never came up" >&2
  exit 1
fi
echo "server is up"

fails=0

# nonempty env var?  have OPENAI_API_KEY
have() { [ -n "${!1:-}" ]; }

# run <description> <probe args...>
run() {
  local desc="$1"; shift
  echo
  echo "============================================================"
  echo "## ${desc}"
  echo "------------------------------------------------------------"
  if "$PROBE" "$@"; then
    echo ">> OK: ${desc}"
  else
    fails=$((fails + 1))
    echo "!! FAILED: ${desc}"
  fi
}

# --------------------------------------------------------------------------------
# chat + tool calls
# --------------------------------------------------------------------------------
chat_tools=()
have OPENAI_API_KEY     && chat_tools+=("openai-chat")
have ANTHROPIC_API_KEY  && chat_tools+=("anthropic-chat")
have GEMINI_API_KEY     && chat_tools+=("gemini-chat")
have COHERE_API_KEY     && chat_tools+=("cohere-chat")
have GROQ_API_KEY       && chat_tools+=("groq-chat")
have OPENROUTER_API_KEY && chat_tools+=("openrouter-chat")
have REQUESTY_API_KEY   && chat_tools+=("requesty-chat")
have DEEPSEEK_API_KEY   && chat_tools+=("deepseek-chat")
have XAI_API_KEY        && chat_tools+=("xai-chat")
have MISTRAL_API_KEY    && chat_tools+=("mistral-chat")
have TOGETHER_API_KEY   && chat_tools+=("together-chat")
have FIREWORKS_API_KEY  && chat_tools+=("fireworks-chat")
have CEREBRAS_API_KEY   && chat_tools+=("cerebras-chat")
have ZAI_API_KEY        && chat_tools+=("zai-chat")
have MOONSHOT_API_KEY   && chat_tools+=("moonshot-chat")
if [ ${#chat_tools[@]} -gt 0 ]; then
  run "chat + tool calls" "${chat_tools[@]}" --only chat,tools
fi

# --------------------------------------------------------------------------------
# chat only (no tool-calling surface)
# --------------------------------------------------------------------------------
chat_only=()
have PERPLEXITY_API_KEY && chat_only+=("perplexity-chat")
if [ ${#chat_only[@]} -gt 0 ]; then
  run "chat only" "${chat_only[@]}" --only chat
fi

# --------------------------------------------------------------------------------
# audio: tts + stt round-trips (real synthesized audio fed into transcription)
# --------------------------------------------------------------------------------
have OPENAI_API_KEY && run "openai audio (tts -> stt)" \
  --only tts,stt --tts-model openai-tts --stt-model openai-stt --voice alloy

# Groq TTS is Orpheus — needs an Orpheus voice (not OpenAI's "alloy") and only emits wav (not mp3).
have GROQ_API_KEY && run "groq audio (tts -> stt)" \
  --only tts,stt --tts-model groq-tts --stt-model groq-stt --voice "autumn" --tts-format wav

# Mistral has both TTS and STT (Voxtral) — a native round-trip on its own models. Voxtral
# TTS voice ids are account-scoped UUIDs (listed by the account's GET /audio/voices), not
# OpenAI's "alloy"; override LLMLEAF_MISTRAL_VOICE with one your own key lists.
have MISTRAL_API_KEY && run "mistral audio (tts -> stt)" \
  --only tts,stt --tts-model mistral-tts --stt-model mistral-stt \
  --voice "${LLMLEAF_MISTRAL_VOICE:-c69964a6-ab8b-4f8a-9465-ec0925096ec8}"

# --------------------------------------------------------------------------------
# nothing configured?
# --------------------------------------------------------------------------------
if [ ${#chat_tools[@]} -eq 0 ] && [ ${#chat_only[@]} -eq 0 ] \
   && ! have OPENAI_API_KEY && ! have GROQ_API_KEY; then
  echo
  echo "No provider API keys set in the environment. Running the offline echo smoke route."
  echo "(chat/tts/stt PASS, tools FAIL — echo never emits a tool call.)"
  run "offline echo smoke" demo --only chat,tts,stt
fi

echo
echo "============================================================"
if [ "$fails" -eq 0 ]; then
  echo "ALL GREEN"
else
  echo "${fails} probe run(s) FAILED"
fi
exit "$fails"
