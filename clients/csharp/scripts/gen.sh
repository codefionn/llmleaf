#!/usr/bin/env bash
# Regenerate the C# typed model from the single proto source of truth.
#
#   clients/csharp/scripts/gen.sh
#
# Toolchain (must be on PATH):
#   - protoc   (libprotoc 35; the schema compiler, with the native --csharp_out backend)
#
# This emits clients/csharp/src/Gen/Llmleaf.cs — the Google.Protobuf typed model that
# proves the schema. It is committed. The SDK's public types + HTTP transport are written
# by hand in System.Text.Json (see src/Models.cs, src/Wire/*.cs): the generated
# Google.Protobuf JSON formatter emits proto canonical JSON (camelCase keys, enum value
# names) which does NOT match the OpenAI/OpenRouter wire (snake_case keys, lowercase enum
# tokens), so it is never used for serialisation — only as the codegen artifact and proof.
set -euo pipefail

# Run from the csharp client root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

if ! command -v protoc >/dev/null 2>&1; then
  echo "error: protoc not found on PATH (need libprotoc with --csharp_out)." >&2
  exit 1
fi

protoc \
  --proto_path=../proto \
  --csharp_out=src/Gen \
  ../proto/llmleaf/v1/llmleaf.proto

echo "generated: src/Gen/Llmleaf.cs"
