#!/usr/bin/env bash
# Regenerate the TypeScript typed model from the single proto source of truth.
#
#   clients/typescript/scripts/gen.sh   (or: npm run gen)
#
# Toolchain:
#   - protoc          (libprotoc 35; the schema compiler, must be on PATH)
#   - protoc-gen-es   (the protobuf-es plugin; provided by the npm devDependency
#                      @bufbuild/protoc-gen-es — run `npm install` first so it lands
#                      in node_modules/.bin)
#
# This emits clients/typescript/src/gen/llmleaf/v1/llmleaf_pb.ts — the protobuf-es
# codegen artifact (committed). The hand-written transport in src/ maps those typed
# messages to / from the OpenAI/OpenRouter-shaped JSON the llmleaf core speaks.
set -euo pipefail

# Run from the typescript client root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

PLUGIN="./node_modules/.bin/protoc-gen-es"

if [ ! -x "$PLUGIN" ]; then
  echo "error: $PLUGIN not found." >&2
  echo "       run 'npm install' first to fetch @bufbuild/protoc-gen-es." >&2
  exit 1
fi

if ! command -v protoc >/dev/null 2>&1; then
  echo "error: protoc not found on PATH (need libprotoc 35)." >&2
  exit 1
fi

mkdir -p src/gen

protoc \
  --plugin=protoc-gen-es="$PLUGIN" \
  --es_out=src/gen \
  --es_opt=target=ts \
  --proto_path=../proto \
  ../proto/llmleaf/v1/llmleaf.proto

echo "generated: src/gen/llmleaf/v1/llmleaf_pb.ts"
