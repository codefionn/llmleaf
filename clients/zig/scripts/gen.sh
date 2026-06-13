#!/usr/bin/env bash
# Regenerate the Zig typed model from the single proto source of truth.
#
#   clients/zig/scripts/gen.sh
#
# IMPORTANT — read this before running:
#
# There is NO official protoc plugin for Zig. The committed file
#   clients/zig/src/gen/llmleaf.zig
# is the AUTHORITATIVE hand-maintained mirror of
#   clients/proto/llmleaf/v1/llmleaf.proto
# and is what `zig build` compiles. It is intentionally shaped for the
# OpenAI/OpenRouter JSON wire (see ../SPEC.md): snake_case fields matching the
# JSON keys, enums whose Zig identifiers equal their lowercase wire tokens, and
# free-form JSON fields carried as raw `[]const u8` strings.
#
# This script documents + attempts regeneration via the THIRD-PARTY
# `protoc-gen-zig` plugin (e.g. github.com/Arwalk/zig-protobuf) IF a contributor
# vendors one. The third-party output is a *starting point*: it emits protobuf-
# binary-oriented types, not the JSON-wire mirror this SDK needs, so the result
# must be reconciled by hand into src/gen/llmleaf.zig. Treat any generated file
# as a diff aid, never a drop-in replacement.
#
# Toolchain (only needed if you actually want to (re)generate):
#   - protoc                 (libprotoc; the schema compiler)
#   - protoc-gen-zig         (third-party; vendor + put on PATH yourself)
#
# Without those tools this script is a no-op that just prints guidance — the
# committed mirror already builds.
set -euo pipefail

# Run from the zig client root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

PROTO_DIR="../proto"
PROTO_FILE="$PROTO_DIR/llmleaf/v1/llmleaf.proto"
GEN_DIR="src/gen"
GEN_OUT="$GEN_DIR/llmleaf.generated.zig" # scratch output, never committed as-is

if ! command -v protoc >/dev/null 2>&1; then
  echo "note: protoc not found — nothing to regenerate."
  echo "      The committed mirror src/gen/llmleaf.zig is authoritative and already builds."
  exit 0
fi

if ! command -v protoc-gen-zig >/dev/null 2>&1; then
  echo "note: protoc-gen-zig (third-party) not found on PATH — skipping codegen."
  echo "      Vendor one (e.g. github.com/Arwalk/zig-protobuf), put it on PATH, and re-run."
  echo "      The committed mirror src/gen/llmleaf.zig is authoritative and already builds."
  exit 0
fi

echo "running third-party protoc-gen-zig (output is a diff aid, NOT a drop-in)..."
mkdir -p "$GEN_DIR"
protoc \
  --proto_path="$PROTO_DIR" \
  --zig_out="$GEN_DIR" \
  "$PROTO_FILE"

echo
echo "generated scratch types under $GEN_DIR/ (plugin-specific filename)."
echo "RECONCILE them by hand into src/gen/llmleaf.zig — the third-party plugin emits"
echo "protobuf-binary types, but this SDK serialises to the OpenAI/OpenRouter JSON wire."
echo "Do not commit raw generated output as src/gen/llmleaf.zig."
