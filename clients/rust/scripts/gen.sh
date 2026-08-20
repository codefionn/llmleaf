#!/usr/bin/env bash
# Regenerate the Rust typed model from the single proto source of truth.
#
#   clients/rust/scripts/gen.sh
#
# This is contributor-only code generation. Consumers build from the checked-in output and
# never need `protoc` or this plugin.
#
# Toolchain (must be on PATH):
#   - protoc
#   - protoc-gen-prost 0.5.0 (`cargo install protoc-gen-prost --version 0.5.0 --locked`)
#
# The generated messages land in `src/gen/llmleaf/v1/llmleaf.v1.rs` and are directly
# `include!`d by `src/pb.rs`, exposed to consumers as `llmleaf_client::pb`.
set -euo pipefail

# Run from the rust client root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

if ! command -v protoc >/dev/null 2>&1 || ! command -v protoc-gen-prost >/dev/null 2>&1; then
  echo "error: protoc and protoc-gen-prost 0.5.0 must be on PATH." >&2
  exit 1
fi

mkdir -p src/gen
protoc --prost_out=src/gen --prost_opt=compile_well_known_types \
  -I ../proto ../proto/llmleaf/v1/llmleaf.proto

echo "generated: src/gen/llmleaf/v1/llmleaf.v1.rs"
