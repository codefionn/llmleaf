#!/usr/bin/env bash
# Regenerate the Rust typed model from the single proto source of truth.
#
#   clients/rust/scripts/gen.sh
#
# Unlike the other clients there is nothing to invoke by hand: codegen is wired into
# the crate's `build.rs`, which runs `prost-build` (and therefore `protoc`) on every
# build. So "regenerate" is just "build". This script makes that explicit and is what
# `make gen-rust` delegates to.
#
# Toolchain (must be on PATH):
#   - protoc  (libprotoc 35; prost-build shells out to it)
#   - cargo / rustc (stable; edition 2021)
#
# The generated messages land in `$OUT_DIR/llmleaf.v1.rs` and are `include!`d by
# `src/pb.rs`, exposed to consumers as `llmleaf_client::pb`.
set -euo pipefail

# Run from the rust client root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

if ! command -v protoc >/dev/null 2>&1; then
  echo "error: protoc not found on PATH (prost-build needs it to compile the proto)." >&2
  echo "       install libprotoc 35, e.g. your distro's 'protobuf' package." >&2
  exit 1
fi

# build.rs runs prost-build → protoc; a clean build forces codegen to re-run.
cargo build

echo "generated: \$OUT_DIR/llmleaf.v1.rs (included by src/pb.rs as llmleaf_client::pb)"
