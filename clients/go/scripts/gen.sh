#!/usr/bin/env bash
# Regenerate the Go typed model from the single proto source of truth.
#
#   clients/go/scripts/gen.sh
#
# Toolchain (must be on PATH):
#   - protoc        (libprotoc 35; the schema compiler)
#   - protoc-gen-go (v1.36+; `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`)
#
# This emits clients/go/llmleafpb/llmleaf.pb.go. The module-rooted `--go_opt=module=...`
# strips the proto's go_package import-path prefix so the file lands under llmleafpb/.
set -euo pipefail

# Run from the go client root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

if ! command -v protoc-gen-go >/dev/null 2>&1; then
  echo "error: protoc-gen-go not found on PATH." >&2
  echo "       install it with: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest" >&2
  echo "       (and ensure \$(go env GOPATH)/bin is on PATH)" >&2
  exit 1
fi

protoc \
  --proto_path=../proto \
  --go_out=. \
  --go_opt=module=github.com/codefionn/llmleaf/clients/go \
  ../proto/llmleaf/v1/llmleaf.proto

echo "generated: llmleafpb/llmleaf.pb.go"
