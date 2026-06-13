#!/usr/bin/env bash
#
# Portable launcher for the self-contained e2e stack. Runs the proxy + probe on a
# dedicated, isolated bridge network (`llmleaf-e2e`) with nothing published to the host.
#
# Prefers a Compose implementation (docker compose / podman compose / podman-compose /
# docker-compose). If none is installed it falls back to driving the containers directly
# with docker or podman — same images, same network, same result.
#
# Exit code = the probe suite's: 0 = all green, N = N failed probe runs.
#
# Usage:
#   ./run.sh                 # build + run the whole stack, then exit with the probe's code
#   ENGINE=podman ./run.sh   # force a container engine for the raw fallback

set -euo pipefail
cd "$(dirname "$0")"

NET="llmleaf-e2e"
SRV="llmleaf-e2e-srv"
PRB="llmleaf-e2e-probe"
SRV_IMAGE="llmleaf:e2e"
PRB_IMAGE="llmleaf-probe:e2e"

# Keep in-network hops (probe -> proxy, loopback) off any host proxy the engine forwards
# into the container — podman/docker pass HTTP_PROXY & co. through by default, and a proxy
# can't reach the internal hostname, so it 502s. Provider egress still honours a real proxy.
NOPROXY="llmleaf,${SRV},localhost,127.0.0.1,::1"

if [ ! -f .env ]; then
  echo "no .env found — copy the template first:  cp .env.example .env" >&2
  exit 1
fi

# ---------------------------------------------------------------------------------
# Preferred path: a real Compose implementation drives docker-compose.yml (which
# already declares the isolated `llmleaf-e2e` network).
# ---------------------------------------------------------------------------------
compose=""
if docker compose version >/dev/null 2>&1; then
  compose="docker compose"
elif podman compose version >/dev/null 2>&1; then
  compose="podman compose"
elif command -v podman-compose >/dev/null 2>&1; then
  compose="podman-compose"
elif command -v docker-compose >/dev/null 2>&1; then
  compose="docker-compose"
fi

if [ -n "$compose" ]; then
  echo ">> using: $compose"
  exec $compose up --build --abort-on-container-exit --exit-code-from probe "$@"
fi

# ---------------------------------------------------------------------------------
# Fallback: no Compose plugin — drive the containers directly on the named network.
# ---------------------------------------------------------------------------------
ENGINE="${ENGINE:-}"
if [ -z "$ENGINE" ]; then
  if command -v docker >/dev/null 2>&1; then ENGINE="docker"
  elif command -v podman >/dev/null 2>&1; then ENGINE="podman"
  else echo "no docker/podman and no compose implementation found" >&2; exit 1
  fi
fi
echo ">> no compose plugin; using raw $ENGINE on network '$NET'"

# podman wants an SELinux label on bind mounts; docker ignores it where not needed.
if [ "$ENGINE" = "podman" ]; then MNT=":ro,z"; else MNT=":ro"; fi

cleanup() {
  "$ENGINE" rm -f "$SRV" "$PRB" >/dev/null 2>&1 || true
  "$ENGINE" network rm "$NET"   >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo ">> building images"
"$ENGINE" build --target runtime       -t "$SRV_IMAGE" -f ../Dockerfile ..
"$ENGINE" build --target probe-runtime -t "$PRB_IMAGE" -f ../Dockerfile ..

cleanup
"$ENGINE" network create "$NET" >/dev/null

echo ">> starting proxy ($SRV)"
"$ENGINE" run -d --name "$SRV" --network "$NET" \
  --env-file .env \
  -e LLMLEAF_CONFIG=/etc/llmleaf/llmleaf.toml \
  -e "RUST_LOG=${RUST_LOG:-info}" \
  -e NO_PROXY="$NOPROXY" -e no_proxy="$NOPROXY" \
  -v "$PWD/llmleaf.e2e.toml:/etc/llmleaf/llmleaf.toml${MNT}" \
  "$SRV_IMAGE" >/dev/null

echo ">> running probe ($PRB)"
set +e
"$ENGINE" run --rm --name "$PRB" --network "$NET" \
  --env-file .env \
  -e LLMLEAF_BASE_URL="http://${SRV}:8080" \
  -e LLMLEAF_CONFIG=/etc/llmleaf/llmleaf.toml \
  -e LLMLEAF_HOST="$SRV" -e LLMLEAF_PORT=8080 \
  -e NO_PROXY="$NOPROXY" -e no_proxy="$NOPROXY" \
  -v "$PWD/llmleaf.e2e.toml:/etc/llmleaf/llmleaf.toml${MNT}" \
  -v "$PWD/probe-suite.sh:/e2e/probe-suite.sh${MNT}" \
  --entrypoint bash "$PRB_IMAGE" /e2e/probe-suite.sh
code=$?
set -e

exit "$code"
