# docker-bake.hcl — orchestrates the multi-stage Dockerfile.
#
# Local usage:
#   docker buildx bake lint            # run formatting + clippy
#   docker buildx bake test            # run the workspace test suite
#   docker buildx bake image           # build the runtime image (host platform)
#   docker buildx bake ci              # lint + test + image, shared layers built once
#   docker buildx bake release         # multi-arch image (amd64 + arm64)
#
# CI usage: docker/metadata-action emits a bake file defining the
# `docker-metadata-action` target (tags + labels); the `image`/`release`
# targets inherit it, so tagging is owned by CI and the defaults below are
# only the local fallback.

variable "REGISTRY" {
  default = "ghcr.io"
}

variable "IMAGE" {
  default = "codefionn/llmleaf"
}

variable "TAG" {
  default = "dev"
}

variable "RUST_VERSION" {
  default = "1.90"
}

# Tags + OCI labels. In CI, docker/metadata-action emits a bake file that REDEFINES this
# target with the real tags (one per registry — ghcr.io / quay.io / docker.io — plus the
# semver/edge/sha matrix) and that file's tags win. The default below is the LOCAL fallback
# only. Crucially, `image`/`release` inherit this target and set NO `tags` of their own: a
# child target's own `tags` would override the inherited metadata tags (bake replaces, not
# merges, list attributes on inherit), which silently pinned CI pushes to `:dev`.
target "docker-metadata-action" {
  tags = ["${REGISTRY}/${IMAGE}:${TAG}"]
}

# Shared settings for every target.
target "_common" {
  dockerfile = "Dockerfile"
  context    = "."
  args = {
    RUST_VERSION = "${RUST_VERSION}"
  }
}

group "default" {
  targets = ["image"]
}

group "ci" {
  targets = ["lint", "test", "image"]
}

# --- Quality gates: build up to a stage and discard the result (no image). ---
target "lint" {
  inherits = ["_common"]
  target   = "lint"
  output   = ["type=cacheonly"]
}

target "test" {
  inherits = ["_common"]
  target   = "test"
  output   = ["type=cacheonly"]
}

# --- The shippable runtime image (single platform by default). ---
# Tags come from the `docker-metadata-action` target (see above); do NOT set `tags` here.
target "image" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime"
}

# --- Multi-arch publish target. ---
target "release" {
  inherits  = ["image"]
  platforms = ["linux/amd64", "linux/arm64"]
}

# --- The capability-probe image used by the e2e-tests/ compose stack. ---
target "probe" {
  inherits = ["_common"]
  target   = "probe-runtime"
  tags     = ["${REGISTRY}/${IMAGE}-probe:${TAG}"]
}
