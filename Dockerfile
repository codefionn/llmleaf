# syntax=docker/dockerfile:1.4
#
# Multi-stage build for the llmleaf proxy binary.
#
# Stages:
#   xx             - BuildKit-aware cross-compilation helper scripts
#   builder-base   - build-platform Rust toolchain + native build deps
#   source         - workspace source + fetched Cargo dependencies
#   lint           - cargo fmt --check + cargo clippy -D warnings
#   test           - cargo test --workspace
#   build          - cross-compile static llmleaf for TARGETPLATFORM
#   probe-build    - cross-compile static probe example for TARGETPLATFORM
#   probe-runtime  - Alpine image with bash for the e2e probe suite
#   runtime-rootfs - CA certs and empty runtime directories for scratch
#   runtime        - scratch image with the static llmleaf binary
#
# Release/probe builds never use FROM --platform=${TARGETPLATFORM}. BuildKit
# runs the Rust toolchain on BUILDPLATFORM, while tonistiigi/xx maps
# TARGETPLATFORM to the correct Rust target triple and C toolchain. The output is
# verified static before being copied into scratch.

ARG RUST_VERSION=1.90
ARG ALPINE_VERSION=3.22
ARG XX_VERSION=1.8.0
ARG APP_NAME=llmleaf

# ---------------------------------------------------------------------------
# xx: BuildKit-aware cross-compilation helpers.
# ---------------------------------------------------------------------------
FROM --platform=${BUILDPLATFORM} tonistiigi/xx:${XX_VERSION} AS xx

# ---------------------------------------------------------------------------
# builder-base: build-platform toolchain + native build dependencies.
# ---------------------------------------------------------------------------
FROM --platform=${BUILDPLATFORM} rust:${RUST_VERSION}-alpine${ALPINE_VERSION} AS builder-base
COPY --from=xx / /
RUN apk add --no-cache \
        build-base \
        ca-certificates \
        cmake \
        clang \
        clang-dev \
        file \
        git \
        lld \
        pax-utils \
        perl \
        pkgconf
WORKDIR /app

# ---------------------------------------------------------------------------
# source: workspace source + dependency fetch. The registry/git caches are
# shared across targets; compiled artifacts live in target-specific cache IDs.
# ---------------------------------------------------------------------------
FROM builder-base AS source
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# Embedded by crates/llmleaf/src/main.rs's test suite via include_str!; without
# it the lint (clippy --all-targets) and test stages fail to compile.
COPY llmleaf.example.toml ./
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    cargo fetch --locked

# ---------------------------------------------------------------------------
# lint: formatting + clippy (fails the build on any warning).
# ---------------------------------------------------------------------------
FROM source AS lint
RUN rustup component add rustfmt clippy
RUN cargo fmt --all -- --check
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/app/target/native-clippy \
    cargo clippy --workspace --all-targets --locked --target-dir /app/target/native-clippy -- -D warnings

# ---------------------------------------------------------------------------
# test: the workspace test suite on the build platform.
# ---------------------------------------------------------------------------
FROM source AS test
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/app/target/native-test \
    cargo test --workspace --locked --target-dir /app/target/native-test

# ---------------------------------------------------------------------------
# build: cross-compile the release binary for TARGETPLATFORM.
# ---------------------------------------------------------------------------
FROM source AS build
ARG APP_NAME
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT
RUN xx-apk add --no-cache gcc musl-dev
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=llmleaf-target-${TARGETOS}-${TARGETARCH}${TARGETVARIANT},target=/app/target \
    xx-cargo build --release --locked -p ${APP_NAME} --target-dir /app/target \
    && cp /app/target/$(xx-cargo --print-target-triple)/release/${APP_NAME} /usr/local/bin/${APP_NAME} \
    && xx-verify --static /usr/local/bin/${APP_NAME}

# ---------------------------------------------------------------------------
# probe-build: cross-compile the capability-probe example.
# ---------------------------------------------------------------------------
FROM source AS probe-build
ARG APP_NAME
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT
RUN xx-apk add --no-cache gcc musl-dev
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=llmleaf-probe-${TARGETOS}-${TARGETARCH}${TARGETVARIANT},target=/app/target \
    xx-cargo build --release --locked -p ${APP_NAME} --example probe --target-dir /app/target \
    && cp /app/target/$(xx-cargo --print-target-triple)/release/examples/probe /usr/local/bin/probe \
    && xx-verify --static /usr/local/bin/probe

# ---------------------------------------------------------------------------
# probe-runtime: the probe binary on Alpine with a shell (the e2e suite script
# is bash) and ca-certificates. NOT scratch: the suite needs /bin/bash.
# ---------------------------------------------------------------------------
FROM alpine:${ALPINE_VERSION} AS probe-runtime
RUN apk add --no-cache bash ca-certificates
COPY --from=probe-build /usr/local/bin/probe /usr/local/bin/probe
ENTRYPOINT ["probe"]

# ---------------------------------------------------------------------------
# runtime-rootfs: arch-independent files copied into scratch.
# ---------------------------------------------------------------------------
FROM alpine:${ALPINE_VERSION} AS runtime-rootfs
RUN apk add --no-cache ca-certificates \
    && mkdir -p /etc/llmleaf /tmp \
    && chmod 1777 /tmp

# ---------------------------------------------------------------------------
# runtime: scratch image containing only the static binary and runtime data.
# ---------------------------------------------------------------------------
FROM scratch AS runtime
ARG APP_NAME

LABEL org.opencontainers.image.title="llmleaf" \
      org.opencontainers.image.description="llmleaf — a high-efficiency LLM proxy" \
      org.opencontainers.image.source="https://github.com/codefionn/llmleaf" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

COPY --from=runtime-rootfs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=runtime-rootfs /etc/llmleaf /etc/llmleaf
COPY --from=runtime-rootfs /tmp /tmp
COPY --from=build /usr/local/bin/${APP_NAME} /usr/local/bin/llmleaf

# llmleaf resolves its config in this order: CLI arg, then $LLMLEAF_CONFIG, then
# ./llmleaf.toml, else a loud embedded dev config. Mount a real config and point
# LLMLEAF_CONFIG at it for production, e.g.:
#   docker run -p 8080:8080 \
#     -v $PWD/llmleaf.toml:/etc/llmleaf/llmleaf.toml:ro \
#     -e LLMLEAF_CONFIG=/etc/llmleaf/llmleaf.toml ghcr.io/codefionn/llmleaf:latest
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
USER 65532:65532
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llmleaf"]
