#!/usr/bin/env bash
# Regenerate the Kotlin typed model from the single proto source of truth.
#
#   clients/kotlin/scripts/gen.sh
#
# Codegen here is Square Wire (com.squareup.wire), wired into the Gradle build and
# pointed at ../proto. This script just runs Wire's generator task so it matches the
# Makefile's `gen-kotlin` target and the other clients' gen.sh contract.
#
# Toolchain (fetched automatically by the Gradle wrapper on first run):
#   - JDK 17+ on PATH (or JAVA_HOME set)
#   - Gradle 8.11.1 (downloaded by ./gradlew; gradle-wrapper.jar is fetched on first run)
#   - The Wire Gradle plugin + Kotlin 2.0.x (resolved from Maven Central)
#
# Output: Wire emits Kotlin types for eu.codefionn.llmleaf.v1 into the build's generated
# sources (build/generated/source/wire/), compiled into commonMain for every target.
set -euo pipefail

# Run from the kotlin client root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

# `generateProtos` is the umbrella Wire task; it also runs as a dependency of `build`.
./gradlew generateProtos

echo "generated: build/generated/source/wire/ (eu.codefionn.llmleaf.v1, compiled into commonMain)"
