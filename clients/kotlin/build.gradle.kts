import com.vanniktech.maven.publish.SonatypeHost
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// ----------------------------------------------------------------------------
// llmleaf — official Kotlin Multiplatform client SDK.
//
// Two codegen stories live in this build, deliberately:
//
//   1. Square Wire (com.squareup.wire) compiles the single-source-of-truth proto at
//      ../proto. This is the REAL proto codegen — proof the schema compiles — and the
//      generated Kotlin types are available to consumers under `eu.codefionn.llmleaf.v1`.
//      Run it on its own with `./gradlew generateProtos` (alias: scripts/gen.sh).
//
//   2. The PUBLIC SDK request/response types and the HTTP transport are hand-written
//      kotlinx.serialization `@Serializable` classes (src/commonMain/.../model). Wire's
//      runtime types do not serialise to the OpenAI/OpenRouter JSON wire (oneof unions,
//      lowercase enum tokens, snake_case keys, free-form raw-JSON splicing), so Wire
//      compiles the proto while kotlinx.serialization drives the wire. See SPEC.md.
//
// NOTE: this project was authored WITHOUT a local Gradle run (no toolchain in the
// authoring environment). A first `./gradlew build` on a normal machine is required to
// confirm it. Versions are pinned to recent stable releases and kept consistent below.
// ----------------------------------------------------------------------------

plugins {
    kotlin("multiplatform") version "2.0.21"
    kotlin("plugin.serialization") version "2.0.21"
    id("com.squareup.wire") version "5.1.0"
    // Maven Central publishing for the whole Kotlin Multiplatform artifact set (the root
    // `kotlinMultiplatform` metadata module plus the per-target jvm / js / linuxX64 modules),
    // including POM generation, in-memory GPG signing and the Central Portal upload. Hand-rolling
    // `maven-publish` + `signing` across every KMP publication is brittle; this plugin owns it.
    // The release workflow drives it with `./gradlew publishToMavenCentral`. See RELEASING.md.
    id("com.vanniktech.maven.publish") version "0.30.0"
    // Declared (not applied) here so the Kotlin/JVM plugin is resolved onto the build classpath
    // once at the root; the :example subproject applies it WITHOUT a version, avoiding the
    // "plugin already on the classpath must not include a version" error.
    kotlin("jvm") version "2.0.21" apply false
}

group = "eu.codefionn.llmleaf"
version = "0.1.0"

// Repositories are centralised in settings.gradle.kts (dependencyResolutionManagement +
// FAIL_ON_PROJECT_REPOS); declaring them here too would fail configuration, so we don't.

// Pinned, mutually-consistent dependency versions.
val ktorVersion = "3.0.3"
val coroutinesVersion = "1.9.0"
val serializationVersion = "1.7.3"

kotlin {
    // Note: explicit-API mode is intentionally NOT enabled. The public types below already
    // carry `public` on their top-level declarations; enabling strict explicit-API would also
    // demand it on every data-class constructor property, and since this project could not be
    // compiled in the authoring environment that is a needless first-build risk. Re-enable with
    // `explicitApi()` once a local Gradle run can confirm every member is annotated.

    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
        }
    }

    // At least one native target (Linux/x64). Mac/Windows hosts can add their own targets
    // the same way; the native `actual` engine lives in src/linuxX64Main (see below).
    linuxX64()

    // JS via the Ktor JS engine. IR is the only backend in Kotlin 2.x. We enable nodejs()
    // only: its tests run under Node with no external tooling, keeping `./gradlew build` green
    // on a fresh machine. Add browser() if you need a browser target (its tests need a browser
    // / headless Chrome on PATH).
    js {
        nodejs()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:$coroutinesVersion")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:$serializationVersion")

                implementation("io.ktor:ktor-client-core:$ktorVersion")
                implementation("io.ktor:ktor-client-content-negotiation:$ktorVersion")
                implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:$coroutinesVersion")
                // MockEngine lets the transport tests run without a network or a platform engine.
                implementation("io.ktor:ktor-client-mock:$ktorVersion")
            }
        }

        val jvmMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-cio:$ktorVersion")
            }
        }

        // CIO on the linuxX64 leaf source set, where the native `actual` engine lives. We
        // attach to the leaf rather than the intermediate `nativeMain` so the build does not
        // depend on the default-hierarchy template having materialised `nativeMain`. Add more
        // native targets by declaring them above and giving each its CIO dependency + actual.
        val linuxX64Main by getting {
            dependencies {
                implementation("io.ktor:ktor-client-cio:$ktorVersion")
            }
        }

        val jsMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-js:$ktorVersion")
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Wire — compile the proto contract from ../proto.
//
// Wire generates Kotlin types into the `commonMain` source set, so they compile for
// every target. The task is named `generateProtos`; `scripts/gen.sh` runs it.
// ----------------------------------------------------------------------------
wire {
    kotlin {
        // Pure Kotlin (no Android), java-interop off — multiplatform-friendly output.
        javaInterop = false
    }
    // The proto tree is the single source of truth at ../proto. It has no external
    // imports, so sourcePath alone is enough; Wire wires the generated Kotlin into
    // commonMain because the Kotlin Multiplatform plugin is applied.
    sourcePath {
        srcDir("../proto")
    }
}

// ----------------------------------------------------------------------------
// Maven Central publishing.
//
// Driven only by the release workflow on a `v*` tag (see ../../.github/workflows/release.yml
// and ../../.github/RELEASING.md). Nothing here runs during a plain `./gradlew build`: the
// signing + upload tasks only execute under `publishToMavenCentral`.
//
// Credentials + the signing key are supplied via Gradle properties / environment at release
// time, NOT committed:
//   ORG_GRADLE_PROJECT_mavenCentralUsername / ...Password  — Central Portal user token
//   ORG_GRADLE_PROJECT_signingInMemoryKey / ...KeyPassword — ASCII-armored GPG secret key
//
// The published version is the project version, overridden from the git tag at release time
// with `-Pversion=<x.y.z>`; locally it stays the `version = "0.1.0"` set above.
// ----------------------------------------------------------------------------
mavenPublishing {
    // Upload to the Central Portal (central.sonatype.com) and auto-release the deployment once
    // its validation passes, so a tag push needs no manual "release" click in the Portal UI.
    publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL, automaticRelease = true)
    // Sign every KMP publication with the in-memory GPG key (Central requires signatures).
    signAllPublications()

    coordinates(group.toString(), "llmleaf-client", version.toString())

    pom {
        name.set("llmleaf-client")
        description.set("Official Kotlin Multiplatform client SDK for the llmleaf LLM proxy.")
        url.set("https://github.com/codefionn/llmleaf")
        licenses {
            license {
                name.set("MIT")
                url.set("https://opensource.org/licenses/MIT")
                distribution.set("repo")
            }
            license {
                name.set("Apache-2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0")
                distribution.set("repo")
            }
        }
        developers {
            developer {
                id.set("codefionn")
                name.set("codefionn")
                url.set("https://github.com/codefionn")
            }
        }
        scm {
            url.set("https://github.com/codefionn/llmleaf")
            connection.set("scm:git:https://github.com/codefionn/llmleaf.git")
            developerConnection.set("scm:git:ssh://git@github.com/codefionn/llmleaf.git")
        }
    }
}
