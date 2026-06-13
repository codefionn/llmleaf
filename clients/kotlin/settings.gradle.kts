// Standalone Gradle build for the llmleaf Kotlin Multiplatform client.
// It is intentionally detached from any parent build: this directory is the root.

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        google()
    }
}

dependencyResolutionManagement {
    // Repositories are declared centrally here for the JVM/native dependency graph.
    //
    // Mode is PREFER_SETTINGS rather than FAIL_ON_PROJECT_REPOS: the Kotlin 2.0 Kotlin/JS
    // toolchain tasks (`kotlinNodeJsSetup`, `kotlinYarnSetup`) unconditionally register their
    // own project-scoped Ivy "distribution" repositories at task time to fetch Node.js / Yarn,
    // and FAIL_ON_PROJECT_REPOS rejects that outright ("repository '...' was added by unknown
    // code"). PREFER_SETTINGS keeps these central repositories authoritative — they still win for
    // every artifact declared here, preserving reproducibility — while permitting the plugin's
    // own toolchain-download repos. The root build and :example still declare no `repositories {}`
    // of their own, so resolution stays centralized in practice.
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        mavenCentral()
        google()

        // The Kotlin/JS toolchain (Node.js + Yarn) is fetched from these Ivy "distribution"
        // repositories. Declaring them centrally (rather than letting the plugin's task-time
        // project repo serve them) keeps resolution under settings control; `exclusiveContent`
        // pins each to its single module so it never shadows mavenCentral/google. The pattern
        // layouts and module ids mirror the Kotlin 2.0 plugin's NodeJsSetupTask / YarnSetupTask.
        exclusiveContent {
            forRepository {
                ivy("https://nodejs.org/dist") {
                    name = "Node.js Distributions"
                    patternLayout { artifact("v[revision]/[artifact](-v[revision]-[classifier]).[ext]") }
                    metadataSources { artifact() }
                    content { includeModule("org.nodejs", "node") }
                }
            }
            filter { includeGroup("org.nodejs") }
        }
        exclusiveContent {
            forRepository {
                ivy("https://github.com/yarnpkg/yarn/releases/download") {
                    name = "Yarn Distributions"
                    patternLayout { artifact("v[revision]/[artifact](-v[revision]).[ext]") }
                    metadataSources { artifact() }
                    content { includeModule("com.yarnpkg", "yarn") }
                }
            }
            filter { includeGroup("com.yarnpkg") }
        }
    }
}

rootProject.name = "llmleaf-client"

include(":example")
