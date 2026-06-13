// Runnable example for the llmleaf KMP client. A plain JVM application that depends on
// the multiplatform library's JVM artifact. Run with:
//
//   LLMLEAF_BASE_URL=https://gateway.example.com LLMLEAF_API_KEY=sk-... \
//     ./gradlew :example:run

plugins {
    // No version here: the Kotlin/JVM plugin is resolved from the root build's classpath
    // (declared at the root with `apply false`). Repositories come from settings.gradle.kts.
    kotlin("jvm")
    application
}

val ktorVersion = "3.0.3"
val coroutinesVersion = "1.9.0"

dependencies {
    // Consume the KMP library; Gradle resolves its JVM ("jvm") variant automatically.
    implementation(project(":"))

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:$coroutinesVersion")
    // The CIO engine the JVM target uses at runtime.
    runtimeOnly("io.ktor:ktor-client-cio:$ktorVersion")
}

application {
    mainClass.set("eu.codefionn.llmleaf.example.MainKt")
}

kotlin {
    jvmToolchain(17)
}
