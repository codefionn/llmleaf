//! Build script for the official Zig SDK of the llmleaf LLM proxy.
//!
//! Targets (Zig 0.16.0):
//!   zig build            -> build the `llmleaf` module + the example (install step)
//!   zig build example    -> run examples/basic.zig (reads LLMLEAF_BASE_URL / LLMLEAF_API_KEY)
//!   zig build test       -> run the unit tests in src/ and test/
//!
//! The SDK is std-only; there are no package dependencies to fetch.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The public library module. Consumers add this package and then
    // `@import("llmleaf")`.
    const llmleaf_mod = b.addModule("llmleaf", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // A static library artifact, mostly so `zig build` produces something
    // installable and type-checks the whole module surface.
    const lib = b.addLibrary(.{
        .name = "llmleaf",
        .root_module = llmleaf_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // --- example -----------------------------------------------------------
    const example_mod = b.createModule(.{
        .root_source_file = b.path("examples/basic.zig"),
        .target = target,
        .optimize = optimize,
    });
    example_mod.addImport("llmleaf", llmleaf_mod);

    const example = b.addExecutable(.{
        .name = "basic",
        .root_module = example_mod,
    });
    b.installArtifact(example);

    const run_example = b.addRunArtifact(example);
    run_example.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_example.addArgs(args);

    const example_step = b.step("example", "Run examples/basic.zig against a live gateway");
    example_step.dependOn(&run_example.step);

    // --- tests -------------------------------------------------------------
    const lib_tests = b.addTest(.{
        .name = "llmleaf-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "Run the SDK unit tests");
    test_step.dependOn(&run_lib_tests.step);
}
