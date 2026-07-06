//! Runnable example for the llmleaf Zig SDK.
//!
//!   zig build example
//!
//! Reads the gateway from the environment:
//!   LLMLEAF_BASE_URL   e.g. https://gateway.example.com   (default http://localhost:8080)
//!   LLMLEAF_API_KEY    your API key                       (required)
//!   LLMLEAF_MODEL      model id          (default gpt-4o-mini)
//!
//! It runs, from SPEC.md: a non-streaming chat (prints the text), a streaming
//! chat (prints deltas), a non-streaming + a streaming Responses call (the
//! OpenAI Responses dialect), and a model list.
//!
//! Zig 0.16 routes I/O and the environment through `std.process.Init`, which the
//! runtime passes to `main`; we read env vars from `init.environ_map` and use
//! `init.gpa` as the allocator.

const std = @import("std");
const llmleaf = @import("llmleaf");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const env = init.environ_map;

    // --- read configuration from the environment --------------------------
    const base_url = env.get("LLMLEAF_BASE_URL") orelse "http://localhost:8080";
    const model = env.get("LLMLEAF_MODEL") orelse "gpt-4o-mini";
    const api_key = env.get("LLMLEAF_API_KEY") orelse {
        std.debug.print("error: set LLMLEAF_API_KEY (and optionally LLMLEAF_BASE_URL / LLMLEAF_MODEL)\n", .{});
        return error.MissingApiKey;
    };

    const client = try llmleaf.Client.init(gpa, base_url, api_key, .{});
    defer client.deinit();

    const stdout_file = std.Io.File.stdout();
    var out_buf: [4096]u8 = undefined;
    var out_writer = stdout_file.writer(init.io, &out_buf);
    const out = &out_writer.interface;

    // --- 1. non-streaming chat -------------------------------------------
    try out.print("== non-streaming chat ==\n", .{});
    {
        var api_err: ?llmleaf.ApiError = null;
        const resp = client.chat(.{
            .model = model,
            .messages = &.{
                llmleaf.ChatMessage.textMsg(.system, "You are concise."),
                llmleaf.ChatMessage.textMsg(.user, "Say hello in one short sentence."),
            },
            .max_completion_tokens = 64,
        }, &api_err) catch |err| {
            try reportApiError(out, err, &api_err, gpa);
            return err;
        };
        defer resp.deinit();
        if (resp.value.choices.len > 0) {
            if (resp.value.choices[0].message.content) |c| switch (c) {
                .text => |t| try out.print("{s}\n", .{t}),
                .parts => try out.print("(multimodal content)\n", .{}),
            };
        }
        if (resp.value.usage) |u| {
            try out.print("(tokens: prompt={d} completion={d} total={d})\n", .{
                u.prompt_tokens, u.completion_tokens, u.total_tokens,
            });
        }
    }

    // --- 2. streaming chat ------------------------------------------------
    try out.print("\n== streaming chat ==\n", .{});
    {
        var api_err: ?llmleaf.ApiError = null;
        var stream = client.chatStream(.{
            .model = model,
            .messages = &.{
                llmleaf.ChatMessage.textMsg(.user, "Count from 1 to 5, one number per line."),
            },
            .max_completion_tokens = 64,
        }, &api_err) catch |err| {
            try reportApiError(out, err, &api_err, gpa);
            return err;
        };
        defer stream.deinit();

        while (try stream.next()) |chunk| {
            if (chunk.choices.len > 0) {
                if (chunk.choices[0].delta.content) |delta| {
                    try out.print("{s}", .{delta});
                    try out.flush(); // show deltas live
                }
            }
        }
        try out.print("\n", .{});
    }

    // --- 3. non-streaming responses (OpenAI Responses dialect) ------------
    try out.print("\n== non-streaming responses ==\n", .{});
    {
        var api_err: ?llmleaf.ApiError = null;
        const resp = client.responses(.{
            .model = model,
            .instructions = "You are concise.",
            .input = .{ .text = "Say hello in one short sentence." },
            .max_output_tokens = 64,
        }, &api_err) catch |err| {
            try reportApiError(out, err, &api_err, gpa);
            return err;
        };
        defer resp.deinit();
        // The assistant's reply is one or more `output` message items with
        // `output_text` parts; print their text.
        for (resp.value.output) |item| switch (item) {
            .message => |m| if (m.content) |c| switch (c) {
                .parts => |parts| for (parts) |p| switch (p) {
                    .output_text => |t| try out.print("{s}\n", .{t.text}),
                    else => {},
                },
                .text => |t| try out.print("{s}\n", .{t}),
            },
            else => {},
        };
        if (resp.value.usage) |u| {
            try out.print("(tokens: input={d} output={d} total={d})\n", .{
                u.input_tokens, u.output_tokens, u.total_tokens,
            });
        }
    }

    // --- 4. streaming responses -------------------------------------------
    try out.print("\n== streaming responses ==\n", .{});
    {
        var api_err: ?llmleaf.ApiError = null;
        var stream = client.responsesStream(.{
            .model = model,
            .input = .{ .text = "Count from 1 to 5, one number per line." },
            .max_output_tokens = 64,
        }, &api_err) catch |err| {
            try reportApiError(out, err, &api_err, gpa);
            return err;
        };
        defer stream.deinit();

        // No `[DONE]` sentinel: `next()` returns null after the terminal
        // response.completed/incomplete/failed event.
        while (try stream.next()) |event| {
            if (std.mem.eql(u8, event.type, "response.output_text.delta")) {
                if (event.delta) |delta| {
                    try out.print("{s}", .{delta});
                    try out.flush(); // show deltas live
                }
            }
        }
        try out.print("\n", .{});
    }

    // --- 5. list models ---------------------------------------------------
    try out.print("\n== models ==\n", .{});
    {
        var api_err: ?llmleaf.ApiError = null;
        const models = client.listModels(.all, null, &api_err) catch |err| {
            try reportApiError(out, err, &api_err, gpa);
            return err;
        };
        defer models.deinit();
        const n = models.value.data.len;
        try out.print("{d} model(s)\n", .{n});
        for (models.value.data[0..@min(n, 10)]) |m| {
            try out.print("  - {s}\n", .{m.id});
        }
        if (n > 10) try out.print("  ... and {d} more\n", .{n - 10});
    }

    try out.flush();
}

fn reportApiError(
    out: *std.Io.Writer,
    err: anyerror,
    api_err: *?llmleaf.ApiError,
    gpa: std.mem.Allocator,
) !void {
    if (api_err.*) |e| {
        try out.print("API error {d}: {s}\n", .{ e.status, e.message });
        gpa.free(e.message);
        api_err.* = null;
    } else {
        try out.print("transport error: {s}\n", .{@errorName(err)});
    }
    try out.flush();
}
