//! Typed errors for the llmleaf SDK.
//!
//! Zig error values cannot carry a payload, so the wire `ApiError`
//! (`{"error":{"message":...}}` on any non-2xx) is surfaced two ways:
//!   * the call returns `error.Api` (or a more specific status error), and
//!   * the structured `{ status, message }` is written through an out-pointer
//!     the caller passes in (`*?ApiError`). See `src/client.zig` call sites.
//!
//! The status->error mapping mirrors SPEC.md:
//!   400 bad request · 401 missing/invalid key · 403 blocked/model-not-allowed
//!   404 no route for model · 429 key suspended · 502 all upstreams failed.

const std = @import("std");

/// The structured API error parsed from the wire envelope. `message` is owned by
/// the arena of the `*?ApiError` slot's owner (the SDK call), and stays valid
/// until that call's `Parsed`/arena is freed — for the bare error path the SDK
/// duplicates it into the caller-supplied allocator; see `client.zig`.
pub const ApiError = struct {
    status: u16,
    message: []const u8,
};

/// Everything a public SDK call can fail with. Transport/parse errors are
/// merged in at the call sites via `||`.
pub const Error = error{
    /// Non-2xx with a parsed `ApiError` available through the out-pointer.
    Api,
    /// 400.
    BadRequest,
    /// 401 — missing or invalid API key.
    Unauthorized,
    /// 403 — request blocked, or model not allowed for this key.
    Forbidden,
    /// 404 — no route for the requested model.
    NotFound,
    /// 429 — key suspended by the limiter.
    RateLimited,
    /// 502 — every upstream failed.
    UpstreamFailed,
    /// Non-2xx whose body did not contain the documented error envelope.
    UnexpectedStatus,
};

/// Map an HTTP status to the most specific error in `Error`. 2xx never reaches
/// here. Any unmapped non-2xx becomes `error.Api` (the generic carrier).
pub fn fromStatus(status: u16) Error {
    return switch (status) {
        400 => error.BadRequest,
        401 => error.Unauthorized,
        403 => error.Forbidden,
        404 => error.NotFound,
        429 => error.RateLimited,
        502 => error.UpstreamFailed,
        else => error.Api,
    };
}

test "status mapping" {
    try std.testing.expectEqual(Error.Unauthorized, fromStatus(401));
    try std.testing.expectEqual(Error.UpstreamFailed, fromStatus(502));
    try std.testing.expectEqual(Error.Api, fromStatus(418));
}
