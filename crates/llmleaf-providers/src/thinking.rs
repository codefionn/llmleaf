//! Shared thinking-effort mapping for the budget-style providers.
//!
//! Several dialects express reasoning effort as a *token budget* rather than a discrete effort string.
//! The canonical [`Thinking`] ladder is abstract (principle 2: the core knows no provider), so the
//! concrete budget for each rung is decided here, in the extension layer, and reused by every provider
//! that speaks budgets (Anthropic, Gemini, Cohere). Each caller still clamps to its own model's range.

use llmleaf_model::Thinking;

/// Map a [`Thinking`] rung to a reasoning token budget. The floor (1024) is the lowest the
/// budget-style upstreams accept; the upper rungs roughly double. Coarse by design.
pub(crate) fn budget_tokens(t: Thinking) -> u32 {
    match t {
        Thinking::Low => 1024,
        Thinking::Med => 4096,
        Thinking::High => 8192,
        Thinking::Highx => 16384,
        Thinking::Max => 32768,
    }
}
