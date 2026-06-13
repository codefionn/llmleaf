use thiserror::Error;

/// Errors that can surface anywhere along the canonical pipeline.
///
/// Kept deliberately small: the core transports and classifies, it does not interpret provider
/// semantics. Provider-specific detail rides along as opaque strings.
#[derive(Debug, Error, Clone)]
pub enum ModelError {
    /// An upstream provider returned a transport- or protocol-level failure.
    #[error("upstream {status}: {message}")]
    Upstream { status: u16, message: String },

    /// The provider could not be reached or the call failed before a response.
    #[error("provider unavailable: {0}")]
    Unavailable(String),

    /// A dialect mapping (edge surface or provider extension) could not translate the payload.
    #[error("mapping error: {0}")]
    Mapping(String),

    /// The selected provider does not implement the requested capability (embeddings, speech
    /// synthesis, transcription). Deliberately distinct from a transient failure: routing falls
    /// through to the next target *without* penalizing this provider's health — it is not degraded,
    /// it simply does not offer this modality.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// The request exceeded its deadline.
    #[error("deadline exceeded")]
    Timeout,

    /// The client or core canceled the in-flight request.
    #[error("canceled")]
    Canceled,
}
