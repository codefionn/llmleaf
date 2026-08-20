//! The async [`Client`] and its [`ClientBuilder`].

use crate::error::{Error, Result};
use crate::stream::{ndjson_lines, sse_chunks, sse_responses};
use crate::transport::{Backend, Body, Method, Request};
use crate::types::*;
use crate::wire::{speech_content_type, RawEmbeddingResponse};
use bytes::Bytes;
use futures::Stream;
use std::time::Duration;

/// Builder for [`Client`]. Obtain via [`Client::builder`].
#[derive(Clone)]
pub struct ClientBuilder {
    base_url: String,
    api_key: String,
    admin_token: Option<String>,
    timeout: Option<Duration>,
    #[cfg(feature = "tokio")]
    http: Option<reqwest::Client>,
    #[cfg(feature = "compio")]
    compio_http: Option<cyper::Client>,
}

impl ClientBuilder {
    fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), api_key: api_key.into(), admin_token: None, timeout: None, #[cfg(feature = "tokio")] http: None, #[cfg(feature = "compio")] compio_http: None }
    }
    /// Set the per-request HTTP timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self { self.timeout = Some(timeout); self }
    /// Set the optional `x-admin-token`, sent on every request.
    pub fn admin_token(mut self, token: impl Into<String>) -> Self { self.admin_token = Some(token.into()); self }
    /// Supply a fully pre-configured `reqwest` client. Available with the default `tokio` feature.
    #[cfg(feature = "tokio")]
    pub fn http_client(mut self, client: reqwest::Client) -> Self { self.http = Some(client); self }
    /// Supply a fully pre-configured Compio HTTP client. Available with the `compio` feature.
    #[cfg(feature = "compio")]
    pub fn compio_http_client(mut self, client: cyper::Client) -> Self { self.compio_http = Some(client); self }
    /// Build the [`Client`].
    pub fn build(self) -> Result<Client> {
        let base_url = self.base_url.trim_end_matches('/').to_string();
        if base_url.is_empty() { return Err(Error::InvalidBaseUrl("base url is empty".to_string())); }
        url::Url::parse(&base_url).map_err(|error| Error::InvalidBaseUrl(error.to_string()))?;
        if self.api_key.bytes().any(|byte| byte == b'\r' || byte == b'\n') { return Err(Error::InvalidBaseUrl("api key contains invalid header bytes".to_string())); }
        if self.admin_token.as_deref().is_some_and(|token| token.bytes().any(|byte| byte == b'\r' || byte == b'\n')) { return Err(Error::InvalidBaseUrl("admin token contains invalid header bytes".to_string())); }
        #[cfg(feature = "tokio")]
        let backend = { let client = match self.http { Some(client) => client, None => { let mut builder = reqwest::Client::builder(); if let Some(timeout) = self.timeout { builder = builder.timeout(timeout); } builder.build()? } }; Backend::Tokio { client } };
        #[cfg(all(not(feature = "tokio"), feature = "compio"))]
        let backend = Backend::Compio { client: self.compio_http.unwrap_or_else(cyper::Client::new), timeout: self.timeout };
        Ok(Client { backend, base_url, bearer: format!("Bearer {}", self.api_key), admin_token: self.admin_token })
    }
}

/// Async client for the llmleaf gateway.
#[derive(Clone)]
pub struct Client { backend: Backend, base_url: String, bearer: String, admin_token: Option<String> }

impl Client {
    /// Construct a client from a base URL and API key with default settings.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> { ClientBuilder::new(base_url, api_key).build() }
    /// Start a [`ClientBuilder`].
    pub fn builder(base_url: impl Into<String>, api_key: impl Into<String>) -> ClientBuilder { ClientBuilder::new(base_url, api_key) }
    fn request(&self, method: Method, path: &str) -> Request { Request::empty(method, format!("{}{}", self.base_url, path), self.bearer.clone(), self.admin_token.clone()) }
    fn json<T: serde::Serialize>(&self, method: Method, path: &str, value: &T) -> Result<Request> { Request::json(method, format!("{}{}", self.base_url, path), self.bearer.clone(), self.admin_token.clone(), value) }
    fn query(&self, method: Method, path: &str, pairs: &[(&str, String)]) -> Result<Request> { let mut url = url::Url::parse(&format!("{}{}", self.base_url, path)).map_err(|error| Error::InvalidBaseUrl(error.to_string()))?; { let mut query = url.query_pairs_mut(); for (key, value) in pairs { query.append_pair(key, value); } } Ok(Request::empty(method, url.into(), self.bearer.clone(), self.admin_token.clone())) }
    async fn send(&self, request: Request) -> Result<crate::transport::Response> { self.backend.send(request).await }
    /// `POST /v1/chat/completions` (non-streaming).
    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse> { request.stream = None; self.send(self.json(Method::Post, "/v1/chat/completions", &request)?).await?.json().await }
    /// `POST /v1/chat/completions` (streaming SSE).
    pub async fn chat_stream(&self, mut request: ChatRequest) -> Result<impl Stream<Item = Result<ChatCompletionChunk>>> { request.stream = Some(true); Ok(sse_chunks(self.send(self.json(Method::Post, "/v1/chat/completions", &request)?).await?.into_stream())) }
    /// `POST /v1/responses` (non-streaming).
    pub async fn responses(&self, mut request: ResponsesRequest) -> Result<ResponsesResponse> { request.stream = None; self.send(self.json(Method::Post, "/v1/responses", &request)?).await?.json().await }
    /// `POST /v1/responses` (streaming SSE).
    pub async fn responses_stream(&self, mut request: ResponsesRequest) -> Result<impl Stream<Item = Result<ResponsesStreamEvent>>> { request.stream = Some(true); Ok(sse_responses(self.send(self.json(Method::Post, "/v1/responses", &request)?).await?.into_stream())) }
    /// `POST /v1/embeddings`, decoding base64 payloads into float vectors.
    pub async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> { let raw: RawEmbeddingResponse = self.send(self.json(Method::Post, "/v1/embeddings", &request)?).await?.json().await?; raw.into_public() }
    /// `POST /v1/rerank`.
    pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse> { self.send(self.json(Method::Post, "/v1/rerank", &request)?).await?.json().await }
    /// `GET /v1/models` with optional type and search filters.
    pub async fn list_models(&self, model_type: Option<ModelType>, search: Option<&str>) -> Result<ListModelsResponse> { let mut pairs = Vec::new(); if let Some(kind) = model_type { pairs.push(("type", kind.as_str().to_string())); } if let Some(search) = search { pairs.push(("search", search.to_string())); } self.send(self.query(Method::Get, "/v1/models", &pairs)?).await?.json().await }
    /// `POST /v1/audio/speech`, returning bytes and content type.
    pub async fn speech(&self, request: SpeechRequest) -> Result<(Bytes, String)> { let fallback = speech_content_type(request.response_format.as_deref()).to_string(); let response = self.send(self.json(Method::Post, "/v1/audio/speech", &request)?).await?; let content_type = response.content_type.clone().unwrap_or(fallback); Ok((response.bytes().await?, content_type)) }
    /// `GET /v1/audio/voices?model=<id>`.
    pub async fn voices(&self, model: &str) -> Result<VoicesResponse> { self.send(self.query(Method::Get, "/v1/audio/voices", &[("model", model.to_string())])?).await?.json().await }
    /// `POST /v1/audio/transcriptions` (multipart).
    pub async fn transcribe(&self, request: TranscriptionRequest, file_name: impl Into<String>, audio: impl Into<Bytes>) -> Result<Transcription> { let multipart = transcription_form(&request, file_name.into(), audio.into()); let mut wire = self.request(Method::Post, "/v1/audio/transcriptions"); wire.body = Body::Bytes { bytes: multipart.0, content_type: multipart.1 }; let response = self.send(wire).await?; if matches!(request.response_format.as_deref(), None | Some("json") | Some("verbose_json")) { Ok(Transcription::Json(response.json().await?)) } else { Ok(Transcription::Text(response.text().await?)) } }
    /// `POST /v1/batches`.
    pub async fn create_batch(&self, request: BatchCreateRequest) -> Result<BatchHandle> { self.send(self.json(Method::Post, "/v1/batches", &request)?).await?.json().await }
    /// `GET /v1/batches/{id}`.
    pub async fn get_batch(&self, id: &str) -> Result<BatchHandle> { self.send(self.request(Method::Get, &format!("/v1/batches/{id}"))).await?.json().await }
    /// `POST /v1/batches/{id}/cancel`.
    pub async fn cancel_batch(&self, id: &str) -> Result<BatchHandle> { self.send(self.request(Method::Post, &format!("/v1/batches/{id}/cancel"))).await?.json().await }
    /// `GET /v1/batches/{id}/results` as an NDJSON stream.
    pub async fn batch_results(&self, id: &str) -> Result<impl Stream<Item = Result<BatchResultLine>>> { Ok(ndjson_lines(self.send(self.request(Method::Get, &format!("/v1/batches/{id}/results"))).await?.into_stream())) }
}

fn transcription_form(request: &TranscriptionRequest, file_name: String, audio: Bytes) -> (Vec<u8>, String) {
    const BOUNDARY: &str = "llmleaf-rust-client-boundary";
    fn field(body: &mut Vec<u8>, name: &str, value: &str) { body.extend_from_slice(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes()); }
    let mut body = Vec::new(); field(&mut body, "model", &request.model); if let Some(value) = &request.language { field(&mut body, "language", value); } if let Some(value) = &request.prompt { field(&mut body, "prompt", value); } if let Some(value) = &request.response_format { field(&mut body, "response_format", value); } if let Some(value) = request.temperature { field(&mut body, "temperature", &value.to_string()); }
    let file_name = file_name.replace('"', "%22").replace('\r', "").replace('\n', ""); body.extend_from_slice(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes()); body.extend_from_slice(&audio); body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes()); (body, format!("multipart/form-data; boundary={BOUNDARY}"))
}
