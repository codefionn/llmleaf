//! Runtime-neutral HTTP transport used by the public client.

use crate::error::{Error, ErrorEnvelope, Result};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::pin::Pin;
#[cfg(feature = "compio")]
use std::time::Duration;

pub(crate) type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;
#[derive(Debug, Clone, Copy)] pub(crate) enum Method { Get, Post }
pub(crate) enum Body { Empty, Bytes { bytes: Vec<u8>, content_type: String } }
pub(crate) struct Request { pub(crate) method: Method, pub(crate) url: String, pub(crate) authorization: String, pub(crate) admin_token: Option<String>, pub(crate) body: Body }
impl Request {
    pub(crate) fn json<T: serde::Serialize>(method: Method, url: String, authorization: String, admin_token: Option<String>, value: &T) -> Result<Self> { Ok(Self { method, url, authorization, admin_token, body: Body::Bytes { bytes: serde_json::to_vec(value)?, content_type: "application/json".to_string() } }) }
    pub(crate) fn empty(method: Method, url: String, authorization: String, admin_token: Option<String>) -> Self { Self { method, url, authorization, admin_token, body: Body::Empty } }
}
pub(crate) struct Response { pub(crate) status: u16, pub(crate) reason: Option<String>, pub(crate) content_type: Option<String>, body: ByteStream }
impl Response {
    pub(crate) fn into_stream(self) -> ByteStream { self.body }
    pub(crate) async fn bytes(self) -> Result<Bytes> { let mut body = self.body; let mut bytes = Vec::new(); while let Some(chunk) = body.next().await { bytes.extend_from_slice(&chunk?); } Ok(Bytes::from(bytes)) }
    pub(crate) async fn text(self) -> Result<String> { String::from_utf8(self.bytes().await?.to_vec()).map_err(|error| Error::Http(format!("response body is not UTF-8: {error}"))) }
    pub(crate) async fn json<T: serde::de::DeserializeOwned>(self) -> Result<T> { Ok(serde_json::from_slice(&self.bytes().await?)?) }
}
#[derive(Clone)] #[allow(dead_code)] pub(crate) enum Backend { #[cfg(feature = "tokio")] Tokio { client: reqwest::Client }, #[cfg(feature = "compio")] Compio { client: cyper::Client, timeout: Option<Duration> } }
impl Backend { pub(crate) async fn send(&self, request: Request) -> Result<Response> { let response = match self { #[cfg(feature = "tokio")] Self::Tokio { client } => tokio_send(client, request).await?, #[cfg(feature = "compio")] Self::Compio { client, timeout } => compio_send(client, *timeout, request).await? }; check_status(response).await } }
async fn check_status(response: Response) -> Result<Response> { if (200..300).contains(&response.status) { return Ok(response); } let status = response.status; let fallback = response.reason.clone().unwrap_or_else(|| "request failed".to_string()); let body = response.bytes().await.unwrap_or_default(); let message = serde_json::from_slice::<ErrorEnvelope>(&body).ok().map(|e| e.error.message).filter(|m| !m.is_empty()).unwrap_or(fallback); Err(Error::Api { status, message }) }
#[cfg(feature = "tokio")]
async fn tokio_send(client: &reqwest::Client, request: Request) -> Result<Response> { let method = match request.method { Method::Get => reqwest::Method::GET, Method::Post => reqwest::Method::POST }; let mut builder = client.request(method, request.url).header(reqwest::header::AUTHORIZATION, request.authorization); if let Some(token) = request.admin_token { builder = builder.header("x-admin-token", token); } if let Body::Bytes { bytes, content_type } = request.body { builder = builder.header(reqwest::header::CONTENT_TYPE, content_type).body(bytes); } let response = builder.send().await?; let status = response.status().as_u16(); let reason = response.status().canonical_reason().map(str::to_string); let content_type = response.headers().get(reqwest::header::CONTENT_TYPE).and_then(|value| value.to_str().ok()).map(str::to_string); let body = Box::pin(response.bytes_stream().map(|chunk| chunk.map_err(Error::from))); Ok(Response { status, reason, content_type, body }) }
#[cfg(feature = "compio")]
async fn compio_send(client: &cyper::Client, timeout: Option<Duration>, request: Request) -> Result<Response> { let method = match request.method { Method::Get => http::Method::GET, Method::Post => http::Method::POST }; let mut builder = client.request(method, request.url)?.header(http::header::AUTHORIZATION, request.authorization)?; if let Some(token) = request.admin_token { builder = builder.header("x-admin-token", token)?; } if let Body::Bytes { bytes, content_type } = request.body { builder = builder.header(http::header::CONTENT_TYPE, content_type)?.body(bytes); } let response = match timeout { Some(timeout) => compio::time::timeout(timeout, builder.send()).await.map_err(|_| Error::Http("request timed out".to_string()))??, None => builder.send().await? }; let status = response.status().as_u16(); let reason = response.status().canonical_reason().map(str::to_string); let content_type = response.headers().get(http::header::CONTENT_TYPE).and_then(|value| value.to_str().ok()).map(str::to_string); let body = Box::pin(response.bytes_stream().map(|chunk| chunk.map_err(Error::from))); Ok(Response { status, reason, content_type, body }) }
