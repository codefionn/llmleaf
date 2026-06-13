//! Read-only client for the core's admin GETs (`/admin/routes`, `/admin/health`). This is the ONLY way
//! the web app calls *into* the core, and it is strictly observational (SOUL.md: "Its only calls into
//! the core are the read-only admin GETs. It has no privileged backdoor."). A failure here never affects
//! anything the web app serves — it just shows the core as unreachable.

use serde::Deserialize;

use crate::config::CoreSection;
use crate::dto::{CoreStatus, HealthRow, RoutesView};

#[derive(Deserialize)]
struct HealthEnvelope {
    #[serde(default)]
    providers: Vec<HealthRow>,
}

/// Probe the configured core's admin surface. Returns a [`CoreStatus`] that is always safe to render:
/// on any error (`no config`, network, non-2xx, bad JSON) `reachable` is false and `error` explains why.
pub async fn fetch_status(http: &reqwest::Client, core: Option<&CoreSection>) -> CoreStatus {
    let Some(core) = core else {
        return CoreStatus {
            reachable: false,
            error: Some("no [core] admin_url configured".into()),
            ..Default::default()
        };
    };
    let base = core.admin_url.trim_end_matches('/');
    let token = core.admin_token.as_ref().and_then(|s| s.resolve());

    let routes: RoutesView =
        match get_json(http, &format!("{base}/admin/routes"), token.as_deref()).await {
            Ok(v) => v,
            Err(e) => {
                return CoreStatus {
                    reachable: false,
                    error: Some(e),
                    ..Default::default()
                }
            }
        };
    let health: HealthEnvelope =
        match get_json(http, &format!("{base}/admin/health"), token.as_deref()).await {
            Ok(v) => v,
            Err(e) => {
                return CoreStatus {
                    reachable: false,
                    routes,
                    error: Some(e),
                    ..Default::default()
                }
            }
        };

    CoreStatus {
        reachable: true,
        routes,
        health: health.providers,
        error: None,
    }
}

async fn get_json<T: serde::de::DeserializeOwned>(
    http: &reqwest::Client,
    url: &str,
    admin_token: Option<&str>,
) -> Result<T, String> {
    let mut req = http.get(url).timeout(std::time::Duration::from_secs(5));
    if let Some(tok) = admin_token {
        req = req.header("x-admin-token", tok);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("core returned {status} for {url}"));
    }
    resp.json::<T>().await.map_err(|e| e.to_string())
}
