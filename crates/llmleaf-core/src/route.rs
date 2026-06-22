//! Routing, fallback, and node-local health (SOUL.md principles 8 and 9).
//!
//! A route maps a logical model to an *ordered* list of provider targets — order is the fallback
//! chain. The engine walks it, skipping providers this node currently considers down.
//!
//! Health decisions are strictly node-local (principle 9): self-contained (from this node's own
//! observations), fast (a local map lookup/compare, never a network round-trip), and specific (a
//! penalty targets one concrete provider — never a cluster-wide mode). Nodes behind a load balancer
//! may converge on the same conclusions because they see the same upstream, never because they talk.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::config::{ProviderConfig, RouteConfig, Target};

/// Resolves logical models to ordered fallback targets. Built from config, then read-only.
#[derive(Debug, Clone, Default)]
pub struct Router {
    table: HashMap<String, Vec<Target>>,
    /// Provider namespace prefixes, longest-first so the most specific prefix wins. A request for
    /// `<prefix>/<model>` with no explicit route resolves to that provider, upstream model = the
    /// segment after the prefix. See [`ProviderConfig::prefix`].
    prefixes: Vec<(String, String)>,
}

impl Router {
    pub fn new(routes: &[RouteConfig], providers: &[ProviderConfig]) -> Self {
        let mut table = HashMap::new();
        for r in routes {
            table.insert(r.model.clone(), r.targets.clone());
        }
        let mut prefixes: Vec<(String, String)> = providers
            .iter()
            .filter_map(|p| p.prefix.clone().map(|pre| (pre, p.name.clone())))
            .collect();
        // Longest prefix first: the most specific claim wins when prefixes nest (e.g. `a` vs `a/b`).
        prefixes.sort_by_key(|(prefix, _)| std::cmp::Reverse(prefix.len()));
        Router { table, prefixes }
    }

    /// The ordered fallback chain for a logical model, or `None` if unrouted. An explicit route is
    /// returned borrowed; a prefix match synthesizes a single-target chain (the prefix is a direct
    /// address, not a fallback chain).
    pub fn resolve(&self, model: &str) -> Option<Cow<'_, [Target]>> {
        if let Some(targets) = self.table.get(model) {
            return Some(Cow::Borrowed(targets.as_slice()));
        }
        for (prefix, provider) in &self.prefixes {
            if let Some(upstream) = strip_namespace(model, prefix) {
                return Some(Cow::Owned(vec![Target {
                    provider: provider.clone(),
                    model: Some(upstream.to_string()),
                }]));
            }
        }
        None
    }

    /// Explicitly routed logical models. Prefix-addressed models are not enumerable — the core does
    /// not know a provider's catalog (principle 2) — so they are reported separately via [`Self::prefixes`].
    pub fn models(&self) -> impl Iterator<Item = &str> {
        self.table.keys().map(String::as_str)
    }

    /// The `(prefix, provider)` namespaces this router will route by prefix, longest-first.
    pub fn prefixes(&self) -> impl Iterator<Item = (&str, &str)> {
        self.prefixes.iter().map(|(p, n)| (p.as_str(), n.as_str()))
    }
}

/// If `model` is `<prefix>/<rest>` with a non-empty `rest`, return `rest`. The separator is a single
/// `/`; the prefix must be a whole leading path segment, never a substring (`oa` must not match
/// `oai/...`).
fn strip_namespace<'a>(model: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = model.strip_prefix(prefix)?.strip_prefix('/')?;
    (!rest.is_empty()).then_some(rest)
}

/// Node-local provider health. A provider that fails gets a short cooldown during which routing
/// skips it; afterwards it is eligible again (a comparison against `now`, no background task, no
/// shared state). This is the entire HA brain — deliberately small.
#[derive(Clone, Default)]
pub struct HealthTable {
    /// provider name -> unix-seconds instant before which the provider is considered down.
    cooldown_until: Arc<RwLock<HashMap<String, u64>>>,
}

impl HealthTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Penalize a provider after an observed failure: skip it until `now + cooldown_secs`.
    pub fn penalize(&self, provider: &str, now: u64, cooldown_secs: u64) {
        self.cooldown_until
            .write()
            .unwrap()
            .insert(provider.to_string(), now + cooldown_secs);
    }

    /// Is this provider currently in cooldown? A single map lookup and comparison.
    pub fn is_down(&self, provider: &str, now: u64) -> bool {
        self.cooldown_until
            .read()
            .unwrap()
            .get(provider)
            .is_some_and(|&until| now < until)
    }

    /// Clear any penalty (e.g. after a successful call). The happy path — nothing penalized — takes
    /// only a read lock and returns, so successful requests never contend on the write lock.
    pub fn clear(&self, provider: &str) {
        if !self.cooldown_until.read().unwrap().contains_key(provider) {
            return;
        }
        self.cooldown_until.write().unwrap().remove(provider);
    }

    /// A `(provider, is_down)` snapshot for the admin health surface.
    pub fn snapshot(&self, now: u64) -> Vec<(String, bool)> {
        self.cooldown_until
            .read()
            .unwrap()
            .iter()
            .map(|(p, &until)| (p.clone(), now < until))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes() -> Vec<RouteConfig> {
        vec![RouteConfig {
            model: "gpt-4o".into(),
            targets: vec![
                Target {
                    provider: "openai".into(),
                    model: Some("gpt-4o".into()),
                },
                Target {
                    provider: "echo".into(),
                    model: None,
                },
            ],
        }]
    }

    fn provider(name: &str, prefix: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            name: name.into(),
            kind: "test".into(),
            endpoint: None,
            credential: None,
            prefix: prefix.map(Into::into),
            settings: Default::default(),
            limits: None,
            model_limits: Default::default(),
        }
    }

    #[test]
    fn resolves_ordered_targets() {
        let r = Router::new(&routes(), &[]);
        let t = r.resolve("gpt-4o").unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].provider, "openai");
        assert!(r.resolve("missing").is_none());
    }

    #[test]
    fn prefix_resolves_to_provider_with_stripped_model() {
        let r = Router::new(&[], &[provider("openai-main", Some("oai"))]);
        let t = r.resolve("oai/gpt-4o").unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].provider, "openai-main");
        assert_eq!(t[0].model.as_deref(), Some("gpt-4o")); // prefix stripped
                                                           // The bare prefix, an empty model, or a substring of the prefix must not match.
        assert!(r.resolve("oai").is_none());
        assert!(r.resolve("oai/").is_none());
        assert!(r.resolve("oa/gpt-4o").is_none());
    }

    #[test]
    fn explicit_route_wins_over_prefix() {
        let r = Router::new(&routes(), &[provider("openai-main", Some("gpt-4o"))]);
        let t = r.resolve("gpt-4o").unwrap();
        assert_eq!(t.len(), 2); // the explicit fallback chain, not the synthetic prefix target
        assert_eq!(t[0].provider, "openai");
    }

    #[test]
    fn longest_prefix_wins() {
        let providers = [
            provider("broad", Some("a")),
            provider("specific", Some("a/b")),
        ];
        let r = Router::new(&[], &providers);
        let t = r.resolve("a/b/model").unwrap();
        assert_eq!(t[0].provider, "specific");
        assert_eq!(t[0].model.as_deref(), Some("model"));
    }

    #[test]
    fn health_cooldown_is_a_comparison() {
        let h = HealthTable::new();
        assert!(!h.is_down("openai", 0));
        h.penalize("openai", 100, 30);
        assert!(h.is_down("openai", 110));
        assert!(!h.is_down("openai", 130)); // cooldown elapsed
        h.clear("openai");
        assert!(!h.is_down("openai", 110));
    }
}
