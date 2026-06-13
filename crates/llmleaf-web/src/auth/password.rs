//! Master-password verification. The stored hash is a crypt(3) MCF string (bcrypt `$2*$`, or a
//! `$1$/$5$/$6$` shadow hash) — the same scheme and verifier the core uses for consumer keys, so an
//! operator can generate the hash with the familiar `htpasswd -bnBC 12`.

/// Constant-work verification (the KDF dominates), conflating every failure into `false`.
pub fn verify(password: &str, hash: &str) -> bool {
    pwhash::unix::verify(password, hash)
}

/// Hash a freshly-issued consumer-key password at bcrypt cost 12 (`$2y$`), matching the core's
/// `[[keys]].pw_hash` expectations.
pub fn hash_consumer_password(password: &str) -> Result<String, pwhash::error::Error> {
    pwhash::bcrypt::hash_with(
        pwhash::bcrypt::BcryptSetup {
            salt: None,
            cost: Some(12),
            variant: Some(pwhash::bcrypt::BcryptVariant::V2y),
        },
        password,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_roundtrip() {
        let hash = hash_consumer_password("s3cret").unwrap();
        assert!(hash.starts_with("$2y$12$"));
        assert!(verify("s3cret", &hash));
        assert!(!verify("wrong", &hash));
    }
}
