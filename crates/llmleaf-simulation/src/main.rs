//! `llmleaf-sim` — the soak runner.
//!
//! Where the `cargo test` sweep covers a fixed band of seeds, this runs seeds until a wall-clock
//! budget elapses, so a problem that only a rare scenario triggers gets found by sheer volume. Every
//! seed is independent and fully reproducible, so a failure is reported as a single reproducer:
//! `llmleaf-sim --seed <N>`.
//!
//! Usage:
//!   llmleaf-sim [DURATION] [--duration D] [--jobs N] [--start S] [--random] [--seed N]
//!
//!   DURATION / --duration   how long to run (e.g. `5m`, `30s`, `2h`, `500ms`). Default 60s.
//!   --jobs N                concurrent workers (default: available parallelism).
//!   --start S               first seed for sequential enumeration (default: random per run).
//!   --random                draw seeds at random (varied across runs) instead of enumerating.
//!   --seed N                run exactly one seed and exit — the reproducer mode.
//!
//! Examples:
//!   cargo run -p llmleaf-simulation --release -- 5m
//!   cargo run -p llmleaf-simulation --release -- --duration 1h --jobs 8
//!   cargo run -p llmleaf-simulation --release -- --seed 123456    # reproduce a reported failure

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use llmleaf_simulation::{run_seed, Rng};

#[tokio::main]
async fn main() -> ExitCode {
    let opts = match Options::parse(std::env::args().skip(1)) {
        Ok(Some(opts)) => opts,
        Ok(None) => return ExitCode::SUCCESS, // --help
        Err(e) => {
            eprintln!("llmleaf-sim: {e}\n\nrun `llmleaf-sim --help` for usage");
            return ExitCode::FAILURE;
        }
    };

    // Reproducer mode: one seed, full report, done.
    if let Some(seed) = opts.single_seed {
        println!("running single seed {seed} …");
        return match run_seed(seed).await {
            Ok(()) => {
                println!("seed {seed}: PASS");
                ExitCode::SUCCESS
            }
            Err(msg) => {
                eprintln!("seed {seed}: FAIL\n{msg}");
                ExitCode::FAILURE
            }
        };
    }

    soak(opts).await
}

async fn soak(opts: Options) -> ExitCode {
    let started = Instant::now();
    let deadline = started + opts.duration;

    let stop = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicU64::new(0));
    let next_seed = Arc::new(AtomicU64::new(opts.start));
    let failure: Arc<Mutex<Option<(u64, String)>>> = Arc::new(Mutex::new(None));

    let source = if opts.random {
        "random seeds".to_string()
    } else {
        format!("sequential seeds from {}", opts.start)
    };
    println!(
        "soak: {source} for {:?} across {} worker(s)",
        opts.duration, opts.jobs
    );

    // Stop early on Ctrl-C, reporting what ran so far.
    {
        let stop = stop.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\ninterrupted — finishing in-flight seeds …");
                stop.store(true, Ordering::SeqCst);
            }
        });
    }

    // Heartbeat: a periodic line so a long run shows it is alive and how fast it is going.
    {
        let stop = stop.clone();
        let completed = completed.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                let done = completed.load(Ordering::SeqCst);
                let elapsed = started.elapsed().as_secs_f64().max(1e-9);
                let remaining = deadline.saturating_duration_since(Instant::now());
                println!(
                    "  … {done} seeds ok · {:.0} seeds/s · {:?} left",
                    done as f64 / elapsed,
                    Duration::from_secs(remaining.as_secs())
                );
            }
        });
    }

    let mut workers = Vec::with_capacity(opts.jobs);
    for w in 0..opts.jobs {
        let stop = stop.clone();
        let completed = completed.clone();
        let next_seed = next_seed.clone();
        let failure = failure.clone();
        let random = opts.random;
        // Each random worker owns an independent SplitMix64 stream so workers don't collide; the base
        // is mixed with a fixed per-run nonce and the worker index.
        let mut local_rng =
            Rng::new(opts.random_base ^ (w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        workers.push(tokio::spawn(async move {
            loop {
                if stop.load(Ordering::SeqCst) || Instant::now() >= deadline {
                    break;
                }
                let seed = if random {
                    local_rng.next_u64()
                } else {
                    next_seed.fetch_add(1, Ordering::Relaxed)
                };
                match run_seed(seed).await {
                    Ok(()) => {
                        completed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(msg) => {
                        stop.store(true, Ordering::SeqCst);
                        let mut slot = failure.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some((seed, msg));
                        }
                        break;
                    }
                }
            }
        }));
    }

    for worker in workers {
        let _ = worker.await;
    }

    let done = completed.load(Ordering::SeqCst);
    let elapsed = started.elapsed();
    let rate = done as f64 / elapsed.as_secs_f64().max(1e-9);

    if let Some((seed, msg)) = failure.lock().unwrap().take() {
        eprintln!(
            "\nFAIL after {done} seeds in {elapsed:?} ({rate:.0} seeds/s)\n\nfailing seed {seed}:\n{msg}\n\nreproduce: llmleaf-sim --seed {seed}"
        );
        return ExitCode::FAILURE;
    }

    println!("\nPASS — {done} seeds in {elapsed:?} ({rate:.0} seeds/s), no divergence found");
    ExitCode::SUCCESS
}

struct Options {
    duration: Duration,
    jobs: usize,
    start: u64,
    random: bool,
    single_seed: Option<u64>,
    /// Per-run nonce that seeds the random workers (kept stable within a run for reproducibility of
    /// the run as a whole; varied across runs from the wall clock).
    random_base: u64,
}

impl Options {
    fn parse(args: impl Iterator<Item = String>) -> Result<Option<Self>, String> {
        let default_jobs = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // The default start is a random per-run value, so sequential enumeration explores a
        // different band each invocation instead of always re-testing seeds near 0. `--start`
        // overrides it; `random_base` reuses the same nonce so `--random` is also varied per run.
        let run_nonce = wall_clock_nonce();
        let mut opts = Options {
            duration: Duration::from_secs(60),
            jobs: default_jobs,
            start: run_nonce,
            random: false,
            single_seed: None,
            random_base: run_nonce,
        };
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_help();
                    return Ok(None);
                }
                "-d" | "--duration" => {
                    opts.duration = parse_duration(&take(&mut args, "--duration")?)?;
                }
                "-j" | "--jobs" => {
                    opts.jobs = take(&mut args, "--jobs")?
                        .parse()
                        .map_err(|_| "--jobs expects a positive integer".to_string())?;
                    if opts.jobs == 0 {
                        return Err("--jobs must be at least 1".to_string());
                    }
                }
                "--start" => {
                    opts.start = take(&mut args, "--start")?
                        .parse()
                        .map_err(|_| "--start expects a u64".to_string())?;
                }
                "--random" => opts.random = true,
                "--seed" => {
                    opts.single_seed = Some(
                        take(&mut args, "--seed")?
                            .parse()
                            .map_err(|_| "--seed expects a u64".to_string())?,
                    );
                }
                other if other.starts_with('-') => {
                    return Err(format!("unknown option {other:?}"));
                }
                // A bare positional is taken as the duration (e.g. `llmleaf-sim 5m`).
                other => opts.duration = parse_duration(other)?,
            }
        }
        Ok(Some(opts))
    }
}

fn take(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} expects a value"))
}

/// Parse `5m` / `30s` / `2h` / `500ms` / `1d` / bare seconds into a [`Duration`].
fn parse_duration(s: &str) -> Result<Duration, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty duration".to_string());
    }
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (digits, unit) = s.split_at(split);
    let n: u64 = digits
        .parse()
        .map_err(|_| format!("invalid duration {s:?}"))?;
    let dur = match unit {
        "" | "s" | "sec" | "secs" => Duration::from_secs(n),
        "ms" => Duration::from_millis(n),
        "m" | "min" | "mins" => Duration::from_secs(n * 60),
        "h" | "hr" | "hrs" => Duration::from_secs(n * 3600),
        "d" | "day" | "days" => Duration::from_secs(n * 86400),
        other => return Err(format!("unknown duration unit {other:?} in {s:?}")),
    };
    Ok(dur)
}

/// A best-effort per-run nonce from the wall clock, so `--random` explores different seeds each run.
fn wall_clock_nonce() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678)
}

fn print_help() {
    println!(
        "llmleaf-sim — seeded simulation soak runner\n\n\
         USAGE:\n  \
           llmleaf-sim [DURATION] [OPTIONS]\n\n\
         OPTIONS:\n  \
           DURATION              how long to run, e.g. 5m, 30s, 2h, 500ms (default 60s)\n  \
           -d, --duration D      same as the positional DURATION\n  \
           -j, --jobs N          concurrent workers (default: available parallelism)\n  \
            --start S             first seed for sequential enumeration (default: random per run)\n  \
           --random              draw seeds at random instead of enumerating\n  \
           --seed N              run exactly one seed and exit (reproducer mode)\n  \
           -h, --help            print this help\n\n\
         EXAMPLES:\n  \
           cargo run -p llmleaf-simulation --release -- 5m\n  \
           cargo run -p llmleaf-simulation --release -- --duration 1h --jobs 8\n  \
           cargo run -p llmleaf-simulation --release -- --seed 123456"
    );
}
