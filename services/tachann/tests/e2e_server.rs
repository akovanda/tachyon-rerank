mod common;

use anyhow::{anyhow, Context, Result};
use common::{
    assert_fixture_orders_match_reference, corpus_matrix, example_names, load_fixture,
    reference_order, sort_ids_by_distances,
};
use once_cell::sync::Lazy;
use reqwest::blocking::Client;
use reqwest::StatusCode;
use serde_json::Value;
use std::fs::{self, File};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tachyon_rerank::api::{ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse};

static SERVER_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

const STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(100);

struct TestServer {
    child: Child,
    base_url: String,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl TestServer {
    fn get_json<T: serde::de::DeserializeOwned>(&self, client: &Client, path: &str) -> Result<T> {
        client
            .get(format!("{}{}", self.base_url, path))
            .send()
            .with_context(|| format!("GET {path}"))?
            .error_for_status()
            .with_context(|| format!("GET {path} returned error status"))?
            .json()
            .with_context(|| format!("decode GET {path} body"))
    }

    fn post_json<S: serde::Serialize, T: serde::de::DeserializeOwned>(
        &self,
        client: &Client,
        path: &str,
        body: &S,
    ) -> Result<T> {
        client
            .post(format!("{}{}", self.base_url, path))
            .json(body)
            .send()
            .with_context(|| format!("POST {path}"))?
            .error_for_status()
            .with_context(|| format!("POST {path} returned error status"))?
            .json()
            .with_context(|| format!("decode POST {path} body"))
    }
}

#[test]
fn cpu_server_e2e_matches_proven_fixture_rankings() -> Result<()> {
    let _guard = SERVER_LOCK.lock().unwrap();
    let server = start_server(&[
        ("TACHANN_BACKEND".to_string(), "cpu".to_string()),
        ("RUST_LOG".to_string(), "error".to_string()),
    ])?;
    let client = http_client()?;

    let info: Value = server.get_json(&client, "/info")?;
    assert_eq!(info["requested_mode"], "cpu");
    assert_eq!(info["actual_backend"], "Cpu");
    assert_eq!(info["route_reason"], "forced cpu");

    for name in example_names() {
        let fixture = load_fixture(name);
        assert_fixture_orders_match_reference(name, &fixture);
        let corpus = corpus_matrix(&fixture);

        let batch_resp: ScoreBatchResponse = server.post_json(
            &client,
            "/score_batch",
            &ScoreBatchRequest {
                qs: fixture
                    .queries
                    .iter()
                    .map(|q| q.embedding.clone())
                    .collect(),
                a: corpus.clone(),
                metric: Some(fixture.metric.clone()),
                max_batch: Some(32),
            },
        )?;

        assert_eq!(
            batch_resp.distances.len(),
            fixture.queries.len(),
            "fixture {name}"
        );
        for (query, distances) in fixture.queries.iter().zip(batch_resp.distances.iter()) {
            assert_eq!(
                sort_ids_by_distances(&fixture.corpus, distances),
                query.expected_order,
                "fixture {name} query {}",
                query.id
            );
            assert_eq!(
                sort_ids_by_distances(&fixture.corpus, distances),
                reference_order(&fixture, query),
                "fixture {name} query {}",
                query.id
            );
        }

        let single_resp: ScoreResponse = server.post_json(
            &client,
            "/score",
            &ScoreRequest {
                q: fixture.queries[0].embedding.clone(),
                a: corpus,
                metric: Some(fixture.metric.clone()),
                max_batch: Some(32),
            },
        )?;

        assert_eq!(
            single_resp.distances.len(),
            batch_resp.distances[0].len(),
            "fixture {name}"
        );
        for (single, batch) in single_resp
            .distances
            .iter()
            .zip(batch_resp.distances[0].iter())
        {
            assert!((single - batch).abs() <= 1e-6, "fixture {name}");
        }
    }

    Ok(())
}

#[test]
fn cpu_server_rejects_invalid_input_without_crashing() -> Result<()> {
    let _guard = SERVER_LOCK.lock().unwrap();
    let server = start_server(&[
        ("TACHANN_BACKEND".to_string(), "cpu".to_string()),
        ("RUST_LOG".to_string(), "error".to_string()),
    ])?;
    let client = http_client()?;

    let resp = client
        .post(format!("{}/score", server.base_url))
        .json(&ScoreRequest {
            q: vec![1.0, 0.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            metric: Some("ip".into()),
            max_batch: Some(0),
        })
        .send()?;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let info: Value = server.get_json(&client, "/info")?;
    assert_eq!(info["actual_backend"], "Cpu");

    Ok(())
}

#[test]
fn optional_qnn_server_e2e_matches_proven_batch_fixture() -> Result<()> {
    if !optional_accel_tests_enabled() {
        eprintln!(
            "optional accelerator tests disabled; set TACHANN_RUN_OPTIONAL_ACCEL_TESTS=1 to run"
        );
        return Ok(());
    }

    for key in [
        "QNN_SDK_ROOT",
        "TACHANN_QNN_LIB",
        "ADSP_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ] {
        assert!(
            std::env::var(key).is_ok(),
            "{key} must be set when TACHANN_RUN_OPTIONAL_ACCEL_TESTS=1"
        );
    }
    assert!(
        Path::new("/dev/adsprpc-smd").exists(),
        "/dev/adsprpc-smd is required"
    );
    assert!(Path::new("/dev/ion").exists(), "/dev/ion is required");

    let _guard = SERVER_LOCK.lock().unwrap();
    let server = start_server(&[
        ("TACHANN_BACKEND".to_string(), "auto".to_string()),
        ("TACHANN_QNN_FP16".to_string(), "0".to_string()),
        ("TACHANN_QNN_STATIC_A".to_string(), "1".to_string()),
        ("RUST_LOG".to_string(), "error".to_string()),
    ])?;
    let client = http_client()?;

    let info: Value = server.get_json(&client, "/info")?;
    assert_eq!(info["requested_mode"], "auto");
    assert_eq!(
        info["actual_backend"], "Qnn",
        "expected QNN backend, got {info}"
    );

    let fixture = load_fixture("code_search");
    assert_fixture_orders_match_reference("code_search", &fixture);
    let corpus = corpus_matrix(&fixture);

    let resp: ScoreBatchResponse = server.post_json(
        &client,
        "/score_batch",
        &ScoreBatchRequest {
            qs: fixture
                .queries
                .iter()
                .map(|q| q.embedding.clone())
                .collect(),
            a: corpus,
            metric: Some(fixture.metric.clone()),
            max_batch: Some(32),
        },
    )?;

    assert_eq!(resp.distances.len(), fixture.queries.len());
    for (query, distances) in fixture.queries.iter().zip(resp.distances.iter()) {
        assert_eq!(
            sort_ids_by_distances(&fixture.corpus, distances),
            query.expected_order,
            "query {}",
            query.id
        );
        assert_eq!(
            sort_ids_by_distances(&fixture.corpus, distances),
            reference_order(&fixture, query),
            "query {}",
            query.id
        );
    }

    Ok(())
}

fn start_server(envs: &[(String, String)]) -> Result<TestServer> {
    let client = http_client()?;
    let binary = binary_path()?;

    let mut last_error = None;
    for attempt in 1..=3 {
        let port = pick_unused_port()?;
        let bind = format!("127.0.0.1:{port}");
        let base_url = format!("http://{bind}");
        let log_path = std::env::temp_dir().join(format!("tachyon-rerank-e2e-{port}.log"));
        let log = File::create(&log_path)
            .with_context(|| format!("create server log file {}", log_path.display()))?;
        let log_err = log
            .try_clone()
            .with_context(|| format!("clone server log handle {}", log_path.display()))?;

        let mut cmd = Command::new(&binary);
        cmd.env("TACHANN_BIND", &bind)
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));
        for (key, value) in envs {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {}", binary.display()))?;
        let deadline = Instant::now() + STARTUP_TIMEOUT;

        loop {
            if let Some(status) = child.try_wait().context("poll child status")? {
                let log = read_log(&log_path);
                last_error = Some(anyhow!(
                    "server exited during startup on attempt {attempt} with status {status}\n{}",
                    log
                ));
                break;
            }

            match client.get(format!("{base_url}/info")).send() {
                Ok(resp) if resp.status().is_success() => {
                    return Ok(TestServer { child, base_url });
                }
                Ok(_) | Err(_) if Instant::now() < deadline => {
                    thread::sleep(POLL_INTERVAL);
                }
                Ok(resp) => {
                    let log = read_log(&log_path);
                    let _ = child.kill();
                    let _ = child.wait();
                    last_error = Some(anyhow!(
                        "server did not become ready on attempt {attempt}; last HTTP status: {}\n{}",
                        resp.status(),
                        log
                    ));
                    break;
                }
                Err(err) => {
                    if Instant::now() < deadline {
                        thread::sleep(POLL_INTERVAL);
                    } else {
                        let log = read_log(&log_path);
                        let _ = child.kill();
                        let _ = child.wait();
                        last_error = Some(anyhow!(
                            "server did not become ready on attempt {attempt}: {err}\n{}",
                            log
                        ));
                        break;
                    }
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("failed to start server")))
}

fn pick_unused_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").context("bind ephemeral port")?;
    let port = listener
        .local_addr()
        .context("lookup ephemeral port")?
        .port();
    drop(listener);
    Ok(port)
}

fn binary_path() -> Result<PathBuf> {
    for key in [
        "CARGO_BIN_EXE_tachyon-rerank",
        "CARGO_BIN_EXE_tachyon_rerank",
    ] {
        if let Some(path) = std::env::var_os(key) {
            return Ok(PathBuf::from(path));
        }
    }

    let fallback = common::repo_root().join("target/debug/tachyon-rerank");
    if fallback.exists() {
        return Ok(fallback);
    }

    Err(anyhow!(
        "missing Cargo binary path env and fallback binary does not exist: {}",
        fallback.display()
    ))
}

fn read_log(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|err| format!("unable to read {}: {err}", path.display()))
}

fn http_client() -> Result<Client> {
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("build HTTP client")
}

fn optional_accel_tests_enabled() -> bool {
    std::env::var("TACHANN_RUN_OPTIONAL_ACCEL_TESTS")
        .ok()
        .as_deref()
        == Some("1")
}
