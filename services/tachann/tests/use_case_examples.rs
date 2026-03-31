use actix_web::test as aw_test;
use actix_web::{web, App};
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use tachyon_rerank::api::{
    configure, AppState, ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse,
};
use tachyon_rerank::backend::{DistMetric, RuntimeRouter};

static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Debug, Deserialize)]
struct ExampleFixture {
    metric: String,
    corpus: Vec<Candidate>,
    queries: Vec<Query>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    id: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Query {
    id: String,
    embedding: Vec<f32>,
    expected_order: Vec<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn example_names() -> [&'static str; 3] {
    ["semantic_search", "code_search", "rag_rerank"]
}

fn load_fixture(name: &str) -> ExampleFixture {
    let path = repo_root().join("examples").join(name).join("example.json");
    serde_json::from_str(&fs::read_to_string(path).expect("example fixture"))
        .expect("valid example fixture")
}

fn metric_from_str(metric: &str) -> DistMetric {
    match metric {
        "cosine" => DistMetric::Cosine,
        "ip" => DistMetric::Ip,
        "l2" | "euclidean" => DistMetric::L2,
        other => panic!("unsupported fixture metric: {other}"),
    }
}

fn cpu_state() -> web::Data<AppState> {
    web::Data::new(AppState::new(RuntimeRouter::forced_cpu()))
}

fn sorted_ids(corpus: &[Candidate], distances: &[f32]) -> Vec<String> {
    let mut ranked: Vec<_> = corpus
        .iter()
        .map(|item| item.id.clone())
        .zip(distances.iter().copied())
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    ranked.into_iter().map(|(id, _)| id).collect()
}

fn corpus_matrix(fixture: &ExampleFixture) -> Vec<Vec<f32>> {
    fixture
        .corpus
        .iter()
        .map(|item| item.embedding.clone())
        .collect()
}

fn with_env_var<T>(key: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let original = std::env::var(key).ok();
    match value {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
    let out = f();
    match original {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
    out
}

fn optional_accel_tests_enabled() -> bool {
    std::env::var("TACHANN_RUN_OPTIONAL_ACCEL_TESTS")
        .ok()
        .as_deref()
        == Some("1")
}

#[actix_web::test]
async fn example_fixtures_match_expected_rankings() {
    let app = aw_test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

    for name in example_names() {
        let fixture = load_fixture(name);
        let req = aw_test::TestRequest::post()
            .uri("/score_batch")
            .set_json(ScoreBatchRequest {
                qs: fixture
                    .queries
                    .iter()
                    .map(|q| q.embedding.clone())
                    .collect(),
                a: corpus_matrix(&fixture),
                metric: Some(fixture.metric.clone()),
                max_batch: Some(32),
            })
            .to_request();
        let resp: ScoreBatchResponse = aw_test::call_and_read_body_json(&app, req).await;

        assert_eq!(
            resp.distances.len(),
            fixture.queries.len(),
            "fixture {name}"
        );
        for (query, distances) in fixture.queries.iter().zip(resp.distances.iter()) {
            assert_eq!(
                sorted_ids(&fixture.corpus, distances),
                query.expected_order,
                "fixture {name} query {}",
                query.id
            );
        }
    }
}

#[actix_web::test]
async fn score_and_score_batch_agree_for_example_fixtures() {
    let app = aw_test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

    for name in example_names() {
        let fixture = load_fixture(name);
        let corpus = corpus_matrix(&fixture);

        let batch_req = aw_test::TestRequest::post()
            .uri("/score_batch")
            .set_json(ScoreBatchRequest {
                qs: fixture
                    .queries
                    .iter()
                    .map(|q| q.embedding.clone())
                    .collect(),
                a: corpus.clone(),
                metric: Some(fixture.metric.clone()),
                max_batch: Some(32),
            })
            .to_request();
        let batch_resp: ScoreBatchResponse =
            aw_test::call_and_read_body_json(&app, batch_req).await;

        for (idx, query) in fixture.queries.iter().enumerate() {
            let single_req = aw_test::TestRequest::post()
                .uri("/score")
                .set_json(ScoreRequest {
                    q: query.embedding.clone(),
                    a: corpus.clone(),
                    metric: Some(fixture.metric.clone()),
                    max_batch: Some(32),
                })
                .to_request();
            let single_resp: ScoreResponse =
                aw_test::call_and_read_body_json(&app, single_req).await;

            assert_eq!(single_resp.distances.len(), batch_resp.distances[idx].len());
            for (lhs, rhs) in single_resp
                .distances
                .iter()
                .zip(batch_resp.distances[idx].iter())
            {
                assert!(
                    (lhs - rhs).abs() <= 1e-6,
                    "fixture {name} query {}",
                    query.id
                );
            }
        }
    }
}

#[test]
fn request_payloads_match_example_fixtures() {
    for name in example_names() {
        let fixture = load_fixture(name);
        let dir = repo_root().join("examples").join(name);
        let corpus = corpus_matrix(&fixture);

        let score_path = dir.join("request.score.json");
        if score_path.exists() {
            let payload: ScoreRequest =
                serde_json::from_str(&fs::read_to_string(score_path).expect("score request"))
                    .expect("valid score request");
            assert_eq!(payload.q, fixture.queries[0].embedding, "fixture {name}");
            assert_eq!(payload.a, corpus, "fixture {name}");
            assert_eq!(payload.metric.as_deref(), Some(fixture.metric.as_str()));
        }

        let batch_path = dir.join("request.score_batch.json");
        if batch_path.exists() {
            let payload: ScoreBatchRequest =
                serde_json::from_str(&fs::read_to_string(batch_path).expect("score batch request"))
                    .expect("valid score batch request");
            assert_eq!(
                payload.qs,
                fixture
                    .queries
                    .iter()
                    .map(|query| query.embedding.clone())
                    .collect::<Vec<_>>(),
                "fixture {name}"
            );
            assert_eq!(payload.a, corpus, "fixture {name}");
            assert_eq!(payload.metric.as_deref(), Some(fixture.metric.as_str()));
        }
    }
}

#[test]
fn optional_ort_matches_cpu_for_semantic_fixture() {
    if !optional_accel_tests_enabled() {
        eprintln!(
            "optional accelerator tests disabled; set TACHANN_RUN_OPTIONAL_ACCEL_TESTS=1 to run"
        );
        return;
    }

    let _guard = ENV_LOCK.lock().unwrap();
    assert!(
        std::env::var("ORT_DYLIB_PATH").is_ok(),
        "ORT_DYLIB_PATH must be set when TACHANN_RUN_OPTIONAL_ACCEL_TESTS=1"
    );

    let fixture = load_fixture("semantic_search");
    let query = fixture.queries[0].embedding.clone();
    let corpus = corpus_matrix(&fixture);
    let metric = metric_from_str(&fixture.metric);

    let cpu = RuntimeRouter::forced_cpu();
    let cpu_res = cpu.score(&query, &corpus, metric, Some(32)).unwrap();

    let models_dir = repo_root().join("models");
    let models_dir = models_dir.to_string_lossy().to_string();
    let ort = with_env_var("MODELS_DIR", Some(&models_dir), || {
        with_env_var("TACHANN_ORT_EP", Some("cpu"), || {
            with_env_var("TACHANN_BACKEND", Some("ort"), RuntimeRouter::from_env)
        })
    });
    let status = ort.status_snapshot();
    assert!(
        matches!(status.actual_backend.as_str(), "OrtCpu" | "OrtQnn"),
        "expected ORT backend, got {} ({:?})",
        status.actual_backend,
        status.fallback_reason
    );
    let ort_res = ort.score(&query, &corpus, metric, Some(32)).unwrap();

    assert_eq!(cpu_res.distances.len(), ort_res.distances.len());
    for (cpu_val, ort_val) in cpu_res.distances.iter().zip(ort_res.distances.iter()) {
        assert!((cpu_val - ort_val).abs() <= 1e-4);
    }
}

#[test]
fn optional_qnn_matches_cpu_for_semantic_fixture() {
    if !optional_accel_tests_enabled() {
        eprintln!(
            "optional accelerator tests disabled; set TACHANN_RUN_OPTIONAL_ACCEL_TESTS=1 to run"
        );
        return;
    }

    let _guard = ENV_LOCK.lock().unwrap();
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
        std::path::Path::new("/dev/adsprpc-smd").exists(),
        "/dev/adsprpc-smd is required"
    );
    assert!(
        std::path::Path::new("/dev/ion").exists(),
        "/dev/ion is required"
    );

    let fixture = load_fixture("semantic_search");
    let query = fixture.queries[0].embedding.clone();
    let corpus = corpus_matrix(&fixture);
    let metric = metric_from_str(&fixture.metric);

    let cpu = RuntimeRouter::forced_cpu();
    let cpu_res = cpu.score(&query, &corpus, metric, Some(32)).unwrap();

    let qnn = with_env_var("TACHANN_BACKEND", Some("qnn"), RuntimeRouter::from_env);
    let status = qnn.status_snapshot();
    assert_eq!(
        status.actual_backend, "Qnn",
        "expected QNN backend, got {} ({:?})",
        status.actual_backend, status.fallback_reason
    );
    let qnn_res = qnn.score(&query, &corpus, metric, Some(32)).unwrap();

    assert_eq!(cpu_res.distances.len(), qnn_res.distances.len());
    for (cpu_val, qnn_val) in cpu_res.distances.iter().zip(qnn_res.distances.iter()) {
        assert!((cpu_val - qnn_val).abs() <= 1e-2);
    }
}
