use serde::Deserialize;
use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use tachyon_rerank::backend::DistMetric;

#[derive(Debug, Deserialize, Clone)]
pub struct ExampleFixture {
    pub metric: String,
    pub corpus: Vec<Candidate>,
    pub queries: Vec<Query>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Candidate {
    pub id: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Query {
    pub id: String,
    pub embedding: Vec<f32>,
    pub expected_order: Vec<String>,
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn example_names() -> [&'static str; 3] {
    ["semantic_search", "code_search", "rag_rerank"]
}

pub fn load_fixture(name: &str) -> ExampleFixture {
    let path = repo_root().join("examples").join(name).join("example.json");
    serde_json::from_str(&fs::read_to_string(path).expect("example fixture"))
        .expect("valid example fixture")
}

pub fn metric_from_str(metric: &str) -> DistMetric {
    match metric {
        "cosine" => DistMetric::Cosine,
        "ip" => DistMetric::Ip,
        "l2" | "euclidean" => DistMetric::L2,
        other => panic!("unsupported fixture metric: {other}"),
    }
}

pub fn corpus_matrix(fixture: &ExampleFixture) -> Vec<Vec<f32>> {
    fixture
        .corpus
        .iter()
        .map(|item| item.embedding.clone())
        .collect()
}

pub fn sort_ids_by_distances(corpus: &[Candidate], distances: &[f32]) -> Vec<String> {
    let mut ranked: Vec<_> = corpus
        .iter()
        .map(|item| item.id.clone())
        .zip(distances.iter().copied())
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    ranked.into_iter().map(|(id, _)| id).collect()
}

pub fn reference_order(fixture: &ExampleFixture, query: &Query) -> Vec<String> {
    let metric = metric_from_str(&fixture.metric);
    let distances = fixture
        .corpus
        .iter()
        .map(|candidate| reference_distance(&query.embedding, &candidate.embedding, metric))
        .collect::<Vec<_>>();
    sort_ids_by_distances(&fixture.corpus, &distances)
}

pub fn assert_fixture_orders_match_reference(name: &str, fixture: &ExampleFixture) {
    for query in &fixture.queries {
        assert_eq!(
            reference_order(fixture, query),
            query.expected_order,
            "fixture {name} query {} expected_order does not match the reference scorer",
            query.id
        );
    }
}

fn reference_distance(query: &[f32], candidate: &[f32], metric: DistMetric) -> f32 {
    assert_eq!(
        query.len(),
        candidate.len(),
        "reference scorer dimension mismatch"
    );

    match metric {
        DistMetric::Ip => {
            let dot = query
                .iter()
                .zip(candidate.iter())
                .map(|(q, a)| f64::from(*q) * f64::from(*a))
                .sum::<f64>();
            (-dot) as f32
        }
        DistMetric::L2 => query
            .iter()
            .zip(candidate.iter())
            .map(|(q, a)| {
                let diff = f64::from(*q) - f64::from(*a);
                diff * diff
            })
            .sum::<f64>() as f32,
        DistMetric::Cosine => {
            let dot = query
                .iter()
                .zip(candidate.iter())
                .map(|(q, a)| f64::from(*q) * f64::from(*a))
                .sum::<f64>();
            let q_norm = query
                .iter()
                .map(|v| {
                    let val = f64::from(*v);
                    val * val
                })
                .sum::<f64>()
                .sqrt();
            let a_norm = candidate
                .iter()
                .map(|v| {
                    let val = f64::from(*v);
                    val * val
                })
                .sum::<f64>()
                .sqrt();
            (1.0 - (dot / (q_norm * a_norm))) as f32
        }
    }
}
