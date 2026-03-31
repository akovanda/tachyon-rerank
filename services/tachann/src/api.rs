use actix_web::{get, post, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::backend::{DistMetric, RuntimeRouter, RuntimeStatus};

#[derive(Clone)]
pub struct AppState {
    router: Arc<RuntimeRouter>,
}

impl AppState {
    pub fn new(router: RuntimeRouter) -> Self {
        Self {
            router: Arc::new(router),
        }
    }

    pub fn status(&self) -> RuntimeStatus {
        self.router.status_snapshot()
    }
}

#[derive(Deserialize, Serialize)]
pub struct ScoreRequest {
    pub q: Vec<f32>,
    pub a: Vec<Vec<f32>>,
    #[serde(default)]
    pub metric: Option<String>,
    #[serde(default)]
    pub max_batch: Option<usize>,
}

#[derive(Deserialize, Serialize)]
pub struct ScoreBatchRequest {
    pub qs: Vec<Vec<f32>>,
    pub a: Vec<Vec<f32>>,
    #[serde(default)]
    pub metric: Option<String>,
    #[serde(default)]
    pub max_batch: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct ScoreResponse {
    pub distances: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct ScoreBatchResponse {
    pub distances: Vec<Vec<f32>>,
}

fn parse_metric(metric: Option<&str>) -> Result<DistMetric, HttpResponse> {
    match metric.unwrap_or("cosine").to_lowercase().as_str() {
        "cosine" => Ok(DistMetric::Cosine),
        "l2" | "euclidean" => Ok(DistMetric::L2),
        "ip" | "inner" | "dot" => Ok(DistMetric::Ip),
        other => Err(HttpResponse::BadRequest()
            .json(serde_json::json!({ "error": format!("unknown metric: {}", other) }))),
    }
}

fn validate_candidates(a: &[Vec<f32>], d: usize) -> Result<(), HttpResponse> {
    if d == 0 || a.is_empty() {
        return Err(HttpResponse::BadRequest()
            .json(serde_json::json!({"error":"q and a must be non-empty"})));
    }
    if a.iter().any(|row| row.len() != d) {
        return Err(HttpResponse::BadRequest()
            .json(serde_json::json!({"error":"all rows in a must have len == len(q)"})));
    }
    Ok(())
}

#[get("/info")]
async fn info(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().json(state.status())
}

#[post("/score")]
async fn score(state: web::Data<AppState>, payload: web::Json<ScoreRequest>) -> impl Responder {
    let metric = match parse_metric(payload.metric.as_deref()) {
        Ok(metric) => metric,
        Err(resp) => return resp,
    };

    if let Err(resp) = validate_candidates(&payload.a, payload.q.len()) {
        return resp;
    }

    match state
        .router
        .score(&payload.q, &payload.a, metric, payload.max_batch)
    {
        Ok(r) => HttpResponse::Ok().json(ScoreResponse {
            distances: r.distances,
        }),
        Err(e) => {
            HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()}))
        }
    }
}

#[post("/score_batch")]
async fn score_batch(
    state: web::Data<AppState>,
    payload: web::Json<ScoreBatchRequest>,
) -> impl Responder {
    let metric = match parse_metric(payload.metric.as_deref()) {
        Ok(metric) => metric,
        Err(resp) => return resp,
    };

    if payload.qs.is_empty() {
        return HttpResponse::BadRequest()
            .json(serde_json::json!({"error":"qs must be non-empty"}));
    }
    let d = payload.qs[0].len();
    if payload.qs.iter().any(|q| q.len() != d) {
        return HttpResponse::BadRequest()
            .json(serde_json::json!({"error":"all queries in qs must have the same dimension"}));
    }
    if let Err(resp) = validate_candidates(&payload.a, d) {
        return resp;
    }

    match state
        .router
        .score_batch(&payload.qs, &payload.a, metric, payload.max_batch)
    {
        Ok(results) => HttpResponse::Ok().json(ScoreBatchResponse {
            distances: results.into_iter().map(|r| r.distances).collect(),
        }),
        Err(e) => {
            HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()}))
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(info).service(score).service(score_batch);
}

#[cfg(test)]
mod tests {
    use actix_web::{http::StatusCode, test, web, App};
    use serde_json::Value;

    use super::{
        configure, AppState, ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse,
    };
    use crate::backend::RuntimeRouter;

    fn cpu_state() -> web::Data<AppState> {
        web::Data::new(AppState::new(RuntimeRouter::forced_cpu()))
    }

    #[actix_web::test]
    async fn info_returns_runtime_status() {
        let app = test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

        let req = test::TestRequest::get().uri("/info").to_request();
        let resp: Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp["requested_mode"], "cpu");
        assert_eq!(resp["actual_backend"], "Cpu");
        assert_eq!(resp["route_reason"], "forced cpu");
    }

    #[actix_web::test]
    async fn score_returns_distances() {
        let app = test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

        let req = test::TestRequest::post()
            .uri("/score")
            .set_json(ScoreRequest {
                q: vec![1.0, 0.0],
                a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                metric: Some("ip".into()),
                max_batch: Some(16),
            })
            .to_request();
        let resp: ScoreResponse = test::call_and_read_body_json(&app, req).await;

        assert_eq!(resp.distances.len(), 2);
        assert!(resp.distances[0] < resp.distances[1]);
    }

    #[actix_web::test]
    async fn score_batch_returns_distances_for_each_query() {
        let app = test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

        let req = test::TestRequest::post()
            .uri("/score_batch")
            .set_json(ScoreBatchRequest {
                qs: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                metric: Some("ip".into()),
                max_batch: Some(16),
            })
            .to_request();
        let resp: ScoreBatchResponse = test::call_and_read_body_json(&app, req).await;

        assert_eq!(resp.distances.len(), 2);
        assert_eq!(resp.distances[0].len(), 2);
        assert_eq!(resp.distances[1].len(), 2);
        assert!(resp.distances[0][0] < resp.distances[0][1]);
        assert!(resp.distances[1][1] < resp.distances[1][0]);
    }

    #[actix_web::test]
    async fn score_rejects_unknown_metric() {
        let app = test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

        let req = test::TestRequest::post()
            .uri("/score")
            .set_json(ScoreRequest {
                q: vec![1.0, 0.0],
                a: vec![vec![1.0, 0.0]],
                metric: Some("bad".into()),
                max_batch: None,
            })
            .to_request();
        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn score_rejects_mismatched_dimensions() {
        let app = test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

        let req = test::TestRequest::post()
            .uri("/score")
            .set_json(ScoreRequest {
                q: vec![1.0, 0.0],
                a: vec![vec![1.0]],
                metric: Some("cosine".into()),
                max_batch: None,
            })
            .to_request();
        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn score_batch_rejects_mismatched_query_dimensions() {
        let app = test::init_service(App::new().app_data(cpu_state()).configure(configure)).await;

        let req = test::TestRequest::post()
            .uri("/score_batch")
            .set_json(ScoreBatchRequest {
                qs: vec![vec![1.0, 0.0], vec![1.0]],
                a: vec![vec![1.0, 0.0]],
                metric: Some("cosine".into()),
                max_batch: None,
            })
            .to_request();
        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
