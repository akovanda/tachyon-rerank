use actix_web::{web, App, HttpServer};

use tachyon_rerank::api::{configure, AppState};
use tachyon_rerank::backend::RuntimeRouter;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize env_logger if not already
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    let router = RuntimeRouter::from_env();
    let state = web::Data::new(AppState::new(router));

    let addr = std::env::var("TACHANN_BIND").unwrap_or_else(|_| "0.0.0.0:8080".into());
    println!("tachyon-rerank listening on http://{}", &addr);
    HttpServer::new(move || App::new().app_data(state.clone()).configure(configure))
        .bind(addr)?
        .workers(num_cpus::get())
        .run()
        .await
}
