use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tachyon_rerank::backend::{
    try_new_ort_backend, Backend, BackendKind, CpuBackend, DistMetric, QnnBackend, RerankResult,
};

use std::sync::Arc;

// -------------------- helpers --------------------
fn getenv_usize(key: &str, default_: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_)
}
fn getenv_string(key: &str, default_: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default_.to_string())
}
fn parse_list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split(',')
        .filter_map(|x| x.trim().parse::<T>().ok())
        .collect()
}
fn parse_chunk_modes(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}
fn chunk_value_for_mode(mode: &str, n: usize, default_small: usize) -> usize {
    if mode.eq_ignore_ascii_case("full") {
        n
    } else {
        mode.parse::<usize>().unwrap_or(default_small)
    }
}

// -------------------- workload --------------------
#[derive(Clone)]
struct Workload {
    chunk: usize,
    metric: DistMetric,
}

fn make_vectors(
    seed: u64,
    n: usize,
    d: usize,
    normalize_for_cosine: bool,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // query
    let mut q = vec![0f32; d];
    for x in &mut q {
        *x = rng.gen::<f32>() * 2.0 - 1.0;
    }
    if normalize_for_cosine {
        let mut s = 0f64;
        for &x in &q {
            s += (x as f64) * (x as f64);
        }
        let nrm = s.sqrt();
        if nrm > 0.0 {
            for x in &mut q {
                *x /= nrm as f32;
            }
        }
    }

    // candidates
    let mut a = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = vec![0f32; d];
        for x in &mut row {
            *x = rng.gen::<f32>() * 2.0 - 1.0;
        }
        if normalize_for_cosine {
            let mut s = 0f64;
            for &x in &row {
                s += (x as f64) * (x as f64);
            }
            let nrm = s.sqrt();
            if nrm > 0.0 {
                for x in &mut row {
                    *x /= nrm as f32;
                }
            }
        }
        a.push(row);
    }
    (q, a)
}

fn bench_backend_once<B: Backend + ?Sized>(
    b: &B,
    wl: &Workload,
    q: &[f32],
    a: &[Vec<f32>],
) -> RerankResult {
    b.distance(q, a, wl.metric, Some(wl.chunk))
        .expect("distance ok")
}

fn spawn_backends(requested: &[&str]) -> Vec<(BackendKind, Arc<dyn Backend + Send + Sync>)> {
    let mut out: Vec<(BackendKind, Arc<dyn Backend + Send + Sync>)> = Vec::new();

    for &name in requested {
        match name {
            "cpu" => {
                let be: Arc<dyn Backend + Send + Sync> = Arc::new(CpuBackend);
                out.push((BackendKind::Cpu, be));
            }
            "ort" => {
                if let Ok(o) = try_new_ort_backend() {
                    let kind = o.kind();
                    let be: Arc<dyn Backend + Send + Sync> = Arc::new(o);
                    out.push((kind, be));
                } else {
                    eprintln!("[bench] ORT backend unavailable (missing model or lib); skipping.");
                }
            }
            "qnn" => {
                if let Ok(q) = QnnBackend::new() {
                    let kind = q.kind();
                    let be: Arc<dyn Backend + Send + Sync> = Arc::new(q);
                    out.push((kind, be));
                } else {
                    eprintln!(
                        "[bench] QNN backend unavailable (shim not found or not ready); skipping."
                    );
                }
            }
            _ => {}
        }
    }
    out
}

// -------------------- main bench --------------------
fn criterion_bench(c: &mut Criterion) {
    // env configs
    let dims_env = getenv_string("TACHANN_BENCH_DIMS", "768,1024,1536");
    let cands_env = getenv_string("TACHANN_BENCH_CANDS", "256,1024,4096");
    let metrics_env = getenv_string("TACHANN_BENCH_METRICS", "cosine,l2,ip");
    let backends_env = getenv_string("TACHANN_BENCH_BACKENDS", "cpu");
    let modes_env = getenv_string("TACHANN_BENCH_CHUNK_MODES", "full,1024");

    let dims: Vec<usize> = parse_list(&dims_env);
    let cands: Vec<usize> = parse_list(&cands_env);
    let metric_names: Vec<&str> = metrics_env.split(',').map(|s| s.trim()).collect();
    let backend_names: Vec<&str> = backends_env.split(',').map(|s| s.trim()).collect();
    let modes: Vec<String> = parse_chunk_modes(&modes_env);

    let default_small = getenv_usize("TACHANN_BENCH_SMALL_CHUNK", 1024);
    let seed = std::env::var("TACHANN_BENCH_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42u64);

    let to_metric = |m: &str| -> Option<DistMetric> {
        match m.to_ascii_lowercase().as_str() {
            "cosine" => Some(DistMetric::Cosine),
            "l2" => Some(DistMetric::L2),
            "ip" | "inner" | "dot" => Some(DistMetric::Ip),
            _ => None,
        }
    };
    let metrics: Vec<DistMetric> = metric_names.into_iter().filter_map(to_metric).collect();
    let backends = spawn_backends(&backend_names);

    if backends.is_empty() {
        eprintln!("[bench] No backends available.");
        return;
    }

    let mut group = c.benchmark_group("tachyon_rerank_distance");
    group.sampling_mode(SamplingMode::Auto);

    // optional CSV
    use std::io::Write as _;
    use std::path::Path;

    let csv_path = getenv_string("TACHANN_BENCH_CSV", "target/tachyon_rerank_bench.csv");
    if let Some(parent) = Path::new(&csv_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut csv = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)
        .expect("open csv");

    for &d in &dims {
        for &n in &cands {
            for &metric in &metrics {
                let normalize = matches!(metric, DistMetric::Cosine);
                let (q, a) = make_vectors(seed, n, d, normalize);
                group.throughput(Throughput::Elements((n * d) as u64));

                for mode in &modes {
                    let chunk = chunk_value_for_mode(mode, n, default_small);
                    let wl = Workload { chunk, metric };

                    for (kind, be) in &backends {
                        let name =
                            format!("{kind:?}/metric={:?}/n={}/d={}/mode={}", metric, n, d, mode);
                        group.bench_with_input(
                            BenchmarkId::new("distance", name.clone()),
                            &wl,
                            |bch, wlref| {
                                bch.iter(|| {
                                    let res = bench_backend_once(be.as_ref(), wlref, &q, &a);
                                    black_box(res.distances.len())
                                });
                            },
                        );

                        // single timing run for CSV
                        let t0 = std::time::Instant::now();
                        let _ = bench_backend_once(be.as_ref(), &wl, &q, &a);
                        let dur = t0.elapsed();
                        writeln!(
                            &mut csv,
                            "backend={:?},metric={:?},n={},d={},mode={},chunk={},micros={}",
                            kind,
                            metric,
                            n,
                            d,
                            mode,
                            chunk,
                            dur.as_micros()
                        )
                        .ok();
                    }
                }
            }
        }
    }

    group.finish();
}

criterion_group!(benches, criterion_bench);
criterion_main!(benches);
