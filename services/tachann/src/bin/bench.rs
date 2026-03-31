use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use tachyon_rerank::backend::{self, Backend, BackendKind, DistMetric};

#[derive(Debug, Clone, ValueEnum)]
enum Metric {
    Cosine,
    L2,
    Ip,
}
impl From<Metric> for DistMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::Cosine => DistMetric::Cosine,
            Metric::L2 => DistMetric::L2,
            Metric::Ip => DistMetric::Ip,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum WhichBackend {
    Auto,
    Cpu,
    Qnn,
    Ort,
    OrtQnn,
}

#[derive(Parser, Debug)]
#[command(
    name = "tachyon-rerank-bench",
    about = "In-process ANN microbench for CPU/QNN/ORT"
)]
struct Args {
    /// Compare these backends; default stays CPU-only for portability, but QNN/auto are supported on hardware
    #[arg(long="backend", value_enum, num_args=1.., default_values_t=[WhichBackend::Cpu])]
    backends: Vec<WhichBackend>,

    /// Rows (candidates)
    #[arg(long, default_value_t = 20000)]
    n: usize,

    /// Dimensionality
    #[arg(long, default_value_t = 768)]
    d: usize,

    /// Max batch for distance()
    #[arg(long, default_value_t = 4096)]
    batch: usize,

    /// Warmup iterations (not measured)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Measured iterations
    #[arg(long, default_value_t = 10)]
    iters: usize,

    /// RNG seed (reproducible data)
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of queries per iteration (batch)
    #[arg(long, default_value_t = 1)]
    q_batch: usize,

    /// Metric to use
    #[arg(long, value_enum, default_value_t = Metric::Cosine)]
    metric: Metric,

    /// Cap Rayon threads for CPU backend
    #[arg(long)]
    rayon_threads: Option<usize>,

    /// Print a few distances from the first run
    #[arg(long)]
    print_sample: bool,
}

#[derive(Clone)]
struct Stats {
    p50_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    rows_per_sec_p50: f64,
    est_gops_p50: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

#[allow(clippy::too_many_arguments)]
fn bench_one(
    label: &str,
    backend: &(dyn Backend + Send + Sync), // <-- trait object (no Sized bound)
    qs: &[Vec<f32>],
    a: &[Vec<f32>],
    metric: DistMetric,
    batch: usize,
    warmup: usize,
    iters: usize,
    print_sample: bool,
) -> anyhow::Result<Stats> {
    // warmup
    for _ in 0..warmup {
        if qs.len() == 1 {
            let _ = backend.distance(&qs[0], a, metric, Some(batch))?;
        } else {
            let _ = backend.distance_batch(qs, a, metric, Some(batch))?;
        }
    }

    let mut times = Vec::with_capacity(iters);
    let mut printed = false;
    for _ in 0..iters {
        let t0 = Instant::now();
        let res = if qs.len() == 1 {
            vec![backend.distance(&qs[0], a, metric, Some(batch))?]
        } else {
            backend.distance_batch(qs, a, metric, Some(batch))?
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        times.push(ms);

        if print_sample && !printed {
            let sample: Vec<f32> = res[0].distances.iter().cloned().take(8).collect();
            println!("# {label} sample distances: {:?}", sample);
            printed = true;
        }
    }

    let mut s = times.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = s.iter().copied().sum::<f64>() / s.len().max(1) as f64;
    let p50 = percentile(&s, 0.50);
    let p90 = percentile(&s, 0.90);
    let p95 = percentile(&s, 0.95);
    let p99 = percentile(&s, 0.99);

    // Approx ops ~2*d per row (IP baseline).
    let d = qs[0].len() as f64;
    let n = a.len() as f64;
    let b = qs.len() as f64;
    let ops_total = 2.0 * d * n * b;
    let rows_per_sec = (n * b) / (p50 / 1e3);
    let gops_per_sec = (ops_total / (p50 / 1e3)) / 1e9;

    Ok(Stats {
        p50_ms: p50,
        p90_ms: p90,
        p95_ms: p95,
        p99_ms: p99,
        mean_ms: mean,
        rows_per_sec_p50: rows_per_sec,
        est_gops_p50: gops_per_sec,
    })
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    let args = Args::parse();
    if let Some(t) = args.rayon_threads {
        std::env::set_var("RAYON_NUM_THREADS", t.to_string());
    }

    // Generate one dataset shared across backends
    let n = args.n;
    let d = args.d;
    let mut rng = StdRng::seed_from_u64(args.seed);
    if args.q_batch == 0 {
        return Err(anyhow::anyhow!("q_batch must be >= 1"));
    }
    let q: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let mut qs: Vec<Vec<f32>> = Vec::with_capacity(args.q_batch);
    for i in 0..args.q_batch {
        if i == 0 {
            qs.push(q.clone());
        } else {
            let qi: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            qs.push(qi);
        }
    }
    let mut a: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        a.push(row);
    }

    println!(
        "# data: n={n} d={d} (~{:.2} MiB)",
        (n as f64 * d as f64 * 4.0) / (1024.0 * 1024.0)
    );
    println!(
        "# metric={:?} batch={} q_batch={} warmup={} iters={}",
        args.metric, args.batch, args.q_batch, args.warmup, args.iters
    );
    println!();
    println!(
        "{:<8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>12} {:>11}",
        "backend", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "rows/s@p50", "gops@p50"
    );

    // Optional: bake A into the ORT-QNN model (weights fixed; q is the only input)
    let baked_qnn = std::env::var("TACHANN_ORT_QNN_BAKED_A").ok().as_deref() == Some("1");
    if baked_qnn {
        let baked_op = std::env::var("TACHANN_ORT_QNN_BAKED_OP").unwrap_or_else(|_| "gemm".into());
        let tmpdir = std::env::temp_dir().join("tachyon_rerank_onnx");
        fs::create_dir_all(&tmpdir)?;
        let a_bin = tmpdir.join("a.bin");
        let mut f = File::create(&a_bin)?;
        for row in &a {
            for &v in row {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        let out_name = if baked_op == "conv" {
            "ip_conv_baked.onnx"
        } else {
            "ip_gemm_baked.onnx"
        };
        let out = tmpdir.join(out_name);
        let status = Command::new("python3")
            .arg("scripts/gen_ip_matmul_onnx.py")
            .arg("--out")
            .arg(&out)
            .arg("--n")
            .arg(n.to_string())
            .arg("--d")
            .arg(d.to_string())
            .arg("--op")
            .arg(&baked_op)
            .arg("--q-row")
            .arg("--bake-a-bin")
            .arg(&a_bin)
            .status()?;
        if !status.success() {
            return Err(anyhow::anyhow!(
                "failed to generate baked ONNX model for ORT-QNN"
            ));
        }
        std::env::set_var("MODELS_DIR", &tmpdir);
        std::env::set_var("TACHANN_ORT_QNN_MODEL", out_name);
        std::env::set_var("TACHANN_ORT_QNN_Q_ROW", "1");
        if baked_op == "conv" {
            std::env::set_var("TACHANN_ORT_QNN_Q4D", "1");
        }
    }

    // Compare requested backends
    for wb in &args.backends {
        let (env_val, ort_ep, shim_direct) = match wb {
            WhichBackend::Cpu => ("cpu", "cpu", "0"),
            WhichBackend::Qnn => ("qnn", "cpu", "1"),
            WhichBackend::Ort => ("ort", "cpu", "0"),
            WhichBackend::OrtQnn => ("ort", "qnn", "0"),
            WhichBackend::Auto => ("auto", "cpu", "0"),
        };
        std::env::set_var("TACHANN_BACKEND", env_val);
        std::env::set_var("TACHANN_ORT_EP", ort_ep);
        std::env::set_var("SHIM_DIRECT_QNN", shim_direct);
        if matches!(wb, WhichBackend::OrtQnn) {
            std::env::set_var("TACHANN_ORT_STATIC_N", n.to_string());
            std::env::set_var("TACHANN_ORT_STATIC_D", d.to_string());
            if std::env::var("TACHANN_ORT_QNN_MODEL").is_err() {
                std::env::set_var("TACHANN_ORT_QNN_MODEL", "ip_matmul_static.onnx");
            }
        }

        let b = backend::select_backend(); // Arc<dyn Backend + Send + Sync>
        let kind = match b.kind() {
            BackendKind::Cpu => "CPU",
            BackendKind::Qnn => "QNN-Graph",
            BackendKind::OrtCpu => "ORT-CPU",
            BackendKind::OrtQnn => "ORT-QNN",
        };

        let label = kind.to_string();
        let stats = bench_one(
            &label,
            b.as_ref(),
            &qs,
            &a,
            args.metric.clone().into(),
            args.batch,
            args.warmup,
            args.iters,
            args.print_sample,
        )?;

        println!(
            "{:<8} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>12.0} {:>11.2}",
            label,
            stats.p50_ms,
            stats.p90_ms,
            stats.p95_ms,
            stats.p99_ms,
            stats.mean_ms,
            stats.rows_per_sec_p50,
            stats.est_gops_p50
        );
    }

    Ok(())
}
