use anyhow::{anyhow, Context, Result};
use libloading::Library;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::{cmp, sync::Arc};

use ort::execution_providers::QNNExecutionProvider;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;

use log::{info, warn};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DistMetric {
    #[default]
    Cosine,
    L2,
    Ip,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Qnn,
    OrtCpu,
    OrtQnn,
    Cpu,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Qnn => "qnn",
            BackendKind::OrtCpu => "ort_cpu",
            BackendKind::OrtQnn => "ort_qnn",
            BackendKind::Cpu => "cpu",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            BackendKind::Qnn => "Qnn",
            BackendKind::OrtCpu => "OrtCpu",
            BackendKind::OrtQnn => "OrtQnn",
            BackendKind::Cpu => "Cpu",
        }
    }
}

pub struct RerankResult {
    pub distances: Vec<f32>,
}

pub trait Backend {
    fn kind(&self) -> BackendKind;
    fn distance(
        &self,
        q: &[f32],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<RerankResult>;
    fn distance_batch(
        &self,
        qs: &[Vec<f32>],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let mut out = Vec::with_capacity(qs.len());
        for q in qs {
            out.push(self.distance(q, a, metric, max_batch)?);
        }
        Ok(out)
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

pub fn try_new_ort_backend() -> Result<OrtBackend> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(OrtBackend::new)) {
        Ok(result) => result,
        Err(payload) => Err(anyhow!(
            "ORT init panicked: {}",
            panic_payload_to_string(payload)
        )),
    }
}

pub fn select_backend() -> Arc<dyn Backend + Send + Sync> {
    let requested = std::env::var("TACHANN_BACKEND").unwrap_or_else(|_| "auto".into());
    info!("backend: requested='{requested}'");

    match requested.as_str() {
        "cpu" => {
            info!("backend: selecting CPU (forced)");
            return Arc::new(CpuBackend);
        }
        "ort" => {
            match try_new_ort_backend() {
                Ok(o) => {
                    match o.kind() {
                        BackendKind::OrtQnn => info!("backend: selected ORT (QNN EP)"),
                        _ => info!("backend: selected ORT (CPU EP)"),
                    }
                    return Arc::new(o);
                }
                Err(e) => warn!("backend: ORT init failed: {e} — falling back to CPU"),
            }
            return Arc::new(CpuBackend);
        }
        "qnn" => {
            match QnnBackend::new() {
                Ok(q) => {
                    info!("backend: selected QNN via shim");
                    return Arc::new(q);
                }
                Err(e) => warn!("backend: QNN init failed: {e} — falling back to CPU"),
            }
            return Arc::new(CpuBackend);
        }
        "adaptive" => info!("backend: adaptive mode defaulting to QNN preference"),
        "auto" => info!("backend: auto mode (trying QNN → CPU)"),
        other => {
            warn!("backend: unknown backend '{other}' — falling back to CPU");
            return Arc::new(CpuBackend);
        }
    }

    if let Ok(q) = QnnBackend::new() {
        info!("backend: auto → QNN");
        return Arc::new(q);
    }
    info!("backend: auto → CPU (fallback)");
    Arc::new(CpuBackend)
}

// ----------------- shared numeric helpers -----------------
#[inline(always)]
fn sumsq64(v: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        let xf = x as f64;
        s += xf * xf;
    }
    s
}

#[inline(always)]
fn l2_sq_from_dot(a_sumsq: f64, q_sumsq: f64, dot: f64) -> f32 {
    (a_sumsq + q_sumsq - 2.0 * dot).max(0.0) as f32
}

// ----------------- CPU -----------------
#[derive(Default)]
pub struct CpuBackend;
impl CpuBackend {
    fn cfg() -> (usize, usize, usize) {
        let max_dim = std::env::var("TACHANN_MAX_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);
        let max_cand = std::env::var("TACHANN_MAX_CAND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);
        let chunk = std::env::var("TACHANN_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        (max_dim, max_cand, chunk)
    }
}
impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }
    fn distance(
        &self,
        q: &[f32],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<RerankResult> {
        let (max_dim, max_cand, default_chunk) = CpuBackend::cfg();
        let d = q.len();
        let n = a.len();
        if d > max_dim {
            return Err(anyhow!("dimension {} exceeds {}", d, max_dim));
        }
        if n > max_cand {
            return Err(anyhow!("candidates {} exceeds {}", n, max_cand));
        }
        let chunk = max_batch.unwrap_or(default_chunk).min(n.max(1));

        let q_sumsq = sumsq64(q);
        let q_norm = if let DistMetric::Cosine = metric {
            q_sumsq.sqrt()
        } else {
            0.0
        };

        let mut out = vec![0f32; n];
        for start in (0..n).step_by(chunk) {
            let end = cmp::min(start + chunk, n);
            out[start..end]
                .par_iter_mut()
                .enumerate()
                .for_each(|(offset, out_ref)| {
                    let i = start + offset;
                    let row = &a[i];
                    match metric {
                        DistMetric::Ip => {
                            let mut dot = 0f64;
                            for j in 0..d {
                                dot += (q[j] as f64) * (row[j] as f64);
                            }
                            *out_ref = -(dot as f32);
                        }
                        DistMetric::L2 => {
                            let mut dot = 0f64;
                            let mut a_sumsq = 0f64;
                            for j in 0..d {
                                let qj = q[j] as f64;
                                let aj = row[j] as f64;
                                dot += qj * aj;
                                a_sumsq += aj * aj;
                            }
                            *out_ref = l2_sq_from_dot(a_sumsq, q_sumsq, dot);
                        }
                        DistMetric::Cosine => {
                            let mut dot = 0f64;
                            let mut a_sumsq = 0f64;
                            for j in 0..d {
                                let qj = q[j] as f64;
                                let aj = row[j] as f64;
                                dot += qj * aj;
                                a_sumsq += aj * aj;
                            }
                            let a_norm = a_sumsq.sqrt();
                            let denom = q_norm * a_norm;
                            let cos_sim = if denom == 0.0 {
                                0.0
                            } else {
                                (dot / denom).clamp(-1.0, 1.0)
                            };
                            *out_ref = (1.0 - cos_sim) as f32;
                        }
                    }
                });
        }
        Ok(RerankResult { distances: out })
    }
}

// ----------------- ORT -----------------
fn qnn_backend_path() -> Result<String> {
    if let Ok(p) = std::env::var("TACHANN_ORT_QNN_BACKEND_PATH") {
        return Ok(p);
    }
    let backend = std::env::var("TACHANN_QNN_BACKEND").unwrap_or_else(|_| "htp".into());
    let lib_name = match backend.as_str() {
        "cpu" => "libQnnCpu.so",
        _ => "libQnnHtp.so",
    };
    if let Ok(root) = std::env::var("QNN_SDK_ROOT") {
        let root = std::path::Path::new(&root);
        let candidates = [
            root.join("lib/aarch64-ubuntu-gcc9.4"),
            root.join("lib/aarch64-oe-linux-gcc9.3"),
            root.join("lib/aarch64-oe-linux-gcc11.2"),
            root.join("lib/aarch64-oe-linux-gcc8.2"),
            root.join("lib"),
        ];
        for dir in candidates {
            let p = dir.join(lib_name);
            if p.exists() {
                return Ok(p.display().to_string());
            }
        }
    }
    Ok(lib_name.to_string())
}

pub struct OrtBackend {
    kind: BackendKind,
    matmul: Mutex<Session>,
    chunk: usize,
    max_dim: usize,
    max_cand: usize,
    static_dims: Option<(usize, usize)>,
    baked_a: bool,
    q_row: bool,
    q4d: bool,
}
impl OrtBackend {
    pub fn new() -> Result<Self> {
        let mut sb = SessionBuilder::new()
            .context("ORT: failed to create SessionBuilder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("ORT: failed to set optimization level")?;

        let ort_ep = std::env::var("TACHANN_ORT_EP").unwrap_or_else(|_| "cpu".into());
        let mut kind = BackendKind::OrtCpu;
        if ort_ep == "qnn" {
            let backend_path = qnn_backend_path()?;
            let mut ep = QNNExecutionProvider::default().with_backend_path(backend_path);
            let fp16 = std::env::var("TACHANN_QNN_FP16").ok().as_deref() == Some("1");
            if fp16 {
                ep = ep.with_htp_fp16_precision(true);
            }
            let offload_q = std::env::var("TACHANN_ORT_QNN_OFFLOAD_IO_QUANT")
                .ok()
                .as_deref()
                == Some("1");
            if offload_q {
                ep = ep.with_offload_graph_io_quantization(true);
            }
            let ep = ep.build();
            sb = sb
                .with_execution_providers([ep])
                .context("ORT: failed to register QNN execution provider")?;
            kind = BackendKind::OrtQnn;
        }

        let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| "./models".into());
        let model_name = if ort_ep == "qnn" {
            std::env::var("TACHANN_ORT_QNN_MODEL")
                .unwrap_or_else(|_| "ip_matmul_static.onnx".into())
        } else {
            std::env::var("TACHANN_ORT_MODEL").unwrap_or_else(|_| "ip_matmul.onnx".into())
        };
        let ip_path = std::path::Path::new(&models_dir).join(model_name);
        info!(
            "ORT: MODELS_DIR='{}', expecting '{}'",
            models_dir,
            ip_path.display()
        );
        if !ip_path.exists() {
            return Err(anyhow!(
                "ORT: missing model file: {}. \
                 Either set MODELS_DIR to a dir containing 'ip_matmul.onnx' \
                 or bake it into the image (e.g., COPY models /app/models && MODELS_DIR=/app/models).",
                ip_path.display()
            ));
        }

        let matmul = sb
            .clone()
            .commit_from_file(ip_path)
            .context("ORT: failed to create session from model file")?;
        let matmul = Mutex::new(matmul);

        let max_dim = std::env::var("TACHANN_MAX_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);
        let max_cand = std::env::var("TACHANN_MAX_CAND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);
        let chunk = std::env::var("TACHANN_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        let static_dims = if ort_ep == "qnn" {
            let n = std::env::var("TACHANN_ORT_STATIC_N")
                .ok()
                .and_then(|s| s.parse().ok());
            let d = std::env::var("TACHANN_ORT_STATIC_D")
                .ok()
                .and_then(|s| s.parse().ok());
            match (n, d) {
                (Some(n), Some(d)) => Some((n, d)),
                _ => None,
            }
        } else {
            None
        };
        let baked_a = std::env::var("TACHANN_ORT_QNN_BAKED_A").ok().as_deref() == Some("1");
        let q_row =
            baked_a || (std::env::var("TACHANN_ORT_QNN_Q_ROW").ok().as_deref() == Some("1"));
        let q4d = std::env::var("TACHANN_ORT_QNN_Q4D").ok().as_deref() == Some("1");
        Ok(Self {
            kind,
            matmul,
            chunk,
            max_dim,
            max_cand,
            static_dims,
            baked_a,
            q_row,
            q4d,
        })
    }

    fn ort_dot_batch(&self, q: &[f32], a_batch: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n = a_batch.len();
        let d = q.len();

        let mut a_flat = Vec::with_capacity(n * d);
        for row in a_batch {
            a_flat.extend_from_slice(&row[..]);
        }

        let a_tensor =
            Tensor::from_array((vec![n, d], a_flat)).context("ORT: failed to make A tensor")?;
        let q_shape = if self.q_row {
            vec![1usize, d]
        } else {
            vec![d, 1usize]
        };
        let q_tensor =
            Tensor::from_array((q_shape, q.to_vec())).context("ORT: failed to make q tensor")?;

        let mut sess = self.matmul.lock().expect("session mutex poisoned");
        let outputs = sess
            .run(ort::inputs!["A" => a_tensor, "q" => q_tensor])
            .context("ORT: session.run failed")?;
        let y = outputs[0]
            .try_extract_tensor::<f32>()
            .context("ORT: failed to extract output tensor<f32>")?;
        let (_shape, data) = y;
        Ok(data[..n].to_vec())
    }

    fn ort_dot_baked(&self, q: &[f32], n: usize) -> Result<Vec<f32>> {
        let d = q.len();
        let q_shape = if self.q4d {
            vec![1usize, d, 1usize, 1usize]
        } else if self.q_row {
            vec![1usize, d]
        } else {
            vec![d, 1usize]
        };
        let q_tensor =
            Tensor::from_array((q_shape, q.to_vec())).context("ORT: failed to make q tensor")?;
        let mut sess = self.matmul.lock().expect("session mutex poisoned");
        let outputs = sess
            .run(ort::inputs!["q" => q_tensor])
            .context("ORT: session.run failed")?;
        let y = outputs[0]
            .try_extract_tensor::<f32>()
            .context("ORT: failed to extract output tensor<f32>")?;
        let (_shape, data) = y;
        Ok(data[..n].to_vec())
    }
}
impl Backend for OrtBackend {
    fn kind(&self) -> BackendKind {
        self.kind
    }
    fn distance(
        &self,
        q: &[f32],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<RerankResult> {
        let d = q.len();
        let n = a.len();
        if self.baked_a {
            let (sn, sd) = self.static_dims.ok_or_else(|| {
                anyhow!("ORT-QNN baked-A requires static dims (set TACHANN_ORT_STATIC_N/D)")
            })?;
            if n != sn || d != sd {
                return Err(anyhow!(
                    "ORT-QNN baked-A requires fixed dims n={}, d={} (got n={}, d={})",
                    sn,
                    sd,
                    n,
                    d
                ));
            }
        }
        if let Some((sn, sd)) = self.static_dims {
            if n != sn || d != sd {
                return Err(anyhow!(
                    "ORT-QNN requires fixed dims n={}, d={} (got n={}, d={})",
                    sn,
                    sd,
                    n,
                    d
                ));
            }
        }
        if d > self.max_dim {
            return Err(anyhow!("dimension {} exceeds {}", d, self.max_dim));
        }
        if n > self.max_cand {
            return Err(anyhow!("candidates {} exceeds {}", n, self.max_cand));
        }
        let chunk = if self.static_dims.is_some() {
            n.max(1)
        } else {
            max_batch.unwrap_or(self.chunk).min(n.max(1))
        };

        let q_sumsq = sumsq64(q);
        let q_norm = if let DistMetric::Cosine = metric {
            q_sumsq.sqrt()
        } else {
            0.0
        };

        let mut out = vec![0f32; n];
        if self.baked_a {
            let dots = self.ort_dot_baked(q, n)?;
            match metric {
                DistMetric::Ip => {
                    for (k, &dot) in dots.iter().enumerate() {
                        out[k] = -dot;
                    }
                }
                DistMetric::L2 => {
                    for (k, row) in a.iter().enumerate() {
                        let a_sumsq = sumsq64(row);
                        out[k] = l2_sq_from_dot(a_sumsq, q_sumsq, dots[k] as f64);
                    }
                }
                DistMetric::Cosine => {
                    for (k, row) in a.iter().enumerate() {
                        let a_norm = sumsq64(row).sqrt();
                        let denom = q_norm * a_norm;
                        let cos_sim = if denom == 0.0 {
                            0.0
                        } else {
                            ((dots[k] as f64) / denom).clamp(-1.0, 1.0)
                        };
                        out[k] = (1.0 - cos_sim) as f32;
                    }
                }
            }
        } else {
            for start in (0..n).step_by(chunk) {
                let end = cmp::min(start + chunk, n);
                let batch = &a[start..end];
                let dots = self.ort_dot_batch(q, batch)?;
                match metric {
                    DistMetric::Ip => {
                        for (k, &dot) in dots.iter().enumerate() {
                            out[start + k] = -dot;
                        }
                    }
                    DistMetric::L2 => {
                        for (k, row) in batch.iter().enumerate() {
                            let a_sumsq = sumsq64(row);
                            out[start + k] = l2_sq_from_dot(a_sumsq, q_sumsq, dots[k] as f64);
                        }
                    }
                    DistMetric::Cosine => {
                        for (k, row) in batch.iter().enumerate() {
                            let a_norm = sumsq64(row).sqrt();
                            let denom = q_norm * a_norm;
                            let cos_sim = if denom == 0.0 {
                                0.0
                            } else {
                                ((dots[k] as f64) / denom).clamp(-1.0, 1.0)
                            };
                            out[start + k] = (1.0 - cos_sim) as f32;
                        }
                    }
                }
            }
        }
        Ok(RerankResult { distances: out })
    }
}

// ----------------- QNN (shim-backed) -----------------
type FnAvail = unsafe extern "C" fn() -> i32;
type FnMatMul =
    unsafe extern "C" fn(a: *const f32, n: i32, d: i32, q: *const f32, out: *mut f32) -> i32;
type FnMatMulBatched = unsafe extern "C" fn(
    a: *const f32,
    n: i32,
    d: i32,
    q: *const f32,
    q_batch: i32,
    out: *mut f32,
) -> i32;
type FnWarmup = unsafe extern "C" fn() -> i32;
type FnCleanup = unsafe extern "C" fn();
type FnFp16Disabled = unsafe extern "C" fn() -> i32;

pub struct QnnBackend {
    _lib: Library,
    matmul: FnMatMul,
    matmul_batched: Option<FnMatMulBatched>,
    cleanup: Option<FnCleanup>,
    fp16_disabled: Option<FnFp16Disabled>,
    chunk: usize,
    max_dim: usize,
    max_cand: usize,
    static_a: bool,
    fp16_requested: bool,
    warmup_requested: bool,
    warmup_ok: bool,
    static_cache: Mutex<Option<Arc<StaticA>>>,
}
struct StaticA {
    n: usize,
    d: usize,
    flat: Vec<f32>,
    sumsq: Vec<f64>,
    norm: Vec<f64>,
}
#[derive(Default, Clone)]
struct QnnTiming {
    pack_ms: f64,
    exec_ms: f64,
    post_ms: f64,
    total_ms: f64,
}
static QNN_TIMING_ONCE: AtomicBool = AtomicBool::new(false);
impl QnnBackend {
    pub fn new() -> Result<Self> {
        let name =
            std::env::var("TACHANN_QNN_LIB").unwrap_or_else(|_| "libtachann_qnnshim.so".into());
        info!("QNN: loading shim '{}'", name);
        let lib = unsafe { Library::new(&name) }
            .with_context(|| format!("QNN: failed to load shim library: {}", name))?;

        // Try current symbol names first; if not present, try legacy names.
        unsafe fn try_symbols(
            lib: &Library,
        ) -> Result<(
            FnAvail,
            FnMatMul,
            Option<FnMatMulBatched>,
            Option<FnCleanup>,
        )> {
            let a: libloading::Symbol<FnAvail> = lib.get(b"tachann_qnn_available")?;
            let m: libloading::Symbol<FnMatMul> = lib.get(b"tachann_qnn_matmul")?;
            let b: Option<FnMatMulBatched> =
                match lib.get::<FnMatMulBatched>(b"tachann_qnn_matmul_batched") {
                    Ok(sym) => Some(*sym),
                    Err(_) => None,
                };
            let c: Option<FnCleanup> = match lib.get::<FnCleanup>(b"tachann_qnn_cleanup_all") {
                Ok(sym) => Some(*sym),
                Err(_) => None,
            };
            Ok((*a, *m, b, c))
        }
        unsafe fn try_compat_symbols(
            lib: &Library,
        ) -> Result<(
            FnAvail,
            FnMatMul,
            Option<FnMatMulBatched>,
            Option<FnCleanup>,
        )> {
            let a: libloading::Symbol<FnAvail> = lib.get(b"tachyon_qnn_avail")?;
            let m: libloading::Symbol<FnMatMul> = lib.get(b"tachyon_qnn_matmul")?;
            let b: Option<FnMatMulBatched> =
                match lib.get::<FnMatMulBatched>(b"tachyon_qnn_matmul_batched") {
                    Ok(sym) => Some(*sym),
                    Err(_) => None,
                };
            let c: Option<FnCleanup> = match lib.get::<FnCleanup>(b"tachyon_qnn_cleanup_all") {
                Ok(sym) => Some(*sym),
                Err(_) => None,
            };
            Ok((*a, *m, b, c))
        }

        let (avail, matmul, matmul_batched, cleanup) = unsafe {
            match try_symbols(&lib).or_else(|e1| {
                warn!("QNN: primary symbols not found ({e1}); trying compatibility names");
                try_compat_symbols(&lib)
            }) {
                Ok(s) => s,
                Err(e2) => {
                    return Err(anyhow!(
                        "QNN: could not locate shim symbols (primary or compat): {e2}"
                    ))
                }
            }
        };

        let warmup: Option<FnWarmup> = unsafe {
            match lib.get(b"tachann_qnn_warmup") {
                Ok(s) => Some(*s),
                Err(e) => {
                    warn!("QNN: warmup symbol not found: {e}");
                    None
                }
            }
        };
        let fp16_disabled: Option<FnFp16Disabled> = unsafe {
            match lib.get(b"tachann_qnn_fp16_disabled") {
                Ok(s) => Some(*s),
                Err(_) => None,
            }
        };

        let is_avail = unsafe { (avail)() } == 1;
        if !is_avail {
            return Err(anyhow!("QNN: shim reports unavailable (check LD_LIBRARY_PATH and QNN_SDK_ROOT/lib/* subdirs)"));
        }

        let do_warmup = std::env::var("TACHANN_QNN_WARMUP")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);
        let mut warmup_ok = false;
        if do_warmup {
            if let Some(f) = warmup {
                let rc = unsafe { (f)() };
                if rc != 0 {
                    return Err(anyhow!("QNN: warmup failed rc={}", rc));
                }
                warmup_ok = true;
            } else {
                warn!("QNN: warmup requested but shim lacks tachann_qnn_warmup; continuing");
            }
        }

        let max_dim = std::env::var("TACHANN_MAX_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);
        let max_cand = std::env::var("TACHANN_MAX_CAND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);
        let chunk = std::env::var("TACHANN_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        let static_a = std::env::var("TACHANN_QNN_STATIC_A").ok().as_deref() == Some("1");
        let fp16_requested = std::env::var("TACHANN_QNN_FP16").ok().as_deref() == Some("1");
        if static_a {
            info!("QNN: static A mode enabled (A cached, q-only execution)");
        }

        info!("QNN: shim ready (max_dim={max_dim}, max_cand={max_cand}, chunk={chunk})");
        Ok(Self {
            _lib: lib,
            matmul,
            matmul_batched,
            cleanup,
            fp16_disabled,
            chunk,
            max_dim,
            max_cand,
            static_a,
            fp16_requested,
            warmup_requested: do_warmup,
            warmup_ok,
            static_cache: Mutex::new(None),
        })
    }

    fn qnn_dot_batch(
        &self,
        q: &[f32],
        a_batch: &[Vec<f32>],
        mut timing: Option<&mut QnnTiming>,
    ) -> Result<Vec<f32>> {
        let n = a_batch.len();
        let d = q.len();
        let t_pack = std::time::Instant::now();
        let mut a_flat = Vec::with_capacity(n * d);
        for row in a_batch {
            a_flat.extend_from_slice(&row[..]);
        }
        if let Some(t) = timing.as_mut() {
            t.pack_ms += t_pack.elapsed().as_secs_f64() * 1e3;
        }
        let mut out = vec![0f32; n];
        let t_exec = std::time::Instant::now();
        let rc = unsafe {
            (self.matmul)(
                a_flat.as_ptr(),
                n as i32,
                d as i32,
                q.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if let Some(t) = timing.as_mut() {
            t.exec_ms += t_exec.elapsed().as_secs_f64() * 1e3;
        }
        if rc != 0 {
            return Err(anyhow!("QNN matmul failed rc={}", rc));
        }
        Ok(out)
    }

    fn qnn_dot_flat(
        &self,
        q: &[f32],
        a_flat: &[f32],
        n: usize,
        d: usize,
        mut timing: Option<&mut QnnTiming>,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0f32; n];
        let t_exec = std::time::Instant::now();
        let rc = unsafe {
            (self.matmul)(
                a_flat.as_ptr(),
                n as i32,
                d as i32,
                q.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if let Some(t) = timing.as_mut() {
            t.exec_ms += t_exec.elapsed().as_secs_f64() * 1e3;
        }
        if rc != 0 {
            return Err(anyhow!("QNN matmul failed rc={}", rc));
        }
        Ok(out)
    }

    fn get_static_a(&self, a: &[Vec<f32>], d: usize) -> Result<Arc<StaticA>> {
        let n = a.len();
        let mut guard = self.static_cache.lock().expect("static cache lock");
        if let Some(cached) = guard.as_ref() {
            if cached.n == n && cached.d == d {
                return Ok(Arc::clone(cached));
            }
        }
        let mut flat = Vec::with_capacity(n * d);
        let mut sumsq = Vec::with_capacity(n);
        let mut norm = Vec::with_capacity(n);
        for row in a.iter() {
            if row.len() != d {
                return Err(anyhow!("row length {} != expected {}", row.len(), d));
            }
            let mut s = 0.0f64;
            for &v in row.iter() {
                let fv = v as f64;
                s += fv * fv;
                flat.push(v);
            }
            sumsq.push(s);
            norm.push(s.sqrt());
        }
        let cached = Arc::new(StaticA {
            n,
            d,
            flat,
            sumsq,
            norm,
        });
        *guard = Some(Arc::clone(&cached));
        info!("QNN: cached static A (n={n}, d={d}, bytes={})", n * d * 4);
        Ok(cached)
    }

    pub fn static_a_enabled(&self) -> bool {
        self.static_a
    }

    pub fn fp16_requested(&self) -> bool {
        self.fp16_requested
    }

    pub fn fp16_effective(&self) -> Option<bool> {
        if !self.fp16_requested {
            return Some(false);
        }
        self.fp16_disabled.map(|f| unsafe { f() == 0 })
    }

    pub fn warmup_requested(&self) -> bool {
        self.warmup_requested
    }

    pub fn warmup_ok(&self) -> bool {
        self.warmup_ok
    }

    pub fn has_batched_api(&self) -> bool {
        self.matmul_batched.is_some()
    }

    pub fn cached_static_a_dims(&self) -> Option<(usize, usize)> {
        let guard = self.static_cache.lock().expect("static cache lock");
        guard.as_ref().map(|cached| (cached.n, cached.d))
    }
}
impl Drop for QnnBackend {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup {
            unsafe { cleanup() };
        }
    }
}
impl Backend for QnnBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Qnn
    }
    fn distance(
        &self,
        q: &[f32],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<RerankResult> {
        let d = q.len();
        let n = a.len();
        if d > self.max_dim {
            return Err(anyhow!("dimension {} exceeds {}", d, self.max_dim));
        }
        if n > self.max_cand {
            return Err(anyhow!("candidates {} exceeds {}", n, self.max_cand));
        }
        let chunk = max_batch.unwrap_or(self.chunk).min(n.max(1));

        let q_sumsq = sumsq64(q);
        let q_norm = if let DistMetric::Cosine = metric {
            q_sumsq.sqrt()
        } else {
            0.0
        };

        let timing_mode = std::env::var("TACHANN_QNN_TIMING").unwrap_or_default();
        let timing_enabled = !timing_mode.is_empty() && timing_mode != "0";
        let timing_all = timing_mode == "all";
        let mut timing = QnnTiming::default();
        let total_start = std::time::Instant::now();

        let mut out = vec![0f32; n];
        if self.static_a {
            let cached = self.get_static_a(a, d)?;
            let dots = self.qnn_dot_flat(
                q,
                &cached.flat,
                n,
                d,
                if timing_enabled {
                    Some(&mut timing)
                } else {
                    None
                },
            )?;
            let t_post = std::time::Instant::now();
            match metric {
                DistMetric::Ip => {
                    for (k, &dot) in dots.iter().enumerate() {
                        out[k] = -dot;
                    }
                }
                DistMetric::L2 => {
                    for (k, &a_sumsq) in cached.sumsq.iter().enumerate() {
                        out[k] = l2_sq_from_dot(a_sumsq, q_sumsq, dots[k] as f64);
                    }
                }
                DistMetric::Cosine => {
                    for (k, &a_norm) in cached.norm.iter().enumerate() {
                        let denom = q_norm * a_norm;
                        let cos_sim = if denom == 0.0 {
                            0.0
                        } else {
                            ((dots[k] as f64) / denom).clamp(-1.0, 1.0)
                        };
                        out[k] = (1.0 - cos_sim) as f32;
                    }
                }
            }
            if timing_enabled {
                timing.post_ms += t_post.elapsed().as_secs_f64() * 1e3;
            }
        } else {
            for start in (0..n).step_by(chunk) {
                let end = cmp::min(start + chunk, n);
                let batch = &a[start..end];
                let dots = self.qnn_dot_batch(
                    q,
                    batch,
                    if timing_enabled {
                        Some(&mut timing)
                    } else {
                        None
                    },
                )?;
                let t_post = std::time::Instant::now();
                match metric {
                    DistMetric::Ip => {
                        for (k, &dot) in dots.iter().enumerate() {
                            out[start + k] = -dot;
                        }
                    }
                    DistMetric::L2 => {
                        for (k, row) in batch.iter().enumerate() {
                            let a_sumsq = sumsq64(row);
                            out[start + k] = l2_sq_from_dot(a_sumsq, q_sumsq, dots[k] as f64);
                        }
                    }
                    DistMetric::Cosine => {
                        for (k, row) in batch.iter().enumerate() {
                            let a_norm = sumsq64(row).sqrt();
                            let denom = q_norm * a_norm;
                            let cos_sim = if denom == 0.0 {
                                0.0
                            } else {
                                ((dots[k] as f64) / denom).clamp(-1.0, 1.0)
                            };
                            out[start + k] = (1.0 - cos_sim) as f32;
                        }
                    }
                }
                if timing_enabled {
                    timing.post_ms += t_post.elapsed().as_secs_f64() * 1e3;
                }
            }
        }
        if timing_enabled {
            timing.total_ms = total_start.elapsed().as_secs_f64() * 1e3;
            let print_ok = timing_all || !QNN_TIMING_ONCE.swap(true, Ordering::Relaxed);
            if print_ok {
                let other =
                    (timing.total_ms - timing.pack_ms - timing.exec_ms - timing.post_ms).max(0.0);
                eprintln!(
                    "[tachann][qnn timing] n={} d={} chunk={} total={:.3}ms pack={:.3}ms exec={:.3}ms post={:.3}ms other={:.3}ms",
                    n, d, chunk, timing.total_ms, timing.pack_ms, timing.exec_ms, timing.post_ms, other
                );
            }
        }
        Ok(RerankResult { distances: out })
    }

    fn distance_batch(
        &self,
        qs: &[Vec<f32>],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let b = qs.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        if b == 1 {
            return Ok(vec![self.distance(&qs[0], a, metric, max_batch)?]);
        }
        let d = qs[0].len();
        let n = a.len();
        if d > self.max_dim {
            return Err(anyhow!("dimension {} exceeds {}", d, self.max_dim));
        }
        if n > self.max_cand {
            return Err(anyhow!("candidates {} exceeds {}", n, self.max_cand));
        }
        if qs.iter().any(|q| q.len() != d) {
            return Err(anyhow!("all queries must have same dimension"));
        }

        let use_batched = self.matmul_batched.is_some();
        if !use_batched {
            // Fallback: run queries sequentially.
            let mut out = Vec::with_capacity(b);
            for q in qs {
                out.push(self.distance(q, a, metric, max_batch)?);
            }
            return Ok(out);
        }

        let timing_enabled = std::env::var("TACHANN_QNN_TIMING").ok().as_deref() == Some("1");
        let mut timing = QnnTiming::default();
        let total_start = std::time::Instant::now();

        // Pack Q into row-major [d, b] so MatMul sees (d x b)
        let t_pack = std::time::Instant::now();
        let mut q_flat = vec![0f32; d * b];
        for (j, q) in qs.iter().enumerate() {
            for i in 0..d {
                q_flat[i * b + j] = q[i];
            }
        }
        if timing_enabled {
            timing.pack_ms += t_pack.elapsed().as_secs_f64() * 1e3;
        }

        // Precompute q norms
        let mut q_sumsq = vec![0f64; b];
        let mut q_norm = vec![0f64; b];
        for (j, q) in qs.iter().enumerate() {
            let ss = sumsq64(q);
            q_sumsq[j] = ss;
            q_norm[j] = if let DistMetric::Cosine = metric {
                ss.sqrt()
            } else {
                0.0
            };
        }

        let dots = if self.static_a {
            let cached = self.get_static_a(a, d)?;
            let mut out = vec![0f32; n * b];
            let t_exec = std::time::Instant::now();
            let rc = unsafe {
                (self.matmul_batched.unwrap())(
                    cached.flat.as_ptr(),
                    n as i32,
                    d as i32,
                    q_flat.as_ptr(),
                    b as i32,
                    out.as_mut_ptr(),
                )
            };
            if timing_enabled {
                timing.exec_ms += t_exec.elapsed().as_secs_f64() * 1e3;
            }
            if rc != 0 {
                return Err(anyhow!("QNN matmul_batched failed rc={}", rc));
            }
            out
        } else {
            let mut a_flat = Vec::with_capacity(n * d);
            for row in a {
                a_flat.extend_from_slice(&row[..]);
            }
            let mut out = vec![0f32; n * b];
            let t_exec = std::time::Instant::now();
            let rc = unsafe {
                (self.matmul_batched.unwrap())(
                    a_flat.as_ptr(),
                    n as i32,
                    d as i32,
                    q_flat.as_ptr(),
                    b as i32,
                    out.as_mut_ptr(),
                )
            };
            if timing_enabled {
                timing.exec_ms += t_exec.elapsed().as_secs_f64() * 1e3;
            }
            if rc != 0 {
                return Err(anyhow!("QNN matmul_batched failed rc={}", rc));
            }
            out
        };

        // Post-process distances
        let t_post = std::time::Instant::now();
        let mut results: Vec<RerankResult> = Vec::with_capacity(b);
        for _ in 0..b {
            results.push(RerankResult {
                distances: vec![0f32; n],
            });
        }
        let a_sumsq_cache = if self.static_a {
            Some(self.get_static_a(a, d)?)
        } else {
            None
        };
        for i in 0..n {
            let a_sumsq = if let Some(ref cached) = a_sumsq_cache {
                cached.sumsq[i]
            } else {
                sumsq64(&a[i])
            };
            let a_norm = if let DistMetric::Cosine = metric {
                a_sumsq.sqrt()
            } else {
                0.0
            };
            for j in 0..b {
                let dot = dots[i * b + j] as f64;
                let v = match metric {
                    DistMetric::Ip => -dot,
                    DistMetric::L2 => l2_sq_from_dot(a_sumsq, q_sumsq[j], dot) as f64,
                    DistMetric::Cosine => {
                        let denom = q_norm[j] * a_norm;
                        let cos_sim = if denom == 0.0 {
                            0.0
                        } else {
                            (dot / denom).clamp(-1.0, 1.0)
                        };
                        1.0 - cos_sim
                    }
                };
                results[j].distances[i] = v as f32;
            }
        }
        if timing_enabled {
            timing.post_ms += t_post.elapsed().as_secs_f64() * 1e3;
        }

        if timing_enabled {
            timing.total_ms = total_start.elapsed().as_secs_f64() * 1e3;
            let other =
                (timing.total_ms - timing.pack_ms - timing.exec_ms - timing.post_ms).max(0.0);
            eprintln!(
                "[tachann][qnn timing batch] n={} d={} b={} total={:.3}ms pack={:.3}ms exec={:.3}ms post={:.3}ms other={:.3}ms",
                n, d, b, timing.total_ms, timing.pack_ms, timing.exec_ms, timing.post_ms, other
            );
        }

        Ok(results)
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestedMode {
    Auto,
    Adaptive,
    Cpu,
    Qnn,
    Ort,
}

impl RequestedMode {
    pub fn from_env() -> Self {
        match std::env::var("TACHANN_BACKEND")
            .unwrap_or_else(|_| "auto".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "adaptive" => RequestedMode::Adaptive,
            "cpu" => RequestedMode::Cpu,
            "qnn" => RequestedMode::Qnn,
            "ort" => RequestedMode::Ort,
            _ => RequestedMode::Auto,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RequestedMode::Auto => "auto",
            RequestedMode::Adaptive => "adaptive",
            RequestedMode::Cpu => "cpu",
            RequestedMode::Qnn => "qnn",
            RequestedMode::Ort => "ort",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub requested_mode: String,
    pub actual_backend: String,
    pub route_reason: String,
    pub fallback_reason: Option<String>,
    pub qnn_available: bool,
    pub static_a: bool,
    pub fp16_requested: bool,
    pub fp16_effective: Option<bool>,
    pub warmup_requested: bool,
    pub warmed: bool,
    pub batched_qnn: bool,
    pub cached_static_a: Option<String>,
    pub adaptive_profile: Option<String>,
    pub last_q_batch: usize,
}

pub struct RuntimeRouter {
    requested_mode: RequestedMode,
    cpu: Arc<CpuBackend>,
    qnn: Option<Arc<QnnBackend>>,
    ort: Option<Arc<OrtBackend>>,
    qnn_unavailable_reason: Option<String>,
    ort_unavailable_reason: Option<String>,
    status: Mutex<RuntimeStatus>,
}

fn adaptive_prefers_cpu(n: usize, d: usize, q_batch: usize) -> bool {
    q_batch < 4 && d >= 512 && n >= 8192
}

impl RuntimeRouter {
    pub fn from_env() -> Self {
        let requested_mode = RequestedMode::from_env();
        let cpu = Arc::new(CpuBackend);
        let mut qnn = None;
        let mut ort = None;
        let mut qnn_unavailable_reason = None;
        let mut ort_unavailable_reason = None;

        match requested_mode {
            RequestedMode::Auto | RequestedMode::Adaptive | RequestedMode::Qnn => {
                match QnnBackend::new() {
                    Ok(backend) => qnn = Some(Arc::new(backend)),
                    Err(e) => qnn_unavailable_reason = Some(e.to_string()),
                }
            }
            RequestedMode::Ort => match try_new_ort_backend() {
                Ok(backend) => ort = Some(Arc::new(backend)),
                Err(e) => ort_unavailable_reason = Some(e.to_string()),
            },
            RequestedMode::Cpu => {}
        }

        let (actual_backend, route_reason, fallback_reason) = match requested_mode {
            RequestedMode::Cpu => (BackendKind::Cpu, "forced cpu".to_string(), None),
            RequestedMode::Qnn => {
                if qnn.is_some() {
                    (BackendKind::Qnn, "forced qnn".to_string(), None)
                } else {
                    (
                        BackendKind::Cpu,
                        "qnn init failed; using cpu fallback".to_string(),
                        qnn_unavailable_reason.clone(),
                    )
                }
            }
            RequestedMode::Ort => {
                if let Some(ort_backend) = ort.as_ref() {
                    (ort_backend.kind(), "forced ort".to_string(), None)
                } else {
                    (
                        BackendKind::Cpu,
                        "ort init failed; using cpu fallback".to_string(),
                        ort_unavailable_reason.clone(),
                    )
                }
            }
            RequestedMode::Auto => {
                if qnn.is_some() {
                    (BackendKind::Qnn, "auto selected qnn".to_string(), None)
                } else {
                    (
                        BackendKind::Cpu,
                        "auto selected cpu because qnn is unavailable".to_string(),
                        qnn_unavailable_reason.clone(),
                    )
                }
            }
            RequestedMode::Adaptive => {
                if qnn.is_some() {
                    (
                        BackendKind::Qnn,
                        "adaptive default preference is qnn".to_string(),
                        None,
                    )
                } else {
                    (
                        BackendKind::Cpu,
                        "adaptive fell back to cpu because qnn is unavailable".to_string(),
                        qnn_unavailable_reason.clone(),
                    )
                }
            }
        };

        let router = Self {
            requested_mode,
            cpu,
            qnn,
            ort,
            qnn_unavailable_reason,
            ort_unavailable_reason,
            status: Mutex::new(RuntimeStatus {
                requested_mode: requested_mode.as_str().to_string(),
                actual_backend: actual_backend.display_name().to_string(),
                route_reason,
                fallback_reason,
                qnn_available: false,
                static_a: false,
                fp16_requested: false,
                fp16_effective: Some(false),
                warmup_requested: false,
                warmed: false,
                batched_qnn: false,
                cached_static_a: None,
                adaptive_profile: if requested_mode == RequestedMode::Adaptive {
                    Some("tachyon_particle_v1".to_string())
                } else {
                    None
                },
                last_q_batch: 1,
            }),
        };
        router.refresh_qnn_fields();
        router
    }

    pub fn forced_cpu() -> Self {
        Self {
            requested_mode: RequestedMode::Cpu,
            cpu: Arc::new(CpuBackend),
            qnn: None,
            ort: None,
            qnn_unavailable_reason: None,
            ort_unavailable_reason: None,
            status: Mutex::new(RuntimeStatus {
                requested_mode: RequestedMode::Cpu.as_str().to_string(),
                actual_backend: BackendKind::Cpu.display_name().to_string(),
                route_reason: "forced cpu".to_string(),
                fallback_reason: None,
                qnn_available: false,
                static_a: false,
                fp16_requested: false,
                fp16_effective: Some(false),
                warmup_requested: false,
                warmed: false,
                batched_qnn: false,
                cached_static_a: None,
                adaptive_profile: None,
                last_q_batch: 1,
            }),
        }
    }

    fn refresh_qnn_fields(&self) {
        let mut status = self.status.lock().expect("runtime status lock");
        if let Some(qnn) = self.qnn.as_ref() {
            status.qnn_available = true;
            status.static_a = qnn.static_a_enabled();
            status.fp16_requested = qnn.fp16_requested();
            status.fp16_effective = qnn.fp16_effective();
            status.warmup_requested = qnn.warmup_requested();
            status.warmed = qnn.warmup_ok();
            status.batched_qnn = qnn.has_batched_api();
            status.cached_static_a = qnn
                .cached_static_a_dims()
                .map(|(n, d)| format!("{}x{}", n, d));
        } else {
            status.qnn_available = false;
            status.static_a = false;
            status.fp16_requested = false;
            status.fp16_effective = Some(false);
            status.warmup_requested = false;
            status.warmed = false;
            status.batched_qnn = false;
            status.cached_static_a = None;
        }
    }

    fn set_last_route(
        &self,
        actual_backend: BackendKind,
        route_reason: impl Into<String>,
        fallback_reason: Option<String>,
        last_q_batch: usize,
    ) {
        self.refresh_qnn_fields();
        let mut status = self.status.lock().expect("runtime status lock");
        status.actual_backend = actual_backend.display_name().to_string();
        status.route_reason = route_reason.into();
        status.fallback_reason = fallback_reason;
        status.last_q_batch = last_q_batch;
        if self.requested_mode == RequestedMode::Adaptive {
            status.adaptive_profile = Some("tachyon_particle_v1".to_string());
        }
    }

    fn run_single_on(
        &self,
        kind: BackendKind,
        q: &[f32],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<RerankResult> {
        match kind {
            BackendKind::Qnn => self
                .qnn
                .as_ref()
                .expect("qnn backend missing")
                .distance(q, a, metric, max_batch),
            BackendKind::OrtCpu | BackendKind::OrtQnn => self
                .ort
                .as_ref()
                .expect("ort backend missing")
                .distance(q, a, metric, max_batch),
            BackendKind::Cpu => self.cpu.distance(q, a, metric, max_batch),
        }
    }

    fn run_batch_on(
        &self,
        kind: BackendKind,
        qs: &[Vec<f32>],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        match kind {
            BackendKind::Qnn => self
                .qnn
                .as_ref()
                .expect("qnn backend missing")
                .distance_batch(qs, a, metric, max_batch),
            BackendKind::OrtCpu | BackendKind::OrtQnn => self
                .ort
                .as_ref()
                .expect("ort backend missing")
                .distance_batch(qs, a, metric, max_batch),
            BackendKind::Cpu => self.cpu.distance_batch(qs, a, metric, max_batch),
        }
    }

    fn choose_backend_for_batch(
        &self,
        n: usize,
        d: usize,
        q_batch: usize,
    ) -> (BackendKind, String, Option<String>) {
        match self.requested_mode {
            RequestedMode::Cpu => (BackendKind::Cpu, "forced cpu".to_string(), None),
            RequestedMode::Ort => {
                if let Some(ort) = self.ort.as_ref() {
                    (ort.kind(), "forced ort".to_string(), None)
                } else {
                    (
                        BackendKind::Cpu,
                        "ort unavailable; cpu fallback".to_string(),
                        self.ort_unavailable_reason.clone(),
                    )
                }
            }
            RequestedMode::Qnn => {
                if self.qnn.is_some() {
                    (BackendKind::Qnn, "forced qnn".to_string(), None)
                } else {
                    (
                        BackendKind::Cpu,
                        "qnn unavailable; cpu fallback".to_string(),
                        self.qnn_unavailable_reason.clone(),
                    )
                }
            }
            RequestedMode::Auto => {
                if self.qnn.is_some() {
                    (BackendKind::Qnn, "auto selected qnn".to_string(), None)
                } else {
                    (
                        BackendKind::Cpu,
                        "auto selected cpu because qnn is unavailable".to_string(),
                        self.qnn_unavailable_reason.clone(),
                    )
                }
            }
            RequestedMode::Adaptive => {
                if self.qnn.is_none() {
                    return (
                        BackendKind::Cpu,
                        "adaptive selected cpu because qnn is unavailable".to_string(),
                        self.qnn_unavailable_reason.clone(),
                    );
                }
                if q_batch >= 4 {
                    return (
                        BackendKind::Qnn,
                        format!(
                            "adaptive selected qnn for batched workload (q_batch={})",
                            q_batch
                        ),
                        None,
                    );
                }
                if adaptive_prefers_cpu(n, d, q_batch) {
                    return (
                        BackendKind::Cpu,
                        format!(
                            "adaptive selected cpu for singleton large-dim workload (n={}, d={})",
                            n, d
                        ),
                        None,
                    );
                }
                (
                    BackendKind::Qnn,
                    format!(
                        "adaptive selected qnn by default (n={}, d={}, q_batch={})",
                        n, d, q_batch
                    ),
                    None,
                )
            }
        }
    }

    pub fn score(
        &self,
        q: &[f32],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<RerankResult> {
        let (kind, route_reason, fallback_reason) =
            self.choose_backend_for_batch(a.len(), q.len(), 1);
        match self.run_single_on(kind, q, a, metric, max_batch) {
            Ok(result) => {
                self.set_last_route(kind, route_reason, fallback_reason, 1);
                Ok(result)
            }
            Err(err) if kind != BackendKind::Cpu => {
                let fallback = format!("{}; qnn/ort execution failed: {}", route_reason, err);
                let result = self.cpu.distance(q, a, metric, max_batch)?;
                self.set_last_route(
                    BackendKind::Cpu,
                    "cpu fallback after primary backend execution failure",
                    Some(fallback),
                    1,
                );
                Ok(result)
            }
            Err(err) => Err(err),
        }
    }

    pub fn score_batch(
        &self,
        qs: &[Vec<f32>],
        a: &[Vec<f32>],
        metric: DistMetric,
        max_batch: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let q_batch = qs.len();
        let d = qs.first().map(|q| q.len()).unwrap_or(0);
        let (kind, route_reason, fallback_reason) =
            self.choose_backend_for_batch(a.len(), d, q_batch);
        match self.run_batch_on(kind, qs, a, metric, max_batch) {
            Ok(results) => {
                self.set_last_route(kind, route_reason, fallback_reason, q_batch.max(1));
                Ok(results)
            }
            Err(err) if kind != BackendKind::Cpu => {
                let fallback = format!("{}; qnn/ort execution failed: {}", route_reason, err);
                let results = self.cpu.distance_batch(qs, a, metric, max_batch)?;
                self.set_last_route(
                    BackendKind::Cpu,
                    "cpu fallback after primary backend batch execution failure",
                    Some(fallback),
                    q_batch.max(1),
                );
                Ok(results)
            }
            Err(err) => Err(err),
        }
    }

    pub fn status_snapshot(&self) -> RuntimeStatus {
        self.refresh_qnn_fields();
        self.status.lock().expect("runtime status lock").clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

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

    #[test]
    fn cpu_backend_inner_product_prefers_closer_vector() {
        let backend = CpuBackend;
        let q = vec![1.0, 0.0];
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let res = backend.distance(&q, &a, DistMetric::Ip, Some(8)).unwrap();

        assert_eq!(res.distances.len(), 2);
        assert!(res.distances[0] < res.distances[1]);
    }

    #[test]
    fn cpu_backend_l2_returns_zero_for_identical_vector() {
        let backend = CpuBackend;
        let q = vec![0.5, -0.5, 1.0];
        let a = vec![q.clone(), vec![1.0, 1.0, 1.0]];

        let res = backend.distance(&q, &a, DistMetric::L2, Some(8)).unwrap();

        assert_eq!(res.distances.len(), 2);
        assert!(res.distances[0] <= 1e-6);
        assert!(res.distances[0] < res.distances[1]);
    }

    #[test]
    fn forced_qnn_without_shim_falls_back_to_cpu() {
        let _guard = ENV_LOCK.lock().unwrap();

        let kind = with_env_var("TACHANN_BACKEND", Some("qnn"), || {
            with_env_var(
                "TACHANN_QNN_LIB",
                Some("/definitely/missing/libtachann_qnnshim.so"),
                || select_backend().kind(),
            )
        });

        assert!(matches!(kind, BackendKind::Cpu));
    }

    #[test]
    fn default_mode_is_auto() {
        let _guard = ENV_LOCK.lock().unwrap();

        let mode = with_env_var("TACHANN_BACKEND", None, RequestedMode::from_env);

        assert_eq!(mode, RequestedMode::Auto);
    }

    #[test]
    fn adaptive_profile_prefers_cpu_for_large_singleton_high_dim() {
        assert!(adaptive_prefers_cpu(8192, 768, 1));
        assert!(!adaptive_prefers_cpu(1024, 128, 4));
    }
}
