//! Tachyon QNN Shim
//! -----------------
//! Exposes two C-ABI functions that the server dlopen()s:
//!   - `tachann_qnn_available() -> i32`    (1 if QNN visible, 0 otherwise)
//!   - `tachann_qnn_matmul(...) -> i32`    (0 on success; <0 on error)
//!
//! Compatibility exports (older names):
//!   - `tachyon_qnn_avail()`
//!   - `tachyon_qnn_matmul(...)`
//!
//! When built with the `direct-qnn` feature and QNN is detected, the shim will
//! attempt direct QNN execution first and fall back to CPU on failure.
//! You can also call `tachann_qnn_warmup()` from the host to force a one-time
//! backend/device/context open+close cycle; this is useful to verify QNN works.
//!
//! Env:
//!   - QNN_SDK_ROOT           (e.g. /opt/qairt)
//!   - SHIM_FORCE_CPU=1       (force CPU even if QNN present)
//!   - SHIM_QNN_WARMUP=1      (run a minimal warmup probe)
//!   - LD_LIBRARY_PATH        (for loader-name detection)

use once_cell::sync::OnceCell;
use std::env;
use std::ffi::c_int;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "detect-qnn")]
use libloading::Library;

// ---- C glue (provided by tach_qnn_fc.c) ----
extern "C" {
    fn tach_qnn_warmup_open_close() -> c_int;
}

// ------------------------- logging helpers -------------------------

fn log_line(msg: &str) {
    eprintln!("[tachyon_qnnshim] {msg}");
}

macro_rules! log_once {
    ($cell:expr, $($arg:tt)*) => {{
        if !$cell.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log_line(&format!($($arg)*));
        }
    }};
}

// ------------------------- error codes (C-friendly) -------------------------

const ERR_NULLPTR: c_int = -1001;
const ERR_BADDIMS: c_int = -1002;
const ERR_OVERFLOW: c_int = -1003;

// ------------------------- backend detection -------------------------

#[derive(Clone, Debug)]
enum Backend {
    Cpu,
    QnnDetected(QnnWhere),
}

#[derive(Clone, Debug)]
struct QnnWhere {
    cpu_path: Option<PathBuf>,
    system_path: Option<PathBuf>,
}

static BACKEND: OnceCell<Backend> = OnceCell::new();
static DETECT_MSG_PRINTED: AtomicBool = AtomicBool::new(false);
static WARMUP_RESULT: OnceCell<c_int> = OnceCell::new();
static FP16_DISABLED: AtomicBool = AtomicBool::new(false);

fn candidate_subdirs(qnn_root: &Path) -> Vec<PathBuf> {
    let lib_root = qnn_root.join("lib");
    let mut v = vec![
        lib_root.join("aarch64-oe-linux-gcc11.2"),
        lib_root.join("aarch64-ubuntu-gcc9.4"),
        lib_root.join("aarch64-oe-linux-gcc9.3"),
        lib_root.join("x86_64-linux-gnu"),
        lib_root.join("linux-clang"),
        lib_root.clone(),
    ];
    if let Ok(rd) = lib_root.read_dir() {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() && !v.iter().any(|x| x == &p) {
                v.push(p);
            }
        }
    }
    v
}

#[cfg(feature = "detect-qnn")]
fn try_open_lib(p: &Path) -> bool {
    unsafe { Library::new(p).is_ok() }
}

#[cfg(feature = "detect-qnn")]
fn try_open_by_name(name: &str) -> bool {
    unsafe { Library::new(name).is_ok() }
}

#[cfg(feature = "detect-qnn")]
fn detect_qnn_impl() -> Option<Backend> {
    if env::var("SHIM_FORCE_CPU").ok().as_deref() == Some("1") {
        log_once!(
            DETECT_MSG_PRINTED,
            "SHIM_FORCE_CPU=1 set; forcing CPU mode."
        );
        return None;
    }

    // 1) Try simple loader resolution (LD_LIBRARY_PATH / system default)
    let name_cpu = "libQnnCpu.so";
    let name_sys = "libQnnSystem.so";
    let by_name_cpu = try_open_by_name(name_cpu);
    let by_name_sys = try_open_by_name(name_sys);

    if by_name_cpu && by_name_sys {
        log_once!(
            DETECT_MSG_PRINTED,
            "Detected QNN via loader paths (LD_LIBRARY_PATH)."
        );
        return Some(Backend::QnnDetected(QnnWhere {
            cpu_path: None,
            system_path: None,
        }));
    }

    // 2) Probe under QNN_SDK_ROOT
    let mut cpu_hit = None;
    let mut sys_hit = None;

    if let Ok(root) = env::var("QNN_SDK_ROOT") {
        let root = PathBuf::from(root);
        let mut searched = String::new();
        for dir in candidate_subdirs(&root) {
            if let Ok(s) = dir.canonicalize() {
                let _ = writeln!(searched, "  - {}", s.display());
            }
            let cpu = dir.join(name_cpu);
            let sys = dir.join(name_sys);

            if cpu_hit.is_none() && cpu.exists() && try_open_lib(&cpu) {
                cpu_hit = Some(cpu);
            }
            if sys_hit.is_none() && sys.exists() && try_open_lib(&sys) {
                sys_hit = Some(sys);
            }
            if cpu_hit.is_some() && sys_hit.is_some() {
                log_once!(
                    DETECT_MSG_PRINTED,
                    "Detected QNN under QNN_SDK_ROOT.\nSearched:\n{}",
                    searched
                );
                break;
            }
        }
        if cpu_hit.is_some() && sys_hit.is_some() {
            return Some(Backend::QnnDetected(QnnWhere {
                cpu_path: cpu_hit,
                system_path: sys_hit,
            }));
        } else {
            log_once!(
                DETECT_MSG_PRINTED,
                "QNN_SDK_ROOT provided, but required libs not found/openable.\n\
                 Needed both: {name_sys} and {name_cpu}\n\
                 Tip: ensure your runtime image sets LD_LIBRARY_PATH to include relevant subdirs \
                 under $QNN_SDK_ROOT/lib, or place those .so files directly on a loader path."
            );
        }
    } else {
        log_once!(
            DETECT_MSG_PRINTED,
            "QNN_SDK_ROOT is unset; relying solely on LD_LIBRARY_PATH/system loader paths."
        );
    }

    None
}

#[cfg(not(feature = "detect-qnn"))]
fn detect_qnn_impl() -> Option<Backend> {
    None
}

fn backend() -> &'static Backend {
    BACKEND.get_or_init(|| detect_qnn_impl().unwrap_or(Backend::Cpu))
}

// ------------------------- CPU implementation -------------------------

#[inline]
fn checked_slice_params(n: i32, d: i32) -> Result<(usize, usize, usize), c_int> {
    if n <= 0 || d <= 0 {
        return Err(ERR_BADDIMS);
    }
    let (n_u, d_u) = (n as usize, d as usize);
    let total = n_u.checked_mul(d_u).ok_or(ERR_OVERFLOW)?;
    Ok((n_u, d_u, total))
}

fn cpu_matmul(a: *const f32, n: i32, d: i32, q: *const f32, out: *mut f32) -> c_int {
    if a.is_null() || q.is_null() || out.is_null() {
        return ERR_NULLPTR;
    }
    let (n_u, d_u, total) = match checked_slice_params(n, d) {
        Ok(t) => t,
        Err(e) => return e,
    };
    unsafe {
        let a = slice::from_raw_parts(a, total);
        let q = slice::from_raw_parts(q, d_u);
        let out = slice::from_raw_parts_mut(out, n_u);

        for i in 0..n_u {
            let row = &a[i * d_u..(i + 1) * d_u];
            let mut acc = 0.0f32;
            for k in 0..d_u {
                acc += row[k] * q[k];
            }
            out[i] = acc;
        }
    }
    0
}

// ------------------------- banner + warmup -------------------------

fn print_detection_banner() {
    static BANNER_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if BANNER_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }

    let ld = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let qroot = env::var("QNN_SDK_ROOT").unwrap_or_else(|_| "<unset>".into());
    log_line(&format!("LD_LIBRARY_PATH={ld}"));
    log_line(&format!("QNN_SDK_ROOT={qroot}"));

    match backend() {
        Backend::Cpu => {
            log_line("Backend detection: CPU (QNN not detected or disabled).");
        }
        Backend::QnnDetected(where_) => {
            log_line("Backend detection: QNN detected.");
            if let Some(p) = &where_.system_path {
                log_line(&format!("  found libQnnSystem.so at {}", p.display()));
            }
            if let Some(p) = &where_.cpu_path {
                log_line(&format!("  found libQnnCpu.so    at {}", p.display()));
            }
            log_line("NOTE: direct QNN execution is enabled by default and will fall back to CPU on failure.");
        }
    }
}

fn maybe_qnn_warmup() {
    if env::var("SHIM_QNN_WARMUP").ok().as_deref() == Some("1") {
        if matches!(backend(), Backend::QnnDetected(_)) {
            log_line("SHIM_QNN_WARMUP=1: attempting QNN warmup (backend/device/context)...");
            let rc = warmup_once();
            log_line(&format!("QNN warmup result = {rc}"));
        } else {
            log_line("SHIM_QNN_WARMUP=1 set but QNN not detected; skipping warmup.");
        }
    }
}

fn warmup_once() -> c_int {
    *WARMUP_RESULT.get_or_init(|| {
        print_detection_banner();
        if !matches!(backend(), Backend::QnnDetected(_)) {
            return -3000;
        }
        // SAFETY: FFI call into our C file; side-effect only
        unsafe { tach_qnn_warmup_open_close() }
    })
}

// ------------------------- public C ABI (exports) -------------------------

fn available_impl() -> c_int {
    print_detection_banner();
    match backend() {
        Backend::Cpu => 0,
        Backend::QnnDetected(_) => 1,
    }
}

/// Implementation for both `tachann_qnn_matmul` and `tachyon_qnn_matmul`.
unsafe fn matmul_impl(a: *const f32, n: i32, d: i32, q: *const f32, out: *mut f32) -> c_int {
    print_detection_banner();
    // Optional probe; still compute on CPU.
    maybe_qnn_warmup();
    #[cfg(feature = "direct-qnn")]
    {
        let want_direct = env::var("SHIM_DIRECT_QNN")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);
        if want_direct && matches!(backend(), Backend::QnnDetected(_)) {
            let static_a = env::var("TACHANN_QNN_STATIC_A").ok().as_deref() == Some("1");
            let q_batch = std::env::var("TACHANN_QNN_Q_BATCH")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(1);
            let fp16_req = env::var("TACHANN_QNN_FP16").ok().as_deref() == Some("1");
            let fp16_ok = fp16_req && !FP16_DISABLED.load(Ordering::Relaxed);

            if fp16_ok {
                if direct::ensure_session(n, d, q_batch, a, static_a, true) {
                    let rc = direct::run(n, d, q_batch, a, q, out, static_a, true);
                    if rc == 0 {
                        return 0;
                    }
                    FP16_DISABLED.store(true, Ordering::Relaxed);
                    log_line(&format!(
                        "direct-qnn fp16 path failed rc={}; retrying fp32",
                        rc
                    ));
                } else {
                    FP16_DISABLED.store(true, Ordering::Relaxed);
                    log_line("direct-qnn fp16 session creation failed; retrying fp32");
                }
            }

            // FP32 path (either requested or fp16 failed)
            if !fp16_req || FP16_DISABLED.load(Ordering::Relaxed) {
                if direct::ensure_session(n, d, q_batch, a, static_a, false) {
                    let rc = direct::run(n, d, q_batch, a, q, out, static_a, false);
                    if rc == 0 {
                        return 0;
                    }
                    log_line(&format!(
                        "direct-qnn fp32 path failed rc={}; falling back to CPU",
                        rc
                    ));
                } else {
                    log_line("direct-qnn fp32 session creation failed; falling back to CPU");
                }
            }
        }
    }
    cpu_matmul(a, n, d, q, out)
}

unsafe fn cpu_matmul_batched(
    a: *const f32,
    n: i32,
    d: i32,
    q: *const f32,
    q_batch: i32,
    out: *mut f32,
) -> c_int {
    if a.is_null() || q.is_null() || out.is_null() || n <= 0 || d <= 0 || q_batch <= 0 {
        return -1;
    }
    let n = n as usize;
    let d = d as usize;
    let b = q_batch as usize;
    for i in 0..n {
        for j in 0..b {
            let mut acc = 0.0f32;
            for k in 0..d {
                let a_val = *a.add(i * d + k);
                let q_val = *q.add(k * b + j);
                acc += a_val * q_val;
            }
            *out.add(i * b + j) = acc;
        }
    }
    0
}

unsafe fn matmul_impl_batched(
    a: *const f32,
    n: i32,
    d: i32,
    q: *const f32,
    q_batch: i32,
    out: *mut f32,
) -> c_int {
    print_detection_banner();
    maybe_qnn_warmup();
    #[cfg(feature = "direct-qnn")]
    {
        let want_direct = env::var("SHIM_DIRECT_QNN")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);
        if want_direct && matches!(backend(), Backend::QnnDetected(_)) {
            let static_a = env::var("TACHANN_QNN_STATIC_A").ok().as_deref() == Some("1");
            let fp16_req = env::var("TACHANN_QNN_FP16").ok().as_deref() == Some("1");
            let fp16_ok = fp16_req && !FP16_DISABLED.load(Ordering::Relaxed);

            if fp16_ok {
                if direct::ensure_session(n, d, q_batch, a, static_a, true) {
                    let rc = direct::run(n, d, q_batch, a, q, out, static_a, true);
                    if rc == 0 {
                        return 0;
                    }
                    FP16_DISABLED.store(true, Ordering::Relaxed);
                    log_line(&format!(
                        "direct-qnn batched fp16 path failed rc={}; retrying fp32",
                        rc
                    ));
                } else {
                    FP16_DISABLED.store(true, Ordering::Relaxed);
                    log_line("direct-qnn batched fp16 session creation failed; retrying fp32");
                }
            }

            if !fp16_req || FP16_DISABLED.load(Ordering::Relaxed) {
                if direct::ensure_session(n, d, q_batch, a, static_a, false) {
                    let rc = direct::run(n, d, q_batch, a, q, out, static_a, false);
                    if rc == 0 {
                        return 0;
                    }
                    log_line(&format!(
                        "direct-qnn batched fp32 path failed rc={}; falling back to CPU",
                        rc
                    ));
                } else {
                    log_line(
                        "direct-qnn batched fp32 session creation failed; falling back to CPU",
                    );
                }
            }
        }
    }
    cpu_matmul_batched(a, n, d, q, q_batch, out)
}

// ---- Primary names the current server expects ----

#[no_mangle]
pub extern "C" fn tachann_qnn_available() -> c_int {
    available_impl()
}

#[no_mangle]
pub unsafe extern "C" fn tachann_qnn_matmul(
    a: *const f32,
    n: i32,
    d: i32,
    q: *const f32,
    out: *mut f32,
) -> c_int {
    matmul_impl(a, n, d, q, out)
}

#[no_mangle]
pub unsafe extern "C" fn tachann_qnn_matmul_batched(
    a: *const f32,
    n: i32,
    d: i32,
    q: *const f32,
    q_batch: i32,
    out: *mut f32,
) -> c_int {
    matmul_impl_batched(a, n, d, q, q_batch, out)
}

/// Force a one-time QNN warmup (backend/device/context open+close).
#[no_mangle]
pub extern "C" fn tachann_qnn_warmup() -> c_int {
    warmup_once()
}

#[no_mangle]
pub extern "C" fn tachann_qnn_fp16_disabled() -> c_int {
    if FP16_DISABLED.load(Ordering::Relaxed) {
        1
    } else {
        0
    }
}

// ---- Compatibility exports (older names some builds looked for) ----

#[no_mangle]
pub extern "C" fn tachyon_qnn_avail() -> c_int {
    available_impl()
}

#[no_mangle]
pub unsafe extern "C" fn tachyon_qnn_matmul(
    a: *const f32,
    n: i32,
    d: i32,
    q: *const f32,
    out: *mut f32,
) -> c_int {
    matmul_impl(a, n, d, q, out)
}

#[no_mangle]
pub unsafe extern "C" fn tachyon_qnn_matmul_batched(
    a: *const f32,
    n: i32,
    d: i32,
    q: *const f32,
    q_batch: i32,
    out: *mut f32,
) -> c_int {
    matmul_impl_batched(a, n, d, q, q_batch, out)
}

#[no_mangle]
pub extern "C" fn tachyon_qnn_warmup() -> c_int {
    warmup_once()
}

#[no_mangle]
pub extern "C" fn tachyon_qnn_fp16_disabled() -> c_int {
    tachann_qnn_fp16_disabled()
}

#[no_mangle]
pub extern "C" fn tachann_qnn_cleanup_all() {
    #[cfg(feature = "direct-qnn")]
    {
        direct::cleanup_all();
    }
}

#[no_mangle]
pub extern "C" fn tachyon_qnn_cleanup_all() {
    tachann_qnn_cleanup_all();
}

// ------------------------- optional future direct path -------------------------
// Keep this disabled unless you have the corresponding C entrypoints compiled in.
// It won’t be used by default CPU path.

#[cfg(feature = "direct-qnn")]
mod direct {
    use once_cell::sync::OnceCell;
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    use std::sync::Mutex;

    #[repr(C)]
    struct TachQnnCtx {
        _p: [u8; 0],
    }
    extern "C" {
        fn tach_qnn_fc_create(
            n: i32,
            d: i32,
            backend: *const c_char,
            use_fp16: c_int,
        ) -> *mut TachQnnCtx;

        fn tach_qnn_fc_create_batched(
            n: i32,
            d: i32,
            q_batch: i32,
            backend: *const c_char,
            use_fp16: c_int,
        ) -> *mut TachQnnCtx;

        fn tach_qnn_fc_create_static_a(
            n: i32,
            d: i32,
            a: *const f32,
            backend: *const c_char,
            use_fp16: c_int,
        ) -> *mut TachQnnCtx;

        fn tach_qnn_fc_create_static_a_batched(
            n: i32,
            d: i32,
            q_batch: i32,
            a: *const f32,
            backend: *const c_char,
            use_fp16: c_int,
        ) -> *mut TachQnnCtx;

        fn tach_qnn_fc_run(
            ctx: *mut TachQnnCtx,
            a: *const f32,
            q: *const f32,
            out: *mut f32,
        ) -> c_int;

        fn tach_qnn_fc_run_q(ctx: *mut TachQnnCtx, q: *const f32, out: *mut f32) -> c_int;

        fn tach_qnn_fc_destroy(ctx: *mut TachQnnCtx);
    }

    struct Session {
        #[allow(dead_code)]
        n: i32,
        #[allow(dead_code)]
        d: i32,
        ctx: *mut TachQnnCtx,
        a_ptr: usize,
    }
    unsafe impl Send for Session {}
    unsafe impl Sync for Session {}

    static SESSIONS: OnceCell<Mutex<HashMap<(i32, i32, i32, bool, bool), Session>>> =
        OnceCell::new();

    fn sessions() -> &'static Mutex<HashMap<(i32, i32, i32, bool, bool), Session>> {
        SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub fn ensure_session(
        n: i32,
        d: i32,
        q_batch: i32,
        a_ptr: *const f32,
        static_a: bool,
        use_fp16: bool,
    ) -> bool {
        let mut map = sessions().lock().unwrap();
        let key = (n, d, q_batch, static_a, use_fp16);
        if let Some(s) = map.get(&key) {
            if static_a && s.a_ptr != a_ptr as usize {
                unsafe { tach_qnn_fc_destroy(s.ctx) };
                map.remove(&key);
            } else {
                return !s.ctx.is_null();
            }
        }
        let backend = std::env::var("TACHANN_QNN_BACKEND").unwrap_or_else(|_| "htp".into());
        let c_backend = match CString::new(backend) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let ctx = if static_a {
            unsafe {
                if q_batch > 1 {
                    tach_qnn_fc_create_static_a_batched(
                        n,
                        d,
                        q_batch,
                        a_ptr,
                        c_backend.as_ptr(),
                        if use_fp16 { 1 } else { 0 },
                    )
                } else {
                    tach_qnn_fc_create_static_a(
                        n,
                        d,
                        a_ptr,
                        c_backend.as_ptr(),
                        if use_fp16 { 1 } else { 0 },
                    )
                }
            }
        } else {
            unsafe {
                if q_batch > 1 {
                    tach_qnn_fc_create_batched(
                        n,
                        d,
                        q_batch,
                        c_backend.as_ptr(),
                        if use_fp16 { 1 } else { 0 },
                    )
                } else {
                    tach_qnn_fc_create(n, d, c_backend.as_ptr(), if use_fp16 { 1 } else { 0 })
                }
            }
        };
        if ctx.is_null() {
            return false;
        }
        map.insert(
            key,
            Session {
                n,
                d,
                ctx,
                a_ptr: a_ptr as usize,
            },
        );
        true
    }

    pub fn run(
        n: i32,
        d: i32,
        q_batch: i32,
        a: *const f32,
        q: *const f32,
        out: *mut f32,
        static_a: bool,
        use_fp16: bool,
    ) -> c_int {
        let map = sessions().lock().unwrap();
        if let Some(s) = map.get(&(n, d, q_batch, static_a, use_fp16)) {
            if static_a {
                unsafe { return tach_qnn_fc_run_q(s.ctx, q, out) }
            }
            unsafe { return tach_qnn_fc_run(s.ctx, a, q, out) }
        }
        -997 // no session
    }

    pub fn cleanup_all() {
        let mut map = sessions().lock().unwrap();
        for (_, s) in map.drain() {
            unsafe { tach_qnn_fc_destroy(s.ctx) }
        }
    }
}
