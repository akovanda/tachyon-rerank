use std::slice;

/// Safe, simple CPU matmul: out[i] = dot(A[i,*], q), A is n x d row-major
pub unsafe fn cpu_matmul(
    a: *const f32, n: i32, d: i32,
    q: *const f32,
    out: *mut f32,
) -> i32 {
    if a.is_null() || q.is_null() || out.is_null() || n <= 0 || d <= 0 { return -1; }

    let (n, d) = (n as usize, d as usize);
    let a = slice::from_raw_parts(a, n * d);
    let q = slice::from_raw_parts(q, d);
    let out = slice::from_raw_parts_mut(out, n);

    for i in 0..n {
        let row = &a[i*d .. (i+1)*d];
        let mut acc = 0.0f32;
        // Simple loop; replace with f32x16 or rayon if you want
        for k in 0..d { acc += row[k] * q[k]; }
        out[i] = acc;
    }
    0
}

/// L2-squared distances: out[i] = sum_k (A[i,k] - q[k])^2
pub unsafe fn cpu_l2sq(
    a: *const f32, n: i32, d: i32,
    q: *const f32,
    out: *mut f32,
) -> i32 {
    if a.is_null() || q.is_null() || out.is_null() || n <= 0 || d <= 0 { return -1; }

    let (n, d) = (n as usize, d as usize);
    let a = slice::from_raw_parts(a, n * d);
    let q = slice::from_raw_parts(q, d);
    let out = slice::from_raw_parts_mut(out, n);

    for i in 0..n {
        let row = &a[i*d .. (i+1)*d];
        let mut acc = 0.0f32;
        for k in 0..d {
            let diff = row[k] - q[k];
            acc += diff * diff;
        }
        out[i] = acc;
    }
    0
}

