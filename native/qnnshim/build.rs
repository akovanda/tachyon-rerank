// native/qnnshim/build.rs
use std::env;
use std::path::Path;

fn main() {
    let qnn_root = env::var("QNN_SDK_ROOT")
        .expect("QNN_SDK_ROOT must point at your QNN SDK (with include/ and lib/)");

    // Re-run when env or sources change
    println!("cargo:rerun-if-env-changed=QNN_SDK_ROOT");
    println!("cargo:rerun-if-changed=src/tach_qnn_fc.c");
    println!("cargo:rerun-if-changed=src/tach_qnn_fc.h");

    // Sanity: the key headers should exist
    for hdr in [
        "include/QNN/QnnCommon.h",
        "include/QNN/QnnInterface.h",
        "include/QNN/QnnTensor.h",
    ] {
        let p = Path::new(&qnn_root).join(hdr);
        if !p.exists() {
            panic!("Expected header missing: {}", p.display());
        }
    }

    // Compile the small C wrapper (C11)
    let mut b = cc::Build::new();
    b.file("src/tach_qnn_fc.c")
        .include(format!("{}/include", qnn_root))
        .include(format!("{}/include/QNN", qnn_root))
        .include(format!("{}/include/QNN/HTP", qnn_root))
        .flag_if_supported("-std=c11")
        .flag_if_supported("-fvisibility=hidden");
    println!("cargo:rustc-link-search=native={}/lib", qnn_root);
    println!(
        "cargo:rustc-link-search=native={}/lib/aarch64-oe-linux-gcc11.2",
        qnn_root
    );
    println!(
        "cargo:rustc-link-search=native={}/lib/x86_64-linux-gnu",
        qnn_root
    );
    println!("cargo:rustc-link-lib=dylib=QnnSystem");
    println!("cargo:rustc-link-lib=dylib=dl");

    b.compile("tach_qnn_fc");

    // Where to search for libs at link time
    println!("cargo:rustc-link-search=native={}/lib", qnn_root);
    println!(
        "cargo:rustc-link-search=native={}/lib/aarch64-oe-linux-gcc11.2",
        qnn_root
    );
    println!(
        "cargo:rustc-link-search=native={}/lib/x86_64-linux-gnu",
        qnn_root
    );

    // We call QnnInterface_getProviders(), which lives in libQnnSystem.so
    println!("cargo:rustc-link-lib=dylib=QnnSystem");

    // On some systems you may also need libdl (usually brought in by libc, but harmless):
    println!("cargo:rustc-link-lib=dylib=dl");
}
