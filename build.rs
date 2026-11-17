use std::path::PathBuf;
use std::process::Command;

/// `DRAMA_CUDA_ARCH` is one arch (`sm_89`) or a comma-separated list (`sm_80,sm_86,sm_89`, for release builds):
/// machine code for each, plus PTX of the newest so later GPUs can JIT it.
fn gencode(arch: &str) -> Vec<String> {
    let archs: Vec<&str> = arch.split(',').map(str::trim).filter(|a| !a.is_empty()).collect();
    let mut args = Vec::new();
    for (i, a) in archs.iter().enumerate() {
        let n = a.trim_start_matches("sm_");
        let code = if i + 1 == archs.len() { format!("[sm_{n},compute_{n}]") } else { format!("sm_{n}") };
        args.push(format!("-gencode=arch=compute_{n},code={code}"));
    }
    args
}

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DRAMA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    // Without the `cuda` feature there is nothing native to compile: src/ffi.rs
    // provides stub functions with identical signatures that report the missing
    // backend at runtime, so the crate still builds and links (e.g. on macOS).
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        if std::env::var("CARGO_FEATURE_METAL").is_err() {
            println!("cargo:warning=drama-rs: building without the `cuda` or `metal` features; inference will fail at runtime");
        }
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".to_string());
    let arch = std::env::var("DRAMA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());

    let mut objs = Vec::new();
    for src in ["drama", "codec"] {
        let obj = out.join(format!("{src}.o"));
        let status = Command::new(format!("{cuda}/bin/nvcc"))
            .args(["-O3", "-std=c++17", "-lineinfo", "-Xcompiler", "-fPIC"])
            .args(gencode(&arch))
            .arg("-c")
            .arg(format!("kernels/{src}.cu"))
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to run nvcc");
        assert!(status.success(), "nvcc failed on {src}.cu");
        objs.push(obj);
    }
    let lib = out.join("libdramakernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=dramakernels");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
