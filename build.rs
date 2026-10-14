use std::path::PathBuf;
use std::process::Command;

/// `DRAMA_CUDA_ARCH` is one arch (`sm_89`) or a comma-separated list (`sm_75,sm_80,sm_86,sm_89,sm_120`, for release
/// builds): machine code for each and no PTX. PTX in a release is JIT-compiled by the driver, and a driver older than
/// the toolkit refuses it: a card the build has no machine code for is unsupported, never a crash on an old driver.
fn gencode(arch: &str) -> Vec<String> {
    arch.split(',')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(|a| {
            let n = a.trim_start_matches("sm_");
            format!("-gencode=arch=compute_{n},code=sm_{n}")
        })
        .collect()
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let windows = std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".to_string());
    let arch = std::env::var("DRAMA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DRAMA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if windows {
        // MSVC: nvcc packs both kernels into one static library against the dynamic C runtime (/MD), as Rust links
        let lib = out.join("dramakernels.lib");
        let _ = std::fs::remove_file(&lib);
        let status = Command::new(format!("{cuda}/bin/nvcc"))
            .args(["-O3", "-std=c++17", "-lineinfo", "-Xcompiler", "/MD"])
            .args(gencode(&arch))
            .arg("-lib")
            .args(["kernels/drama.cu", "kernels/codec.cu"])
            .arg("-o")
            .arg(&lib)
            .status()
            .expect("failed to run nvcc");
        assert!(status.success(), "nvcc failed");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=dramakernels");
        println!("cargo:rustc-link-search=native={cuda}/lib/x64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        return;
    }

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
