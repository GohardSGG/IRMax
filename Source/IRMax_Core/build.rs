#[cfg(target_os = "windows")]
use std::env;
#[cfg(target_os = "windows")]
use std::path::PathBuf;

fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut candidates: Vec<PathBuf> = Vec::new();

        if let Ok(root) = env::var("IPP_ROOT") {
            candidates.push(PathBuf::from(root).join("lib"));
        }
        if let Ok(root) = env::var("ONEAPI_ROOT") {
            candidates.push(PathBuf::from(&root).join("ipp").join("latest").join("lib"));
            candidates.push(PathBuf::from(&root).join("ipp").join("2022.3").join("lib"));
        }

        candidates.push(PathBuf::from(
            r"C:\Program Files (x86)\Intel\oneAPI\ipp\latest\lib",
        ));
        candidates.push(PathBuf::from(
            r"C:\Program Files (x86)\Intel\oneAPI\ipp\2022.3\lib",
        ));

        let mut found = None;
        for path in candidates {
            if path.exists() {
                found = Some(path);
                break;
            }
        }

        let Some(lib_path) = found else {
            panic!("IPP library path not found. Set IPP_ROOT or ONEAPI_ROOT.");
        };

        println!("cargo:rerun-if-changed=src/shaders/fused_convolution.cu");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm60.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm61.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm70.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm75.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm80.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm86.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm89.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm90.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm100.ptx");
        println!("cargo:rerun-if-changed=src/shaders/fused_convolution_sm120.ptx");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=dylib=ippcore");
        println!("cargo:rustc-link-lib=dylib=ipps");
        println!("cargo:rustc-link-lib=dylib=ippvm");
    }
}
