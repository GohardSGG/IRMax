#![cfg(not(target_os = "macos"))]

use cudarc::driver::sys::CUdevice_attribute_enum;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

pub struct CudaContext {
    pub device: Arc<CudaDevice>,
}

impl CudaContext {
    pub fn new() -> Option<Self> {
        // Initialize CUDA Device 0
        println!("🚀 调试: 初始化 CudaDevice::new(0)...");
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                println!("❌ 调试: CudaDevice::new(0) 失败: {:?}", e);
                return None;
            }
        };

        // Resolve compute capability (major.minor) for PTX selection.
        let cc_major = device
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let cc_minor = device
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);

        // Select PTX based on compute capability.
        // Precision routing for optimal performance on newer architectures.
        let (ptx_label, ptx_primary) = match (cc_major, cc_minor) {
            (12, _) => (
                "sm_120",
                include_str!("../shaders/lib/fused_convolution_sm120.ptx"),
            ),
            (10, _) => (
                "sm_100",
                include_str!("../shaders/lib/fused_convolution_sm100.ptx"),
            ),
            (9, 0) => (
                "sm_90",
                include_str!("../shaders/lib/fused_convolution_sm90.ptx"),
            ),
            (8, 9) => (
                "sm_89",
                include_str!("../shaders/lib/fused_convolution_sm89.ptx"),
            ),
            (8, 6) => (
                "sm_86",
                include_str!("../shaders/lib/fused_convolution_sm86.ptx"),
            ),
            (8, 0) => (
                "sm_80",
                include_str!("../shaders/lib/fused_convolution_sm80.ptx"),
            ),
            (7, 5) => (
                "sm_75",
                include_str!("../shaders/lib/fused_convolution_sm75.ptx"),
            ),
            (7, 0) => (
                "sm_70",
                include_str!("../shaders/lib/fused_convolution_sm70.ptx"),
            ),
            (6, 1) => (
                "sm_61",
                include_str!("../shaders/lib/fused_convolution_sm61.ptx"),
            ),
            (6, 0) => (
                "sm_60",
                include_str!("../shaders/lib/fused_convolution_sm60.ptx"),
            ),
            // Fallbacks for unusual variations
            (m, _) if m >= 12 => (
                "sm_120",
                include_str!("../shaders/lib/fused_convolution_sm120.ptx"),
            ),
            (m, _) if m >= 10 => (
                "sm_100",
                include_str!("../shaders/lib/fused_convolution_sm100.ptx"),
            ),
            (m, _) if m >= 9 => (
                "sm_90",
                include_str!("../shaders/lib/fused_convolution_sm90.ptx"),
            ),
            (m, _) if m >= 8 => (
                "sm_86",
                include_str!("../shaders/lib/fused_convolution_sm86.ptx"),
            ),
            (7, _) => (
                "sm_75",
                include_str!("../shaders/lib/fused_convolution_sm75.ptx"),
            ),
            _ => (
                "sm_61",
                include_str!("../shaders/lib/fused_convolution_sm61.ptx"),
            ),
        };

        let ptx_fallback = if cc_major >= 8 {
            include_str!("../shaders/lib/fused_convolution_sm75.ptx")
        } else {
            include_str!("../shaders/lib/fused_convolution_sm61.ptx")
        };

        println!(
            "🚀 调试: CUDA 计算能力 {}.{} -> 选择 PTX {} (长度={})",
            cc_major,
            cc_minor,
            ptx_label,
            ptx_primary.len()
        );

        // Load module "irmax" with kernel "fused_mimo"
        println!("🚀 调试: device.load_ptx...");
        if let Err(e) = device.load_ptx(ptx_primary.into(), "irmax", &["fused_mimo"]) {
            println!("❌ 调试: load_ptx 失败 ({}) : {:?}", ptx_label, e);
            // Fallback to the other PTX to maximize compatibility.
            if let Err(e2) = device.load_ptx(ptx_fallback.into(), "irmax", &["fused_mimo"]) {
                println!("❌ 调试: load_ptx 备用版本失败: {:?}", e2);
                return None;
            }
        }

        println!("🚀 调试: CudaContext 初始化成功。");
        Some(Self { device })
    }
}

pub fn probe_cuda_available() -> bool {
    // Probe using the same path as real runtime initialization (device + PTX load).
    CudaContext::new().is_some()
}
