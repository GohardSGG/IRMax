// --- 1. FFT Context (Shim: dsp::fft) ---
#[cfg(target_os = "macos")]
pub mod fft_vdsp;
#[cfg(target_os = "macos")]
pub use fft_vdsp as fft;

#[cfg(target_os = "windows")]
pub mod ipp {
    pub use crate::framework::ipp::*;
}

#[cfg(target_os = "windows")]
pub mod fft_ipp;
#[cfg(target_os = "windows")]
pub use fft_ipp as fft;

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
pub mod fft_rustfft;
#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
pub use fft_rustfft as fft;

// --- 2. Head Processor (Shim: dsp::head_titan) ---
// pub mod head_cpu; // Removed

// Windows: Use IPP implementation
#[cfg(target_os = "windows")]
pub mod head_ipp;

#[cfg(target_os = "windows")]
pub mod head_titan {
    pub use crate::dsp::head_ipp::HeadRowIpp as TitanHeadRow;
}

// Non-Windows platforms: provide a compatible TitanHeadRow wrapper over HeadVdsp.
#[cfg(not(target_os = "windows"))]
pub mod head_titan {
    use crate::dsp::head_vdsp::HeadVdsp;

    pub struct TitanHeadRow {
        inner: HeadVdsp,
    }

    impl TitanHeadRow {
        pub fn new(row_irs: &[&[f32]], head_len: usize) -> Self {
            const DEFAULT_MAX_BUFFER: usize = 2048;
            let mut owned: Vec<Vec<f32>> = Vec::with_capacity(row_irs.len());
            for ir in row_irs {
                let len = head_len.min(ir.len());
                owned.push(ir[0..len].to_vec());
            }
            let inner = HeadVdsp::new(&owned, DEFAULT_MAX_BUFFER);
            Self { inner }
        }

        pub fn process(&mut self, input: &[f32], outputs: &mut [&mut [f32]]) {
            self.inner.process(input, outputs);
        }

        pub fn reset_state(&mut self) {
            self.inner.reset_state();
        }
    }
}

// Low-Priority: vDSP Head Shim (Dead code on Windows usually)
#[cfg(target_os = "macos")]
pub mod head_vdsp;

#[cfg(not(target_os = "macos"))]
pub mod head_vdsp {
    pub struct HeadVdsp;
    impl HeadVdsp {
        pub fn new(_: &[Vec<f32>], _: usize) -> Self {
            Self
        }
        pub fn process(&mut self, _: &[f32], _: &mut [&mut [f32]]) {}
        pub fn reset_state(&mut self) {}
    }
}

// --- 3. GPU/Tail Processor (Shim: dsp::tail_gpu & dsp::metal_context) ---

#[cfg(target_os = "macos")]
pub mod metal_context {
    pub use crate::framework::metal::MetalContext;
}

#[cfg(target_os = "windows")]
pub mod metal_context {
    use crate::framework::cuda::CudaContext;
    use crate::framework::vulkan::VulkanContext;
    use std::sync::Arc;

    pub enum MetalContext {
        Cuda(Arc<CudaContext>),
        Vulkan(Arc<VulkanContext>),
    }

    impl MetalContext {
        pub fn new() -> Option<Self> {
            if let Some(ctx) = Self::new_cuda() {
                return Some(ctx);
            }
            Self::new_vulkan()
        }

        pub fn new_vulkan() -> Option<Self> {
            VulkanContext::new().map(|ctx| MetalContext::Vulkan(ctx))
        }

        pub fn new_cuda() -> Option<Self> {
            CudaContext::new().map(|ctx| MetalContext::Cuda(Arc::new(ctx)))
        }

        pub fn backend_label(&self) -> &'static str {
            match self {
                MetalContext::Cuda(_) => "CUDA",
                MetalContext::Vulkan(_) => "VULKAN",
            }
        }
    }
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
pub mod metal_context {
    // Shim MetalContext -> CudaContext (non-Windows fallback)
    pub use crate::framework::cuda::CudaContext as MetalContext;
}

#[cfg(not(target_os = "macos"))]
pub mod cuda_context {
    pub use crate::framework::cuda::CudaContext;
}

#[cfg(not(target_os = "macos"))]
pub mod tail_cuda;
#[cfg(not(target_os = "macos"))]
pub mod tail_vulkan;

#[cfg(target_os = "macos")]
pub mod tail_metal; // Renamed from tail_gpu.rs

// Shim tail_gpu -> tail_metal (Mac) or tail_cuda (Win)
#[cfg(target_os = "macos")]
pub mod tail_gpu {
    pub use crate::dsp::tail_metal::GpuTailProcessor;
}

#[cfg(target_os = "windows")]
pub mod tail_gpu {
    use super::metal_context::MetalContext;
    use crate::dsp::tail_cuda::CudaTailProcessor;
    use crate::dsp::tail_vulkan::VulkanTailProcessor;
    use std::sync::Arc;

    enum TailImpl {
        Cuda(CudaTailProcessor),
        Vulkan(VulkanTailProcessor),
    }

    pub struct GpuTailProcessor {
        inner: TailImpl,
        pub last_driver_us: u64,
        pub last_gpu_us: u64,
    }

    impl GpuTailProcessor {
        pub fn new(
            context: Arc<MetalContext>,
            num_inputs: usize,
            num_outputs: usize,
            block_size: usize,
            ir_data: &[f32],
            active_inputs: &[u32],
        ) -> Option<Self> {
            match context.as_ref() {
                MetalContext::Cuda(ctx) => {
                    let tail = CudaTailProcessor::new(
                        ctx.clone(),
                        num_inputs,
                        num_outputs,
                        block_size,
                        ir_data,
                        active_inputs,
                    )?;
                    return Some(Self {
                        inner: TailImpl::Cuda(tail),
                        last_driver_us: 0,
                        last_gpu_us: 0,
                    });
                }
                MetalContext::Vulkan(ctx) => {
                    let tail = VulkanTailProcessor::new(
                        ctx.clone(),
                        num_inputs,
                        num_outputs,
                        block_size,
                        ir_data,
                        active_inputs,
                    )?;
                    return Some(Self {
                        inner: TailImpl::Vulkan(tail),
                        last_driver_us: 0,
                        last_gpu_us: 0,
                    });
                }
            }
        }

        pub fn process(&mut self, input: &[f32]) -> &[f32] {
            match &mut self.inner {
                TailImpl::Cuda(tail) => {
                    let tail_ptr = tail as *const CudaTailProcessor;
                    let out = tail.process(input);
                    unsafe {
                        self.last_driver_us = (*tail_ptr).last_driver_us;
                        self.last_gpu_us = (*tail_ptr).last_gpu_us;
                    }
                    out
                }
                TailImpl::Vulkan(tail) => {
                    let tail_ptr = tail as *const VulkanTailProcessor;
                    let out = tail.process(input);
                    unsafe {
                        self.last_driver_us = (*tail_ptr).last_driver_us;
                        self.last_gpu_us = (*tail_ptr).last_gpu_us;
                    }
                    out
                }
            }
        }

        pub fn reset_state(&mut self) {
            match &mut self.inner {
                TailImpl::Cuda(tail) => tail.reset_state(),
                TailImpl::Vulkan(tail) => tail.reset_state(),
            }
        }
    }
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
pub mod tail_gpu {
    pub use crate::dsp::tail_cuda::CudaTailProcessor as GpuTailProcessor;
}

// pub mod tail_cpu; // Removed
// #[cfg(not(target_os = "macos"))]
// pub mod tail_cpu; // Removed
