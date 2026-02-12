use lazy_static::lazy_static;
use std::sync::Arc;

#[cfg(any(target_os = "macos", target_os = "windows"))]
use crate::dsp::metal_context::MetalContext;

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
use crate::framework::cuda::CudaContext;

pub struct GlobalBackend {
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    pub context: Arc<MetalContext>,

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    pub context: Arc<CudaContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Auto,
    Metal,
    Cuda,
    Vulkan,
}

impl GpuBackend {
    pub fn label(self) -> &'static str {
        match self {
            GpuBackend::Auto => "AUTO",
            GpuBackend::Metal => "METAL",
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Vulkan => "VULKAN",
        }
    }
}

pub fn available_backends() -> Vec<GpuBackend> {
    #[cfg(target_os = "macos")]
    {
        return vec![GpuBackend::Metal];
    }

    #[cfg(target_os = "windows")]
    {
        let mut list = Vec::new();
        if crate::framework::cuda::probe_cuda_available() {
            list.push(GpuBackend::Cuda);
        }
        if crate::framework::vulkan::probe_vulkan_available() {
            list.push(GpuBackend::Vulkan);
        }
        return list;
    }

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        vec![GpuBackend::Cuda]
    }
}

pub fn create_context(preferred: GpuBackend) -> Option<(Arc<MetalContext>, GpuBackend)> {
    #[cfg(target_os = "macos")]
    {
        let _ = preferred;
        return MetalContext::new().map(|ctx| (Arc::new(ctx), GpuBackend::Metal));
    }

    #[cfg(target_os = "windows")]
    {
        match preferred {
            GpuBackend::Auto => {
                if let Some(ctx) = MetalContext::new_cuda() {
                    return Some((Arc::new(ctx), GpuBackend::Cuda));
                }
                if let Some(ctx) = MetalContext::new_vulkan() {
                    return Some((Arc::new(ctx), GpuBackend::Vulkan));
                }
                None
            }
            GpuBackend::Cuda => {
                MetalContext::new_cuda().map(|ctx| (Arc::new(ctx), GpuBackend::Cuda))
            }
            GpuBackend::Vulkan => {
                MetalContext::new_vulkan().map(|ctx| (Arc::new(ctx), GpuBackend::Vulkan))
            }
            GpuBackend::Metal => None,
        }
    }

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        if matches!(preferred, GpuBackend::Auto | GpuBackend::Cuda) {
            return CudaContext::new().map(|ctx| (Arc::new(ctx), GpuBackend::Cuda));
        }
        None
    }
}

lazy_static! {
    pub static ref BACKEND: Option<Arc<GlobalBackend>> = {
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        {
            MetalContext::new().map(|ctx| {
                Arc::new(GlobalBackend {
                    context: Arc::new(ctx),
                })
            })
        }

        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            CudaContext::new().map(|ctx| {
                Arc::new(GlobalBackend {
                    context: Arc::new(ctx),
                })
            })
        }
    };
}

impl GlobalBackend {
    pub fn get() -> Option<Arc<GlobalBackend>> {
        BACKEND.clone()
    }
}
