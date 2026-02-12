#![cfg(target_os = "macos")]

use metal::*;

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub pipeline: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        // Keep source-first behavior for maximum compatibility across macOS/Metal versions.
        // If runtime compilation fails, fallback to embedded metallib.
        const LIB_METALLIB: &[u8] = include_bytes!("../shaders/lib/fused_convolution.metallib");
        const LIB_SOURCE: &str = include_str!("../shaders/fused_convolution.metal");

        let pipeline = Self::build_pipeline_from_source(&device, LIB_SOURCE)
            .or_else(|| Self::build_pipeline_from_metallib(&device, LIB_METALLIB))?;

        Some(Self {
            device,
            queue,
            pipeline,
        })
    }

    fn build_pipeline_from_source(device: &Device, source: &str) -> Option<ComputePipelineState> {
        let options = CompileOptions::new();
        let library = device.new_library_with_source(source, &options).ok()?;
        Self::build_pipeline_from_library(device, &library)
    }

    fn build_pipeline_from_metallib(device: &Device, data: &[u8]) -> Option<ComputePipelineState> {
        let library = device.new_library_with_data(data).ok()?;
        Self::build_pipeline_from_library(device, &library)
    }

    fn build_pipeline_from_library(
        device: &Device,
        library: &Library,
    ) -> Option<ComputePipelineState> {
        let function = library.get_function("fused_mimo_convolution", None).ok()?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .ok()
    }
}
