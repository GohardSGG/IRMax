// Re-export
pub mod framework;
pub mod backend;
pub mod dsp;
pub mod engine;
pub mod loader;

// Back-compat re-exports (keep existing paths)
pub use framework::accelerate;
#[cfg(target_os = "windows")]
pub use framework::ipp;
pub use backend::{available_backends, GpuBackend};

// Preamble: Export Core Types
pub use engine::matrix_scheduler::{EngineType, HeadEngine, TailEngine};
pub use engine::matrix_scheduler::{MatrixAudioProcessor, MatrixScheduler, MatrixWorker};
pub use engine::matrix_scheduler::TailProbe;
// Alias for Plugin
pub type IRMaxProcessor = MatrixAudioProcessor;

// Temporary compatibility for benchmarks (until they are updated)
// Re-export head so old code works? Or update benchmarks?
// Ideally update benchmarks. But let's verify what `tail_gpu.rs` and `lib.rs` contains.
// `tail_gpu.rs` is currently a file in src/.
// pub mod tail_cpu; // Moved to dsp

// Remove old head export
// pub mod head;::HeadProcessor;
