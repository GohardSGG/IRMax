// Framework-level OS bindings and low-level DSP runtimes

pub mod accelerate;

pub mod r#type;

#[cfg(target_os = "windows")]
pub mod ipp;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(not(target_os = "macos"))]
pub mod cuda;

#[cfg(not(target_os = "macos"))]
pub mod vulkan;

// Re-export for convenience (allow crate::framework::* access)
pub use accelerate::*;
pub use r#type::*;

#[cfg(target_os = "windows")]
pub use ipp::*;

#[cfg(target_os = "macos")]
pub use metal::*;

#[cfg(not(target_os = "macos"))]
pub use cuda::*;

#[cfg(not(target_os = "macos"))]
pub use vulkan::*;
