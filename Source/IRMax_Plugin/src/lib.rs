#![allow(non_snake_case)] // Crate name matches project branding
use irmax_core::backend::GlobalBackend;
use irmax_core::{available_backends, GpuBackend};
use log::{error, info};
use nih_plug::prelude::*;
use std::sync::{Arc, OnceLock};
#[cfg(target_os = "windows")]
use std::sync::Once;

use crossbeam_channel::Receiver;
use irmax_core::IRMaxProcessor;
use parking_lot::RwLock; // Use parking_lot for better mutex
use serde::{Deserialize, Serialize}; // For Persistence
use std::path::PathBuf;

const SLOT_COUNT: usize = 5;
const SLOT_MAX_INDEX: i32 = (SLOT_COUNT as i32) - 1;
const GUARD_PROFILE_MIN: i32 = -2;
const GUARD_PROFILE_MAX: i32 = 1;
const GUARD_PROFILE_BASELINE: i32 = 0;

#[inline]
fn clamp_slot(slot: i32) -> i32 {
    slot.clamp(0, SLOT_MAX_INDEX)
}

#[inline]
pub(crate) fn clamp_guard_profile(profile: i32) -> i32 {
    profile.clamp(GUARD_PROFILE_MIN, GUARD_PROFILE_MAX)
}

#[inline]
pub(crate) fn format_guard_profile_label(profile: i32) -> String {
    match clamp_guard_profile(profile) {
        -2 => "Aggressive 2".to_string(),
        -1 => "Aggressive 1".to_string(),
        0 => "Baseline".to_string(),
        1 => "Stable 1".to_string(),
        _ => "Baseline".to_string(),
    }
}

#[cfg(target_os = "windows")]
fn preload_ipp_runtime() {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, LoadLibraryExW,
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };

    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        let mut module: HMODULE = 0;
        let addr = preload_ipp_runtime as *const () as *const u16;
        let flags =
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        if GetModuleHandleExW(flags, addr, &mut module) == 0 {
            return;
        }

        let mut buf = vec![0u16; 1024];
        let len = GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32);
        if len == 0 {
            return;
        }
        buf.truncate(len as usize);
        let mut path = std::path::PathBuf::from(OsString::from_wide(&buf));
        path.pop();

        let dlls = ["ippcore.dll", "ipps.dll", "ippvm.dll"];
        for dll in dlls {
            let dll_path = path.join(dll);
            let wide: Vec<u16> = dll_path.as_os_str().encode_wide().chain(Some(0)).collect();
            let _ = LoadLibraryExW(
                wide.as_ptr(),
                0,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            );
        }
    });
}

mod editor;
pub mod loader;
pub mod logger;
pub mod preset;
mod worker;

pub struct IRMaxPlugin {
    params: Arc<IRMaxPluginParams>,

    // Logger
    pub logger: Arc<logger::InstanceLogger>,

    // Hot-Swap: Receiver for new Audio Processors from Loader
    pub processor_inbox: Receiver<IRMaxProcessor>,
    // The active processor (Owned by Audio Thread)
    pub current_processor: Option<IRMaxProcessor>,
    // Sender to verify we can initialize it (Logic in Default/New)
    // We need to keep the Sender end somewhere? No, the Loader needs it.
    // The Plugin creates the channel and gives Sender to Loader.

    // Command Interface
    pub loader: loader::Loader,

    // Background Worker for GPU
    pub worker_thread: worker::WorkerThread,

    // Scratch Buffers for Planar adaptation
    scratch_input: Vec<Vec<f32>>,
    scratch_output: Vec<Vec<f32>>,

    // Runtime State (Transient, not serialized)
    pub runtime: Arc<RwLock<RuntimePluginState>>,

    last_playing: bool,
    pending_reset: bool,
    last_pos_samples: Option<i64>,
    last_block_samples: Option<i64>,
    last_process_instant: Option<std::time::Instant>,
    last_diag_log_instant: Option<std::time::Instant>,
    diag_prev_snapshot: Option<DiagPerfSnapshot>,
    last_requested_sample_rate: u32,
    last_slot_param_value: i32,
    last_slot_load_path: String,
    last_guard_profile_value: i32,
}

pub fn default_slot_param() -> i32 {
    0
}

fn pick_default_backend(available: &[GpuBackend]) -> GpuBackend {
    if available.contains(&GpuBackend::Cuda) {
        GpuBackend::Cuda
    } else if available.contains(&GpuBackend::Vulkan) {
        GpuBackend::Vulkan
    } else if available.contains(&GpuBackend::Metal) {
        GpuBackend::Metal
    } else {
        GpuBackend::Auto
    }
}

fn current_build_kind() -> &'static str {
    if cfg!(feature = "production") {
        "Production"
    } else if cfg!(debug_assertions) {
        "Dev"
    } else {
        "Release"
    }
}

#[derive(Clone)]
struct DiagPerfSnapshot {
    at: std::time::Instant,
    tail_pushed: u64,
    tail_consumed: u64,
    tail_results: u64,
    body_pushed: u64,
    body_consumed: u64,
    body_results: u64,
}

fn diag_log_interval() -> std::time::Duration {
    static INTERVAL_MS: OnceLock<u64> = OnceLock::new();
    let ms = *INTERVAL_MS.get_or_init(|| {
        std::env::var("IRMAX_DIAG_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v.clamp(100, 5000))
            .unwrap_or(500)
    });
    std::time::Duration::from_millis(ms)
}

#[inline]
fn rate_per_sec(now: u64, prev: u64, elapsed_s: f64) -> f64 {
    if elapsed_s <= 0.0 {
        0.0
    } else {
        now.saturating_sub(prev) as f64 / elapsed_s
    }
}

#[derive(Serialize, Deserialize)]
pub struct SharedPluginState {
    #[serde(default)]
    pub last_folder: String,

    #[serde(default = "default_slot_paths")]
    pub slot_paths: [String; SLOT_COUNT],

    #[serde(default)]
    pub preset_root: String,

    #[serde(default)]
    pub ui_dark_mode: bool,
}

pub fn default_slot_paths() -> [String; SLOT_COUNT] {
    std::array::from_fn(|_| String::new())
}

impl Default for SharedPluginState {
    fn default() -> Self {
        Self {
            last_folder: String::new(),
            slot_paths: default_slot_paths(),
            preset_root: String::new(),
            ui_dark_mode: false,
        }
    }
}

#[inline]
fn slot_path_from_shared(shared: &SharedPluginState, slot: i32) -> String {
    let index = clamp_slot(slot) as usize;
    shared.slot_paths[index].clone()
}

#[inline]
fn format_slot_label(slot: i32) -> String {
    format!("Slot {}", clamp_slot(slot) + 1)
}

pub struct RuntimePluginState {
    pub loaded_info: Option<loader::LoadedInfo>,
    pub init_load_triggered: bool,
    pub preview_mode: bool,
    pub host_sample_rate: u32,
    pub host_outputs: usize,
    pub host_max_buffer: usize,
    /// Set to true after initialize() completes, signaling that SharedPluginState is fully deserialized
    pub persistence_ready: bool,
    pub gpu_backend: GpuBackend,
    pub gpu_backend_pending: Option<GpuBackend>,
    pub gpu_backends: Vec<GpuBackend>,
    pub gpu_can_switch: bool,
}

impl Default for RuntimePluginState {
    fn default() -> Self {
        Self {
            loaded_info: None,
            init_load_triggered: false,
            preview_mode: false,
            host_sample_rate: 0,
            host_outputs: 0,
            host_max_buffer: 0,
            persistence_ready: false,
            gpu_backend: GpuBackend::Auto,
            gpu_backend_pending: None,
            gpu_backends: Vec::new(),
            gpu_can_switch: false,
        }
    }
}

#[derive(Params)]
pub struct IRMaxPluginParams {
    #[persist = "shared_v3"]
    pub shared: Arc<RwLock<SharedPluginState>>,

    #[id = "preset_slot_select"]
    pub preset_slot_select: IntParam,

    #[id = "mix"]
    pub mix: FloatParam,

    #[id = "output_gain"]
    pub output_gain: FloatParam,

    #[id = "guard_profile"]
    pub guard_profile: IntParam,

    #[id = "dirty_trigger"]
    pub dirty_trigger: BoolParam,
}

impl Default for IRMaxPlugin {
    fn default() -> Self {
        // Initialize Logger
        let instance_id = logger::generate_instance_id();
        let logger = logger::InstanceLogger::new(&instance_id);
        logger.info("Plugin", "Creating new IRMaxPlugin instance...");

        let (proc_tx, proc_rx) = crossbeam_channel::bounded(1);

        // Start Background Worker
        let worker_thread = worker::WorkerThread::new(Some(logger.clone()));
        let worker_tx = worker_thread.get_sender();

        let loader = loader::Loader::new(
            proc_tx,
            worker_tx,
            Some(logger.clone()),
        );

        Self {
            params: Arc::new(IRMaxPluginParams::default()),
            logger,
            processor_inbox: proc_rx,
            current_processor: None,
            loader,
            worker_thread,
            scratch_input: Vec::new(),
            scratch_output: Vec::new(),
            runtime: Arc::new(RwLock::new(RuntimePluginState::default())),
            last_playing: false,
            pending_reset: false,
            last_pos_samples: None,
            last_block_samples: None,
            last_process_instant: None,
            last_diag_log_instant: None,
            diag_prev_snapshot: None,
            last_requested_sample_rate: 0,
            last_slot_param_value: default_slot_param(),
            last_slot_load_path: String::new(),
            last_guard_profile_value: GUARD_PROFILE_BASELINE,
        }
    }
}

impl Default for IRMaxPluginParams {
    fn default() -> Self {
        Self {
            preset_slot_select: IntParam::new(
                "Preset Slot",
                default_slot_param(),
                IntRange::Linear {
                    min: 0,
                    max: SLOT_MAX_INDEX,
                },
            )
            .with_value_to_string(std::sync::Arc::new(format_slot_label))
            .with_string_to_value(std::sync::Arc::new(|text: &str| {
                let trimmed = text.trim();
                if let Ok(raw) = trimmed.parse::<i32>() {
                    let zero_based = raw - 1;
                    if (0..=SLOT_MAX_INDEX).contains(&zero_based) {
                        return Some(zero_based);
                    }
                }

                let lower = trimmed.to_ascii_lowercase();
                let parsed = lower
                    .strip_prefix("slot")
                    .map(str::trim)
                    .and_then(|v| v.parse::<i32>().ok())
                    .or_else(|| lower.strip_prefix('#').and_then(|v| v.parse::<i32>().ok()))
                    .or_else(|| lower.parse::<i32>().ok())?;

                let zero_based = if parsed >= 1 { parsed - 1 } else { parsed };
                if (0..=SLOT_MAX_INDEX).contains(&zero_based) {
                    Some(zero_based)
                } else {
                    None
                }
            })),

            mix: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            output_gain: FloatParam::new(
                "Output Gain",
                util::db_to_gain(0.0), // Default 0 dB (Unity)
                FloatRange::Skewed {
                    min: util::db_to_gain(-100.0), // "Silence" (Effectively)
                    max: util::db_to_gain(6.0),    // +6 dB
                    factor: FloatRange::gain_skew_factor(-100.0, 6.0),
                },
            )
            .with_unit(" dB")
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2)),

            guard_profile: IntParam::new(
                "Guard Profile",
                GUARD_PROFILE_BASELINE,
                IntRange::Linear {
                    min: GUARD_PROFILE_MIN,
                    max: GUARD_PROFILE_MAX,
                },
            )
            .with_value_to_string(std::sync::Arc::new(format_guard_profile_label))
            .with_string_to_value(std::sync::Arc::new(|text: &str| {
                let trimmed = text.trim();
                if let Ok(v) = trimmed.parse::<i32>() {
                    let clamped = clamp_guard_profile(v);
                    if (GUARD_PROFILE_MIN..=GUARD_PROFILE_MAX).contains(&clamped) {
                        return Some(clamped);
                    }
                }

                let lower = trimmed.to_ascii_lowercase();
                match lower.as_str() {
                    "aggressive2" | "aggressive_2" | "fast2" => Some(-2),
                    "aggressive1" | "aggressive_1" | "fast1" => Some(-1),
                    "baseline" | "base" | "default" => Some(0),
                    "stable1" | "stable_1" | "safe" => Some(1),
                    _ => None,
                }
            })),

            dirty_trigger: BoolParam::new("Internal State", false),

            shared: Arc::new(RwLock::new(SharedPluginState::default())),
        }
    }
}

impl Plugin for IRMaxPlugin {
    const NAME: &'static str = "IRMax";
    const VENDOR: &'static str = "GoHardSGG"; // Or User's Name
    const URL: &'static str = "https://github.com/SGGGGG/IRMax";
    const EMAIL: &'static str = "info@example.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Standard VST3/CLAP Layouts
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(4), // Quad
            main_output_channels: NonZeroU32::new(4),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(6), // 5.1
            main_output_channels: NonZeroU32::new(6),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(8), // 7.1
            main_output_channels: NonZeroU32::new(8),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(10), // 7.1.2
            main_output_channels: NonZeroU32::new(10),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(12), // 7.1.4 (Atmos)
            main_output_channels: NonZeroU32::new(12),
            ..AudioIOLayout::const_default()
        },
        // Extended Discrete Layouts (Even numbers 14 - 64)
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(14),
            main_output_channels: NonZeroU32::new(14),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(16),
            main_output_channels: NonZeroU32::new(16),
            ..AudioIOLayout::const_default()
        }, // 9.1.6
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(18),
            main_output_channels: NonZeroU32::new(18),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(20),
            main_output_channels: NonZeroU32::new(20),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(22),
            main_output_channels: NonZeroU32::new(22),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(24),
            main_output_channels: NonZeroU32::new(24),
            ..AudioIOLayout::const_default()
        }, // 22.2
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(26),
            main_output_channels: NonZeroU32::new(26),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(28),
            main_output_channels: NonZeroU32::new(28),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(30),
            main_output_channels: NonZeroU32::new(30),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(32),
            main_output_channels: NonZeroU32::new(32),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(34),
            main_output_channels: NonZeroU32::new(34),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(36),
            main_output_channels: NonZeroU32::new(36),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(38),
            main_output_channels: NonZeroU32::new(38),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(40),
            main_output_channels: NonZeroU32::new(40),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(42),
            main_output_channels: NonZeroU32::new(42),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(44),
            main_output_channels: NonZeroU32::new(44),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(46),
            main_output_channels: NonZeroU32::new(46),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(48),
            main_output_channels: NonZeroU32::new(48),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(50),
            main_output_channels: NonZeroU32::new(50),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(52),
            main_output_channels: NonZeroU32::new(52),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(54),
            main_output_channels: NonZeroU32::new(54),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(56),
            main_output_channels: NonZeroU32::new(56),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(58),
            main_output_channels: NonZeroU32::new(58),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(60),
            main_output_channels: NonZeroU32::new(60),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(62),
            main_output_channels: NonZeroU32::new(62),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(64),
            main_output_channels: NonZeroU32::new(64),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        if std::env::var("IRMAX_DISABLE_EDITOR").ok().as_deref() == Some("1") {
            self.logger.warn(
                "Editor",
                "Editor disabled by IRMAX_DISABLE_EDITOR=1 (safe mode).",
            );
            return None;
        }

        editor::create(
            self.params.clone(),
            self.loader.clone(),
            self.logger.clone(),
            self.runtime.clone(),
        )
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.logger.info("Plugin", "Initializing...");
        self.logger.info(
            "Logger",
            &format!(
                "Detailed diagnostics: {} (IRMAX_DETAILED_LOG can override)",
                if self.logger.detailed_enabled() {
                    "enabled"
                } else {
                    "disabled"
                }
            ),
        );
        self.logger.info(
            "Build",
            &format!(
                "Build kind={}, branch={}",
                current_build_kind(),
                option_env!("IRMAX_GIT_BRANCH").unwrap_or("nogit")
            ),
        );
        info!("IRMax: Initializing...");
        #[cfg(target_os = "windows")]
        {
            preload_ipp_runtime();
        }

        // Auto-Reload Persistence via Params
        let slot_param = clamp_slot(self.params.preset_slot_select.value());
        let path_str = {
            let guard = self.params.shared.read();
            slot_path_from_shared(&guard, slot_param)
        };
        self.last_slot_param_value = slot_param;
        self.last_slot_load_path = path_str.clone();

        let host_sample_rate = buffer_config.sample_rate as u32;
        let host_max_buffer = buffer_config.max_buffer_size as usize;
        let negotiated_channels = audio_io_layout
            .main_output_channels
            .map(|c| c.get())
            .unwrap_or(2) as usize;

        {
            let mut rt = self.runtime.write();
            let prev_outputs = rt.host_outputs;
            let prev_max_buffer = rt.host_max_buffer;
            let prev_sample_rate = rt.host_sample_rate;

            rt.host_sample_rate = host_sample_rate;
            rt.host_outputs = negotiated_channels;
            rt.host_max_buffer = host_max_buffer;

            if rt.gpu_backends.is_empty() {
                let available = available_backends();
                rt.gpu_backends = available.clone();
                rt.gpu_can_switch = cfg!(target_os = "windows") && available.len() >= 2;
                if rt.gpu_backend == GpuBackend::Auto {
                    rt.gpu_backend = pick_default_backend(&available);
                }
                let backend_list = if available.is_empty() {
                    "none".to_string()
                } else {
                    available
                        .iter()
                        .map(|b| b.label())
                        .collect::<Vec<_>>()
                        .join(",")
                };
                self.logger.info(
                    "Backend",
                    &format!(
                        "Detected backends: [{}], default={}, can_switch={}",
                        backend_list,
                        rt.gpu_backend.label(),
                        rt.gpu_can_switch
                    ),
                );
            }
            rt.gpu_backend_pending = None;
            let backend = rt.gpu_backend;

            let guard_profile = clamp_guard_profile(self.params.guard_profile.value());

            let layout_changed = prev_outputs != 0
                && (prev_outputs != negotiated_channels
                    || prev_max_buffer != host_max_buffer
                    || prev_sample_rate != host_sample_rate);

            // Check if we need to load (using Runtime State)
            let needs_load = if !rt.init_load_triggered && !path_str.is_empty() {
                rt.init_load_triggered = true;
                true
            } else {
                false
            };

            // Trigger reload if host layout changed after initial load
            if layout_changed && !path_str.is_empty() {
                self.logger.info(
                    "Persistence",
                    &format!(
                        "Host layout changed ({}ch -> {}ch, maxbuf {} -> {}, sr {} -> {}). Reloading...",
                        prev_outputs,
                        negotiated_channels,
                        prev_max_buffer,
                        host_max_buffer,
                        prev_sample_rate,
                        host_sample_rate
                    ),
                );
                let _ = self.loader.tx.send(loader::LoaderCommand::LoadFolder(
                    PathBuf::from(path_str.clone()),
                    host_sample_rate,
                    negotiated_channels,
                    host_max_buffer,
                    backend,
                    None,
                    guard_profile,
                ));
                self.last_requested_sample_rate = host_sample_rate;
            } else if needs_load {
                self.logger
                    .info("Persistence", &format!("Restoring folder: {}", path_str));
                let _ = self.loader.tx.send(loader::LoaderCommand::LoadFolder(
                    PathBuf::from(path_str),
                    host_sample_rate,
                    negotiated_channels,
                    host_max_buffer,
                    backend,
                    None,
                    guard_profile,
                ));
                self.last_requested_sample_rate = host_sample_rate;
            } else if !path_str.is_empty() {
                self.logger
                    .info("Persistence", "Skipping Restore (Already Triggered).");
            }
        }

        // Signal that SharedPluginState is now fully deserialized and ready
        {
            let mut rt = self.runtime.write();
            rt.persistence_ready = true;
        }
        self.last_guard_profile_value = clamp_guard_profile(self.params.guard_profile.value());

        let max_samples = buffer_config.max_buffer_size as usize;

        // ALLOCATE SPARE CHANNELS FOR SILENCE PADDING
        // We allocate 64 channels max to support practically any IR without reallocation.
        let allocation_channels = 64;

        info!(
            "IRMax: Allocating Scratch Buffers ({} ch negotiated, {} alloc x {} samples)",
            negotiated_channels, allocation_channels, max_samples
        );
        self.logger.info(
            "Plugin",
            &format!(
                "Allocating Scratch: {} negotiated, {} alloc, {} samples",
                negotiated_channels, allocation_channels, max_samples
            ),
        );

        self.scratch_input = vec![vec![0.0; max_samples]; allocation_channels];
        self.scratch_output = vec![vec![0.0; max_samples]; allocation_channels];

        if let Some(_backend) = GlobalBackend::get() {
            info!("IRMax: Global Backend Initialized (GPU Ready).");
        } else {
            error!("IRMax: Global Backend Failed to Initialize!");
        }

        true
    }

    fn reset(&mut self) {
        // Host has requested a reset (e.g. Seek, Stop, or Panic)
        // We must ensure the engine state is cleared.
        self.last_playing = false; // Reset edge detector
        self.pending_reset = true;
        self.last_pos_samples = None;
        self.last_block_samples = None;
        self.last_process_instant = None;
        self.last_diag_log_instant = None;
        self.diag_prev_snapshot = None;
        if let Some(processor) = &mut self.current_processor {
            processor.reset_state();
            self.pending_reset = false;
        }
    }

    fn deactivate(&mut self) {
        self.logger.info("Plugin", "Deactivating...");
        self.current_processor = None;
        self.worker_thread.clear_current_worker();
        self.pending_reset = false;
        self.last_playing = false;
        self.last_pos_samples = None;
        self.last_block_samples = None;
        self.last_process_instant = None;
        self.last_diag_log_instant = None;
        self.diag_prev_snapshot = None;
        self.logger.info("Plugin", "Deactivated: released current processor.");
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let current_sr = context.transport().sample_rate as u32;

        // Detect host sample-rate change during active processing
        if current_sr != 0 && current_sr != self.last_requested_sample_rate {
            // Update runtime state (for UI + downstream reload)
            let host_outputs = buffer.channels();
            let host_max_buffer = {
                let mut rt = self.runtime.write();
                rt.host_sample_rate = current_sr;
                rt.host_outputs = host_outputs;
                rt.host_max_buffer = rt.host_max_buffer.max(buffer.samples() as usize);
                rt.host_max_buffer
            };
            let backend = {
                let rt = self.runtime.read();
                rt.gpu_backend_pending.unwrap_or(rt.gpu_backend)
            };
            let guard_profile = clamp_guard_profile(self.params.guard_profile.value());

            let path_str = {
                let guard = self.params.shared.read();
                let slot_param = clamp_slot(self.params.preset_slot_select.value());
                slot_path_from_shared(&guard, slot_param)
            };

            if !path_str.is_empty() {
                // Trigger a reload at the new sample rate
                let _ = self.loader.tx.send(loader::LoaderCommand::LoadFolder(
                    PathBuf::from(path_str),
                    current_sr,
                    host_outputs,
                    host_max_buffer,
                    backend,
                    None,
                    guard_profile,
                ));
                self.last_requested_sample_rate = current_sr;
                if !self.pending_reset {
                    self.pending_reset = true;
                    self.logger
                        .detailed_warn("Reset", "Reason: sample_rate_change");
                }
            }
        }
        let playing = context.transport().playing;
        let pos_samples = context.transport().pos_samples();
        let now = std::time::Instant::now();
        if playing {
            if let Some(last) = self.last_process_instant {
                let elapsed = now.duration_since(last).as_secs_f64();
                let block_seconds =
                    buffer.samples() as f64 / context.transport().sample_rate as f64;
                let threshold = (block_seconds * 4.0).max(0.05);
                if elapsed > threshold {
                    if !self.pending_reset {
                        self.pending_reset = true;
                        self.logger.detailed_warn("Reset", "Reason: callback_gap");
                    }
                }
            }
        }
        self.last_process_instant = Some(now);
        let block_samples = buffer.samples() as i64;
        if playing {
            if let (Some(pos), Some(last_pos), Some(last_block)) =
                (pos_samples, self.last_pos_samples, self.last_block_samples)
            {
                let expected = last_pos + last_block;
                let diff = (pos - expected).abs();
                if diff > 64 {
                    if !self.pending_reset {
                        self.pending_reset = true;
                        self.logger.detailed_warn("Reset", "Reason: pos_jump");
                    }
                }
            }
        }
        if playing && !self.last_playing {
            if !self.pending_reset {
                self.pending_reset = true;
                self.logger.detailed_warn("Reset", "Reason: transport_start");
            }
        }
        self.last_playing = playing;
        self.last_pos_samples = pos_samples;
        self.last_block_samples = Some(block_samples);

        let slot_param = clamp_slot(self.params.preset_slot_select.value());
        if slot_param != self.last_slot_param_value {
            let host_outputs = buffer.channels();
            let host_max_buffer = {
                let mut rt = self.runtime.write();
                rt.host_sample_rate = current_sr;
                rt.host_outputs = host_outputs;
                rt.host_max_buffer = rt.host_max_buffer.max(buffer.samples() as usize);
                rt.host_max_buffer
            };
            let backend = {
                let rt = self.runtime.read();
                rt.gpu_backend_pending.unwrap_or(rt.gpu_backend)
            };
            let guard_profile = clamp_guard_profile(self.params.guard_profile.value());

            let path_str = {
                let guard = self.params.shared.read();
                slot_path_from_shared(&guard, slot_param)
            };

            if !path_str.is_empty() {
                let effective_sr = if current_sr == 0 {
                    let rt = self.runtime.read();
                    rt.host_sample_rate.max(48000)
                } else {
                    current_sr
                };

                let already_requested_by_ui = {
                    let guard = self.params.shared.read();
                    guard.last_folder == path_str
                };

                let guard_changed = guard_profile != self.last_guard_profile_value;

                if guard_changed
                    || (!already_requested_by_ui
                        && (self.last_slot_load_path != path_str
                            || self.last_requested_sample_rate != effective_sr))
                {
                    {
                        let mut guard = self.params.shared.write();
                        guard.last_folder = path_str.clone();
                    }
                    let _ = self.loader.tx.send(loader::LoaderCommand::LoadFolder(
                        PathBuf::from(path_str.clone()),
                        effective_sr,
                        host_outputs.max(1),
                        host_max_buffer.max(64),
                        backend,
                        None,
                        guard_profile,
                    ));
                    self.last_requested_sample_rate = effective_sr;
                    self.last_slot_load_path = path_str.clone();
                    if !self.pending_reset {
                        self.pending_reset = true;
                        self.logger.detailed_warn(
                            "Reset",
                            &format!("Reason: slot_change_to_{}", slot_param + 1),
                        );
                    }
                    self.logger.info(
                        "Preset",
                        &format!(
                            "Automation slot switch -> {} path={}",
                            slot_param + 1,
                            path_str
                        ),
                    );
                } else {
                    self.logger.detailed_info(
                        "Preset",
                        &format!(
                            "Automation slot switch -> {} deduped path={}",
                            slot_param + 1,
                            path_str
                        ),
                    );
                    self.last_slot_load_path = path_str.clone();
                    self.last_requested_sample_rate = effective_sr;
                }
            } else {
                {
                    let mut guard = self.params.shared.write();
                    guard.last_folder.clear();
                }
                {
                    let mut rt = self.runtime.write();
                    rt.loaded_info = None;
                }
                self.current_processor = None;
                self.pending_reset = false;
                self.last_slot_load_path.clear();
                self.logger.info(
                    "Preset",
                    &format!(
                        "Automation slot switch -> {} cleared: slot path is empty",
                        slot_param + 1
                    ),
                );
                self.logger.warn(
                    "Preset",
                    &format!(
                        "Automation slot switch -> {} ignored: slot path is empty",
                        slot_param + 1
                    ),
                );
            }

            self.last_slot_param_value = slot_param;
            self.last_guard_profile_value = guard_profile;
        }

        let guard_profile = clamp_guard_profile(self.params.guard_profile.value());
        if guard_profile != self.last_guard_profile_value {
            let host_outputs = buffer.channels();
            let host_max_buffer = {
                let mut rt = self.runtime.write();
                rt.host_sample_rate = current_sr;
                rt.host_outputs = host_outputs;
                rt.host_max_buffer = rt.host_max_buffer.max(buffer.samples() as usize);
                rt.host_max_buffer
            };
            let backend = {
                let rt = self.runtime.read();
                rt.gpu_backend_pending.unwrap_or(rt.gpu_backend)
            };
            let slot_param = clamp_slot(self.params.preset_slot_select.value());
            let path_str = {
                let guard = self.params.shared.read();
                slot_path_from_shared(&guard, slot_param)
            };

            if !path_str.is_empty() {
                let effective_sr = if current_sr == 0 {
                    let rt = self.runtime.read();
                    rt.host_sample_rate.max(48000)
                } else {
                    current_sr
                };

                let _ = self.loader.tx.send(loader::LoaderCommand::LoadFolder(
                    PathBuf::from(path_str.clone()),
                    effective_sr,
                    host_outputs.max(1),
                    host_max_buffer.max(64),
                    backend,
                    None,
                    guard_profile,
                ));
                self.last_requested_sample_rate = effective_sr;
                self.last_slot_load_path = path_str;
                if !self.pending_reset {
                    self.pending_reset = true;
                    self.logger.detailed_warn(
                        "Reset",
                        &format!("Reason: guard_profile_change_to_{}", guard_profile),
                    );
                }
            }

            self.last_guard_profile_value = guard_profile;
        }

        // 1. Hot-Swap
        if let Ok(new_processor) = self.processor_inbox.try_recv() {
            self.logger.detailed_info(
                "HotSwap",
                &format!("Activated engine={}.", new_processor.diag_engine_id()),
            );
            self.current_processor = Some(new_processor);
            self.last_diag_log_instant = None;
            self.diag_prev_snapshot = None;
        }

        if self.pending_reset {
            if let Some(processor) = &mut self.current_processor {
                processor.reset_state();
                self.pending_reset = false;
                self.last_diag_log_instant = None;
                self.diag_prev_snapshot = None;
                self.logger
                    .detailed_info("Reset", "Applied pending reset to processor");
            }
        }

        if let Some(processor) = &mut self.current_processor {
            let samples = buffer.samples();
            let track_channels = buffer.channels();
            let max_buf_len = self.scratch_input[0].len();

            // Safety Check
            if samples > max_buf_len {
                return ProcessStatus::Normal;
            }

            // 2. Buffer Adaptation (Vectorized)
            // Copy Track Audio to Scratch
            // Only copy what we have (track_channels).
            // The rest of scratch_input is already Zero (Silence).
            let safe_channels = track_channels.min(self.scratch_input.len());
            let samples = buffer.samples();
            let engine_outputs = processor.num_outputs();

            #[cfg(target_os = "macos")]
            unsafe {
                for c in 0..safe_channels {
                    // Wait, existing code uses buffer.iter_samples() which implies Interleaved?
                    // Or nih_plug abstracts it.
                    // If we cannot assume Planar vs Interleaved, we better stick to iter_samples for safety?
                    // NO. The user wants performance.
                    // nih_plug::Buffer DOES NOT have as_slice().
                    // But it implicitly derefs to something? No.
                    // Use standard loop for now? No, user wants vectorization.
                    // Let's assume Planar access via pointers is possible if we can get the raw pointer.
                    // buffer.as_ptr() ?
                    // Actually, let's look at `input_refs` construction later.
                    // `input_refs` uses `scratch_input`.
                    // To copy FROM buffer:
                    // If we can't get a slice, we can't memcpy.
                    // BUT: buffer[c] works? `ops::Index`?
                    // If not, we have to use iter_samples or unsafe access.
                    // Inspecting `nih_plug` source is not possible.
                    // Let's TRY `buffer.as_array_of_slices()` pattern?
                    // Revert to SCALAR loop for Input Copy on macOS for now to fix build?
                    // The user demands performance.
                    // Let's blindly trust that buffer is Planar and we can get a pointer?
                    // ACTUALLY: The error message said "no method named `as_mut_slice`".
                    // It said "help: there is a method `as_slice`".
                    // So `buffer.as_slice()` exists and returns `&[f32]`.
                    // This implies the buffer IS contiguous (Interleaved or Flattened Planar).
                    // Given `iter_samples` iterates frames, it's likely Interleaved.
                    // So my original logic was correct, except I used `as_mut_slice` which failed.
                    // So `as_slice` (immutable) is fine for INPUT.
                    // I used `raw_buffer.as_ptr()` in input copy. That was valid!
                    // So correcting `accelerate.rs` signature might fix the Input Copy *if* strict casting is fixed.

                    // RETRY Vectorized Interleave Copy (Input)
                    // Assuming buffer.as_slice() works.
                    // Error: `expected *const f32, found *const &mut [f32]`
                    // This means `buffer.as_slice()` returned `&[&mut [f32]]` (slice of slices)??
                    // If so, it IS Planar!
                    // If `as_slice()` returns `&[&mut [f32]]`, then `raw_buffer[c]` gives the slide for channel C.
                    // So `raw_buffer[c].as_ptr()` gives `*const f32` for that channel.
                    // Correct!

                    // So:
                    // let channels_slices = buffer.as_slice(); // &[&mut [f32]] ? or something similar.
                    // If checking error log: `note: expected raw pointer *const f32 found raw pointer *const &mut [f32]`
                    // This confirms `as_slice()` returns a slice of references to slices (Planar list).

                    // So we do:
                    // let channels = buffer.as_slice();
                    // let src_ptr = channels[c].as_ptr();
                    // std::ptr::copy_nonoverlapping(src_ptr, dst, samples);

                    // buffer.as_slice() returns &[f32] (interleaved) OR something else?
                    // The error says "found *const &mut [f32]".
                    // This means `raw_buffer` is `&[&mut [f32]]`.
                    // So `raw_buffer` is a slice of channel slices.
                    // This confirms PLANAR.

                    // Planar Copy Logic:
                    // We don't need vDSP_mmov (Strided). We need straight copy.
                    // Because src is Planar (step 1) and dst is Planar (step 1).
                    // src stride = 1. dst stride = 1.

                    let src_channels = buffer.as_slice();
                    if c < src_channels.len() {
                        let src_ptr = src_channels[c].as_ptr();
                        let dst_ptr = self.scratch_input[c].as_mut_ptr();
                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, samples);
                    }
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                // Fallback: Safe Iter-based Copy (Input)
                // Note: buffer.as_slice() causing type ambiguity on Windows.
                // Using iter_samples is safe and reasonably fast for memory copy.
                for (s, frame) in buffer.iter_samples().enumerate() {
                    for (c, sample) in frame.into_iter().enumerate() {
                        if c < safe_channels {
                            self.scratch_input[c][s] = *sample;
                        }
                    }
                }
            }

            // 3. Prepare Slices (Stack Allocation / Zero Allocation)
            // Strategy: Use MaybeUninit array on stack (Max 64 channels).
            // Input Refs
            let mut input_refs_storage: [std::mem::MaybeUninit<&[f32]>; 64] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };

            let mut in_count = 0;
            for ch_buf in &self.scratch_input {
                if in_count >= 64 {
                    break;
                }
                input_refs_storage[in_count] = std::mem::MaybeUninit::new(&ch_buf[0..samples]);
                in_count += 1;
            }
            // Transmute slice of MaybeUninit to slice of T
            let input_refs: &[&[f32]] = unsafe {
                std::slice::from_raw_parts(input_refs_storage.as_ptr() as *const &[f32], in_count)
            };

            // Output Refs (Mutable)
            let mut out_refs_storage: [std::mem::MaybeUninit<&mut [f32]>; 64] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };

            let mut out_count = 0;
            for ch_buf in &mut self.scratch_output {
                if out_count >= 64 {
                    break;
                }
                out_refs_storage[out_count] = std::mem::MaybeUninit::new(&mut ch_buf[0..samples]);
                out_count += 1;
            }

            // We need &mut [&mut [f32]] for the processor
            // Cast the array pointer to *mut &mut [f32]
            let out_refs: &mut [&mut [f32]] = unsafe {
                std::slice::from_raw_parts_mut(
                    out_refs_storage.as_mut_ptr() as *mut &mut [f32],
                    out_count,
                )
            };

            // 4. Fire Engine
            // Now safe: No Vec allocation.
            processor.process(input_refs, out_refs);

            if self.logger.detailed_enabled() {
                let should_log_diag = self
                    .last_diag_log_instant
                    .map_or(true, |last| now.duration_since(last) >= diag_log_interval());
                if should_log_diag {
                    let (tail_underrun, body_underrun, tail_drop, body_drop) = processor.diag_counts();
                    let (
                        tail_pushed,
                        tail_consumed,
                        tail_results,
                        body_pushed,
                        body_consumed,
                        body_results,
                    ) = processor.diag_task_counts();
                    let (
                        body_worker_started,
                        body_pushed_w,
                        body_consumed_w,
                        body_results_w,
                    ) = processor.diag_body_worker();
                    let (tail_task_slots, tail_result_slots) = processor.diag_tail_slots();
                    let (body_task_slots, body_result_slots) =
                        processor.diag_body_slots().unwrap_or((0, 0));
                    let (body_parallel_threads, body_outputs_per_task) =
                        processor.diag_body_parallel_profile();
                    let (gpu_exec_us, gpu_driver_us, gpu_compute_us) = processor.diag_gpu_times_us();
                    let (
                        tail_block,
                        tail_latency_blocks,
                        head_len,
                        tail_start,
                        body_block,
                        body_latency_blocks,
                    ) = processor.diag_pipeline_layout();

                    self.logger.detailed_warn(
                        "Diag",
                        &format!(
                            "Diag: engine={} tail_underrun={} body_underrun={} tail_drop={} body_drop={} tail_pushed={} tail_consumed={} tail_results={} body_pushed={} body_consumed={} body_results={} body_worker_started={} body_pushed_w={} body_consumed_w={} body_results_w={} tail_slots(task={},result={}) body_slots(task={},result={}) body_parallel(threads={},outputs_per_task={}) gpu_us(total={},driver={},compute={}) layout(tail_block={},tail_latency_blocks={},head_len={},tail_start={},body_block={},body_latency_blocks={})",
                            processor.diag_engine_id(),
                            tail_underrun,
                            body_underrun,
                            tail_drop,
                            body_drop,
                            tail_pushed,
                            tail_consumed,
                            tail_results,
                            body_pushed,
                            body_consumed,
                            body_results,
                            body_worker_started,
                            body_pushed_w,
                            body_consumed_w,
                            body_results_w,
                            tail_task_slots,
                            tail_result_slots,
                            body_task_slots,
                            body_result_slots,
                            body_parallel_threads,
                            body_outputs_per_task,
                            gpu_exec_us,
                            gpu_driver_us,
                            gpu_compute_us,
                            tail_block,
                            tail_latency_blocks,
                            head_len,
                            tail_start,
                            body_block,
                            body_latency_blocks,
                        ),
                    );

                    let (tail_push_hz, tail_consume_hz, tail_result_hz, body_push_hz, body_consume_hz, body_result_hz, dt_ms) =
                        if let Some(prev) = &self.diag_prev_snapshot {
                            let elapsed_s = now.duration_since(prev.at).as_secs_f64();
                            (
                                rate_per_sec(tail_pushed, prev.tail_pushed, elapsed_s),
                                rate_per_sec(tail_consumed, prev.tail_consumed, elapsed_s),
                                rate_per_sec(tail_results, prev.tail_results, elapsed_s),
                                rate_per_sec(body_pushed, prev.body_pushed, elapsed_s),
                                rate_per_sec(body_consumed, prev.body_consumed, elapsed_s),
                                rate_per_sec(body_results, prev.body_results, elapsed_s),
                                elapsed_s * 1000.0,
                            )
                        } else {
                            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        };

                    let tail_inflight = tail_pushed.saturating_sub(tail_consumed);
                    let body_inflight = body_pushed.saturating_sub(body_consumed);
                    let tail_result_gap = tail_results.saturating_sub(tail_consumed);
                    let body_result_gap = body_results.saturating_sub(body_consumed);

                    self.logger.detailed_info(
                        "Perf",
                        &format!(
                            "Perf: dt_ms={:.1} rate_hz tail(p/c/r)={:.1}/{:.1}/{:.1} body(p/c/r)={:.1}/{:.1}/{:.1} backlog tail_inflight={} tail_result_gap={} body_inflight={} body_result_gap={}",
                            dt_ms,
                            tail_push_hz,
                            tail_consume_hz,
                            tail_result_hz,
                            body_push_hz,
                            body_consume_hz,
                            body_result_hz,
                            tail_inflight,
                            tail_result_gap,
                            body_inflight,
                            body_result_gap,
                        ),
                    );

                    self.diag_prev_snapshot = Some(DiagPerfSnapshot {
                        at: now,
                        tail_pushed,
                        tail_consumed,
                        tail_results,
                        body_pushed,
                        body_consumed,
                        body_results,
                    });
                    self.last_diag_log_instant = Some(now);
                }
            }

            // 4.1 Clear channels beyond engine outputs (deterministic, per-frame)
            let clear_start = engine_outputs.min(safe_channels);
            for c in clear_start..safe_channels {
                self.scratch_output[c][0..samples].fill(0.0);
            }

            // 5. Copy Scratch -> Buffer (With Mix & Gain)
            // Only copy back channels that exist on Track.
            let mix = self.params.mix.value();
            let gain = self.params.output_gain.value();

            #[cfg(target_os = "macos")]
            unsafe {
                use irmax_core::accelerate;
                let len = samples as i64;
                // Note: mix and gain are f32
                // We need pointers to them for vDSP.
                // vDSP_vsmul expects *const c_float for the scalar.
                let g_wet = mix * gain;
                let g_dry = (1.0 - mix) * gain;

                for c in 0..safe_channels {
                    let dry_ptr = self.scratch_input[c].as_mut_ptr();
                    let wet_ptr = self.scratch_output[c].as_mut_ptr();

                    // A. Calculate Blended Result in-place (Vectorized)
                    // TempA (Dry) = Dry * G_dry
                    accelerate::vDSP_vsmul(dry_ptr, 1, &g_dry, dry_ptr, 1, len);
                    // TempB (Wet) = Wet * G_wet
                    accelerate::vDSP_vsmul(wet_ptr, 1, &g_wet, wet_ptr, 1, len);
                    // Result (Wet) = TempA + TempB
                    accelerate::vDSP_vadd(dry_ptr, 1, wet_ptr, 1, wet_ptr, 1, len);
                }
            }

            // B. Write to Output Buffer (Planar Zero-Copy)
            // Use std::ptr::copy_nonoverlapping (effectively memcpy) which is fully vectorized.
            #[cfg(target_os = "macos")]
            unsafe {
                // buffer.as_slice() returns &[&mut [f32]] (Planar)
                let out_channels = buffer.as_slice();
                for c in 0..safe_channels {
                    if c < out_channels.len() {
                        // Src: Scratch Output (fully mixed)
                        let src_ptr = self.scratch_output[c].as_ptr();
                        // Dst: Buffer Channel
                        let dst_ptr = out_channels[c].as_mut_ptr();

                        // 100% Accelerated Memory Copy
                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, samples);
                    }
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                let g_wet = mix * gain;
                let g_dry = (1.0 - mix) * gain;

                // 1. Vectorized Mix (Scratch Input -> Scratch Output)
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                unsafe {
                    use std::arch::x86_64::*;
                    let v_g_wet = _mm256_set1_ps(g_wet);
                    let v_g_dry = _mm256_set1_ps(g_dry);

                    for c in 0..safe_channels {
                        let dry_ptr = self.scratch_input[c].as_mut_ptr();
                        let wet_ptr = self.scratch_output[c].as_mut_ptr();

                        let mut i = 0;
                        while i + 8 <= samples {
                            let d = _mm256_loadu_ps(dry_ptr.add(i));
                            let w = _mm256_loadu_ps(wet_ptr.add(i));

                            // Dry = Dry * G_dry
                            let d_res = _mm256_mul_ps(d, v_g_dry);
                            // Wet = Wet * G_wet
                            let w_res = _mm256_mul_ps(w, v_g_wet);
                            // Sum
                            let res = _mm256_add_ps(d_res, w_res);

                            _mm256_storeu_ps(wet_ptr.add(i), res);
                            i += 8;
                        }
                        // Remainder
                        while i < samples {
                            *wet_ptr.add(i) = *dry_ptr.add(i) * g_dry + *wet_ptr.add(i) * g_wet;
                            i += 1;
                        }
                    }
                }

                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    // Scalar fallback loop for ARM/etc
                    for c in 0..safe_channels {
                        let dry = &mut self.scratch_input[c];
                        let wet = &mut self.scratch_output[c];
                        for i in 0..samples {
                            wet[i] = dry[i] * g_dry + wet[i] * g_wet;
                        }
                    }
                }

                // 2. Safe Copy to Output (Iter-based)
                // scratch_output now contains the fully mixed Wet+Dry signal.
                for (s, frame) in buffer.iter_samples().enumerate() {
                    for (c, sample) in frame.into_iter().enumerate() {
                        if c < safe_channels {
                            *sample = self.scratch_output[c][s];
                        }
                    }
                }
            }
        }

        ProcessStatus::Normal
    }
}
impl ClapPlugin for IRMaxPlugin {
    const CLAP_ID: &'static str = "com.irmax.plugin";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("High Performance IR Loader");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for IRMaxPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"IRMaxByAntigrav."; // 16 chars unique ID
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Reverb,
        Vst3SubCategory::Spatial, // Replacing Stereo with Spatial for Multi-Channel support
    ];
}

nih_export_clap!(IRMaxPlugin);
nih_export_vst3!(IRMaxPlugin);
