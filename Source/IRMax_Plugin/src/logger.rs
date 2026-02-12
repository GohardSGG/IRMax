//! Instance-level logging system for IRMax
//!
//! Each VST instance has its own independent log file.
//! No global state - fully instance-isolated.
//!
//! # Real-time Safety
//! This logger is designed for audio thread usage.
//! - `info/warn/error` methods are non-blocking (push to channel).
//! - File IO happens in a dedicated background thread.
//! - String formatting is minimized in the hot path.

use chrono::Local;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

/// File logging is enabled for dev/release, disabled for production builds.
#[cfg(feature = "production")]
const FILE_LOGGING_ENABLED: bool = false;
#[cfg(not(feature = "production"))]
const FILE_LOGGING_ENABLED: bool = true;

/// Detailed diagnostics are available in dev/release, disabled in production.
#[cfg(feature = "production")]
const DETAILED_LOGGING_DEFAULT: bool = false;
#[cfg(not(feature = "production"))]
const DETAILED_LOGGING_DEFAULT: bool = true;
/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error,
    Warn,
    Info,
    #[allow(dead_code)]
    Debug,
    #[allow(dead_code)]
    Trace,
}

impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Level::Error => write!(f, "ERROR"),
            Level::Warn => write!(f, "WARN "),
            Level::Info => write!(f, "INFO "),
            Level::Debug => write!(f, "DEBUG"),
            Level::Trace => write!(f, "TRACE"),
        }
    }
}

/// Log Message sent to background thread
enum LogMsg {
    /// Standard log entry
    Entry {
        level: Level,
        module: &'static str,
        message: String,
        show_in_ui: bool,
    },
    /// Flush signal (for shutdown)
    Flush,
}

/// Maximum number of log entries to keep in memory for UI display
const MAX_RECENT_LOGS: usize = 50;

/// Generate a unique instance ID using timestamp + random bits
pub fn generate_instance_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:08x}", (nanos & 0xFFFFFFFF) as u32)
}

/// Instance-level logger - each VST instance owns one
pub struct InstanceLogger {
    /// Channel sender for non-blocking logging
    tx: Sender<LogMsg>,
    #[allow(dead_code)]
    pub instance_id: String,
    log_path: PathBuf,
    /// Recent log entries for UI display (thread-safe)
    recent_logs: Arc<RwLock<VecDeque<String>>>,
    /// Handle to the background thread (joined on drop implicitly via detach or we can store it)
    /// We detach for simplicity as Drop order of Arc<Logger> is complex
    _thread_handle: Option<thread::JoinHandle<()>>,
}

impl InstanceLogger {
    fn parse_env_bool(name: &str) -> Option<bool> {
        let value = std::env::var(name).ok()?;
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        }
    }

    fn detailed_logging_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            Self::parse_env_bool("IRMAX_DETAILED_LOG").unwrap_or(DETAILED_LOGGING_DEFAULT)
        })
    }

    /// Create a new logger for a specific instance
    /// Spawns a background thread for file IO
    pub fn new(instance_id: &str) -> Arc<Self> {
        let path = Self::get_log_path(instance_id);
        let path_for_thread = path.clone();
        let (tx, rx) = bounded::<LogMsg>(4096);
        let recent_logs = Arc::new(RwLock::new(VecDeque::with_capacity(MAX_RECENT_LOGS)));

        let recent_logs_clone = Arc::clone(&recent_logs);
        let instance_id_clone = instance_id.to_string();

        // Spawn background worker
        let thread_handle = thread::spawn(move || {
            Self::log_worker(rx, path_for_thread, instance_id_clone, recent_logs_clone);
        });

        Arc::new(Self {
            tx,
            instance_id: instance_id.to_string(),
            log_path: path,
            recent_logs,
            _thread_handle: Some(thread_handle),
        })
    }

    /// Background worker function
    fn log_worker(
        rx: Receiver<LogMsg>,
        path: PathBuf,
        instance_id: String,
        recent_logs: Arc<RwLock<VecDeque<String>>>,
    ) {
        // Open file safely
        let mut file = if FILE_LOGGING_ENABLED {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| eprintln!("[IRMax] Failed to open log: {}", e))
                .ok()
        } else {
            None
        };

        // Write header
        if let Some(ref mut f) = file {
            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let _ = writeln!(
                f,
                "\n=================================================================="
            );
            let _ = writeln!(f, "[{}] [INFO ] IRMax Logger Initialized", timestamp);
            let _ = writeln!(f, "[{}] [INFO ] Instance ID: {}", timestamp, instance_id);
            let _ = writeln!(
                f,
                "=================================================================="
            );
        }

        // Event loop
        while let Ok(msg) = rx.recv() {
            match msg {
                LogMsg::Entry {
                    level,
                    module,
                    message,
                    show_in_ui,
                } => {
                    let timestamp_str = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

                    // 1. Write to file
                    if let Some(ref mut f) = file {
                        let _ = writeln!(
                            f,
                            "[{}] [{}] [{}] {}",
                            timestamp_str, level, module, message
                        );
                    }

                    // 2. Update UI buffer (if needed)
                    if show_in_ui {
                        let ui_time = Local::now().format("%H:%M:%S");
                        let log_line = format!("[{}] [{}] {}", ui_time, module, message);
                        let mut logs = recent_logs.write();
                        if logs.len() >= MAX_RECENT_LOGS {
                            logs.pop_front();
                        }
                        logs.push_back(log_line);
                    }
                }
                LogMsg::Flush => {
                    if let Some(ref mut f) = file {
                        let _ = f.flush();
                    }
                }
            }
        }
    }

    /// Get the log file path for this instance
    fn get_log_path(instance_id: &str) -> PathBuf {
        fn expand_user_dir(path: &str) -> PathBuf {
            if let Some(stripped) = path.strip_prefix("~/") {
                if let Some(home) = dirs::home_dir() {
                    return home.join(stripped);
                }
            }
            PathBuf::from(path)
        }

        let mut candidates: Vec<PathBuf> = Vec::new();

        if let Ok(env_dir) = std::env::var("IRMAX_LOG_DIR") {
            if !env_dir.trim().is_empty() {
                candidates.push(expand_user_dir(env_dir.trim()));
            }
        }

        #[cfg(target_os = "windows")]
        {
            candidates.push(PathBuf::from(r"C:\Plugins\IRMax_Logs"));
        }

        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            if let Some(home) = dirs::home_dir() {
                candidates.push(home.join("IRMax_Logs"));
            }
        }

        if let Some(base_dir) = dirs::data_local_dir() {
            candidates.push(base_dir.join("IRMax").join("Logs"));
        }

        for dir in candidates {
            if let Err(e) = fs::create_dir_all(&dir) {
                eprintln!("[IRMax] Failed to create log dir {:?}: {}", dir, e);
                continue;
            }
            let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
            return dir.join(format!("Instance_{}_{}.log", timestamp, instance_id));
        }

        std::env::temp_dir()
            .join("IRMax_Logs")
            .join(format!("IRMax_{}.log", instance_id))
    }

    /// Internal log function (Non-blocking)
    fn log(&self, level: Level, module: &'static str, message: String, show_in_ui: bool) {
        // Try send to avoid blocking audio thread if queue is full
        let msg = LogMsg::Entry {
            level,
            module,
            message,
            show_in_ui,
        };
        let _ = self.tx.try_send(msg);
    }

    /// Get recent log entries for UI display
    pub fn get_recent_logs(&self) -> Vec<String> {
        self.recent_logs.read().iter().cloned().collect()
    }

    /// Get the full log file path for this instance
    pub fn log_path(&self) -> &PathBuf {
        &self.log_path
    }

    /// Get the full log file path as a String (for UI)
    pub fn log_path_string(&self) -> String {
        self.log_path.to_string_lossy().into_owned()
    }

    /// Check if file logging is enabled (disabled in production)
    pub fn file_logging_enabled(&self) -> bool {
        FILE_LOGGING_ENABLED
    }

    pub fn detailed_enabled(&self) -> bool {
        Self::detailed_logging_enabled()
    }

    /// Important log entry shown in UI
    pub fn important(&self, module: &'static str, message: &str) {
        self.log(Level::Info, module, message.to_string(), true);
    }

    /// Log at INFO level (file only)
    pub fn info(&self, module: &'static str, message: &str) {
        self.log(Level::Info, module, message.to_string(), false);
    }

    /// Log at INFO level only when detailed diagnostics are enabled.
    pub fn detailed_info(&self, module: &'static str, message: &str) {
        if Self::detailed_logging_enabled() {
            self.log(Level::Info, module, message.to_string(), false);
        }
    }

    /// Log at WARN level (file only)
    pub fn warn(&self, module: &'static str, message: &str) {
        self.log(Level::Warn, module, message.to_string(), false);
    }

    /// Log at WARN level only when detailed diagnostics are enabled.
    pub fn detailed_warn(&self, module: &'static str, message: &str) {
        if Self::detailed_logging_enabled() {
            self.log(Level::Warn, module, message.to_string(), false);
        }
    }

    /// Log at ERROR level (file only)
    pub fn error(&self, module: &'static str, message: &str) {
        self.log(Level::Error, module, message.to_string(), false);
    }

    #[allow(dead_code)]
    pub fn debug(&self, module: &'static str, message: &str) {
        self.log(Level::Debug, module, message.to_string(), false);
    }

    /// Explicit flush (e.g. on shutdown)
    pub fn flush(&self) {
        let _ = self.tx.send(LogMsg::Flush);
    }
}

impl Drop for InstanceLogger {
    fn drop(&mut self) {
        // Best effort flush
        let _ = self.tx.send(LogMsg::Flush);
    }
}
