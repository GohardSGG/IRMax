use std::any::Any;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use nih_plug::editor::ParentWindowHandle;
use nih_plug::prelude::{Editor, GuiContext, Param, ParamSetter};
use nih_plug_slint::editor::SlintEditor;
use nih_plug_slint::handle::SlintHostHandle;
use nih_plug_slint::param_events::{ParamEventAdapter, ParamEventDispatcher};
use nih_plug_slint::plugin_canvas::window::WindowAttributes;
use nih_plug_slint::plugin_canvas::LogicalSize;
use nih_plug_slint::resize::{HostResizeCoordinator, ResizeDebounce, ResizePolicy};
use nih_plug_slint::view::PluginView;
use slint::ComponentHandle;

use crate::IRMaxPluginParams;
use crate::loader::{Loader, LoaderCommand, SPECTRUM_HEIGHT, SPECTRUM_WIDTH};
use crate::logger::InstanceLogger;
use crate::preset::{self, PresetPack, PresetTreeNode};
use irmax_core::{available_backends, GpuBackend};
use nih_plug::util;
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};

const OUTPUT_MIN_DB: f32 = -100.0;
const OUTPUT_MAX_DB: f32 = 6.0;
const OUTPUT_ZERO_DB_KNOB_POS: f32 = 2.0 / 3.0;
const OUTPUT_NEG_INF_THRESHOLD_DB: f32 = -99.9;
const SLOT_COUNT: usize = 5;
const SLINT_DEFAULT_WIDTH: u32 = 980;
const SLINT_DEFAULT_HEIGHT: u32 = 654;
const SLINT_ASPECT_RATIO: f32 = SLINT_DEFAULT_WIDTH as f32 / SLINT_DEFAULT_HEIGHT as f32;
const SLINT_MIN_WIDTH: u32 = 760;
const SLINT_MIN_HEIGHT: u32 = 460;
const SPECTRUM_PANEL_MIN_WIDTH: usize = 1280;
const SPECTRUM_PANEL_MIN_HEIGHT: usize = 720;
const SPECTRUM_PANEL_MAX_WIDTH: usize = 4096;
const SPECTRUM_PANEL_MAX_HEIGHT: usize = 3072;


include!("slint_host/common.rs");
include!("slint_host/spectrum.rs");
include!("slint_host/preset.rs");
include!("slint_host/bridge.rs");
include!("slint_host/view.rs");
include!("slint_host/runtime.rs");
