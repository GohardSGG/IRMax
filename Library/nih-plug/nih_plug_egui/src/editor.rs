//! An [`Editor`] implementation for egui.

use crate::egui::Vec2;
use crate::egui::ViewportCommand;
use crate::EguiState;
use baseview::gl::GlConfig;
use baseview::{Size, WindowHandle, WindowOpenOptions, WindowScalePolicy};
use crossbeam::atomic::AtomicCell;
use egui_baseview::egui::Context;
use egui_baseview::EguiWindow;
use nih_plug::prelude::{Editor, GuiContext, ParamSetter, ParentWindowHandle};
use parking_lot::RwLock;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
const RESIZE_SUBMIT_MIN_INTERVAL: Duration = Duration::from_millis(33);

/// An [`Editor`] implementation that calls an egui draw loop.
pub(crate) struct EguiEditor<T> {
    pub(crate) egui_state: Arc<EguiState>,
    /// The plugin's state. This is kept in between editor openenings.
    pub(crate) user_state: Arc<RwLock<T>>,

    /// The user's build function. Applied once at the start of the application.
    pub(crate) build: Arc<dyn Fn(&Context, &mut T) + 'static + Send + Sync>,
    /// The user's update function.
    pub(crate) update: Arc<dyn Fn(&Context, &ParamSetter, &mut T) + 'static + Send + Sync>,

    /// The scaling factor reported by the host, if any. On macOS this will never be set and we
    /// should use the system scaling factor instead.
    pub(crate) scaling_factor: AtomicCell<Option<f32>>,
}

/// This version of `baseview` uses a different version of `raw_window_handle than NIH-plug, so we
/// need to adapt it ourselves.
struct ParentWindowHandleAdapter(nih_plug::editor::ParentWindowHandle);

unsafe impl HasRawWindowHandle for ParentWindowHandleAdapter {
    fn raw_window_handle(&self) -> RawWindowHandle {
        match self.0 {
            ParentWindowHandle::X11Window(window) => {
                let mut handle = raw_window_handle::XcbWindowHandle::empty();
                handle.window = window;
                RawWindowHandle::Xcb(handle)
            }
            ParentWindowHandle::AppKitNsView(ns_view) => {
                let mut handle = raw_window_handle::AppKitWindowHandle::empty();
                handle.ns_view = ns_view;
                RawWindowHandle::AppKit(handle)
            }
            ParentWindowHandle::Win32Hwnd(hwnd) => {
                let mut handle = raw_window_handle::Win32WindowHandle::empty();
                handle.hwnd = hwnd;
                RawWindowHandle::Win32(handle)
            }
        }
    }
}

impl<T> Editor for EguiEditor<T>
where
    T: 'static + Send + Sync,
{
    fn spawn(
        &self,
        parent: ParentWindowHandle,
        context: Arc<dyn GuiContext>,
    ) -> Box<dyn std::any::Any + Send> {
        let build = self.build.clone();
        let update = self.update.clone();
        let state = self.user_state.clone();
        let egui_state = self.egui_state.clone();
        #[cfg(target_os = "macos")]
        let mut last_resize_submit = Instant::now() - RESIZE_SUBMIT_MIN_INTERVAL;

        let (unscaled_width, unscaled_height) = self.egui_state.size();
        let scaling_factor = self.scaling_factor.load();
        let window = EguiWindow::open_parented(
            &ParentWindowHandleAdapter(parent),
            WindowOpenOptions {
                title: String::from("egui window"),
                // Baseview should be doing the DPI scaling for us
                size: Size::new(unscaled_width as f64, unscaled_height as f64),
                // NOTE: For some reason passing 1.0 here causes the UI to be scaled on macOS but
                //       not the mouse events.
                scale: scaling_factor
                    .map(|factor| WindowScalePolicy::ScaleFactor(factor as f64))
                    .unwrap_or(WindowScalePolicy::SystemScaleFactor),

                #[cfg(feature = "opengl")]
                gl_config: Some(GlConfig {
                    version: (3, 2),
                    red_bits: 8,
                    blue_bits: 8,
                    green_bits: 8,
                    alpha_bits: 8,
                    depth_bits: 24,
                    stencil_bits: 8,
                    samples: None,
                    srgb: true,
                    double_buffer: true,
                    vsync: true,
                    ..Default::default()
                }),
            },
            Default::default(),
            state,
            move |egui_ctx, _queue, state| build(egui_ctx, &mut state.write()),
            move |egui_ctx, _queue, state| {
                let setter = ParamSetter::new(context.as_ref());
                (update)(egui_ctx, &setter, &mut state.write());

                // Apply host-confirmed size to actual platform viewport.
                if let Some(confirmed_size) = egui_state.host_confirmed_size.swap(None) {
                    egui_ctx.send_viewport_cmd(ViewportCommand::InnerSize(Vec2::new(
                        confirmed_size.0 as f32,
                        confirmed_size.1 as f32,
                    )));
                    egui_ctx.request_repaint();
                }

                // Resize handshake for VST3: request host resize, but do not apply local resize
                // until host confirmation arrives through `host_confirmed_size`.
                if let Some(requested_size) = egui_state.requested_size.load() {
                    let current_size = egui_state.size.load();
                    #[cfg(target_os = "macos")]
                    let urgent_submit = egui_state.take_resize_commit_urgent();
                    #[cfg(not(target_os = "macos"))]
                    let _ = egui_state.take_resize_commit_urgent();

                    // If the requested size is already the current size, then no host callback is
                    // expected. Clear the resize handshake state immediately.
                    if requested_size == current_size {
                        egui_state.requested_size.store(None);
                        egui_state.resize_in_flight.store(false, Ordering::Release);
                        egui_state.in_flight_resize_size.store(None);
                    } else if !egui_state.resize_in_flight.load(Ordering::Acquire) {
                        #[cfg(target_os = "macos")]
                        let should_submit_now = {
                            urgent_submit || last_resize_submit.elapsed() >= RESIZE_SUBMIT_MIN_INTERVAL
                        };
                        #[cfg(not(target_os = "macos"))]
                        let should_submit_now = true;

                        if should_submit_now {
                            if context.request_resize() {
                                egui_state.resize_in_flight.store(true, Ordering::Release);
                                egui_state
                                    .in_flight_resize_size
                                    .store(Some(requested_size));
                                #[cfg(target_os = "macos")]
                                {
                                    last_resize_submit = Instant::now();
                                }
                                egui_ctx.request_repaint();
                            }
                        } else {
                            // Keep rendering preview smoothly while host commits are throttled.
                            egui_ctx.request_repaint();
                        }
                    } else {
                        // Keep repainting while waiting for host onSize() callback.
                        egui_ctx.request_repaint();
                    }
                }
            },
        );

        self.egui_state.open.store(true, Ordering::Release);
        Box::new(EguiEditorHandle {
            egui_state: self.egui_state.clone(),
            window,
        })
    }

    /// Size of the editor window
    fn size(&self) -> (u32, u32) {
        let new_size = self.egui_state.requested_size_hint();
        // This method will be used to ask the host for new size.
        // If the editor is currently being resized and new size hasn't been consumed and set yet, return new requested size.
        if let Some(new_size) = new_size {
            new_size
        } else {
            self.egui_state.size()
        }
    }

    fn set_scale_factor(&self, factor: f32) -> bool {
        // If the editor is currently open then the host must not change the current HiDPI scale as
        // we don't have a way to handle that. Ableton Live does this.
        if self.egui_state.is_open() {
            return false;
        }

        self.scaling_factor.store(Some(factor));
        true
    }

    fn min_size(&self) -> Option<(u32, u32)> {
        Some(self.egui_state.min_size_hint())
    }

    fn aspect_ratio(&self) -> Option<f32> {
        let (w, h) = self.egui_state.size();
        if h == 0 {
            None
        } else {
            Some((w as f32) / (h as f32))
        }
    }

    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {
        // As mentioned above, for now we'll always force a redraw to allow meter widgets to work
        // correctly. In the future we can use an `Arc<AtomicBool>` and only force a redraw when
        // that boolean is set.
    }

    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {}

    fn param_values_changed(&self) {
        // Same
    }

    fn host_resized(&self, width: u32, height: u32) {
        let confirmed_size = (width.max(1), height.max(1));
        self.egui_state.size.store(confirmed_size);
        self.egui_state.host_confirmed_size.store(Some(confirmed_size));
        let pending = self.egui_state.requested_size_hint();
        if pending == Some(confirmed_size) {
            self.egui_state.requested_size.store(None);
        }

        let in_flight = self.egui_state.in_flight_resize_size.load();
        if in_flight == Some(confirmed_size) || in_flight.is_none() {
            self.egui_state.resize_in_flight.store(false, Ordering::Release);
            self.egui_state.in_flight_resize_size.store(None);
        }

        // If the user kept dragging while host confirmation was in-flight, keep the latest
        // requested size pending for the next handshake step.
        if let Some(next_requested) = self.egui_state.requested_size_hint() {
            if next_requested != confirmed_size {
                self.egui_state.resize_in_flight.store(false, Ordering::Release);
            }
        }
    }
}

/// The window handle used for [`EguiEditor`].
struct EguiEditorHandle {
    egui_state: Arc<EguiState>,
    window: WindowHandle,
}

/// The window handle enum stored within 'WindowHandle' contains raw pointers. Is there a way around
/// having this requirement?
unsafe impl Send for EguiEditorHandle {}

impl Drop for EguiEditorHandle {
    fn drop(&mut self) {
        self.egui_state.open.store(false, Ordering::Release);
        // XXX: This should automatically happen when the handle gets dropped, but apparently not
        self.window.close();
    }
}
