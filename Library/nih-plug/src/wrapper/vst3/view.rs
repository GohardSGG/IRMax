#![allow(unused)]
use atomic_float::AtomicF32;
use crossbeam::atomic::AtomicCell;
use parking_lot::{Mutex, RwLock};
use std::any::Any;
use std::ffi::{c_void, CStr};
use std::mem;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use vst3_sys::base::{
    kInvalidArgument, kNotImplemented, kResultFalse, kResultOk, kResultTrue, tresult, TBool,
};
use vst3_sys::gui::{IPlugFrame, IPlugView, IPlugViewContentScaleSupport, ViewRect};
use vst3_sys::utils::SharedVstPtr;
use vst3_sys::VST3;

use super::inner::{Task, WrapperInner};
use super::util::{ObjectPtr, VstPtr};
use crate::plugin::vst3::Vst3Plugin;
use crate::prelude::{Editor, ParentWindowHandle};

// Alias needed for the VST3 attribute macro
use vst3_sys as vst3_com;

// Windows-specific imports for keyboard forwarding
#[cfg(target_os = "windows")]
use std::sync::atomic::AtomicIsize;

// Thanks for putting this behind a platform-specific ifdef...
// NOTE: This should also be used on the BSDs, but vst3-sys exposes these interfaces only for Linux
#[cfg(target_os = "linux")]
use {
    crate::event_loop::{EventLoop, MainThreadExecutor, TASK_QUEUE_CAPACITY},
    crossbeam::queue::ArrayQueue,
    libc,
    vst3_sys::gui::linux::{FileDescriptor, IEventHandler, IRunLoop},
};

// Window handle type constants missing from vst3-sys
#[allow(unused)]
const VST3_PLATFORM_HWND: &str = "HWND";
#[allow(unused)]
const VST3_PLATFORM_HIVIEW: &str = "HIView";
#[allow(unused)]
const VST3_PLATFORM_NSVIEW: &str = "NSView";
#[allow(unused)]
const VST3_PLATFORM_UIVIEW: &str = "UIView";
#[allow(unused)]
const VST3_PLATFORM_X11_WINDOW: &str = "X11EmbedWindowID";

/// FIXME: vst3-sys does not allow you to conditionally define fields with #[cfg()], so this is a
///        workaround to define the field outside of the struct
#[cfg(target_os = "linux")]
struct RunLoopEventHandlerWrapper<P: Vst3Plugin>(RwLock<Option<Box<RunLoopEventHandler<P>>>>);
#[cfg(not(target_os = "linux"))]
struct RunLoopEventHandlerWrapper<P: Vst3Plugin>(std::marker::PhantomData<P>);

/// Wrapper for parent window handle on Windows
#[cfg(target_os = "windows")]
struct ParentHwndWrapper(AtomicIsize);
#[cfg(not(target_os = "windows"))]
struct ParentHwndWrapper(());

/// The plugin's [`IPlugView`] instance created in [`IEditController::create_view()`] if `P` has an
/// editor. This is managed separately so the lifetime bounds match up.
#[VST3(implements(IPlugView, IPlugViewContentScaleSupport))]
pub(crate) struct WrapperView<P: Vst3Plugin> {
    inner: Arc<WrapperInner<P>>,
    editor: Arc<Mutex<Box<dyn Editor>>>,
    editor_handle: RwLock<Option<Box<dyn Any>>>,

    /// The `IPlugFrame` instance passed by the host during [IPlugView::set_frame()].
    plug_frame: RwLock<Option<VstPtr<dyn IPlugFrame>>>,
    /// Allows handling events events on the host's GUI thread when using Linux. Needed because
    /// otherwise REAPER doesn't like us very much. The event handler could be implemented directly
    /// on this object but vst3-sys does not let us conditionally implement interfaces.
    run_loop_event_handler: RunLoopEventHandlerWrapper<P>,

    /// The DPI scaling factor as passed to the [IPlugViewContentScaleSupport::set_scale_factor()]
    /// function. Defaults to 1.0, and will be kept there on macOS. When reporting and handling size
    /// the sizes communicated to and from the DAW should be scaled by this factor since NIH-plug's
    /// APIs only deal in logical pixels.
    scaling_factor: AtomicF32,

    /// Last host-confirmed editor size in logical pixels.
    ///
    /// This is used for `IPlugView::get_size()` so we report a stable, host-accepted size while a
    /// resize request is in flight.
    last_known_logical_size: AtomicCell<(u32, u32)>,

    /// Pending resize request initiated by the plugin's own internal resize handle.
    ///
    /// When this is `None`, incoming `on_size()` callbacks are treated as host-initiated outer
    /// frame resize attempts and will be rejected to keep the editor size host-locked.
    pending_internal_resize_target: AtomicCell<Option<(u32, u32)>>,

    /// The parent window handle (HWND on Windows) for keyboard forwarding
    parent_hwnd: ParentHwndWrapper,
}

/// Allow handling tasks on the host's GUI thread on Linux. This doesn't need to be a separate
/// struct, but vst3-sys does not let us implement interfaces conditionally and the interface is
/// only exposed when compiling on Linux. The struct will register itself when calling
/// [`RunLoopEventHandler::new()`] and it will unregister itself when it gets dropped.
#[cfg(target_os = "linux")]
#[VST3(implements(IEventHandler))]
struct RunLoopEventHandler<P: Vst3Plugin> {
    /// We need access to the inner wrapper so we that we can post any outstanding tasks there when
    /// this object gets dropped so no work is lost.
    inner: Arc<WrapperInner<P>>,

    /// The host's run loop interface. This lets us run tasks on the same thread as the host's UI.
    run_loop: VstPtr<dyn IRunLoop>,

    /// We need a Unix domain socket the host can poll to know that we have an event to handle. In
    /// theory eventfd would be much better suited for this, but Ardour doesn't respond to fds that
    /// aren't sockets. So instead, we will write a single byte here for every message we should
    /// handle.
    socket_read_fd: i32,
    socket_write_fd: i32,

    /// A queue of tasks that still need to be performed. Because CLAP lets the plugin request a
    /// host callback directly, we don't need to use the OsEventLoop we use in our other plugin
    /// implementations. Instead, we'll post tasks to this queue, ask the host to call
    /// [`on_main_thread()`][Self::on_main_thread()] on the main thread, and then continue to pop
    /// tasks off this queue there until it is empty.
    tasks: ArrayQueue<Task<P>>,
}

impl<P: Vst3Plugin> WrapperView<P> {
    pub fn new(inner: Arc<WrapperInner<P>>, editor: Arc<Mutex<Box<dyn Editor>>>) -> Box<Self> {
        let initial_size = {
            let (w, h) = editor.lock().size();
            (w.max(1), h.max(1))
        };

        Self::allocate(
            inner,
            editor,
            RwLock::new(None),
            RwLock::new(None),
            #[cfg(target_os = "linux")]
            RunLoopEventHandlerWrapper(RwLock::new(None)),
            #[cfg(not(target_os = "linux"))]
            RunLoopEventHandlerWrapper(Default::default()),
            AtomicF32::new(1.0),
            AtomicCell::new(initial_size),
            AtomicCell::new(None),
            #[cfg(target_os = "windows")]
            ParentHwndWrapper(AtomicIsize::new(0)),
            #[cfg(not(target_os = "windows"))]
            ParentHwndWrapper(()),
        )
    }

    /// Ask the host to resize the view to the size specified by [`Editor::size()`]. Will return false
    /// if the host doesn't like you. This **needs** to be run from the GUI thread.
    ///
    /// # Safety
    ///
    /// May cause memory corruption in Linux REAPER when called from outside of the `IRunLoop`.
    #[must_use]
    pub unsafe fn request_resize(&self) -> bool {
        // Don't do anything if the editor is not open, because that would be strange
        if self
            .editor_handle
            .try_read()
            .map(|e| e.is_none())
            .unwrap_or(true)
        {
            return false;
        }

        match &*self.plug_frame.read() {
            Some(plug_frame) => {
                let (unscaled_width, unscaled_height) = self.editor.lock().size();
                let scaling_factor = self.scaling_factor.load(Ordering::Relaxed);
                let mut size = ViewRect {
                    right: (unscaled_width as f32 * scaling_factor).round() as i32,
                    bottom: (unscaled_height as f32 * scaling_factor).round() as i32,
                    ..Default::default()
                };

                self.pending_internal_resize_target
                    .store(Some((unscaled_width.max(1), unscaled_height.max(1))));

                // The argument types are a bit wonky here because you can't construct a
                // `SharedVstPtr`. This _should_ work however.
                let plug_view: SharedVstPtr<dyn IPlugView> =
                    mem::transmute(&self.__iplugviewvptr as *const *const _);
                let result = plug_frame.resize_view(plug_view, &mut size);

                #[cfg(target_os = "macos")]
                if result == kResultOk {
                    // Some macOS hosts can resize the outer host frame first and delay/skip the
                    // immediate `on_size()` callback. Optimistically mirror the accepted size into
                    // the editor state so inner rendering and the resize handle stay in sync on the
                    // first drag.
                    let logical_width = unscaled_width.max(1);
                    let logical_height = unscaled_height.max(1);
                    self.last_known_logical_size
                        .store((logical_width, logical_height));
                    self.editor
                        .lock()
                        .host_resized(logical_width, logical_height);
                    self.pending_internal_resize_target.store(None);
                }

                debug_assert_eq!(
                    result, kResultOk,
                    "The host denied the resize, we currently don't handle this for VST3 plugins"
                );

                if result != kResultOk {
                    self.pending_internal_resize_target.store(None);
                }

                result == kResultOk
            }
            None => false,
        }
    }

    /// If the host supports `IRunLoop`, then this will post the task to a task queue that will be
    /// run on the host's UI thread. If not, then this will return an `Err` value containing the
    /// task so it can be run elsewhere.
    #[cfg(target_os = "linux")]
    pub fn do_maybe_in_run_loop(&self, task: Task<P>) -> Result<(), Task<P>> {
        match &*self.run_loop_event_handler.0.read() {
            Some(run_loop) => run_loop.post_task(task),
            None => Err(task),
        }
    }

    /// If the host supports `IRunLoop`, then this will post the task to a task queue that will be
    /// run on the host's UI thread. If not, then this will return an `Err` value containing the
    /// task so it can be run elsewhere.
    #[cfg(not(target_os = "linux"))]
    pub fn do_maybe_in_run_loop(&self, task: Task<P>) -> Result<(), Task<P>> {
        Err(task)
    }

    fn constrain_size_logical(
        &self,
        requested_width: u32,
        requested_height: u32,
        min_size: Option<(u32, u32)>,
        aspect_ratio: Option<f32>,
    ) -> (u32, u32) {
        let mut width = requested_width.max(1);
        let mut height = requested_height.max(1);

        if let Some((min_w, min_h)) = min_size {
            width = width.max(min_w.max(1));
            height = height.max(min_h.max(1));
        }

        if let Some(ratio) = aspect_ratio {
            let ratio = ratio.max(0.0001);
            let current_ratio = (width as f32) / (height as f32);

            if current_ratio > ratio {
                width = ((height as f32) * ratio).round().max(1.0) as u32;
            } else {
                height = ((width as f32) / ratio).round().max(1.0) as u32;
            }

            if let Some((min_w, min_h)) = min_size {
                if width < min_w || height < min_h {
                    let min_width_from_height = ((min_h as f32) * ratio).ceil() as u32;
                    let min_height_from_width = ((min_w as f32) / ratio).ceil() as u32;

                    width = width.max(min_w).max(min_width_from_height);
                    height = height.max(min_h).max(min_height_from_width);

                    let corrected_ratio = (width as f32) / (height as f32);
                    if corrected_ratio > ratio {
                        width = ((height as f32) * ratio).round().max(1.0) as u32;
                    } else {
                        height = ((width as f32) / ratio).round().max(1.0) as u32;
                    }
                }
            }
        }

        (width.max(1), height.max(1))
    }
}

#[cfg(target_os = "linux")]
impl<P: Vst3Plugin> RunLoopEventHandler<P> {
    pub fn new(inner: Arc<WrapperInner<P>>, run_loop: VstPtr<dyn IRunLoop>) -> Box<Self> {
        let mut sockets = [0i32; 2];
        assert_eq!(
            unsafe {
                libc::socketpair(
                    libc::AF_UNIX,
                    libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                    0,
                    sockets.as_mut_ptr(),
                )
            },
            0
        );
        let [socket_read_fd, socket_write_fd] = sockets;

        let handler = RunLoopEventHandler::allocate(
            inner,
            run_loop,
            socket_read_fd,
            socket_write_fd,
            ArrayQueue::new(TASK_QUEUE_CAPACITY),
        );

        // vst3-sys provides no way to convert to a SharedVstPtr, so, uh, yeah. These are pointers
        // to vtable poitners.
        let event_handler: SharedVstPtr<dyn IEventHandler> =
            unsafe { mem::transmute(&handler.__ieventhandlervptr as *const *const _) };
        assert_eq!(
            unsafe {
                handler
                    .run_loop
                    .register_event_handler(event_handler, handler.socket_read_fd)
            },
            kResultOk
        );

        handler
    }

    /// Post a task to the tasks queue so it will be run on the host's GUI thread later. Returns the
    /// task if the queue is full and the task could not be posted.
    pub fn post_task(&self, task: Task<P>) -> Result<(), Task<P>> {
        self.tasks.push(task)?;

        // We need to use a Unix domain socket to let the host know to call our event handler. In
        // theory eventfd would be more suitable here, but Ardour does not support that. This is
        // read again in `Self::on_fd_is_set()`.
        let notify_value = 1i8;
        const NOTIFY_VALUE_SIZE: usize = std::mem::size_of::<i8>();
        assert_eq!(
            unsafe {
                libc::write(
                    self.socket_write_fd,
                    &notify_value as *const _ as *const c_void,
                    NOTIFY_VALUE_SIZE,
                )
            },
            NOTIFY_VALUE_SIZE as isize
        );

        Ok(())
    }
}

impl<P: Vst3Plugin> IPlugView for WrapperView<P> {
    #[cfg(all(target_family = "unix", not(target_os = "macos")))]
    unsafe fn is_platform_type_supported(&self, type_: vst3_sys::base::FIDString) -> tresult {
        let type_ = CStr::from_ptr(type_);
        match type_.to_str() {
            Ok(type_) if type_ == VST3_PLATFORM_X11_WINDOW => kResultOk,
            _ => {
                nih_debug_assert_failure!("Invalid window handle type: {:?}", type_);
                kResultFalse
            }
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn is_platform_type_supported(&self, type_: vst3_sys::base::FIDString) -> tresult {
        let type_ = CStr::from_ptr(type_);
        match type_.to_str() {
            Ok(type_) if type_ == VST3_PLATFORM_NSVIEW => kResultOk,
            _ => {
                nih_debug_assert_failure!("Invalid window handle type: {:?}", type_);
                kResultFalse
            }
        }
    }

    #[cfg(target_os = "windows")]
    unsafe fn is_platform_type_supported(&self, type_: vst3_sys::base::FIDString) -> tresult {
        let type_ = CStr::from_ptr(type_);
        match type_.to_str() {
            Ok(type_) if type_ == VST3_PLATFORM_HWND => kResultOk,
            _ => {
                nih_debug_assert_failure!("Invalid window handle type: {:?}", type_);
                kResultFalse
            }
        }
    }

    unsafe fn attached(&self, parent: *mut c_void, type_: vst3_sys::base::FIDString) -> tresult {
        let mut editor_handle = self.editor_handle.write();
        if editor_handle.is_none() {
            let type_ = CStr::from_ptr(type_);
            let parent_handle = match type_.to_str() {
                Ok(type_) if type_ == VST3_PLATFORM_X11_WINDOW => {
                    ParentWindowHandle::X11Window(parent as usize as u32)
                }
                Ok(type_) if type_ == VST3_PLATFORM_NSVIEW => {
                    ParentWindowHandle::AppKitNsView(parent)
                }
                Ok(type_) if type_ == VST3_PLATFORM_HWND => {
                    // Save the parent HWND for keyboard forwarding on Windows
                    #[cfg(target_os = "windows")]
                    self.parent_hwnd.0.store(parent as isize, Ordering::SeqCst);
                    ParentWindowHandle::Win32Hwnd(parent)
                }
                _ => {
                    nih_debug_assert_failure!("Unknown window handle type: {:?}", type_);
                    return kInvalidArgument;
                }
            };

            *editor_handle = Some(
                self.editor
                    .lock()
                    .spawn(parent_handle, self.inner.clone().make_gui_context()),
            );
            *self.inner.plug_view.write() = Some(ObjectPtr::from(self));

            kResultOk
        } else {
            nih_debug_assert_failure!(
                "Host tried to attach editor while the editor is already attached"
            );

            kResultFalse
        }
    }

    unsafe fn removed(&self) -> tresult {
        let mut editor_handle = self.editor_handle.write();
        if editor_handle.is_some() {
            *self.inner.plug_view.write() = None;
            *editor_handle = None;

            kResultOk
        } else {
            nih_debug_assert_failure!("Host tried to remove the editor without an active editor");

            kResultFalse
        }
    }

    unsafe fn on_wheel(&self, _distance: f32) -> tresult {
        // We'll let the plugin use the OS' input mechanisms because not all DAWs (or very few
        // actually) implement these functions
        kNotImplemented
    }

    unsafe fn on_key_down(
        &self,
        _key: vst3_sys::base::char16,
        _key_code: i16,
        _modifiers: i16,
    ) -> tresult {
        // JUCE and upstream nih-plug both rely on native OS keyboard delivery for editor windows.
        // Host-level VST3 key codes are not Win32 virtual-key values and should not be forwarded as
        // WM_KEYDOWN/WM_KEYUP directly.
        kNotImplemented
    }

    unsafe fn on_key_up(
        &self,
        _key: vst3_sys::base::char16,
        _key_code: i16,
        _modifiers: i16,
    ) -> tresult {
        kNotImplemented
    }

    unsafe fn get_size(&self, size: *mut ViewRect) -> tresult {
        check_null_ptr!(size);

        *size = mem::zeroed();

        let (unscaled_width, unscaled_height) = self.last_known_logical_size.load();
        let scaling_factor = self.scaling_factor.load(Ordering::Relaxed);
        let size = &mut *size;
        size.left = 0;
        size.right = (unscaled_width as f32 * scaling_factor).round() as i32;
        size.top = 0;
        size.bottom = (unscaled_height as f32 * scaling_factor).round() as i32;

        kResultOk
    }

    unsafe fn on_size(&self, new_size: *mut ViewRect) -> tresult {
        check_null_ptr!(new_size);

        let scaling_factor = self.scaling_factor.load(Ordering::Relaxed).max(0.0001);
        let previous_logical = self.last_known_logical_size.load();
        let pending_internal = self.pending_internal_resize_target.load();

        if pending_internal.is_none() {
            let enforced_width = (previous_logical.0 as f32 * scaling_factor).round().max(1.0) as i32;
            let enforced_height = (previous_logical.1 as f32 * scaling_factor).round().max(1.0) as i32;

            let requested_width = ((*new_size).right - (*new_size).left).max(1);
            let requested_height = ((*new_size).bottom - (*new_size).top).max(1);

            (*new_size).right = (*new_size).left + enforced_width;
            (*new_size).bottom = (*new_size).top + enforced_height;

            if (requested_width != enforced_width || requested_height != enforced_height)
                && self.plug_frame.read().is_some()
            {
                if let Some(plug_frame) = &*self.plug_frame.read() {
                    let mut corrected_rect = ViewRect {
                        left: 0,
                        top: 0,
                        right: enforced_width,
                        bottom: enforced_height,
                    };

                    let plug_view: SharedVstPtr<dyn IPlugView> =
                        mem::transmute(&self.__iplugviewvptr as *const *const _);
                    let _ = plug_frame.resize_view(plug_view, &mut corrected_rect);
                }
            }

            return kResultOk;
        }

        // Accept host-confirmed size and push it back into the editor state as authoritative.
        // This closes the host->editor size loop and avoids ending a drag with one-step mismatch.
        let width = ((*new_size).right - (*new_size).left).max(1);
        let height = ((*new_size).bottom - (*new_size).top).max(1);

        let incoming_logical_width = ((width as f32) / scaling_factor).round().max(1.0) as u32;
        let incoming_logical_height = ((height as f32) / scaling_factor).round().max(1.0) as u32;

        let (min_size, aspect_ratio) = {
            let editor = self.editor.lock();
            (editor.min_size(), editor.aspect_ratio())
        };
        let (logical_width, logical_height) = self.constrain_size_logical(
            incoming_logical_width,
            incoming_logical_height,
            min_size,
            aspect_ratio,
        );

        // If host reports a size that does not match the editor constraints, request correction.
        if logical_width != incoming_logical_width || logical_height != incoming_logical_height {
            let corrected_width = (logical_width as f32 * scaling_factor).round().max(1.0) as i32;
            let corrected_height = (logical_height as f32 * scaling_factor).round().max(1.0) as i32;
            (*new_size).right = (*new_size).left + corrected_width;
            (*new_size).bottom = (*new_size).top + corrected_height;

            if let Some(plug_frame) = &*self.plug_frame.read() {
                let mut corrected_rect = ViewRect {
                    left: 0,
                    top: 0,
                    right: corrected_width,
                    bottom: corrected_height,
                };
                let plug_view: SharedVstPtr<dyn IPlugView> =
                    mem::transmute(&self.__iplugviewvptr as *const *const _);
                let _ = plug_frame.resize_view(plug_view, &mut corrected_rect);
            }
        }

        self.last_known_logical_size
            .store((logical_width, logical_height));
        self.pending_internal_resize_target.store(None);
        self.editor.lock().host_resized(logical_width, logical_height);
        kResultOk
    }

    unsafe fn on_focus(&self, _state: TBool) -> tresult {
        kNotImplemented
    }

    unsafe fn set_frame(&self, frame: *mut c_void) -> tresult {
        // The correct argument type is missing from the bindings
        let frame: SharedVstPtr<dyn IPlugFrame> = mem::transmute(frame);
        match frame.upgrade() {
            Some(frame) => {
                // On Linux the host will expose another interface that lets us run code on the
                // host's GUI thread. REAPER will segfault when we don't do this for resizes.
                #[cfg(target_os = "linux")]
                {
                    *self.run_loop_event_handler.0.write() = frame.cast().map(|run_loop| {
                        RunLoopEventHandler::new(self.inner.clone(), VstPtr::from(run_loop))
                    });
                }
                *self.plug_frame.write() = Some(VstPtr::from(frame));
            }
            None => {
                #[cfg(target_os = "linux")]
                {
                    *self.run_loop_event_handler.0.write() = None;
                }
                *self.plug_frame.write() = None;
            }
        }

        kResultOk
    }

    unsafe fn can_resize(&self) -> tresult {
        // Keep this enabled so plugin-initiated `request_resize()` is consistently honored by
        // hosts. Host outer-drag attempts are filtered in `check_size_constraint()`.
        kResultTrue
    }

    unsafe fn check_size_constraint(&self, rect: *mut ViewRect) -> tresult {
        check_null_ptr!(rect);

        let left = (*rect).left;
        let top = (*rect).top;

        let scaling_factor = self.scaling_factor.load(Ordering::Relaxed).max(0.0001);

        if let Some((pending_w, pending_h)) = self.pending_internal_resize_target.load() {
            let (min_size, aspect_ratio) = {
                let editor = self.editor.lock();
                (editor.min_size(), editor.aspect_ratio())
            };
            let (logical_width, logical_height) =
                self.constrain_size_logical(pending_w, pending_h, min_size, aspect_ratio);

            let width = ((logical_width as f32) * scaling_factor).round().max(1.0) as i32;
            let height = ((logical_height as f32) * scaling_factor).round().max(1.0) as i32;

            (*rect).right = left + width;
            (*rect).bottom = top + height;
            return kResultOk;
        }

        let (current_w, current_h) = self.last_known_logical_size.load();
        let width = ((current_w.max(1) as f32) * scaling_factor).round().max(1.0) as i32;
        let height = ((current_h.max(1) as f32) * scaling_factor).round().max(1.0) as i32;

        (*rect).right = left + width;
        (*rect).bottom = top + height;
        kResultOk
    }
}

impl<P: Vst3Plugin> IPlugViewContentScaleSupport for WrapperView<P> {
    unsafe fn set_scale_factor(&self, factor: f32) -> tresult {
        // TODO: So apparently Ableton Live doesn't call this function. Right now we'll hardcode the
        //       default scale to 1.0 on Linux and Windows since we can't easily get the scale from
        //       baseview. A better alternative would be to do the fallback DPI scale detection
        //       within NIH-plug itself. Then we can still only use baseview's system scale policy
        //       on macOS and both the editor implementation and the wrappers would know about the
        //       correct scale.

        // On macOS scaling is done by the OS, and all window sizes are in logical pixels
        if cfg!(target_os = "macos") {
            nih_debug_assert_failure!("Ignoring host request to set explicit DPI scaling factor");
            return kResultFalse;
        }

        if self.editor.lock().set_scale_factor(factor) {
            self.scaling_factor.store(factor, Ordering::Relaxed);
            kResultOk
        } else {
            kResultFalse
        }
    }
}

#[cfg(target_os = "linux")]
impl<P: Vst3Plugin> IEventHandler for RunLoopEventHandler<P> {
    unsafe fn on_fd_is_set(&self, _fd: FileDescriptor) {
        // There should be a one-to-one correlation to bytes being written to `self.socket_read_fd`
        // and events being pushed to `self.tasks`, but because the process of pushing a task and
        // notifying this thread through the socket is not atomic we can't reliably just read a byte
        // from this socket for every task we process. For instance, if `Self::post_task()` gets
        // called while this loop is already running, it could happen that we pop and execute the
        // task before the byte gets written to the socket. To avoid this, we'll clear the socket
        // upfront, and then execute the tasks afterwards. If this situation does happen, then the
        // worst thing that can happen is that this function is called a second time while
        // `self.tasks()` is already empty.
        let mut notify_value = [0; 32];
        loop {
            let read_result = libc::read(
                self.socket_read_fd,
                &mut notify_value as *mut _ as *mut c_void,
                std::mem::size_of_val(&notify_value),
            );

            // If after the first loop the socket contains no more data, then the `read()` call will
            // return -1 and `errno` will have been set to `EAGAIN`
            if read_result <= 0 {
                break;
            }
        }

        // This gets called from the host's UI thread because we wrote some bytes to the Unix domain
        // socket. We'll read that data from the socket again just to make REAPER happy.
        while let Some(task) = self.tasks.pop() {
            self.inner.execute(task, true);
        }
    }
}

#[cfg(target_os = "linux")]
impl<P: Vst3Plugin> Drop for RunLoopEventHandler<P> {
    fn drop(&mut self) {
        // When this object gets dropped and there are still unprocessed tasks left, then we'll
        // handle those in the regular event loop so no work gets lost
        let mut posting_failed = false;
        while let Some(task) = self.tasks.pop() {
            posting_failed |= !self
                .inner
                .event_loop
                .borrow()
                .as_ref()
                .unwrap()
                .schedule_gui(task);
        }

        if posting_failed {
            nih_debug_assert_failure!(
                "Outstanding tasks have been dropped when closing the editor as the task queue \
                 was full"
            );
        }

        unsafe {
            libc::close(self.socket_read_fd);
            libc::close(self.socket_write_fd);
        }

        let event_handler: SharedVstPtr<dyn IEventHandler> =
            unsafe { mem::transmute(&self.__ieventhandlervptr as *const _) };
        unsafe { self.run_loop.unregister_event_handler(event_handler) };
    }
}
