use std::rc::Rc;
use std::sync::Arc;

use nih_plug::editor::ParentWindowHandle;
use plugin_canvas_slint::plugin_canvas::window::WindowAttributes;
use plugin_canvas_slint::view::PluginView;
use raw_window_handle::{AppKitWindowHandle, RawWindowHandle, Win32WindowHandle, XcbWindowHandle};

pub type EditorHandle = plugin_canvas_slint::editor::EditorHandle;

pub struct SlintEditor;

impl SlintEditor {
    pub fn open<C, B>(
        parent: ParentWindowHandle,
        window_attributes: WindowAttributes,
        view_builder: B,
    ) -> Rc<EditorHandle>
    where
        C: PluginView + 'static,
        B: Fn(Arc<plugin_canvas_slint::plugin_canvas::Window>) -> C + 'static,
    {
        let raw_parent = convert_parent_window_handle(parent);
        plugin_canvas_slint::editor::SlintEditor::open(raw_parent, window_attributes, view_builder)
    }
}

pub fn convert_parent_window_handle(parent: ParentWindowHandle) -> RawWindowHandle {
    match parent {
        ParentWindowHandle::X11Window(window) => {
            let handle = XcbWindowHandle::new(std::num::NonZeroU32::new(window).unwrap());
            RawWindowHandle::Xcb(handle)
        }
        ParentWindowHandle::AppKitNsView(ns_view) => {
            let handle = AppKitWindowHandle::new(std::ptr::NonNull::new(ns_view.cast()).unwrap());
            RawWindowHandle::AppKit(handle)
        }
        ParentWindowHandle::Win32Hwnd(hwnd) => {
            let handle = Win32WindowHandle::new(std::num::NonZeroIsize::new(hwnd as isize).unwrap());
            RawWindowHandle::Win32(handle)
        }
    }
}
