use std::rc::Rc;

use crate::editor::EditorHandle;

pub struct SlintHostHandle {
    editor_handle: Rc<EditorHandle>,
}

impl SlintHostHandle {
    pub fn new(editor_handle: Rc<EditorHandle>) -> Self {
        Self { editor_handle }
    }
}

unsafe impl Send for SlintHostHandle {}

impl Drop for SlintHostHandle {
    fn drop(&mut self) {
        if let Some(editor_handle) = Rc::get_mut(&mut self.editor_handle) {
            editor_handle.on_frame();
        }
    }
}
