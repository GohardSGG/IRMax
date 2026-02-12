use std::sync::Arc;

use nih_plug::prelude::Editor;
use parking_lot::RwLock;

use crate::IRMaxPluginParams;
use crate::loader::Loader;

pub mod slint_host;

pub mod slint_preview_ui {
    slint::include_modules!();
}

pub fn create(
    params: Arc<IRMaxPluginParams>,
    loader: Loader,
    logger: Arc<crate::logger::InstanceLogger>,
    runtime: Arc<RwLock<crate::RuntimePluginState>>,
) -> Option<Box<dyn Editor>> {
    logger.info("Editor", "Using Slint host editor.");

    let editor = slint_host::SlintHostEditor::new((980, 654), params, loader, runtime, logger);
    Some(Box::new(editor))
}

