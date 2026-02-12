impl SlintHostView {
    fn request_spectrum_render(&self, info: crate::loader::LoadedInfo) {
        let generation = self
            .spectrum_render_generation
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        let _ = self
            .spectrum_render_request_tx
            .send(SpectrumRenderRequest { generation, info });
    }

    fn flush_spectrum_render_results(&self) {
        while let Ok(result) = self.spectrum_render_result_rx.try_recv() {
            let latest_generation = self.spectrum_render_generation.load(Ordering::Acquire);
            if result.generation != latest_generation {
                continue;
            }

            let preview_mode = {
                let rt = self.runtime.read();
                rt.preview_mode
            };
            apply_loaded_info_to_spectrum_ui(&self.app, &result.info, result.panel, preview_mode);
        }
    }

    fn flush_loader_messages(&self) {
        while let Ok(message) = self.loader.rx.try_recv() {
            match message {
                crate::loader::LoaderResponse::Loading(text) => {
                    let has_previous_data = self.app.get_spectrum_has_data();
                    self.app.set_spectrum_loading(true);
                    if !has_previous_data {
                        self.app.set_spectrum_has_data(false);
                        self.app.set_spectrum_image(empty_spectrum_image());
                    }
                    let loading_detail = text.trim();
                    let loading_text = if loading_detail.is_empty() {
                        "LOADING IR MATRIX\nPreparing selected slot...".to_string()
                    } else {
                        format!("LOADING IR MATRIX\n{loading_detail}")
                    };
                    self.app.set_spectrum_status_text(loading_text.into());
                    self.app
                        .set_ir_meta_text("LEN: -- | RATE: -- | BIT: -- | FILES: --".into());
                }
                crate::loader::LoaderResponse::Loaded(info, _) => {
                    let mut rt = self.runtime.write();
                    rt.loaded_info = Some(info.clone());

                    self.app.set_spectrum_loading(true);
                    self.app
                        .set_spectrum_status_text("LOADING IR MATRIX\nRendering spectrum...".into());
                    self.request_spectrum_render(info);
                }
                crate::loader::LoaderResponse::Error(err) => {
                    self.spectrum_render_generation.fetch_add(1, Ordering::AcqRel);
                    self.app.set_spectrum_loading(false);
                    self.app.set_spectrum_has_data(false);
                    self.app.set_channel_text("CH --x--".into());
                    self.app.set_folder_path_text("No Folder Loaded".into());
                    self.app.set_spectrum_image(empty_spectrum_image());
                    self.app.set_spectrum_rows(1);
                    self.app.set_spectrum_cols(1);
                    self.app
                        .set_spectrum_row_labels(slint::ModelRc::new(slint::VecModel::from(Vec::<slint::SharedString>::new())));

                    let backend_failure = err.starts_with(crate::loader::BACKEND_FAILURE_PREFIX);
                    if backend_failure {
                        let detail = err
                            .trim_start_matches(crate::loader::BACKEND_FAILURE_PREFIX)
                            .trim();
                        let dialog_message = if detail.is_empty() {
                            "CUDA failed, Vulkan fallback also failed. Please update GPU drivers or switch backend settings.".to_string()
                        } else {
                            format!(
                                "CUDA failed, Vulkan fallback also failed.\n{}\n\nPlease update GPU drivers, then reopen the plugin.",
                                detail
                            )
                        };
                        self.app.set_slot_full_dialog_title("GPU BACKEND FAILED".into());
                        self.app.set_slot_full_dialog_message(dialog_message.into());
                        self.app.set_slot_full_dialog_open(true);
                        self.app
                            .set_spectrum_status_text("Load Error: GPU backend initialization failed".into());
                    } else {
                        self.app
                            .set_spectrum_status_text(format!("Load Error: {err}").into());
                    }

                    self.app
                        .set_ir_meta_text("LEN: -- | RATE: -- | BIT: -- | FILES: --".into());
                    self.logger.warn("SlintSpectrum", &format!("Loader error: {err}"));
                }
                crate::loader::LoaderResponse::BackendReady(backend) => {
                    let mut rt = self.runtime.write();
                    rt.gpu_backend = backend;
                    rt.gpu_backend_pending = None;

                    refresh_backend_ui(&self.app, &rt);
                }
            }
        }
    }

    fn flush_ui_messages(&self) {
        while let Ok(message) = self.ui_rx.try_recv() {
            match message {
                UiMessage::SetMix(value) => self.app.set_mix_value(value.clamp(0.0, 1.0)),
                UiMessage::SetMixText(text) => self.app.set_mix_text(text.into()),
                UiMessage::SetOutput(value) => self.app.set_output_value(value.clamp(0.0, 1.0)),
                UiMessage::SetOutputText(text) => self.app.set_output_text(text.into()),
                UiMessage::SetGuardProfile { value, text } => {
                    self.app.set_guard_profile_value(crate::clamp_guard_profile(value));
                    self.app.set_guard_profile_text(text.into());
                }
                UiMessage::SetActiveSlot(slot) => {
                    self.app
                        .set_active_slot(slot.clamp(-1, (SLOT_COUNT as i32) - 1))
                }
                UiMessage::SetSlotOccupied(occupied) => {
                    self.app.set_slot1_occupied(occupied[0]);
                    self.app.set_slot2_occupied(occupied[1]);
                    self.app.set_slot3_occupied(occupied[2]);
                    self.app.set_slot4_occupied(occupied[3]);
                    self.app.set_slot5_occupied(occupied[4]);
                }
                UiMessage::SetChannelText(text) => self.app.set_channel_text(text.into()),
                UiMessage::SetBackendUi {
                    label,
                    enabled,
                    pending,
                } => {
                    self.app.set_backend_label(label.into());
                    self.app.set_backend_switch_enabled(enabled);
                    self.app.set_backend_switch_pending(pending);
                }
                UiMessage::SetLogPathLabel(text) => self.app.set_log_path_label(text.into()),
                UiMessage::SetFolderPathText(text) => self.app.set_folder_path_text(text.into()),
                UiMessage::SetWindowSize(width, height) => {
                    self.app
                        .window()
                        .set_size(slint::LogicalSize::new(width as f32, height as f32));
                }
            }
        }
    }
}

impl PluginView for SlintHostView {
    fn window(&self) -> &slint::Window {
        self.app.window()
    }

    fn on_event(
        &self,
        _event: &nih_plug_slint::plugin_canvas::Event,
    ) -> nih_plug_slint::plugin_canvas::event::EventResponse {
        self.flush_spectrum_render_results();
        self.flush_loader_messages();
        self.flush_spectrum_render_results();
        self.flush_ui_messages();
        nih_plug_slint::plugin_canvas::event::EventResponse::Ignored
    }
}


