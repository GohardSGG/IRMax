enum UiMessage {
    SetMix(f32),
    SetMixText(String),
    SetOutput(f32),
    SetOutputText(String),
    SetGuardProfile {
        value: i32,
        text: String,
    },
    SetActiveSlot(i32),
    SetSlotOccupied([bool; SLOT_COUNT]),
    SetChannelText(String),
    SetFolderPathText(String),
    SetBackendUi {
        label: String,
        enabled: bool,
        pending: bool,
    },
    SetLogPathLabel(String),
    SetWindowSize(u32, u32),
}

struct SlintBridgeState {
    ui_events: ParamEventDispatcher<UiMessage>,
}

pub struct SlintHostEditor {
    params: Arc<IRMaxPluginParams>,
    loader: Loader,
    runtime: Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
    logger: Arc<InstanceLogger>,
    state: Mutex<SlintBridgeState>,
    resize_coordinator: HostResizeCoordinator,
}

impl SlintHostEditor {
    pub fn new(
        size: (u32, u32),
        params: Arc<IRMaxPluginParams>,
        loader: Loader,
        runtime: Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
        logger: Arc<InstanceLogger>,
    ) -> Self {
        let resize_coordinator = HostResizeCoordinator::new(
            size,
            ResizePolicy {
                min_width: SLINT_MIN_WIDTH,
                min_height: SLINT_MIN_HEIGHT,
                aspect_ratio: SLINT_ASPECT_RATIO,
            },
            ResizeDebounce::default(),
        );

        Self {
            params,
            loader,
            runtime,
            logger,
            state: Mutex::new(SlintBridgeState {
                ui_events: ParamEventDispatcher::default(),
            }),
            resize_coordinator,
        }
    }
}

impl Editor for SlintHostEditor {
    fn spawn(&self, parent: ParentWindowHandle, gui_context: Arc<dyn GuiContext>) -> Box<dyn Any + Send> {
        let (ui_tx, ui_rx) = mpsc::channel::<UiMessage>();
        {
            let mut guard = self.state.lock().expect("slint state lock poisoned");
            guard.ui_events.set_sender(ui_tx);
        }

        let initial_size = self.size();
        let window_attributes = WindowAttributes::new(
            LogicalSize::new(initial_size.0 as f64, initial_size.1 as f64),
            1.0,
        );

        let params = self.params.clone();
        let loader = self.loader.clone();
        let runtime = self.runtime.clone();
        let logger = self.logger.clone();
        let resize_coordinator = self.resize_coordinator.clone();
        let gui_for_closure = gui_context.clone();
        let ui_rx_cell = Arc::new(Mutex::new(Some(ui_rx)));

        let editor_handle = SlintEditor::open(parent, window_attributes, move |_window| {
            let rx = {
                let mut guard = ui_rx_cell
                    .lock()
                    .expect("slint rx lock poisoned");
                guard
                    .take()
                    .expect("slint receiver already consumed")
            };

            SlintHostView::new(
                params.clone(),
                loader.clone(),
                runtime.clone(),
                logger.clone(),
                gui_for_closure.clone(),
                resize_coordinator.clone(),
                rx,
            )
        });

        Box::new(SlintHostHandle::new(editor_handle))
    }

    fn size(&self) -> (u32, u32) {
        self.resize_coordinator.size()
    }

    fn set_scale_factor(&self, _factor: f32) -> bool {
        false
    }

    fn min_size(&self) -> Option<(u32, u32)> {
        Some(self.resize_coordinator.min_size())
    }

    fn aspect_ratio(&self) -> Option<f32> {
        Some(self.resize_coordinator.aspect_ratio())
    }

    fn host_resized(&self, width: u32, height: u32) {
        let constrained = self.resize_coordinator.on_host_resized(width, height);

        let maybe_tx = {
            let guard = self.state.lock().expect("slint state lock poisoned");
            guard.ui_events.sender()
        };

        if let Some(tx) = maybe_tx {
            let _ = tx.send(UiMessage::SetWindowSize(constrained.0, constrained.1));
        }
    }

    fn param_value_changed(&self, id: &str, _normalized_value: f32) {
        ParamEventAdapter::on_param_value_changed(self, id);
    }

    fn param_modulation_changed(&self, id: &str, modulation_offset: f32) {
        ParamEventAdapter::on_param_modulation_changed(self, id, modulation_offset);
    }

    fn param_values_changed(&self) {
        ParamEventAdapter::on_param_values_changed(self);
    }
}

impl SlintHostEditor {
    fn push_mix_to_ui(&self) {
        let mix = self.params.mix.value().clamp(0.0, 1.0);
        let mix_text = format!("{:.0}%", mix * 100.0);
        let maybe_tx = {
            let guard = self.state.lock().expect("slint state lock poisoned");
            guard.ui_events.sender()
        };

        if let Some(tx) = maybe_tx {
            let _ = tx.send(UiMessage::SetMix(mix));
            let _ = tx.send(UiMessage::SetMixText(mix_text));
        }
    }

    fn push_output_to_ui(&self) {
        let output_gain = self.params.output_gain.modulated_plain_value();
        let output_db = util::gain_to_db(output_gain);
        let output_knob_value = output_gain_to_knob_value(output_gain);
        let output_text = format_output_text(output_db);
        let maybe_tx = {
            let guard = self.state.lock().expect("slint state lock poisoned");
            guard.ui_events.sender()
        };

        if let Some(tx) = maybe_tx {
            let _ = tx.send(UiMessage::SetOutput(output_knob_value));
            let _ = tx.send(UiMessage::SetOutputText(output_text));
        }
    }

    fn push_guard_profile_to_ui(&self) {
        let value = crate::clamp_guard_profile(self.params.guard_profile.value());
        let text = crate::format_guard_profile_label(value);
        let maybe_tx = {
            let guard = self.state.lock().expect("slint state lock poisoned");
            guard.ui_events.sender()
        };

        if let Some(tx) = maybe_tx {
            let _ = tx.send(UiMessage::SetGuardProfile { value, text });
        }
    }

    fn push_slot_state_to_ui(&self) {
        let active_slot = current_ui_active_slot(&self.params, &self.runtime);
        let slot_occupied = current_slot_occupied(&self.params);
        let maybe_tx = {
            let guard = self.state.lock().expect("slint state lock poisoned");
            guard.ui_events.sender()
        };

        let (channel_text, folder_path_text) = {
            let rt = self.runtime.read();
            (
                runtime_channel_text(&rt),
                runtime_folder_path_text(self.params.as_ref(), &rt),
            )
        };

        let log_path_label = if self.logger.file_logging_enabled() {
            format_log_path_for_ui(&self.logger.log_path_string())
        } else {
            String::new()
        };

        let (backend_label, backend_enabled, backend_pending) = {
            let rt = self.runtime.read();
            let effective_backend = rt.gpu_backend_pending.unwrap_or(rt.gpu_backend);
            let backends = if rt.gpu_backends.is_empty() {
                available_backends()
            } else {
                rt.gpu_backends.clone()
            };
            let can_switch = if rt.gpu_backends.is_empty() {
                cfg!(target_os = "windows") && backends.len() >= 2
            } else {
                rt.gpu_can_switch
            };
            let has_switch_target = next_backend(effective_backend, &backends).is_some();
            let enabled = can_switch && rt.gpu_backend_pending.is_none() && has_switch_target;
            (effective_backend.label().to_string(), enabled, rt.gpu_backend_pending.is_some())
        };

        if let Some(tx) = maybe_tx {
            let _ = tx.send(UiMessage::SetActiveSlot(active_slot));
            let _ = tx.send(UiMessage::SetSlotOccupied(slot_occupied));
            let _ = tx.send(UiMessage::SetChannelText(channel_text));
            let _ = tx.send(UiMessage::SetFolderPathText(folder_path_text));
            let _ = tx.send(UiMessage::SetBackendUi {
                label: backend_label,
                enabled: backend_enabled,
                pending: backend_pending,
            });
            let _ = tx.send(UiMessage::SetLogPathLabel(log_path_label));
        }
    }
}

impl ParamEventAdapter for SlintHostEditor {
    fn on_param_value_changed(&self, id: &str) {
        match id {
            "mix" => self.push_mix_to_ui(),
            "output_gain" => self.push_output_to_ui(),
            "guard_profile" => self.push_guard_profile_to_ui(),
            "preset_slot_select" => self.push_slot_state_to_ui(),
            _ => {}
        }
    }

    fn on_param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {}

    fn on_param_values_changed(&self) {
        self.push_mix_to_ui();
        self.push_output_to_ui();
        self.push_guard_profile_to_ui();
        self.push_slot_state_to_ui();
    }
}

