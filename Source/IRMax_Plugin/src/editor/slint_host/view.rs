struct SlintHostView {
    app: crate::editor::slint_preview_ui::AppWindow,
    ui_rx: mpsc::Receiver<UiMessage>,
    loader: Loader,
    spectrum_render_request_tx: mpsc::Sender<SpectrumRenderRequest>,
    spectrum_render_result_rx: mpsc::Receiver<SpectrumRenderResult>,
    spectrum_render_generation: AtomicU64,
    runtime: Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
    logger: Arc<InstanceLogger>,
}

impl SlintHostView {
    fn new(
        params: Arc<IRMaxPluginParams>,
        loader: Loader,
        runtime: Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
        logger: Arc<InstanceLogger>,
        gui_context: Arc<dyn GuiContext>,
        resize_coordinator: HostResizeCoordinator,
        ui_rx: mpsc::Receiver<UiMessage>,
    ) -> Self {
        let (spectrum_render_request_tx, spectrum_render_request_rx) =
            mpsc::channel::<SpectrumRenderRequest>();
        let (spectrum_render_result_tx, spectrum_render_result_rx) =
            mpsc::channel::<SpectrumRenderResult>();
        spawn_spectrum_render_worker(spectrum_render_request_rx, spectrum_render_result_tx);

        let app = crate::editor::slint_preview_ui::AppWindow::new().unwrap();
        let mix = params.mix.value().clamp(0.0, 1.0);
        app.set_mix_value(mix);
        app.set_mix_text(format!("{:.0}%", mix * 100.0).into());

        let output_gain = params.output_gain.modulated_plain_value();
        let output_db = util::gain_to_db(output_gain);
        let output_knob_value = output_gain_to_knob_value(output_gain);
        app.set_output_value(output_knob_value);
        app.set_output_text(format_output_text(output_db).into());

        let guard_profile = crate::clamp_guard_profile(params.guard_profile.value());
        app.set_guard_profile_value(guard_profile);
        app.set_guard_profile_text(crate::format_guard_profile_label(guard_profile).into());

        let active_slot = current_ui_active_slot(&params, &runtime);
        let slot_occupied = current_slot_occupied(&params);
        app.set_active_slot(active_slot);
        app.set_slot1_occupied(slot_occupied[0]);
        app.set_slot2_occupied(slot_occupied[1]);
        app.set_slot3_occupied(slot_occupied[2]);
        app.set_slot4_occupied(slot_occupied[3]);
        app.set_slot5_occupied(slot_occupied[4]);
        app.set_selected_preset_label("PRESET".into());
        app.set_preset_insert_enabled(false);
        app.set_spectrum_image(empty_spectrum_image());
        app.set_spectrum_status_text("No IR Loaded".into());
        app.set_ir_meta_text("LEN: -- | RATE: -- | BIT: -- | FILES: --".into());
        app.set_channel_text("CH --x--".into());
        app.set_folder_path_text("No Folder Loaded".into());
        app.set_backend_label("Auto".into());
        app.set_backend_switch_enabled(false);
        app.set_backend_switch_pending(false);
        app.set_log_path_label(format_log_path_for_ui(&logger.log_path_string()).into());
        {
            let shared = params.shared.read();
            app.set_dark_mode_enabled(shared.ui_dark_mode);
        }
        app.set_spectrum_has_data(false);
        app.set_spectrum_loading(false);
        app.set_save_mode_active(false);
        app.set_save_name_text("".into());

        let initial_loaded_info = runtime.read().loaded_info.clone();

        {
            let rt = runtime.read();
            app.set_channel_text(runtime_channel_text(&rt).into());
            app.set_folder_path_text(runtime_folder_path_text(params.as_ref(), &rt).into());
            refresh_backend_ui(&app, &rt);
        }

        let preset_root = sync_or_default_preset_root(params.as_ref());
        let mut preset_ui_state = PresetUiState {
            preset_root,
            preset_tree: PresetTreeNode::new_root(),
            preset_pack: PresetPack::default(),
            selected_leaf_id: None,
            preview_loaded_leaf_id: None,
            picker_path_chain: vec![String::new()],
            picker_selected_rows: vec![-1],
            picker_column_rows: Vec::new(),
            picker_render_entries: Vec::new(),
        };
        refresh_preset_sources(&mut preset_ui_state, logger.as_ref());
        let preset_state = Arc::new(Mutex::new(preset_ui_state));
        let save_flow_state = Arc::new(Mutex::new(SaveFlowState::default()));

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            app.on_mix_start_change(move || {
                let setter = ParamSetter::new(gui_context.as_ref());
                setter.begin_set_parameter(&params.mix);
            });
        }

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            let app_weak = app.as_weak();
            app.on_mix_changed(move |next| {
                let value = next.clamp(0.0, 1.0);
                if let Some(app) = app_weak.upgrade() {
                    app.set_mix_text(format!("{:.0}%", value * 100.0).into());
                }
                let setter = ParamSetter::new(gui_context.as_ref());
                setter.set_parameter(&params.mix, value);
            });
        }

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            app.on_mix_end_change(move || {
                let setter = ParamSetter::new(gui_context.as_ref());
                setter.end_set_parameter(&params.mix);
            });
        }

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            app.on_output_start_change(move || {
                let setter = ParamSetter::new(gui_context.as_ref());
                setter.begin_set_parameter(&params.output_gain);
            });
        }

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            let app_weak = app.as_weak();
            app.on_output_changed(move |next| {
                let knob_value = next.clamp(0.0, 1.0);
                let output_db = knob_value_to_output_db(knob_value);
                if let Some(app) = app_weak.upgrade() {
                    app.set_output_text(format_output_text(output_db).into());
                }

                let output_gain = knob_value_to_output_gain(knob_value);
                let setter = ParamSetter::new(gui_context.as_ref());
                setter.set_parameter(&params.output_gain, output_gain);
            });
        }

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            let loader = loader.clone();
            let runtime = runtime.clone();
            let logger = logger.clone();
            let app_weak = app.as_weak();
            app.on_guard_profile_changed(move |profile| {
                let clamped = crate::clamp_guard_profile(profile);
                let current = crate::clamp_guard_profile(params.guard_profile.value());
                if clamped == current {
                    if let Some(app) = app_weak.upgrade() {
                        app.set_guard_profile_value(clamped);
                        app.set_guard_profile_text(crate::format_guard_profile_label(clamped).into());
                    }
                    return;
                }

                let setter = ParamSetter::new(gui_context.as_ref());
                setter.begin_set_parameter(&params.guard_profile);
                setter.set_parameter(&params.guard_profile, clamped);
                setter.end_set_parameter(&params.guard_profile);

                if let Some(app) = app_weak.upgrade() {
                    app.set_guard_profile_value(clamped);
                    app.set_guard_profile_text(crate::format_guard_profile_label(clamped).into());
                }

                let active_slot = params.preset_slot_select.value().clamp(0, (SLOT_COUNT as i32) - 1) as usize;
                let slot_path = {
                    let shared = params.shared.read();
                    shared.slot_paths[active_slot].trim().to_string()
                };

                if slot_path.is_empty() {
                    logger.warn(
                        "SlintGuard",
                        &format!("Guard profile changed to {} but active slot {} has no path; skipping reload.", clamped, active_slot + 1),
                    );
                    return;
                }

                let (safe_sr, safe_outputs, safe_max_buffer, backend) = {
                    let rt = runtime.read();
                    let safe_sr = if rt.host_sample_rate == 0 { 48000 } else { rt.host_sample_rate };
                    let safe_outputs = if rt.host_outputs == 0 { 2 } else { rt.host_outputs };
                    let safe_max_buffer = if rt.host_max_buffer == 0 { 2048 } else { rt.host_max_buffer };
                    let backend = rt.gpu_backend_pending.unwrap_or(rt.gpu_backend);
                    (safe_sr, safe_outputs, safe_max_buffer, backend)
                };

                {
                    let mut shared = params.shared.write();
                    shared.last_folder = slot_path.clone();
                }

                {
                    let mut rt = runtime.write();
                    rt.preview_mode = false;
                }

                let _ = loader.tx.send(LoaderCommand::LoadFolder(
                    PathBuf::from(slot_path.clone()),
                    safe_sr,
                    safe_outputs,
                    safe_max_buffer,
                    backend,
                    None,
                    clamped,
                ));

                logger.info(
                    "SlintGuard",
                    &format!(
                        "Guard profile switched to {} ({}) and reloaded slot {}",
                        clamped,
                        crate::format_guard_profile_label(clamped),
                        active_slot + 1
                    ),
                );
            });
        }

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            let app_weak = app.as_weak();
            app.on_toggle_theme_mode(move || {
                if let Some(app) = app_weak.upgrade() {
                    let next_dark_mode = !app.get_dark_mode_enabled();
                    app.set_dark_mode_enabled(next_dark_mode);

                    {
                        let mut shared = params.shared.write();
                        shared.ui_dark_mode = next_dark_mode;
                    }

                    let setter = ParamSetter::new(gui_context.as_ref());
                    let current_dirty = params.dirty_trigger.value();
                    setter.begin_set_parameter(&params.dirty_trigger);
                    setter.set_parameter(&params.dirty_trigger, !current_dirty);
                    setter.end_set_parameter(&params.dirty_trigger);
                }
            });
        }

        {
            let params = params.clone();
            let gui_context = gui_context.clone();
            let loader = loader.clone();
            let runtime = runtime.clone();
            let logger = logger.clone();
            let preset_state = preset_state.clone();
            let app_weak = app.as_weak();
            app.on_select_slot(move |slot_index| {
                let selected_slot = slot_index.clamp(0, (SLOT_COUNT as i32) - 1);
                if let Some(app) = app_weak.upgrade() {
                    app.set_active_slot(selected_slot);
                }

                {
                    let mut rt = runtime.write();
                    rt.preview_mode = false;
                }

                if let Ok(mut state) = preset_state.lock() {
                    state.selected_leaf_id = None;
                    state.preview_loaded_leaf_id = None;
                    state.picker_path_chain.clear();
                    state.picker_path_chain.push(String::new());
                    state.picker_selected_rows.clear();
                    state.picker_selected_rows.push(-1);
                    state.picker_column_rows.clear();
                    state.picker_render_entries.clear();
                }

                if let Some(app) = app_weak.upgrade() {
                    app.set_selected_preset_label("PRESET".into());
                    app.set_preset_insert_enabled(false);
                    app.set_preset_selected_index(-1);
                    clear_picker_ui(&app);
                    app.set_preset_picker_open(false);

                    let slot_path_for_status = {
                        let shared = params.shared.read();
                        shared.slot_paths[selected_slot as usize].trim().to_string()
                    };

                    if !slot_path_for_status.is_empty() {
                        app.set_spectrum_loading(true);
                        app.set_spectrum_has_data(true);
                        app.set_spectrum_status_text("LOADING IR MATRIX\nSwitching slot...".into());
                    }
                }

                let setter = ParamSetter::new(gui_context.as_ref());
                setter.begin_set_parameter(&params.preset_slot_select);
                setter.set_parameter(&params.preset_slot_select, selected_slot);
                setter.end_set_parameter(&params.preset_slot_select);

                let slot_path = {
                    let shared = params.shared.read();
                    shared.slot_paths[selected_slot as usize].trim().to_string()
                };

                if !slot_path.is_empty() {
                    let (safe_sr, safe_outputs, safe_max_buffer, backend) = {
                        let rt = runtime.read();
                        let safe_sr = if rt.host_sample_rate == 0 { 48000 } else { rt.host_sample_rate };
                        let safe_outputs = if rt.host_outputs == 0 { 2 } else { rt.host_outputs };
                        let safe_max_buffer = if rt.host_max_buffer == 0 { 2048 } else { rt.host_max_buffer };
                        let backend = rt.gpu_backend_pending.unwrap_or(rt.gpu_backend);
                        (safe_sr, safe_outputs, safe_max_buffer, backend)
                    };

                    {
                        let mut shared = params.shared.write();
                        shared.last_folder = slot_path.clone();
                    }

                    let _ = loader.tx.send(LoaderCommand::LoadFolder(
                        PathBuf::from(slot_path.clone()),
                        safe_sr,
                        safe_outputs,
                        safe_max_buffer,
                        backend,
                        None,
                        crate::clamp_guard_profile(params.guard_profile.value()),
                    ));

                    logger.info(
                        "SlintSlot",
                        &format!("Slot {} selected -> reload '{}'", selected_slot + 1, slot_path),
                    );
                } else {
                    if let Some(app) = app_weak.upgrade() {
                        set_slot_occupied_flag(&app, selected_slot as usize, false);
                        app.set_channel_text("CH --x--".into());
                        app.set_folder_path_text("No Folder Loaded".into());
                        app.set_spectrum_has_data(false);
                        app.set_spectrum_loading(false);
                        app.set_spectrum_image(empty_spectrum_image());
                        app.set_spectrum_rows(1);
                        app.set_spectrum_cols(1);
                        app.set_spectrum_status_text("No IR Loaded".into());
                        app.set_ir_meta_text("LEN: -- | RATE: -- | BIT: -- | FILES: --".into());
                        app.set_spectrum_row_labels(slint::ModelRc::new(
                            slint::VecModel::from(Vec::<slint::SharedString>::new()),
                        ));
                    }

                    {
                        let mut rt = runtime.write();
                        rt.loaded_info = None;
                    }

                    {
                        let mut shared = params.shared.write();
                        shared.last_folder.clear();
                    }

                    logger.warn(
                        "SlintSlot",
                        &format!("Slot {} is empty; cleared current preview", selected_slot + 1),
                    );
                }

                let current_dirty = params.dirty_trigger.value();
                setter.begin_set_parameter(&params.dirty_trigger);
                setter.set_parameter(&params.dirty_trigger, !current_dirty);
                setter.end_set_parameter(&params.dirty_trigger);
            });
        }

        {
            let params = params.clone();
            let runtime = runtime.clone();
            let gui_context = gui_context.clone();
            let app_weak = app.as_weak();
            let logger = logger.clone();
            let preset_state = preset_state.clone();
            app.on_clear_slot(move |slot_index| {
                let slot = slot_index.clamp(0, (SLOT_COUNT as i32) - 1) as usize;
                let was_active = params.preset_slot_select.value().clamp(0, (SLOT_COUNT as i32) - 1) as usize == slot;

                let previous_path = {
                    let shared = params.shared.read();
                    shared.slot_paths[slot].clone()
                };

                {
                    let mut shared = params.shared.write();
                    shared.slot_paths[slot].clear();
                    if shared.last_folder == previous_path {
                        shared.last_folder.clear();
                    }
                }

                if let Some(app) = app_weak.upgrade() {
                    set_slot_occupied_flag(&app, slot, false);
                }

                if was_active {
                    if let Some(app) = app_weak.upgrade() {
                        app.set_channel_text("CH --x--".into());
                        app.set_folder_path_text("No Folder Loaded".into());
                        app.set_selected_preset_label("PRESET".into());
                        app.set_preset_insert_enabled(false);
                        app.set_preset_selected_index(-1);
                        app.set_preset_picker_open(false);
                        app.set_spectrum_has_data(false);
                        app.set_spectrum_loading(false);
                        app.set_spectrum_image(empty_spectrum_image());
                        app.set_spectrum_rows(1);
                        app.set_spectrum_cols(1);
                        app.set_spectrum_status_text("No IR Loaded".into());
                        app.set_ir_meta_text("LEN: -- | RATE: -- | BIT: -- | FILES: --".into());
                        app.set_spectrum_row_labels(slint::ModelRc::new(
                            slint::VecModel::from(Vec::<slint::SharedString>::new()),
                        ));
                    }

                    if let Ok(mut state) = preset_state.lock() {
                        state.selected_leaf_id = None;
                        state.preview_loaded_leaf_id = None;
                        state.picker_path_chain.clear();
                        state.picker_path_chain.push(String::new());
                        state.picker_selected_rows.clear();
                        state.picker_selected_rows.push(-1);
                        state.picker_column_rows.clear();
                        state.picker_render_entries.clear();
                    }

                    let mut rt = runtime.write();
                    rt.loaded_info = None;
                }

                let setter = ParamSetter::new(gui_context.as_ref());
                let current_dirty = params.dirty_trigger.value();
                setter.begin_set_parameter(&params.dirty_trigger);
                setter.set_parameter(&params.dirty_trigger, !current_dirty);
                setter.end_set_parameter(&params.dirty_trigger);

                logger.info("SlintSlot", &format!("Cleared slot {}", slot + 1));
            });
        }

        {
            let preset_state = preset_state.clone();
            let app_weak = app.as_weak();
            let logger = logger.clone();
            app.on_request_preset_leaf_items(move || {
                if let Ok(mut state) = preset_state.lock() {
                    refresh_preset_sources(&mut state, logger.as_ref());
                    reset_picker_navigation_to_root(&mut state);
                    rebuild_picker_render_state(&mut state);
                }

                if let Some(app) = app_weak.upgrade() {
                    if let Ok(state) = preset_state.lock() {
                        apply_picker_state_to_ui(&app, &state);
                    } else {
                        clear_picker_ui(&app);
                    }
                }
            });
        }

        {
            let params = params.clone();
            let logger = logger.clone();
            let preset_state = preset_state.clone();
            app.on_open_preset_folder(move || {
                let mut preset_root = sync_or_default_preset_root(params.as_ref());
                if let Ok(mut state) = preset_state.lock() {
                    state.preset_root = sync_or_default_preset_root(params.as_ref());
                    preset_root = state.preset_root.clone();
                    refresh_preset_sources(&mut state, logger.as_ref());
                }

                let _ = preset::ensure_preset_root_exists(&preset_root);
                if let Err(err) = open_folder_in_file_manager(&preset_root) {
                    logger.warn(
                        "SlintPreset",
                        &format!(
                            "Failed to open preset folder '{}': {}",
                            preset_root.to_string_lossy(),
                            err
                        ),
                    );
                }
            });
        }

        {
            let params = params.clone();
            let runtime = runtime.clone();
            let logger = logger.clone();
            let app_weak = app.as_weak();
            let preset_state = preset_state.clone();
            let save_flow_state = save_flow_state.clone();
            app.on_begin_save_preset(move || {
                let host_sr = {
                    let rt = runtime.read();
                    if rt.host_sample_rate == 0 {
                        48_000
                    } else {
                        rt.host_sample_rate
                    }
                };

                let ui_active_slot = app_weak
                    .upgrade()
                    .map(|app| app.get_active_slot())
                    .unwrap_or(-1);

                let preview_selected_leaf = if ui_active_slot < 0 {
                    preset_state
                        .lock()
                        .ok()
                        .and_then(|state| state.selected_leaf_id.clone())
                } else {
                    None
                };

                if ui_active_slot < 0 && preview_selected_leaf.is_none() {
                    if let Some(app) = app_weak.upgrade() {
                        app.set_save_mode_active(false);
                        app.set_save_name_text("".into());
                    }
                    logger.warn(
                        "SlintSave",
                        "Begin rename ignored because no preset leaf is selected.",
                    );
                    return;
                }

                let mut rename_parent_rel: Option<String> = None;
                let mut rename_original_leaf_id: Option<String> = None;
                let mut default_name_override: Option<String> = None;

                let mut resolved_source = if let Some(leaf_id) = preview_selected_leaf.clone() {
                    let preset_root = sync_or_default_preset_root(params.as_ref());
                    let source_root = preset::canonicalize_sample_rate_root(&preset::resolve_leaf_path(
                        &preset_root,
                        &leaf_id,
                    ));
                    rename_parent_rel = Some(preset_leaf_parent_rel(&leaf_id));
                    rename_original_leaf_id = Some(preset::normalize_leaf_id(&leaf_id));
                    default_name_override = Some(preset::leaf_label(&leaf_id));
                    Some((0usize, source_root, host_sr))
                } else if ui_active_slot >= 0 {
                    let slot_index = (ui_active_slot as usize).min(SLOT_COUNT - 1);
                    let source_path = {
                        let shared = params.shared.read();
                        shared.slot_paths[slot_index].trim().to_string()
                    };
                    if source_path.is_empty() {
                        None
                    } else {
                        Some((
                            slot_index,
                            preset::canonicalize_sample_rate_root(Path::new(&source_path)),
                            host_sr,
                        ))
                    }
                } else {
                    None
                };

                if resolved_source.is_none() {
                    resolved_source = resolve_active_slot_source(&params, &runtime);
                }

                if resolved_source.is_none() {
                    let last_folder = {
                        let shared = params.shared.read();
                        shared.last_folder.trim().to_string()
                    };
                    if !last_folder.is_empty() {
                        let fallback_slot = if ui_active_slot >= 0 {
                            (ui_active_slot as usize).min(SLOT_COUNT - 1)
                        } else {
                            current_active_slot(params.as_ref()) as usize
                        };
                        resolved_source = Some((
                            fallback_slot,
                            preset::canonicalize_sample_rate_root(Path::new(&last_folder)),
                            host_sr,
                        ));
                    }
                }

                let Some((source_slot, source_root, _)) = resolved_source else {
                    if let Some(app) = app_weak.upgrade() {
                        app.set_save_mode_active(false);
                        app.set_save_name_text("".into());
                    }
                    logger.warn("SlintSave", "Begin save ignored because no source path could be resolved.");
                    return;
                };

                let default_name = default_name_override
                    .unwrap_or_else(|| default_save_name_for_path(&source_root));

                if let Ok(mut state) = save_flow_state.lock() {
                    state.source_slot = Some(source_slot);
                    state.source_root = Some(source_root.clone());
                    state.rename_parent_rel = rename_parent_rel.clone();
                    state.rename_original_leaf_id = rename_original_leaf_id.clone();
                }

                if let Some(app) = app_weak.upgrade() {
                    app.set_save_name_text(default_name.into());
                    app.set_save_mode_active(true);
                    app.set_preset_picker_open(false);
                    app.set_slot_full_dialog_open(false);
                }

                let flow_label = if rename_parent_rel.is_some() {
                    "rename"
                } else {
                    "save"
                };
                logger.info(
                    "SlintSave",
                    &format!(
                        "Save mode entered (flow={}, slot={}, source='{}')",
                        flow_label,
                        source_slot + 1,
                        source_root.to_string_lossy()
                    ),
                );
            });
        }

        {
            let app_weak = app.as_weak();
            let save_flow_state = save_flow_state.clone();
            app.on_cancel_save_preset(move || {
                if let Ok(mut state) = save_flow_state.lock() {
                    *state = SaveFlowState::default();
                }

                if let Some(app) = app_weak.upgrade() {
                    app.set_save_mode_active(false);
                    app.set_save_name_text("".into());
                }
            });
        }

        {
            let params = params.clone();
            let runtime = runtime.clone();
            let logger = logger.clone();
            let app_weak = app.as_weak();
            let preset_state = preset_state.clone();
            let save_flow_state = save_flow_state.clone();
            app.on_confirm_save_preset(move || {
                let requested_name = app_weak
                    .upgrade()
                    .map(|app| app.get_save_name_text().to_string())
                    .unwrap_or_default();

                let host_sr = {
                    let rt = runtime.read();
                    if rt.host_sample_rate == 0 {
                        48_000
                    } else {
                        rt.host_sample_rate
                    }
                };

                let (cached_source, rename_parent_rel, rename_original_leaf_id) = save_flow_state
                    .lock()
                    .ok()
                    .map(|state| {
                        (
                            state.source_slot.zip(state.source_root.clone()),
                            state.rename_parent_rel.clone(),
                            state.rename_original_leaf_id.clone(),
                        )
                    })
                    .unwrap_or((None, None, None));

                let save_result: Result<(PathBuf, Option<String>), String> =
                    if let Some((source_slot, source_root)) = cached_source {
                        if let Some(parent_rel) = rename_parent_rel.as_deref() {
                            if let Some(original_leaf_id) = rename_original_leaf_id.as_deref() {
                                true_rename_preset_branch(
                                    source_slot,
                                    &source_root,
                                    host_sr,
                                    &requested_name,
                                    parent_rel,
                                    original_leaf_id,
                                    &params,
                                    &logger,
                                )
                                .map(|(path, leaf_id)| (path, Some(leaf_id)))
                            } else {
                                save_slot_source_into_preset_branch(
                                    source_slot,
                                    &source_root,
                                    host_sr,
                                    &requested_name,
                                    parent_rel,
                                    &params,
                                    &logger,
                                )
                                .map(|(path, leaf_id)| (path, Some(leaf_id)))
                            }
                        } else {
                            save_slot_source_into_user_preset(
                                source_slot,
                                &source_root,
                                host_sr,
                                &requested_name,
                                &params,
                                &logger,
                            )
                            .map(|path| (path, None))
                        }
                    } else {
                        save_active_slot_into_user_preset(&params, &runtime, &requested_name, &logger)
                            .map(|path| (path, None))
                    };

                match save_result {
                    Ok((target_path, renamed_leaf_id)) => {
                        let success_toast = if let Some(ref new_leaf_id) = renamed_leaf_id {
                            format!("Preset renamed to '{}'.", preset::leaf_label(new_leaf_id))
                        } else {
                            let user_name = target_path
                                .parent()
                                .and_then(|path| path.file_name())
                                .and_then(|name| name.to_str())
                                .filter(|name| !name.trim().is_empty())
                                .unwrap_or("Preset");
                            format!("Preset saved to User/{}.", user_name)
                        };

                        if let Ok(mut state) = preset_state.lock() {
                            if let Some(ref leaf_id) = renamed_leaf_id {
                                state.selected_leaf_id = Some(leaf_id.clone());
                                state.preview_loaded_leaf_id = Some(leaf_id.clone());
                            }
                            refresh_preset_sources(&mut state, logger.as_ref());
                            rebuild_picker_render_state(&mut state);
                        }

                        if let Some(app) = app_weak.upgrade() {
                            app.set_save_mode_active(false);
                            app.set_save_name_text("".into());
                            if let Ok(state) = preset_state.lock() {
                                apply_picker_state_to_ui(&app, &state);
                            }
                            trigger_ui_toast(&app, success_toast, false);
                        }

                        if let Ok(mut state) = save_flow_state.lock() {
                            *state = SaveFlowState::default();
                        }

                        if let Some(new_leaf_id) = renamed_leaf_id {
                            let rename_from = rename_original_leaf_id
                                .as_deref()
                                .unwrap_or("(unknown)");
                            logger.info(
                                "SlintSave",
                                &format!(
                                    "Preset branch saved '{}' -> '{}' at '{}'",
                                    rename_from,
                                    new_leaf_id,
                                    target_path.to_string_lossy()
                                ),
                            );
                        } else {
                            logger.info(
                                "SlintSave",
                                &format!("User preset saved to '{}'", target_path.to_string_lossy()),
                            );
                        }
                    }
                    Err(err) => {
                        logger.warn("SlintSave", &format!("Failed to save preset: {err}"));
                        if let Some(app) = app_weak.upgrade() {
                            let short_err = err.lines().next().unwrap_or(err.as_str());
                            trigger_ui_toast(&app, format!("Save failed: {}", short_err), true);
                        }
                    }
                }
            });
        }

        {
            let params = params.clone();
            let loader = loader.clone();
            let runtime = runtime.clone();
            let gui_context = gui_context.clone();
            let logger = logger.clone();
            let app_weak = app.as_weak();
            let preset_state = preset_state.clone();
            let app_weak_for_select = app_weak.clone();

            app.on_select_preset_leaf(move || {
                if let Some(app) = app_weak_for_select.upgrade() {
                    app.invoke_request_preset_leaf_items();

                    let selected_index = app.get_preset_selected_index();
                    let next_index = if selected_index < 0 {
                        0
                    } else {
                        selected_index + 1
                    };
                    app.invoke_choose_preset_leaf(next_index);
                }
            });

            let params_for_choose = params.clone();
            let loader_for_choose = loader.clone();
            let runtime_for_choose = runtime.clone();
            let gui_context_for_choose = gui_context.clone();
            let logger_for_choose = logger.clone();
            let app_weak_for_choose = app_weak.clone();
            let preset_state_for_choose = preset_state.clone();
            let app_weak_for_hover = app_weak.clone();
            let preset_state_for_hover = preset_state.clone();

            app.on_hover_preset_leaf(move |hovered_index| {
                let mut changed = false;

                if let Ok(mut state) = preset_state_for_hover.lock() {
                    if state.picker_render_entries.is_empty() {
                        return;
                    }

                    let Some(entry) =
                        resolve_picker_entry(&state.picker_render_entries, hovered_index)
                    else {
                        return;
                    };
                    let depth = entry.depth.max(0) as usize;

                    if state.picker_selected_rows.len() < depth + 1 {
                        state.picker_selected_rows.resize(depth + 1, -1);
                    }

                    if depth < state.picker_selected_rows.len()
                        && state.picker_selected_rows[depth] != entry.row
                    {
                        state.picker_selected_rows[depth] = entry.row;
                        changed = true;
                    }

                    match entry.kind {
                        PresetPickerEntryKind::Folder { rel_path } => {
                            if state.picker_path_chain.len() < depth + 1 {
                                state.picker_path_chain.resize(depth + 1, String::new());
                            }

                            let already_open =
                                state.picker_path_chain.len() > depth + 1
                                    && state.picker_path_chain[depth + 1] == rel_path;

                            if !already_open {
                                if state.picker_path_chain.len() > depth + 1 {
                                    state.picker_path_chain.truncate(depth + 1);
                                }
                                state.picker_path_chain.push(rel_path);

                                let chain_len = state.picker_path_chain.len();
                                if state.picker_selected_rows.len() < chain_len {
                                    state.picker_selected_rows.resize(chain_len, -1);
                                } else if state.picker_selected_rows.len() > chain_len {
                                    state.picker_selected_rows.truncate(chain_len);
                                }

                                if depth + 1 < state.picker_selected_rows.len() {
                                    state.picker_selected_rows[depth + 1] = -1;
                                }

                                changed = true;
                            }
                        }
                        PresetPickerEntryKind::Leaf { .. } => {
                            // 鍙跺瓙浠呴珮浜紝涓嶅湪 hover 闃舵鏀瑰啓璺緞閾撅紝
                            // 閬垮厤鐐瑰嚮鐬棿鍒楄〃閲嶅缓瀵艰嚧绱㈠紩婕傜Щ銆?
                        }
                    }

                    if changed {
                        rebuild_picker_render_state(&mut state);
                    }
                }

                if changed {
                    if let Some(app) = app_weak_for_hover.upgrade() {
                        if let Ok(state) = preset_state_for_hover.lock() {
                            apply_picker_state_to_ui(&app, &state);
                        }
                    }
                }
            });

            app.on_choose_preset_leaf(move |picked_index| {
                let mut selected_leaf: Option<String> = None;
                let mut selected_label = "PRESET".to_string();
                let mut insert_enabled = false;
                let mut should_auto_preview = false;
                let mut preset_root = PathBuf::new();
                let mut preset_pack = PresetPack::default();
                let mut close_picker = false;

                if let Ok(mut state) = preset_state_for_choose.lock() {
                    if !state.picker_render_entries.is_empty() {
                        if let Some(entry) =
                            resolve_picker_entry(&state.picker_render_entries, picked_index)
                        {
                            match &entry.kind {
                                PresetPickerEntryKind::Folder { rel_path } => {
                                    let depth = entry.depth.max(0) as usize;
                                    if state.picker_path_chain.len() > depth + 1 {
                                        state.picker_path_chain.truncate(depth + 1);
                                    }
                                    state.picker_path_chain.push(rel_path.clone());

                                    let chain_len = state.picker_path_chain.len();
                                    if state.picker_selected_rows.len() < chain_len {
                                        state.picker_selected_rows.resize(chain_len, -1);
                                    } else if state.picker_selected_rows.len() > chain_len {
                                        state.picker_selected_rows.truncate(chain_len);
                                    }
                                    if depth < state.picker_selected_rows.len() {
                                        state.picker_selected_rows[depth] = entry.row;
                                    }
                                    if depth + 1 < state.picker_selected_rows.len() {
                                        state.picker_selected_rows[depth + 1] = -1;
                                    }
                                    rebuild_picker_render_state(&mut state);
                                }
                                PresetPickerEntryKind::Leaf { leaf_id } => {
                                    let leaf_id = leaf_id.clone();
                                    selected_label = preset::leaf_label(&leaf_id);
                                    insert_enabled = true;
                                    should_auto_preview = state.preview_loaded_leaf_id.as_deref()
                                        != Some(leaf_id.as_str());
                                    preset_root = state.preset_root.clone();
                                    preset_pack = state.preset_pack.clone();
                                    state.selected_leaf_id = Some(leaf_id.clone());
                                    selected_leaf = Some(leaf_id);
                                    close_picker = true;

                                    let depth = entry.depth.max(0) as usize;
                                    if state.picker_selected_rows.len() < depth + 1 {
                                        state.picker_selected_rows.resize(depth + 1, -1);
                                    }
                                    if depth < state.picker_selected_rows.len() {
                                        state.picker_selected_rows[depth] = entry.row;
                                    }
                                }
                            }
                        }
                    } else {
                        state.selected_leaf_id = None;
                        state.preview_loaded_leaf_id = None;
                    }
                }

                if let Some(app) = app_weak_for_choose.upgrade() {
                    if let Ok(state) = preset_state_for_choose.lock() {
                        apply_picker_state_to_ui(&app, &state);
                    } else {
                        clear_picker_ui(&app);
                    }

                    let resolved_index = if let Ok(state) = preset_state_for_choose.lock() {
                        selected_index_in_picker(&state.picker_render_entries, state.selected_leaf_id.as_deref())
                    } else {
                        -1
                    };

                    let mut display_label = selected_label.clone();
                    let mut display_insert_enabled = insert_enabled;
                    if !close_picker {
                        if let Ok(state) = preset_state_for_choose.lock() {
                            if let Some(current_leaf) = state.selected_leaf_id.as_ref() {
                                display_label = preset::leaf_label(current_leaf);
                                display_insert_enabled = true;
                            }
                        }
                    }

                    app.set_selected_preset_label(display_label.into());
                    app.set_preset_insert_enabled(display_insert_enabled);
                    app.set_preset_selected_index(resolved_index);
                    if close_picker {
                        app.set_active_slot(-1);
                        app.set_preset_picker_open(false);
                    } else {
                        app.set_preset_picker_open(true);
                    }
                }

                if should_auto_preview {
                    if let Some(leaf_id) = selected_leaf.clone() {
                        let loaded = load_selected_leaf(
                            &leaf_id,
                            None,
                            &preset_root,
                            &preset_pack,
                            &params_for_choose,
                            &loader_for_choose,
                            &runtime_for_choose,
                            &gui_context_for_choose,
                            &logger_for_choose,
                            &app_weak_for_choose,
                        );
                        if loaded {
                            if let Ok(mut state) = preset_state_for_choose.lock() {
                                state.preview_loaded_leaf_id = Some(leaf_id);
                            }
                        }
                    }
                }
            });
        }

        {
            let params = params.clone();
            let loader = loader.clone();
            let runtime = runtime.clone();
            let gui_context = gui_context.clone();
            let logger = logger.clone();
            let app_weak = app.as_weak();
            let preset_state = preset_state.clone();

            app.on_insert_selected_preset(move || {
                let (selected_leaf, preset_root, preset_pack) = if let Ok(state) = preset_state.lock() {
                    (
                        state.selected_leaf_id.clone(),
                        state.preset_root.clone(),
                        state.preset_pack.clone(),
                    )
                } else {
                    (None, PathBuf::new(), PresetPack::default())
                };

                let Some(leaf_id) = selected_leaf else {
                    if let Some(app) = app_weak.upgrade() {
                        app.set_preset_insert_enabled(false);
                    }
                    return;
                };

                let Some(target_slot) = first_empty_slot(params.as_ref()) else {
                    if let Some(app) = app_weak.upgrade() {
                        app.set_slot_full_dialog_title("SLOT BANK FULL".into());
                        app.set_slot_full_dialog_message(
                            "All 5 slots are occupied. Please clear one slot and try INSERT again.".into(),
                        );
                        app.set_slot_full_dialog_open(true);
                    }

                    logger.warn(
                        "SlintPreset",
                        "Insert requested but no empty slot available.",
                    );
                    return;
                };

                let loaded = load_selected_leaf(
                    &leaf_id,
                    Some(target_slot),
                    &preset_root,
                    &preset_pack,
                    &params,
                    &loader,
                    &runtime,
                    &gui_context,
                    &logger,
                    &app_weak,
                );

                if loaded {
                    let setter = ParamSetter::new(gui_context.as_ref());
                    setter.begin_set_parameter(&params.preset_slot_select);
                    setter.set_parameter(&params.preset_slot_select, target_slot as i32);
                    setter.end_set_parameter(&params.preset_slot_select);

                    if let Ok(mut state) = preset_state.lock() {
                        state.preview_loaded_leaf_id = Some(leaf_id.clone());
                    }

                    if let Some(app) = app_weak.upgrade() {
                        app.set_active_slot(target_slot as i32);
                        app.set_selected_preset_label(preset::leaf_label(&leaf_id).into());
                        app.set_preset_insert_enabled(true);
                        app.set_slot_full_dialog_open(false);
                        if let Ok(state) = preset_state.lock() {
                            apply_picker_state_to_ui(&app, &state);
                        }
                        app.set_preset_picker_open(false);
                    }
                }
            });
        }

        {
            let params = params.clone();
            let loader = loader.clone();
            let runtime = runtime.clone();
            let gui_context = gui_context.clone();
            let logger = logger.clone();
            let app_weak = app.as_weak();
            app.on_load_ir_folder(move || {
                let Some(folder_path) = rfd::FileDialog::new().pick_folder() else {
                    return;
                };

                let canonical_folder = preset::canonicalize_sample_rate_root(&folder_path);

                let folder_str = canonical_folder.to_string_lossy().to_string();
                let active_slot = params.preset_slot_select.value().clamp(0, (SLOT_COUNT as i32) - 1) as usize;

                {
                    let mut shared = params.shared.write();
                    shared.last_folder = folder_str.clone();
                    shared.slot_paths[active_slot] = folder_str.clone();
                }

                {
                    let mut rt = runtime.write();
                    rt.preview_mode = false;
                }

                let (safe_sr, safe_outputs, safe_max_buffer, backend) = {
                    let rt = runtime.read();
                    let safe_sr = if rt.host_sample_rate == 0 {
                        48000
                    } else {
                        rt.host_sample_rate
                    };
                    let safe_outputs = if rt.host_outputs == 0 { 2 } else { rt.host_outputs };
                    let safe_max_buffer = if rt.host_max_buffer == 0 {
                        2048
                    } else {
                        rt.host_max_buffer
                    };
                    let backend = rt.gpu_backend_pending.unwrap_or(rt.gpu_backend);
                    (safe_sr, safe_outputs, safe_max_buffer, backend)
                };

                let _ = loader.tx.send(LoaderCommand::LoadFolder(
                    canonical_folder,
                    safe_sr,
                    safe_outputs,
                    safe_max_buffer,
                    backend,
                    None,
                    crate::clamp_guard_profile(params.guard_profile.value()),
                ));

                let setter = ParamSetter::new(gui_context.as_ref());
                let current_dirty = params.dirty_trigger.value();
                setter.begin_set_parameter(&params.dirty_trigger);
                setter.set_parameter(&params.dirty_trigger, !current_dirty);
                setter.end_set_parameter(&params.dirty_trigger);

                if let Some(app) = app_weak.upgrade() {
                    set_slot_occupied_flag(&app, active_slot, true);
                }

                logger.info(
                    "Slint",
                    &format!(
                        "LOAD FOLDER loaded '{}' into slot {}",
                        folder_str,
                        active_slot + 1
                    ),
                );
            });
        }

        {
            let loader = loader.clone();
            let runtime = runtime.clone();
            let logger = logger.clone();
            let app_weak = app.as_weak();
            app.on_switch_backend(move || {
                let (current_backend, pending_backend, mut backend_list, can_switch, host_sr, host_outputs, host_max_buffer, loaded_path) = {
                    let rt = runtime.read();
                    (
                        rt.gpu_backend,
                        rt.gpu_backend_pending,
                        rt.gpu_backends.clone(),
                        rt.gpu_can_switch,
                        rt.host_sample_rate,
                        rt.host_outputs,
                        rt.host_max_buffer,
                        rt.loaded_info.as_ref().map(|info| info.full_path.clone()),
                    )
                };

                if backend_list.is_empty() {
                    backend_list = available_backends();
                }

                let can_switch_now = if backend_list.is_empty() {
                    false
                } else {
                    can_switch || (cfg!(target_os = "windows") && backend_list.len() >= 2)
                };

                let effective_backend = pending_backend.unwrap_or(current_backend);
                let Some(next) = next_backend(effective_backend, &backend_list) else {
                    logger.warn(
                        "SlintBackend",
                        &format!("No switch target from backend {}", effective_backend.label()),
                    );
                    return;
                };

                if !can_switch_now || pending_backend.is_some() {
                    logger.warn(
                        "SlintBackend",
                        &format!(
                            "Switch ignored (can_switch={}, pending={})",
                            can_switch_now,
                            pending_backend.is_some()
                        ),
                    );
                    return;
                }

                let Some(path) = loaded_path else {
                    {
                        let mut rt = runtime.write();
                        if rt.gpu_backends.is_empty() {
                            rt.gpu_backends = backend_list.clone();
                            rt.gpu_can_switch = cfg!(target_os = "windows") && rt.gpu_backends.len() >= 2;
                        }
                        rt.gpu_backend = next;
                        rt.gpu_backend_pending = None;
                    }

                    if let Some(app) = app_weak.upgrade() {
                        let rt = runtime.read();
                        refresh_backend_ui(&app, &rt);
                    }

                    logger.info(
                        "SlintBackend",
                        &format!(
                            "Backend preselected without loaded IR: {} -> {}",
                            effective_backend.label(),
                            next.label()
                        ),
                    );
                    return;
                };

                {
                    let mut rt = runtime.write();
                    rt.gpu_backend_pending = Some(next);
                }

                if let Some(app) = app_weak.upgrade() {
                    app.set_backend_label(next.label().into());
                    app.set_backend_switch_enabled(false);
                    app.set_backend_switch_pending(true);
                }

                let safe_sr = if host_sr == 0 { 48000 } else { host_sr };
                let safe_outputs = if host_outputs == 0 { 2 } else { host_outputs };
                let safe_max_buffer = if host_max_buffer == 0 { 2048 } else { host_max_buffer };

                let _ = loader.tx.send(LoaderCommand::LoadFolder(
                    path,
                    safe_sr,
                    safe_outputs,
                    safe_max_buffer,
                    next,
                    None,
                    crate::clamp_guard_profile(params.guard_profile.value()),
                ));

                logger.info(
                    "SlintBackend",
                    &format!("Backend switch requested: {} -> {}", effective_backend.label(), next.label()),
                );
            });
        }

        {
            let logger = logger.clone();
            app.on_open_log_file(move || {
                let log_path = PathBuf::from(logger.log_path_string());
                if let Err(err) = open_file_in_default_app(&log_path) {
                    logger.warn(
                        "SlintLog",
                        &format!("Failed to open log file '{}': {}", log_path.to_string_lossy(), err),
                    );
                }
            });
        }

        {
            let gui_context = gui_context.clone();
            let logger = logger.clone();
            let resize_coordinator = resize_coordinator.clone();
            app.on_handle_resize_request(move |width_px, height_px| {
                let requested_w = width_px.max(1.0).round() as u32;
                let requested_h = height_px.max(1.0).round() as u32;
                let now = Instant::now();
                let Some(constrained) =
                    resize_coordinator.begin_request_from_ui(requested_w, requested_h, now)
                else {
                    return;
                };

                if !gui_context.request_resize() {
                    resize_coordinator.reject_pending_request();

                    logger.warn(
                        "SlintResize",
                        &format!(
                            "Host denied resize request to {}x{}",
                            constrained.0, constrained.1
                        ),
                    );
                }
            });
        }

        let view = Self {
            app,
            ui_rx,
            loader,
            spectrum_render_request_tx,
            spectrum_render_result_rx,
            spectrum_render_generation: AtomicU64::new(0),
            runtime,
            logger,
        };

        if let Some(info) = initial_loaded_info {
            view.app.set_spectrum_loading(true);
            view.app.set_spectrum_status_text("LOADING IR MATRIX\nRendering spectrum...".into());
            view.request_spectrum_render(info);
        }

        view
    }

}

