#[derive(Clone)]
enum PresetPickerEntryKind {
    Folder { rel_path: String },
    Leaf { leaf_id: String },
}

#[derive(Clone)]
struct PresetPickerEntry {
    label: String,
    kind: PresetPickerEntryKind,
}

fn is_terminal_preset_node(node: &PresetTreeNode) -> bool {
    node.has_ir_group && node.children.is_empty()
}

fn find_tree_node_by_rel_path<'a>(
    node: &'a PresetTreeNode,
    rel_path: &str,
) -> Option<&'a PresetTreeNode> {
    if rel_path.is_empty() {
        return Some(node);
    }
    if node.rel_path == rel_path {
        return Some(node);
    }
    for child in &node.children {
        if let Some(found) = find_tree_node_by_rel_path(child, rel_path) {
            return Some(found);
        }
    }
    None
}

#[derive(Clone)]
struct PresetPickerRenderEntry {
    label: String,
    depth: i32,
    row: i32,
    is_folder: bool,
    kind: PresetPickerEntryKind,
}

const PICKER_ENTRY_KEY_STRIDE: i32 = 10_000;

fn build_entries_for_node(tree: &PresetTreeNode, rel_path: &str) -> Vec<PresetPickerEntry> {
    let normalized = preset::normalize_leaf_id(rel_path);
    let Some(current_node) = find_tree_node_by_rel_path(tree, &normalized) else {
        return Vec::new();
    };

    let mut entries = Vec::new();
    for child in &current_node.children {
        if is_terminal_preset_node(child) {
            entries.push(PresetPickerEntry {
                label: child.name.clone(),
                kind: PresetPickerEntryKind::Leaf {
                    leaf_id: child.rel_path.clone(),
                },
            });
        } else {
            entries.push(PresetPickerEntry {
                label: child.name.clone(),
                kind: PresetPickerEntryKind::Folder {
                    rel_path: child.rel_path.clone(),
                },
            });
        }
    }

    entries
}

fn selected_index_in_picker(
    entries: &[PresetPickerRenderEntry],
    selected_leaf_id: Option<&str>,
) -> i32 {
    let Some(selected_leaf) = selected_leaf_id else {
        return -1;
    };

    entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.kind,
                PresetPickerEntryKind::Leaf { leaf_id } if leaf_id == selected_leaf
            )
        })
        .map(|index| index as i32)
        .unwrap_or(-1)
}

fn resolve_picker_entry(
    entries: &[PresetPickerRenderEntry],
    encoded_or_index: i32,
) -> Option<PresetPickerRenderEntry> {
    if encoded_or_index >= PICKER_ENTRY_KEY_STRIDE + 1 {
        let packed = encoded_or_index - 1;
        let depth = (packed / PICKER_ENTRY_KEY_STRIDE) - 1;
        let row = packed % PICKER_ENTRY_KEY_STRIDE;

        if depth >= 0 && row >= 0 {
            if let Some(entry) = entries
                .iter()
                .find(|entry| entry.depth == depth && entry.row == row)
            {
                return Some(entry.clone());
            }
        }
    }

    if encoded_or_index >= 0 {
        return entries.get(encoded_or_index as usize).cloned();
    }

    None
}

fn rebuild_picker_render_state(state: &mut PresetUiState) {
    if state.picker_path_chain.is_empty() {
        state.picker_path_chain.push(String::new());
    }

    state.picker_path_chain[0].clear();

    let mut valid_chain: Vec<String> = Vec::new();
    valid_chain.push(String::new());

    for path in state.picker_path_chain.iter().skip(1) {
        let normalized = preset::normalize_leaf_id(path);
        if normalized.is_empty() {
            continue;
        }

        let Some(parent_path) = valid_chain.last() else {
            break;
        };
        let parent_entries = build_entries_for_node(&state.preset_tree, parent_path);

        let can_descend = parent_entries.iter().any(|entry| {
            matches!(
                &entry.kind,
                PresetPickerEntryKind::Folder { rel_path } if rel_path == &normalized
            )
        });

        if can_descend {
            valid_chain.push(normalized);
        } else {
            break;
        }
    }

    state.picker_path_chain = valid_chain;

    if state.picker_selected_rows.len() < state.picker_path_chain.len() {
        state
            .picker_selected_rows
            .resize(state.picker_path_chain.len(), -1);
    } else if state.picker_selected_rows.len() > state.picker_path_chain.len() {
        state.picker_selected_rows.truncate(state.picker_path_chain.len());
    }

    state.picker_column_rows.clear();
    state.picker_render_entries.clear();

    for (depth, rel_path) in state.picker_path_chain.iter().enumerate() {
        let entries = build_entries_for_node(&state.preset_tree, rel_path);
        state.picker_column_rows.push(entries.len() as i32);

        for (row, entry) in entries.into_iter().enumerate() {
            let is_folder = matches!(entry.kind, PresetPickerEntryKind::Folder { .. });
            state.picker_render_entries.push(PresetPickerRenderEntry {
                label: entry.label,
                depth: depth as i32,
                row: row as i32,
                is_folder,
                kind: entry.kind,
            });
        }
    }
}

fn reset_picker_navigation_to_root(state: &mut PresetUiState) {
    state.picker_path_chain.clear();
    state.picker_path_chain.push(String::new());
    state.picker_selected_rows.clear();
    state.picker_selected_rows.push(-1);
    state.picker_column_rows.clear();
    state.picker_render_entries.clear();
}

fn apply_picker_state_to_ui(
    app: &crate::editor::slint_preview_ui::AppWindow,
    state: &PresetUiState,
) {
    let labels: Vec<slint::SharedString> = state
        .picker_render_entries
        .iter()
        .map(|entry| slint::SharedString::from(entry.label.clone()))
        .collect();

    let depths: Vec<i32> = state
        .picker_render_entries
        .iter()
        .map(|entry| entry.depth)
        .collect();

    let rows: Vec<i32> = state
        .picker_render_entries
        .iter()
        .map(|entry| entry.row)
        .collect();

    let is_folder: Vec<bool> = state
        .picker_render_entries
        .iter()
        .map(|entry| entry.is_folder)
        .collect();

    let max_rows = state
        .picker_column_rows
        .iter()
        .copied()
        .max()
        .unwrap_or(0);

    app.set_preset_leaf_items(slint::ModelRc::new(slint::VecModel::from(labels)));
    app.set_preset_menu_entry_depths(slint::ModelRc::new(slint::VecModel::from(depths)));
    app.set_preset_menu_entry_rows(slint::ModelRc::new(slint::VecModel::from(rows)));
    app.set_preset_menu_entry_is_folder(slint::ModelRc::new(slint::VecModel::from(is_folder)));
    app.set_preset_menu_column_rows(slint::ModelRc::new(slint::VecModel::from(
        state.picker_column_rows.clone(),
    )));
    app.set_preset_menu_selected_rows(slint::ModelRc::new(slint::VecModel::from(
        state.picker_selected_rows.clone(),
    )));
    app.set_preset_menu_column_count(state.picker_path_chain.len() as i32);
    app.set_preset_menu_max_rows(max_rows.max(0));
    app.set_preset_selected_index(selected_index_in_picker(
        &state.picker_render_entries,
        state.selected_leaf_id.as_deref(),
    ));
}

fn clear_picker_ui(app: &crate::editor::slint_preview_ui::AppWindow) {
    app.set_preset_leaf_items(slint::ModelRc::new(slint::VecModel::from(Vec::<slint::SharedString>::new())));
    app.set_preset_menu_entry_depths(slint::ModelRc::new(slint::VecModel::from(Vec::<i32>::new())));
    app.set_preset_menu_entry_rows(slint::ModelRc::new(slint::VecModel::from(Vec::<i32>::new())));
    app.set_preset_menu_entry_is_folder(slint::ModelRc::new(slint::VecModel::from(Vec::<bool>::new())));
    app.set_preset_menu_column_rows(slint::ModelRc::new(slint::VecModel::from(Vec::<i32>::new())));
    app.set_preset_menu_selected_rows(slint::ModelRc::new(slint::VecModel::from(Vec::<i32>::new())));
    app.set_preset_menu_column_count(1);
    app.set_preset_menu_max_rows(0);
    app.set_preset_selected_index(-1);
}


fn output_gain_to_knob_value(output_gain: f32) -> f32 {
    output_db_to_knob_value(util::gain_to_db(output_gain))
}

fn knob_value_to_output_gain(knob_value: f32) -> f32 {
    util::db_to_gain(knob_value_to_output_db(knob_value))
}

fn current_active_slot(params: &IRMaxPluginParams) -> i32 {
    params.preset_slot_select.value().clamp(0, (SLOT_COUNT as i32) - 1)
}

fn current_ui_active_slot(
    params: &IRMaxPluginParams,
    runtime: &Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
) -> i32 {
    let is_preview_mode = {
        let rt = runtime.read();
        rt.preview_mode
    };
    if is_preview_mode {
        return -1;
    }

    let active_slot = current_active_slot(params);
    let active_slot_index = active_slot as usize;

    let (active_slot_path, last_folder) = {
        let shared = params.shared.read();
        (
            shared.slot_paths[active_slot_index].trim().to_string(),
            shared.last_folder.trim().to_string(),
        )
    };

    if !last_folder.is_empty() && active_slot_path != last_folder {
        -1
    } else {
        active_slot
    }
}

fn current_slot_occupied(params: &IRMaxPluginParams) -> [bool; SLOT_COUNT] {
    let shared = params.shared.read();
    std::array::from_fn(|index| !shared.slot_paths[index].trim().is_empty())
}

fn first_empty_slot(params: &IRMaxPluginParams) -> Option<usize> {
    let shared = params.shared.read();
    shared
        .slot_paths
        .iter()
        .position(|path| path.trim().is_empty())
}

#[derive(Clone)]
struct PresetUiState {
    preset_root: PathBuf,
    preset_tree: PresetTreeNode,
    preset_pack: PresetPack,
    selected_leaf_id: Option<String>,
    preview_loaded_leaf_id: Option<String>,
    picker_path_chain: Vec<String>,
    picker_selected_rows: Vec<i32>,
    picker_column_rows: Vec<i32>,
    picker_render_entries: Vec<PresetPickerRenderEntry>,
}

#[derive(Default, Clone)]
struct SaveFlowState {
    source_slot: Option<usize>,
    source_root: Option<PathBuf>,
    rename_parent_rel: Option<String>,
    rename_original_leaf_id: Option<String>,
}

fn default_save_name_for_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "Preset".to_string())
}

fn sanitize_preset_name(raw: &str) -> String {
    let mut sanitized = String::new();
    for ch in raw.chars() {
        let mapped = if matches!(ch, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') || ch.is_control() {
            '_'
        } else {
            ch
        };
        sanitized.push(mapped);
    }

    let collapsed = sanitized
        .split_whitespace()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    let trimmed = collapsed.trim().trim_matches('.').to_string();
    if trimmed.is_empty() {
        "Preset".to_string()
    } else {
        trimmed
    }
}


fn trigger_ui_toast(
    app: &crate::editor::slint_preview_ui::AppWindow,
    message: impl Into<String>,
    is_error: bool,
) {
    let next_nonce = app.get_toast_nonce().wrapping_add(1);
    app.set_toast_message(message.into().into());
    app.set_toast_is_error(is_error);
    app.set_toast_nonce(next_nonce);
}

fn format_sample_rate_folder_name(sample_rate: u32) -> String {
    let khz = (sample_rate.max(8_000) as f64) / 1000.0;
    let mut label = if (khz - khz.round()).abs() < 0.000_1 {
        format!("{:.0}", khz.round())
    } else {
        format!("{:.1}", khz)
    };

    if label.ends_with(".0") {
        label.truncate(label.len() - 2);
    }

    format!("{}k", label)
}

fn resolve_active_slot_source(
    params: &Arc<IRMaxPluginParams>,
    runtime: &Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
) -> Option<(usize, PathBuf, u32)> {
    let active_slot = current_active_slot(params.as_ref())
        .clamp(0, (SLOT_COUNT as i32) - 1) as usize;
    let source_path = {
        let shared = params.shared.read();
        shared.slot_paths[active_slot].trim().to_string()
    };

    if source_path.is_empty() {
        return None;
    }

    let host_sr = {
        let rt = runtime.read();
        if rt.host_sample_rate == 0 {
            48_000
        } else {
            rt.host_sample_rate
        }
    };

    let source_root = preset::canonicalize_sample_rate_root(Path::new(&source_path));
    Some((active_slot, source_root, host_sr))
}

fn preset_leaf_parent_rel(leaf_id: &str) -> String {
    let normalized = preset::normalize_leaf_id(leaf_id);
    Path::new(&normalized)
        .parent()
        .map(|parent| parent.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default()
}

fn same_directory_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }

    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}


fn count_wav_files(path: &Path) -> Result<usize, String> {
    let entries = fs::read_dir(path)
        .map_err(|err| format!("Failed to read folder '{}': {}", path.to_string_lossy(), err))?;

    let mut count = 0usize;
    for entry in entries {
        let entry = entry.map_err(|err| format!("Failed to iterate folder '{}': {}", path.to_string_lossy(), err))?;
        let candidate = entry.path();
        let is_wav = candidate
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("wav"))
            .unwrap_or(false);
        if candidate.is_file() && is_wav {
            count += 1;
        }
    }

    Ok(count)
}

fn copy_dir_recursive_all(source_dir: &Path, target_dir: &Path) -> Result<(), String> {
    if !source_dir.exists() {
        return Err(format!(
            "Source folder does not exist: {}",
            source_dir.to_string_lossy()
        ));
    }
    if !source_dir.is_dir() {
        return Err(format!(
            "Source path is not a directory: {}",
            source_dir.to_string_lossy()
        ));
    }

    fs::create_dir_all(target_dir).map_err(|err| {
        format!(
            "Failed to create target folder '{}': {}",
            target_dir.to_string_lossy(),
            err
        )
    })?;

    let entries = fs::read_dir(source_dir).map_err(|err| {
        format!(
            "Failed to read source folder '{}': {}",
            source_dir.to_string_lossy(),
            err
        )
    })?;

    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "Failed to iterate source folder '{}': {}",
                source_dir.to_string_lossy(),
                err
            )
        })?;
        let source_path = entry.path();
        let target_path = target_dir.join(entry.file_name());

        if source_path.is_dir() {
            copy_dir_recursive_all(&source_path, &target_path)?;
        } else if source_path.is_file() {
            fs::copy(&source_path, &target_path).map_err(|err| {
                format!(
                    "Failed to copy '{}' -> '{}': {}",
                    source_path.to_string_lossy(),
                    target_path.to_string_lossy(),
                    err
                )
            })?;
        }
    }

    Ok(())
}

fn copy_wav_payload(source_dir: &Path, target_dir: &Path) -> Result<usize, String> {
    if !source_dir.exists() {
        return Err(format!("Source folder does not exist: {}", source_dir.to_string_lossy()));
    }
    if !source_dir.is_dir() {
        return Err(format!("Source path is not a directory: {}", source_dir.to_string_lossy()));
    }

    fs::create_dir_all(target_dir)
        .map_err(|err| format!("Failed to create target folder '{}': {}", target_dir.to_string_lossy(), err))?;

    if same_directory_path(source_dir, target_dir) {
        let existing = count_wav_files(source_dir)?;
        if existing == 0 {
            return Err(format!(
                "No WAV files found in source folder '{}'",
                source_dir.to_string_lossy()
            ));
        }
        return Ok(existing);
    }

    if let Ok(entries) = fs::read_dir(target_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_wav = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false);
            if path.is_file() && is_wav {
                let _ = fs::remove_file(path);
            }
        }
    }

    let mut copied = 0usize;
    let entries = fs::read_dir(source_dir)
        .map_err(|err| format!("Failed to read source folder '{}': {}", source_dir.to_string_lossy(), err))?;

    for entry in entries {
        let entry = entry.map_err(|err| format!("Failed to iterate source folder: {}", err))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let is_wav = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("wav"))
            .unwrap_or(false);
        if !is_wav {
            continue;
        }

        let target_path = target_dir.join(entry.file_name());
        fs::copy(&path, &target_path).map_err(|err| {
            format!(
                "Failed to copy '{}' -> '{}': {}",
                path.to_string_lossy(),
                target_path.to_string_lossy(),
                err
            )
        })?;
        copied += 1;
    }

    if copied == 0 {
        return Err(format!(
            "No WAV files found in source folder '{}'",
            source_dir.to_string_lossy()
        ));
    }

    Ok(copied)
}

fn save_slot_source_into_user_preset(
    source_slot: usize,
    source_root: &Path,
    host_sr: u32,
    requested_name: &str,
    params: &Arc<IRMaxPluginParams>,
    logger: &Arc<InstanceLogger>,
) -> Result<PathBuf, String> {
    let source_variant = preset::resolve_sample_rate_variant_path(source_root, host_sr);
    if !source_variant.exists() {
        return Err(format!(
            "Resolved source folder does not exist: {}",
            source_variant.to_string_lossy()
        ));
    }

    let fallback_name = default_save_name_for_path(source_root);
    let desired_name = if requested_name.trim().is_empty() {
        fallback_name
    } else {
        requested_name.trim().to_string()
    };
    let normalized_name = sanitize_preset_name(&desired_name);

    let preset_root = sync_or_default_preset_root(params.as_ref());
    let target_variant = preset_root
        .join("User")
        .join(&normalized_name)
        .join(format_sample_rate_folder_name(host_sr));

    let copied_count = copy_wav_payload(&source_variant, &target_variant)?;

    logger.info(
        "SlintSave",
        &format!(
            "Saved slot {} into '{}' ({} file(s), source='{}', target='{}')",
            source_slot + 1,
            normalized_name,
            copied_count,
            source_variant.to_string_lossy(),
            target_variant.to_string_lossy()
        ),
    );

    Ok(target_variant)
}

fn save_active_slot_into_user_preset(
    params: &Arc<IRMaxPluginParams>,
    runtime: &Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
    requested_name: &str,
    logger: &Arc<InstanceLogger>,
) -> Result<PathBuf, String> {
    let (active_slot, source_root, host_sr) = resolve_active_slot_source(params, runtime)
        .ok_or_else(|| "Current slot is empty. Load or insert an IR first.".to_string())?;

    save_slot_source_into_user_preset(
        active_slot,
        &source_root,
        host_sr,
        requested_name,
        params,
        logger,
    )
}

fn true_rename_preset_branch(
    source_slot: usize,
    source_root: &Path,
    host_sr: u32,
    requested_name: &str,
    target_parent_rel: &str,
    original_leaf_id: &str,
    params: &Arc<IRMaxPluginParams>,
    logger: &Arc<InstanceLogger>,
) -> Result<(PathBuf, String), String> {
    let preset_root = sync_or_default_preset_root(params.as_ref());
    let source_leaf_root = preset::canonicalize_sample_rate_root(source_root);
    if !source_leaf_root.exists() {
        return Err(format!(
            "Resolved source folder does not exist: {}",
            source_leaf_root.to_string_lossy()
        ));
    }

    let old_leaf_id = preset::normalize_leaf_id(original_leaf_id);
    if old_leaf_id.is_empty() {
        return Err("Original preset leaf id is empty; cannot rename.".to_string());
    }

    let fallback_name = default_save_name_for_path(source_root);
    let desired_name = if requested_name.trim().is_empty() {
        fallback_name
    } else {
        requested_name.trim().to_string()
    };
    let normalized_name = sanitize_preset_name(&desired_name);

    let normalized_parent = preset::normalize_leaf_id(target_parent_rel);
    let target_leaf_id = if normalized_parent.is_empty() {
        normalized_name.clone()
    } else {
        format!("{}/{}", normalized_parent, normalized_name)
    };

    if target_leaf_id == old_leaf_id {
        let target_variant = preset::resolve_sample_rate_variant_path(&source_leaf_root, host_sr);
        return Ok((target_variant, target_leaf_id));
    }

    let target_leaf_root = preset::resolve_leaf_path(&preset_root, &target_leaf_id);
    if target_leaf_root.exists() {
        return Err(format!(
            "Target preset '{}' already exists. Please choose another name.",
            target_leaf_id
        ));
    }

    if let Some(parent) = target_leaf_root.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create target parent '{}': {}",
                parent.to_string_lossy(),
                err
            )
        })?;
    }

    let moved = match fs::rename(&source_leaf_root, &target_leaf_root) {
        Ok(_) => true,
        Err(err) => {
            logger.warn(
                "SlintSave",
                &format!(
                    "Direct rename failed (source='{}', target='{}'): {}. Falling back to copy+delete.",
                    source_leaf_root.to_string_lossy(),
                    target_leaf_root.to_string_lossy(),
                    err
                ),
            );

            copy_dir_recursive_all(&source_leaf_root, &target_leaf_root)?;
            fs::remove_dir_all(&source_leaf_root).map_err(|remove_err| {
                format!(
                    "Rename fallback copied files but failed to remove original folder '{}': {}",
                    source_leaf_root.to_string_lossy(),
                    remove_err
                )
            })?;
            false
        }
    };

    let target_variant = preset::resolve_sample_rate_variant_path(&target_leaf_root, host_sr);
    let wav_count = count_wav_files(&target_variant).unwrap_or(0);

    logger.info(
        "SlintSave",
        &format!(
            "Preset renamed '{}' -> '{}' (mode={}, slot={}, wav_files_in_variant={}, target='{}')",
            old_leaf_id,
            target_leaf_id,
            if moved { "move" } else { "copy_delete" },
            source_slot + 1,
            wav_count,
            target_leaf_root.to_string_lossy()
        ),
    );

    Ok((target_variant, target_leaf_id))
}

fn save_slot_source_into_preset_branch(
    source_slot: usize,
    source_root: &Path,
    host_sr: u32,
    requested_name: &str,
    target_parent_rel: &str,
    params: &Arc<IRMaxPluginParams>,
    logger: &Arc<InstanceLogger>,
) -> Result<(PathBuf, String), String> {
    let source_variant = preset::resolve_sample_rate_variant_path(source_root, host_sr);
    if !source_variant.exists() {
        return Err(format!(
            "Resolved source folder does not exist: {}",
            source_variant.to_string_lossy()
        ));
    }

    let fallback_name = default_save_name_for_path(source_root);
    let desired_name = if requested_name.trim().is_empty() {
        fallback_name
    } else {
        requested_name.trim().to_string()
    };
    let normalized_name = sanitize_preset_name(&desired_name);

    let preset_root = sync_or_default_preset_root(params.as_ref());
    let normalized_parent = preset::normalize_leaf_id(target_parent_rel);
    let target_leaf_id = if normalized_parent.is_empty() {
        normalized_name.clone()
    } else {
        format!("{}/{}", normalized_parent, normalized_name)
    };
    let target_leaf_root = preset::resolve_leaf_path(&preset_root, &target_leaf_id);
    let target_variant = target_leaf_root.join(format_sample_rate_folder_name(host_sr));

    let copied_count = copy_wav_payload(&source_variant, &target_variant)?;

    logger.info(
        "SlintSave",
        &format!(
            "Saved preset branch '{}' (slot={}, {} file(s), source='{}', target='{}')",
            target_leaf_id,
            source_slot + 1,
            copied_count,
            source_variant.to_string_lossy(),
            target_variant.to_string_lossy()
        ),
    );

    Ok((target_variant, target_leaf_id))
}


fn sync_or_default_preset_root(params: &IRMaxPluginParams) -> PathBuf {
    let saved_root = {
        let shared = params.shared.read();
        shared.preset_root.trim().to_string()
    };

    let preset_root = if saved_root.is_empty() {
        preset::default_preset_root()
    } else {
        PathBuf::from(saved_root)
    };

    let _ = preset::ensure_preset_root_exists(&preset_root);

    if {
        let shared = params.shared.read();
        shared.preset_root.trim().is_empty()
    } {
        let mut shared = params.shared.write();
        shared.preset_root = preset_root.to_string_lossy().to_string();
    }

    preset_root
}

fn refresh_preset_sources(state: &mut PresetUiState, logger: &InstanceLogger) {
    match preset::build_preset_tree(&state.preset_root) {
        Ok(tree) => {
            state.preset_tree = tree;
            if let Some(selected) = state.selected_leaf_id.clone() {
                if !state.preset_tree.contains_group(&selected) {
                    state.selected_leaf_id = None;
                    state.preview_loaded_leaf_id = None;
                }
            }
        }
        Err(err) => {
            logger.warn(
                "SlintPreset",
                &format!(
                    "build_preset_tree failed for '{}': {}",
                    state.preset_root.to_string_lossy(),
                    err
                ),
            );
            state.preset_tree = PresetTreeNode::new_root();
            state.selected_leaf_id = None;
            state.preview_loaded_leaf_id = None;
        }
    }

    match preset::load_preset_pack(&state.preset_root) {
        Ok(pack) => {
            state.preset_pack = pack;
        }
        Err(err) => {
            logger.warn(
                "SlintPreset",
                &format!(
                    "load_preset_pack failed for '{}': {}",
                    state.preset_root.to_string_lossy(),
                    err
                ),
            );
            state.preset_pack = PresetPack::default();
        }
    }

    rebuild_picker_render_state(state);
}

fn open_folder_in_file_manager(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(path)
            .spawn()
            .map_err(|err| err.to_string())?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|err| err.to_string())?;
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|err| err.to_string())?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err("Unsupported platform for opening folders".to_string())
}

fn open_file_in_default_app(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        // On Windows, if the file doesn't exist yet (very common at startup),
        // try to open the parent folder instead of failing.
        let target = if path.exists() {
            path.to_path_buf()
        } else if let Some(parent) = path.parent() {
            if parent.exists() {
                parent.to_path_buf()
            } else {
                path.to_path_buf()
            }
        } else {
            path.to_path_buf()
        };

        // Use explorer to open the file/folder. This is generally more reliable than 'start'.
        std::process::Command::new("explorer")
            .arg(&target)
            .spawn()
            .map_err(|err| err.to_string())?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|err| err.to_string())?;
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|err| err.to_string())?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err("Unsupported platform for opening files".to_string())
}


fn set_slot_occupied_flag(app: &crate::editor::slint_preview_ui::AppWindow, slot: usize, occupied: bool) {
    match slot {
        0 => app.set_slot1_occupied(occupied),
        1 => app.set_slot2_occupied(occupied),
        2 => app.set_slot3_occupied(occupied),
        3 => app.set_slot4_occupied(occupied),
        4 => app.set_slot5_occupied(occupied),
        _ => {}
    }
}

fn refresh_backend_ui(app: &crate::editor::slint_preview_ui::AppWindow, runtime: &crate::RuntimePluginState) {
    let effective_backend = runtime.gpu_backend_pending.unwrap_or(runtime.gpu_backend);
    let backends = if runtime.gpu_backends.is_empty() {
        available_backends()
    } else {
        runtime.gpu_backends.clone()
    };
    let can_switch = if runtime.gpu_backends.is_empty() {
        cfg!(target_os = "windows") && backends.len() >= 2
    } else {
        runtime.gpu_can_switch
    };
    let has_switch_target = next_backend(effective_backend, &backends).is_some();
    let enabled = can_switch && runtime.gpu_backend_pending.is_none() && has_switch_target;

    app.set_backend_label(effective_backend.label().into());
    app.set_backend_switch_enabled(enabled);
    app.set_backend_switch_pending(runtime.gpu_backend_pending.is_some());
}

#[allow(clippy::too_many_arguments)]
fn load_selected_leaf(
    leaf_id: &str,
    insert_to_slot: Option<usize>,
    preset_root: &Path,
    preset_pack: &PresetPack,
    params: &Arc<IRMaxPluginParams>,
    loader: &Loader,
    runtime: &Arc<parking_lot::RwLock<crate::RuntimePluginState>>,
    gui_context: &Arc<dyn GuiContext>,
    logger: &Arc<InstanceLogger>,
    app_weak: &slint::Weak<crate::editor::slint_preview_ui::AppWindow>,
) -> bool {
    let requested_leaf_path = preset::canonicalize_sample_rate_root(&preset::resolve_leaf_path(preset_root, leaf_id));
    let resolved_leaf_path = {
        let host_sr = {
            let rt = runtime.read();
            if rt.host_sample_rate == 0 {
                48000
            } else {
                rt.host_sample_rate
            }
        };
        preset::resolve_sample_rate_variant_path(&requested_leaf_path, host_sr)
    };

    if !resolved_leaf_path.exists() {
        logger.warn(
            "SlintPreset",
            &format!(
                "Selected leaf path does not exist: {}",
                resolved_leaf_path.to_string_lossy()
            ),
        );

        if requested_leaf_path != resolved_leaf_path {
            logger.warn(
                "SlintPreset",
                &format!(
                    "Resolved base leaf path was: {}",
                    requested_leaf_path.to_string_lossy()
                ),
            );
        }
        return false;
    }

    let (safe_sr, safe_outputs, safe_max_buffer, backend) = {
        let rt = runtime.read();
        let safe_sr = if rt.host_sample_rate == 0 {
            48000
        } else {
            rt.host_sample_rate
        };
        let safe_outputs = if rt.host_outputs == 0 {
            2
        } else {
            rt.host_outputs
        };
        let safe_max_buffer = if rt.host_max_buffer == 0 {
            2048
        } else {
            rt.host_max_buffer
        };
        let backend = rt.gpu_backend_pending.unwrap_or(rt.gpu_backend);
        (safe_sr, safe_outputs, safe_max_buffer, backend)
    };

    let setter = ParamSetter::new(gui_context.as_ref());

    if let Some(group) = preset_pack.groups.get(leaf_id) {
        let mix = group.mix.clamp(0.0, 1.0);
        setter.begin_set_parameter(&params.mix);
        setter.set_parameter(&params.mix, mix);
        setter.end_set_parameter(&params.mix);

        let gain = util::db_to_gain(group.output_gain_db);
        setter.begin_set_parameter(&params.output_gain);
        setter.set_parameter(&params.output_gain, gain);
        setter.end_set_parameter(&params.output_gain);
    }

    let _ = loader.tx.send(LoaderCommand::LoadFolder(
        requested_leaf_path.clone(),
        safe_sr,
        safe_outputs,
        safe_max_buffer,
        backend,
        None,
        crate::clamp_guard_profile(params.guard_profile.value()),
    ));

    let path_str = requested_leaf_path.to_string_lossy().to_string();

    {
        let mut rt = runtime.write();
        rt.preview_mode = insert_to_slot.is_none();
    }

    {
        let mut shared = params.shared.write();
        shared.last_folder = path_str.clone();
        if let Some(slot) = insert_to_slot {
            shared.slot_paths[slot] = path_str.clone();
        }
    }

    let current_dirty = params.dirty_trigger.value();
    setter.begin_set_parameter(&params.dirty_trigger);
    setter.set_parameter(&params.dirty_trigger, !current_dirty);
    setter.end_set_parameter(&params.dirty_trigger);

    if let Some(slot) = insert_to_slot {
        if let Some(app) = app_weak.upgrade() {
            set_slot_occupied_flag(&app, slot, true);
        }

        logger.info(
            "SlintPreset",
            &format!("Inserted leaf '{}' into slot {}", leaf_id, slot + 1),
        );
    } else {
        logger.info(
            "SlintPreset",
            &format!("Loaded leaf '{}' as active preset preview", leaf_id),
        );
    }

    true
}


