fn format_log_path_for_ui(full_path: &str) -> String {
    let path = Path::new(full_path);
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(full_path);
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("");

    if parent.is_empty() {
        format!("LOG: {file}")
    } else {
        format!("LOG: {parent}/{file}")
    }
}

fn next_backend(current: GpuBackend, available: &[GpuBackend]) -> Option<GpuBackend> {
    if available.len() < 2 {
        return None;
    }

    let idx = available.iter().position(|backend| *backend == current).unwrap_or(0);
    let next = available[(idx + 1) % available.len()];
    if next == current {
        None
    } else {
        Some(next)
    }
}

fn output_db_to_knob_value(output_db: f32) -> f32 {
    let db = output_db.clamp(OUTPUT_MIN_DB, OUTPUT_MAX_DB);

    if db <= 0.0 {
        ((db - OUTPUT_MIN_DB) / (0.0 - OUTPUT_MIN_DB)) * OUTPUT_ZERO_DB_KNOB_POS
    } else {
        OUTPUT_ZERO_DB_KNOB_POS + (db / OUTPUT_MAX_DB) * (1.0 - OUTPUT_ZERO_DB_KNOB_POS)
    }
}

fn knob_value_to_output_db(knob_value: f32) -> f32 {
    let value = knob_value.clamp(0.0, 1.0);

    if value <= OUTPUT_ZERO_DB_KNOB_POS {
        OUTPUT_MIN_DB + (value / OUTPUT_ZERO_DB_KNOB_POS) * (0.0 - OUTPUT_MIN_DB)
    } else {
        ((value - OUTPUT_ZERO_DB_KNOB_POS) / (1.0 - OUTPUT_ZERO_DB_KNOB_POS)) * OUTPUT_MAX_DB
    }
}

fn format_output_text(output_db: f32) -> String {
    if output_db <= OUTPUT_NEG_INF_THRESHOLD_DB {
        "-inf".to_string()
    } else {
        let rounded_db = (output_db * 10.0).round() / 10.0;
        let display_db = if rounded_db.abs() < 0.05 {
            0.0
        } else {
            rounded_db
        };

        if display_db > 0.0 {
            format!("+{display_db:.1}")
        } else {
            format!("{display_db:.1}")
        }
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let t = t.clamp(0.0, 1.0);
    (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
}

fn get_heatmap_color_rgba(val: f32) -> Rgba8Pixel {
    let clamped = val.clamp(0.0, 1.0);

    let (r1, g1, b1, r2, g2, b2, t) = if clamped < 0.02 {
        (25, 25, 25, 25, 25, 25, 0.0)
    } else if clamped < 0.30 {
        let t = (clamped - 0.02) / (0.30 - 0.02);
        (0, 0, 100, 0, 200, 255, t)
    } else if clamped < 0.55 {
        let t = (clamped - 0.30) / (0.55 - 0.30);
        (0, 200, 255, 0, 255, 0, t)
    } else if clamped < 0.80 {
        let t = (clamped - 0.55) / (0.80 - 0.55);
        (0, 255, 0, 255, 255, 0, t)
    } else {
        let t = (clamped - 0.80) / (1.0 - 0.80);
        (255, 255, 0, 255, 0, 0, t)
    };

    Rgba8Pixel {
        r: lerp_u8(r1, r2, t),
        g: lerp_u8(g1, g2, t),
        b: lerp_u8(b1, b2, t),
        a: 255,
    }
}

fn partition_span(index: usize, total_parts: usize, total_size: usize) -> (usize, usize) {
    if total_parts == 0 || total_size == 0 {
        return (0, 0);
    }

    let start = index.saturating_mul(total_size) / total_parts;
    let mut end = (index.saturating_add(1)).saturating_mul(total_size) / total_parts;
    if end <= start {
        end = (start + 1).min(total_size);
    }

    (start.min(total_size), end.min(total_size))
}

fn choose_spectrum_panel_size(rows: usize, cols: usize) -> (u32, u32) {
    let safe_rows = rows.max(1);
    let safe_cols = cols.max(1);

    let desired_w = (safe_cols * SPECTRUM_WIDTH * 2) as f32;
    let desired_h = (safe_rows * SPECTRUM_HEIGHT * 2) as f32;

    let min_scale = (SPECTRUM_PANEL_MIN_WIDTH as f32 / desired_w)
        .max(SPECTRUM_PANEL_MIN_HEIGHT as f32 / desired_h)
        .max(1.0);
    let max_scale = (SPECTRUM_PANEL_MAX_WIDTH as f32 / desired_w)
        .min(SPECTRUM_PANEL_MAX_HEIGHT as f32 / desired_h)
        .max(0.01);

    let scale = min_scale.min(max_scale);

    let panel_w = (desired_w * scale)
        .round()
        .clamp(safe_cols as f32, SPECTRUM_PANEL_MAX_WIDTH as f32) as usize;
    let panel_h = (desired_h * scale)
        .round()
        .clamp(safe_rows as f32, SPECTRUM_PANEL_MAX_HEIGHT as f32) as usize;

    (panel_w as u32, panel_h as u32)
}

#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn sample_spectrum_bilinear(channel_data: &[f32], x_norm: f32, y_norm: f32) -> f32 {
    let x = x_norm.clamp(0.0, 1.0) * (SPECTRUM_WIDTH as f32 - 1.0);
    let y = y_norm.clamp(0.0, 1.0) * (SPECTRUM_HEIGHT as f32 - 1.0);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(SPECTRUM_WIDTH - 1);
    let y1 = (y0 + 1).min(SPECTRUM_HEIGHT - 1);

    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let i00 = y0 * SPECTRUM_WIDTH + x0;
    let i10 = y0 * SPECTRUM_WIDTH + x1;
    let i01 = y1 * SPECTRUM_WIDTH + x0;
    let i11 = y1 * SPECTRUM_WIDTH + x1;

    let v00 = *channel_data.get(i00).unwrap_or(&0.0);
    let v10 = *channel_data.get(i10).unwrap_or(&v00);
    let v01 = *channel_data.get(i01).unwrap_or(&v00);
    let v11 = *channel_data.get(i11).unwrap_or(&v00);

    let top = lerp_f32(v00, v10, tx);
    let bottom = lerp_f32(v01, v11, tx);
    lerp_f32(top, bottom, ty)
}

fn format_ir_meta_text(sample_rate: u32, bit_depth: u16, duration_seconds: f32, file_count: usize) -> String {
    format!(
        "LEN: {:.2}s | RATE: {}Hz | BIT: {}bit | FILES: {}",
        duration_seconds.max(0.0),
        sample_rate,
        bit_depth,
        file_count
    )
}

fn runtime_channel_text(runtime: &crate::RuntimePluginState) -> String {
    if let Some(info) = runtime.loaded_info.as_ref() {
        let rows = info.files.len().max(1);
        let cols = info.max_channels_per_file.max(1) as usize;
        format!("CH {}x{}", rows, cols)
    } else {
        "CH --x--".to_string()
    }
}

fn runtime_folder_path_text(params: &IRMaxPluginParams, runtime: &crate::RuntimePluginState) -> String {
    if let Some(info) = runtime.loaded_info.as_ref() {
        if runtime.preview_mode {
            return info.folder_name.clone();
        }
        return info.full_path.to_string_lossy().to_string();
    }

    let shared = params.shared.read();
    let last_folder = shared.last_folder.trim();
    if last_folder.is_empty() {
        "No Folder Loaded".to_string()
    } else {
        last_folder.to_string()
    }
}

fn apply_loaded_info_to_spectrum_ui(
    app: &crate::editor::slint_preview_ui::AppWindow,
    info: &crate::loader::LoadedInfo,
    prebuilt_panel: Option<SpectrumPanelPixels>,
    preview_mode: bool,
) {
    let image = prebuilt_panel
        .map(spectrum_panel_pixels_to_image)
        .or_else(|| build_spectrum_panel_image(&info.files, info.max_channels_per_file))
        .unwrap_or_else(empty_spectrum_image);
    app.set_spectrum_image(image);

    app.set_spectrum_loading(false);
    app.set_spectrum_has_data(!info.files.is_empty());

    let rows = info.files.len().max(1);
    let cols = info.max_channels_per_file.max(1) as i32;
    app.set_spectrum_rows(rows as i32);
    app.set_spectrum_cols(cols);

    let mut cell_values: Vec<f32> = Vec::new();
    for r in 0..info.files.len() {
        for c in 0..(cols as usize) {
            let value = info
                .files
                .get(r)
                .and_then(|(_, _, channels)| channels.get(c))
                .and_then(|channel| {
                    if channel.is_empty() {
                        None
                    } else {
                        let center_idx = (SPECTRUM_HEIGHT / 2) * SPECTRUM_WIDTH + (SPECTRUM_WIDTH / 2);
                        channel.get(center_idx).copied().or_else(|| channel.first().copied())
                    }
                })
                .unwrap_or(0.0);
            cell_values.push(value);
        }
    }
    app.set_spectrum_cell_values(slint::ModelRc::new(slint::VecModel::from(cell_values)));

    let mut row_labels: Vec<slint::SharedString> = Vec::new();
    for (index, (path, _, _)) in info.files.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown");
        row_labels.push(format!("[{}] {}", index + 1, name).into());
    }
    app.set_spectrum_row_labels(slint::ModelRc::new(slint::VecModel::from(row_labels)));

    app.set_spectrum_status_text(
        format!(
            "{} | {} files | {} Hz | {} bit",
            info.folder_name, info.file_count, info.sample_rate, info.bit_depth
        )
        .into(),
    );

    let rows = info.files.len().max(1);
    let cols = info.max_channels_per_file.max(1) as usize;
    app.set_channel_text(format!("CH {}x{}", rows, cols).into());
    let folder_text = if preview_mode {
        info.folder_name.clone()
    } else {
        info.full_path.to_string_lossy().to_string()
    };
    app.set_folder_path_text(folder_text.into());

    app.set_ir_meta_text(
        format_ir_meta_text(
            info.sample_rate,
            info.bit_depth,
            info.duration_seconds,
            info.file_count,
        )
        .into(),
    );
}

fn empty_spectrum_image() -> Image {
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(1, 1);
    {
        let pixels = buffer.make_mut_slice();
        pixels[0] = Rgba8Pixel {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
    }
    Image::from_rgba8(buffer)
}

type SpectrumPanelPixels = (u32, u32, Vec<Rgba8Pixel>);

fn spectrum_panel_pixels_to_image((width, height, pixels): SpectrumPanelPixels) -> Image {
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    {
        let target = buffer.make_mut_slice();
        if target.len() == pixels.len() {
            target.copy_from_slice(&pixels);
        }
    }
    Image::from_rgba8(buffer)
}


