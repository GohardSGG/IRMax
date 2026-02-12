struct SpectrumRenderRequest {
    generation: u64,
    info: crate::loader::LoadedInfo,
}

#[derive(Debug)]
struct SpectrumRenderResult {
    generation: u64,
    info: crate::loader::LoadedInfo,
    panel: Option<SpectrumPanelPixels>,
}

fn spawn_spectrum_render_worker(
    request_rx: mpsc::Receiver<SpectrumRenderRequest>,
    result_tx: mpsc::Sender<SpectrumRenderResult>,
) {
    std::thread::spawn(move || {
        while let Ok(mut request) = request_rx.recv() {
            while let Ok(newer_request) = request_rx.try_recv() {
                request = newer_request;
            }

            let panel = build_spectrum_panel_pixels(
                &request.info.files,
                request.info.max_channels_per_file,
            );

            if result_tx
                .send(SpectrumRenderResult {
                    generation: request.generation,
                    info: request.info,
                    panel,
                })
                .is_err()
            {
                break;
            }
        }
    });
}

fn build_spectrum_panel_pixels(
    files: &[(PathBuf, u16, Vec<Vec<f32>>)],
    max_channels_per_file: u16,
) -> Option<SpectrumPanelPixels> {
    if files.is_empty() {
        return None;
    }

    let rows = files.len();
    let cols = max_channels_per_file.max(1) as usize;
    let (panel_w_u32, panel_h_u32) = choose_spectrum_panel_size(rows, cols);
    let panel_w = panel_w_u32 as usize;
    let panel_h = panel_h_u32 as usize;

    let gray20 = Rgba8Pixel {
        r: 20,
        g: 20,
        b: 20,
        a: 255,
    };
    let transparent = Rgba8Pixel {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };

    let mut pixels = vec![transparent; panel_w.saturating_mul(panel_h)];

    for (row_index, (_, _, channels_data)) in files.iter().enumerate() {
        let (row_y0, row_y1) = partition_span(row_index, rows, panel_h);

        for col_index in 0..cols {
            let Some(channel_data) = channels_data.get(col_index) else {
                continue;
            };

            if channel_data.len() < SPECTRUM_WIDTH * SPECTRUM_HEIGHT {
                continue;
            }

            let (col_x0, col_x1) = partition_span(col_index, cols, panel_w);
            if row_y1 <= row_y0 || col_x1 <= col_x0 {
                continue;
            }

            for y in row_y0..row_y1 {
                for x in col_x0..col_x1 {
                    pixels[y * panel_w + x] = gray20;
                }
            }

            let cell_w = col_x1 - col_x0;
            let cell_h = row_y1 - row_y0;
            if cell_w == 0 || cell_h == 0 {
                continue;
            }

            let den_x = (cell_w.saturating_sub(1)).max(1) as f32;
            let den_y = (cell_h.saturating_sub(1)).max(1) as f32;

            for y in row_y0..row_y1 {
                let local_y = y - row_y0;
                let norm_y = local_y as f32 / den_y;
                for x in col_x0..col_x1 {
                    let local_x = x - col_x0;
                    let norm_x = local_x as f32 / den_x;
                    let val = sample_spectrum_bilinear(channel_data, norm_x, norm_y);
                    pixels[y * panel_w + x] = get_heatmap_color_rgba(val);
                }
            }
        }
    }

    Some((panel_w_u32, panel_h_u32, pixels))
}

fn build_spectrum_panel_image(
    files: &[(PathBuf, u16, Vec<Vec<f32>>)],
    max_channels_per_file: u16,
) -> Option<Image> {
    build_spectrum_panel_pixels(files, max_channels_per_file).map(spectrum_panel_pixels_to_image)
}


