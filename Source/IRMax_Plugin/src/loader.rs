use crossbeam_channel::{unbounded, Receiver, Sender};
use hound::WavReader;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use rustfft::{num_complex::Complex, FftPlanner};
use std::path::PathBuf;

use std::thread;
use walkdir::WalkDir;

pub const SPECTRUM_WIDTH: usize = 256; // Time slices (X-Axis)
pub const SPECTRUM_HEIGHT: usize = 128; // Frequency bins (Y-Axis)
pub const EXECUTION_QUANTUM_MAX_SAMPLES: usize = 384;
pub const OVERSIZED_FALLBACK_SLICE_SAMPLES: usize = 128;

#[inline]
pub fn planning_buffer_from_host_max(host_max_buffer: usize) -> usize {
    host_max_buffer.max(1).min(EXECUTION_QUANTUM_MAX_SAMPLES)
}

#[derive(Debug, Clone)]
pub struct LoadedInfo {
    pub folder_name: String,
    pub full_path: PathBuf,
    pub total_channels: u32,
    pub file_count: usize,
    /// (Path, Channels, Spectrogram[Channel][Flattened Grid])
    /// Flattened Grid = SPECTRUM_WIDTH * SPECTRUM_HEIGHT floats (0.0 - 1.0)
    pub files: Vec<(PathBuf, u16, Vec<Vec<f32>>)>,
    pub max_channels_per_file: u16,
    // Metadata for UI
    pub sample_rate: u32,
    pub bit_depth: u16,
    pub duration_seconds: f32,
}

#[derive(Debug)]
pub enum LoaderCommand {
    LoadFolder(PathBuf, u32, usize, usize, GpuBackend, Option<usize>, i32), // (path, target_sample_rate, host_outputs, host_max_buffer, backend, slot_link_target, guard_profile)
}

#[derive(Debug)]
pub enum LoaderResponse {
    Loading(String), // e.g. "Scanning..."
    Loaded(LoadedInfo, Option<usize>),
    Error(String),
    BackendReady(GpuBackend),
}

use irmax_core::{GpuBackend, IRMaxProcessor, MatrixWorker};

#[derive(Clone)]
pub struct Loader {
    pub tx: Sender<LoaderCommand>,
    pub rx: Receiver<LoaderResponse>,
    pub processor_tx: Sender<IRMaxProcessor>,
    pub worker_tx: Sender<MatrixWorker>,
}

#[inline]
fn drain_to_latest_command(
    cmd_rx: &Receiver<LoaderCommand>,
    mut current: LoaderCommand,
) -> (LoaderCommand, usize) {
    let mut dropped = 0usize;
    while let Ok(next) = cmd_rx.try_recv() {
        current = next;
        dropped += 1;
    }
    (current, dropped)
}

#[inline]
fn drain_pending_command(cmd_rx: &Receiver<LoaderCommand>) -> (Option<LoaderCommand>, usize) {
    let mut latest = None;
    let mut dropped = 0usize;
    while let Ok(next) = cmd_rx.try_recv() {
        latest = Some(next);
        dropped += 1;
    }
    (latest, dropped)
}

fn resolve_scan_paths(requested_path: &PathBuf, target_sr: u32) -> (PathBuf, PathBuf) {
    let canonical_request = crate::preset::canonicalize_sample_rate_root(requested_path);
    let resolved_scan_path = crate::preset::resolve_sample_rate_variant_path(&canonical_request, target_sr);
    (canonical_request, resolved_scan_path)
}

pub const BACKEND_FAILURE_PREFIX: &str = "[BackendFailure]";

#[inline]
fn backend_attempt_chain(requested: GpuBackend) -> Vec<GpuBackend> {
    let mut chain = vec![requested];

    #[cfg(target_os = "windows")]
    if requested == GpuBackend::Cuda {
        chain.push(GpuBackend::Vulkan);
    }

    chain
}

impl Loader {
    pub fn new(
        processor_tx: Sender<IRMaxProcessor>,
        worker_tx: Sender<MatrixWorker>,
        logger: Option<std::sync::Arc<crate::logger::InstanceLogger>>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = unbounded::<LoaderCommand>();
        let (resp_tx, resp_rx) = unbounded::<LoaderResponse>();

        let processor_tx_clone = processor_tx.clone();
        let worker_tx_clone = worker_tx.clone();

        thread::spawn(move || {
            let mut pending_cmd: Option<LoaderCommand> = None;

            loop {
                let initial_cmd = if let Some(cmd) = pending_cmd.take() {
                    cmd
                } else {
                    match cmd_rx.recv() {
                        Ok(cmd) => cmd,
                        Err(_) => break, // Channel disconnected
                    }
                };

                let (cmd, dropped_before_work) = drain_to_latest_command(&cmd_rx, initial_cmd);
                if dropped_before_work > 0 {
                    if let Some(l) = &logger {
                        l.warn(
                            "Loader",
                            &format!(
                                "Coalesced {} queued load requests before work (keeping latest).",
                                dropped_before_work
                            ),
                        );
                    }
                }

                match cmd {
                    LoaderCommand::LoadFolder(
                        path,
                        target_sr,
                        host_outputs,
                        host_max_buffer,
                        backend,
                        slot_link_target,
                        guard_profile,
                    ) => {
                        let planning_buffer = planning_buffer_from_host_max(host_max_buffer);
                        let effective_target_sr = if target_sr == 0 { 48000 } else { target_sr };

                        if target_sr == 0 {
                            if let Some(l) = &logger {
                                l.warn(
                                    "Loader",
                                    "Received target sample rate 0Hz. Falling back to 48000Hz for load.",
                                );
                            }
                        }

                        if let Some(l) = &logger {
                            l.info(
                                "Loader",
                                &format!(
                                    "Received LoadFolder: {:?} @ {}Hz, outputs={}, max_buffer={}, planning_buffer={}, requested_backend={}, guard_profile={}",
                                    path,
                                    effective_target_sr,
                                    host_outputs,
                                    host_max_buffer,
                                    planning_buffer,
                                    backend.label(),
                                    guard_profile
                                ),
                            );
                        }

                        let (canonical_request_path, scan_path) =
                            resolve_scan_paths(&path, effective_target_sr);

                        if let Some(l) = &logger {
                            if scan_path != path {
                                l.info(
                                    "Loader",
                                    &format!(
                                        "Resolved load path {:?} -> {:?} for target_sr={}Hz",
                                        path, scan_path, effective_target_sr
                                    ),
                                );
                            }
                        }

                        let _ = resp_tx.send(LoaderResponse::Loading(
                            scan_path.to_string_lossy().to_string(),
                        ));

                        let scan_result = scan_folder(&scan_path, effective_target_sr);

                        // If a newer command arrived while scanning, skip stale result entirely.
                        let (superseding_cmd, dropped_after_scan) = drain_pending_command(&cmd_rx);
                        if let Some(next_cmd) = superseding_cmd {
                            if let Some(l) = &logger {
                                l.warn(
                                    "Loader",
                                    &format!(
                                        "Skipped stale scan result for {:?}; {} newer request(s) queued.",
                                        path, dropped_after_scan
                                    ),
                                );
                            }
                            pending_cmd = Some(next_cmd);
                            continue;
                        }

                        match scan_result {
                            Ok((mut info, audio_buffers)) => {
                                info.full_path = canonical_request_path.clone();
                                info.folder_name = canonical_request_path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("Unknown")
                                    .to_string();
                                if let Some(l) = &logger {
                                    l.info(
                                        "Loader",
                                        &format!(
                                            "Scan Success! Found {} files. Sending Response.",
                                            info.files.len()
                                        ),
                                    );
                                }
                                if let Some(l) = &logger {
                                    let file_count = audio_buffers.len();
                                    let max_file_outputs =
                                        audio_buffers.iter().map(|f| f.len()).max().unwrap_or(0);
                                    l.info(
                                        "Topology",
                                        &format!(
                                            "files={} max_file_outputs={} host_outputs={} host_max_buffer={} planning_buffer={}",
                                            file_count,
                                            max_file_outputs,
                                            host_outputs,
                                            host_max_buffer,
                                            planning_buffer
                                        ),
                                    );
                                }

                                // 1. Send UI Update
                                let _ = resp_tx
                                    .send(LoaderResponse::Loaded(info.clone(), slot_link_target));

                                // 2. Build Engine
                                use irmax_core::loader::ResourceLoader;

                                let attempt_chain = backend_attempt_chain(backend);
                                let show_backend_dialog = attempt_chain.len() > 1;
                                let mut build_error_log: Vec<String> = Vec::new();
                                let mut built_engine = None;

                                for candidate_backend in attempt_chain {
                                    match ResourceLoader::build_engine_from_buffers_with_probe_and_backend(
                                        &audio_buffers,
                                        info.sample_rate,
                                        host_outputs,
                                        host_max_buffer,
                                        None,
                                        candidate_backend,
                                        guard_profile,
                                    ) {
                                        Ok((processor, worker, resolved_backend)) => {
                                            if candidate_backend != backend {
                                                if let Some(l) = &logger {
                                                    l.warn(
                                                        "Loader",
                                                        &format!(
                                                            "Backend fallback succeeded: requested={} -> active={}",
                                                            backend.label(),
                                                            resolved_backend.label()
                                                        ),
                                                    );
                                                }
                                            }
                                            built_engine = Some((processor, worker, resolved_backend));
                                            break;
                                        }
                                        Err(e) => {
                                            let err_text = e.to_string();
                                            build_error_log.push(format!(
                                                "{}: {}",
                                                candidate_backend.label(),
                                                err_text
                                            ));
                                            if let Some(l) = &logger {
                                                l.warn(
                                                    "Loader",
                                                    &format!(
                                                        "Engine build attempt failed for backend {}: {}",
                                                        candidate_backend.label(),
                                                        err_text
                                                    ),
                                                );
                                            }
                                        }
                                    }
                                }

                                if let Some((processor, worker, resolved_backend)) = built_engine {
                                    // If a newer command arrived during engine build, discard stale engine.
                                    let (superseding_cmd, dropped_after_build) =
                                        drain_pending_command(&cmd_rx);
                                    if let Some(next_cmd) = superseding_cmd {
                                        if let Some(l) = &logger {
                                            l.warn(
                                                "Loader",
                                                &format!(
                                                    "Discarded stale built engine for {:?}; {} newer request(s) queued.",
                                                    path, dropped_after_build
                                                ),
                                            );
                                        }
                                        pending_cmd = Some(next_cmd);
                                        continue;
                                    }

                                    // 3. Send to Audio Thread
                                    let _ = processor_tx_clone.send(processor);
                                    // 4. Send to Worker Thread
                                    let _ = worker_tx_clone.send(worker);
                                    let _ = resp_tx.send(LoaderResponse::BackendReady(resolved_backend));
                                    if let Some(l) = &logger {
                                        l.info(
                                            "Loader",
                                            &format!(
                                                "Engine Built & Sent to Audio/Worker threads. backend={} (requested={})",
                                                resolved_backend.label(),
                                                backend.label()
                                            ),
                                        );
                                    }
                                } else {
                                    let details = if build_error_log.is_empty() {
                                        "unknown build failure".to_string()
                                    } else {
                                        build_error_log.join(" | ")
                                    };

                                    let error_text = if show_backend_dialog {
                                        format!(
                                            "{} CUDA failed and Vulkan fallback also failed. requested={} attempts={}",
                                            BACKEND_FAILURE_PREFIX,
                                            backend.label(),
                                            details
                                        )
                                    } else {
                                        format!("Engine Build Failed: {}", details)
                                    };

                                    if let Some(l) = &logger {
                                        l.error("Loader", &error_text);
                                    }
                                    let _ = resp_tx.send(LoaderResponse::Error(error_text));
                                }
                            }
                            Err(e) => {
                                if let Some(l) = &logger {
                                    l.error("Loader", &format!("Scan Failed: {}", e));
                                }
                                let _ = resp_tx.send(LoaderResponse::Error(e.to_string()));
                            }
                        }
                    }
                }
            }
        });

        Self {
            tx: cmd_tx,
            rx: resp_rx,
            processor_tx,
            worker_tx,
        }
    }
}

fn scan_folder(path: &PathBuf, target_sr: u32) -> anyhow::Result<(LoadedInfo, Vec<Vec<Vec<f32>>>)> {
    let folder_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Unknown")
        .to_string();

    // 1. First Pass: Collect and Validate Candidates (Non-Recursive)
    let candidates: Vec<_> = WalkDir::new(path)
        .max_depth(1) // Requested: Only scan current directory
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().is_file()
                && e.path()
                    .extension()
                    .map_or(false, |ext| ext.to_string_lossy().to_lowercase() == "wav")
        })
        .collect();

    let candidate_count = candidates.len();

    // Requested Limit: Max 64 files
    if candidate_count > 64 {
        return Err(anyhow::anyhow!(
            "Too many files! Found {} WAVs. Limit is 64.\nPlease select a folder with fewer files.",
            candidate_count
        ));
    }

    if candidate_count == 0 {
        return Err(anyhow::anyhow!("No valid .wav files found in this folder."));
    }

    // Sort Candidates
    let mut sorted_candidates = candidates;
    sorted_candidates
        .sort_by(|a, b| natural_cmp(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    let mut total_channels = 0;
    let mut file_count = 0;
    let mut files = Vec::new();
    let mut max_channels_per_file = 0;
    let mut all_audio_buffers = Vec::new(); // Store [File][Channel][Samples]

    // Removed sample rate detection logic, we force target_sr
    // let mut sample_rate = 0;
    let sample_rate = target_sr;
    let mut bit_depth = 0;
    let mut duration_seconds = 0.0;

    for entry in sorted_candidates {
        let p = entry.path();
        if let Ok(mut reader) = WavReader::open(p) {
            let spec = reader.spec();
            let channels = spec.channels;

            if channels > 0 {
                // Requested Limit: Max 64 channels per file? (User said "64 files of 64 channels")
                // Let's enforce 64 channel limit per file too for safety.
                if channels > 64 {
                    // We can either skip or error. Let's warn/skip or error?
                    // User said "allowed max quantity is 64 [files]... limit checks... error".
                    return Err(anyhow::anyhow!(
                        "File {:?} has {} channels. Max supported is 64.",
                        p.file_name().unwrap_or_default(),
                        channels
                    ));
                }

                if file_count == 0 {
                    // sample_rate = spec.sample_rate; // No longer set from file
                    bit_depth = spec.bits_per_sample;
                    // Duration needs to be calculated AFTER resampling potentially
                    // But for initial estimate we can use source duration
                    let frames = reader.duration();
                    duration_seconds = frames as f32 / spec.sample_rate as f32;
                } else {
                    // Consistency check?
                    // If we are resampling, we don't strictly require source files to match EACH OTHER's sample rate,
                    // as long as we resample everything to target.
                    // But for safety/sanity let's warn or strict specific files?
                    // Let's relax this: allow mixed sample rates, we resample all to target.
                }

                // Read ALL Samples (Interleaved)
                let len = reader.duration() as usize;
                // Pre-allocate conservative size
                let mut all_samples = Vec::with_capacity(len * channels as usize);
                let norm_factor = match spec.bits_per_sample {
                    16 => 32768.0,
                    24 => 8388608.0,
                    32 if spec.sample_format == hound::SampleFormat::Int => 2147483648.0,
                    _ => 1.0,
                };

                match spec.sample_format {
                    hound::SampleFormat::Float => {
                        for s in reader.samples::<f32>() {
                            all_samples.push(s.unwrap_or(0.0));
                        }
                    }
                    hound::SampleFormat::Int => {
                        if spec.bits_per_sample == 16 {
                            for s in reader.samples::<i16>() {
                                all_samples.push(s.unwrap_or(0) as f32 / norm_factor);
                            }
                        } else {
                            for s in reader.samples::<i32>() {
                                all_samples.push(s.unwrap_or(0) as f32 / norm_factor);
                            }
                        }
                    }
                }

                // De-interleave
                let mut channel_data = vec![Vec::with_capacity(len); channels as usize];
                for (i, &sample) in all_samples.iter().enumerate() {
                    channel_data[i % channels as usize].push(sample);
                }

                let file_sr = spec.sample_rate;

                // ==================== 🔥 Resampling 🔥 ====================
                let final_data = if file_sr != target_sr {
                    resample_audio(channel_data, file_sr, target_sr)
                } else {
                    channel_data
                };
                // =========================================================

                // Compute Spectrogram from final samples
                let spectrograms = compute_spectrogram_from_memory(&final_data, target_sr);

                // Store
                let file_duration = final_data
                    .get(0)
                    .map(|ch0| ch0.len() as f32 / target_sr as f32)
                    .unwrap_or(0.0);

                files.push((p.to_path_buf(), channels, spectrograms));
                all_audio_buffers.push(final_data);

                // Update duration based on resampled length (use max across files)
                if file_count == 0 {
                    duration_seconds = file_duration;
                } else if file_duration > duration_seconds {
                    duration_seconds = file_duration;
                }

                total_channels += channels as u32;
                file_count += 1;
                max_channels_per_file = max_channels_per_file.max(channels);
            }
        }
    }

    if file_count == 0 {
        return Err(anyhow::anyhow!("No valid .wav files found in this folder."));
    }

    Ok((
        LoadedInfo {
            folder_name,
            full_path: path.clone(),
            total_channels,
            file_count,
            files,
            max_channels_per_file,
            sample_rate, // Now implies target_sr
            bit_depth,
            duration_seconds,
        },
        all_audio_buffers,
    ))
}

fn resample_audio(input: Vec<Vec<f32>>, source_sr: u32, target_sr: u32) -> Vec<Vec<f32>> {
    if source_sr == target_sr || input.is_empty() {
        return input;
    }

    if source_sr == 0 || target_sr == 0 {
        eprintln!(
            "[IRMax] Skipping resample due to invalid sample rate source={} target={}",
            source_sr, target_sr
        );
        return input;
    }

    let channels = input.len();
    let frames = input[0].len();

    // Sinc Interpolation Parameters
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    // SincFixedIn: Process entire chunk at once
    let mut resampler = match SincFixedIn::<f32>::new(
        target_sr as f64 / source_sr as f64, // Ratio
        2.0,                                 // Max relative ratio
        params,
        frames, // Chunk size
        channels,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[IRMax] Resampler init failed: {}", e);
            return input;
        }
    };

    let mut result = match resampler.process(&input, None) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[IRMax] Resampling failed: {}", e);
            return input;
        }
    };

    // Flush tail to avoid truncation of the last samples (safest offline fix)
    let tail: Vec<Vec<f32>> = match resampler.process_partial::<Vec<f32>>(None, None) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[IRMax] Resampling tail flush failed: {}", e);
            return result;
        }
    };

    // Append tail per channel
    let channels = result.len().min(tail.len());
    for ch in 0..channels {
        result[ch].extend_from_slice(&tail[ch]);
    }

    // Compensate group delay to align IR onset (optional but requested)
    let delay = resampler.output_delay();
    if delay > 0 {
        for ch in 0..result.len() {
            let len = result[ch].len();
            if len > delay {
                // Shift left by delay, keep total length by zero-padding the end
                result[ch].copy_within(delay..len, 0);
                result[ch].truncate(len - delay);
                result[ch].resize(len, 0.0);
            } else {
                // If delay exceeds length, zero out to avoid out-of-bounds
                result[ch].fill(0.0);
            }
        }
    }

    // Gain compensation for sample-rate change:
    // Preserve IR integral/overall gain across SR changes.
    let gain_comp = source_sr as f32 / target_sr as f32;
    if (gain_comp - 1.0).abs() > f32::EPSILON {
        for ch in 0..result.len() {
            for v in &mut result[ch] {
                *v *= gain_comp;
            }
        }
    }

    result
}

// Compute Log-Log Spectrogram from Memory
fn compute_spectrogram_from_memory(channel_data: &[Vec<f32>], _sample_rate: u32) -> Vec<Vec<f32>> {
    let channels = channel_data.len();
    if channels == 0 {
        return Vec::new();
    }
    let len = channel_data[0].len();

    // Reuse existing constants
    let fft_size = 1024;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut results = vec![vec![0.0; SPECTRUM_WIDTH * SPECTRUM_HEIGHT]; channels];

    // Pre-calculate Hanning Window
    let window: Vec<f32> = (0..fft_size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos())
        })
        .collect();

    for c in 0..channels {
        for t in 0..SPECTRUM_WIDTH {
            let start_sample = (t * len) / SPECTRUM_WIDTH;
            let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(fft_size);
            for i in 0..fft_size {
                let s_idx = start_sample + i;
                let val = if s_idx < channel_data[c].len() {
                    channel_data[c][s_idx] * window[i]
                } else {
                    0.0
                };
                buffer.push(Complex::new(val, 0.0));
            }
            fft.process(&mut buffer);

            // ... Process Magnitude logic (Same as before) ...
            let nyquist_idx = fft_size / 2;
            for y in 0..SPECTRUM_HEIGHT {
                let norm_y = y as f32 / (SPECTRUM_HEIGHT as f32 - 1.0);
                let min_bin = 1.0;
                let max_bin = nyquist_idx as f32;
                let bin_idx_float = min_bin * (max_bin / min_bin).powf(norm_y);
                let bin_idx = bin_idx_float as usize;
                let next_bin_idx = (min_bin
                    * (max_bin / min_bin).powf((y + 1) as f32 / (SPECTRUM_HEIGHT as f32 - 1.0)))
                    as usize;

                let range_end = next_bin_idx.max(bin_idx + 1).min(nyquist_idx);
                let mut sum_mag = 0.0;
                let mut count = 0;
                for b in bin_idx..range_end {
                    sum_mag += buffer[b].norm();
                    count += 1;
                }
                let avg_mag = if count > 0 {
                    sum_mag / count as f32
                } else {
                    0.0
                };
                let log_mag = (avg_mag * 10.0).ln_1p();

                let row_idx = SPECTRUM_HEIGHT - 1 - y;
                results[c][row_idx * SPECTRUM_WIDTH + t] = log_mag;
            }
        }
    }

    // Normalization
    let global_max = results
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0f32, |a, &b| a.max(b));
    if global_max > 1e-6 {
        for c in 0..channels {
            for v in &mut results[c] {
                *v /= global_max;
            }
        }
    }

    results
}

/// Natural comparison for filenames (handles numbers correctly)
/// "cut1" < "cut2" < "cut10" instead of "cut1" < "cut10" < "cut2"
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let mut a_chars = a.chars().peekable();
    let mut b_chars = b.chars().peekable();

    loop {
        match (a_chars.peek(), b_chars.peek()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ac), Some(bc)) => {
                let ac = *ac;
                let bc = *bc;

                // If both are digits, compare numerically
                if ac.is_ascii_digit() && bc.is_ascii_digit() {
                    // Extract full numbers
                    let mut a_num = String::new();
                    while let Some(&c) = a_chars.peek() {
                        if c.is_ascii_digit() {
                            a_num.push(c);
                            a_chars.next();
                        } else {
                            break;
                        }
                    }

                    let mut b_num = String::new();
                    while let Some(&c) = b_chars.peek() {
                        if c.is_ascii_digit() {
                            b_num.push(c);
                            b_chars.next();
                        } else {
                            break;
                        }
                    }

                    // Compare as numbers
                    let a_val: u64 = a_num.parse().unwrap_or(0);
                    let b_val: u64 = b_num.parse().unwrap_or(0);

                    match a_val.cmp(&b_val) {
                        Ordering::Equal => continue,
                        other => return other,
                    }
                } else {
                    // Compare characters case-insensitively
                    let ac_lower = ac.to_ascii_lowercase();
                    let bc_lower = bc.to_ascii_lowercase();

                    match ac_lower.cmp(&bc_lower) {
                        Ordering::Equal => {
                            a_chars.next();
                            b_chars.next();
                            continue;
                        }
                        other => return other,
                    }
                }
            }
        }
    }
}
