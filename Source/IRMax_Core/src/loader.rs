use crate::backend::{create_context, GpuBackend};
use crate::engine::matrix_scheduler::{
    EngineType, MatrixAudioProcessor, MatrixScheduler, MatrixWorker, TailProbe,
};
use hound::WavReader;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub inputs: Vec<InputConfig>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct InputConfig {
    pub id: usize, // Maps to cut{id}.wav
    pub enabled: bool,
}

pub struct ResourceLoader {
    base_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct PartitionPlan {
    tail_block_size: usize,
    head_len: usize,
    body_block_size: usize,
    tail_start: usize,
}

impl PartitionPlan {
    #[inline]
    fn body_len(self) -> usize {
        self.tail_start.saturating_sub(self.head_len)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuardProfile {
    Aggressive2,
    Aggressive1,
    Baseline,
    Stable1,
}

impl GuardProfile {
    #[inline]
    fn from_i32(raw: i32) -> Self {
        match raw {
            -2 => Self::Aggressive2,
            -1 => Self::Aggressive1,
            0 => Self::Baseline,
            1 => Self::Stable1,
            _ => Self::Baseline,
        }
    }

    #[inline]
    fn delta_blocks(self) -> isize {
        match self {
            Self::Aggressive2 => -2,
            Self::Aggressive1 => -1,
            Self::Baseline => 0,
            Self::Stable1 => 1,
        }
    }
}

#[inline]
fn align_up(value: usize, step: usize) -> usize {
    if step == 0 {
        return value;
    }
    let rem = value % step;
    if rem == 0 {
        value
    } else {
        value + (step - rem)
    }
}

#[inline]
fn apply_guard_profile(base: PartitionPlan, profile: GuardProfile) -> PartitionPlan {
    if base.body_block_size == 0 || base.tail_start <= base.body_block_size {
        return base;
    }

    let body_block = base.body_block_size as isize;
    let min_head = body_block;
    let max_head = (base.tail_start - base.body_block_size) as isize;

    let mut head = base.head_len as isize + profile.delta_blocks() * body_block;
    if head < min_head {
        head = min_head;
    }
    if head > max_head {
        head = max_head;
    }

    let mut head_len = head as usize;
    // Keep head/body boundary aligned to body block for deterministic latency blocks.
    head_len = (head_len / base.body_block_size) * base.body_block_size;
    head_len = head_len
        .max(base.body_block_size)
        .min(base.tail_start - base.body_block_size);

    PartitionPlan {
        head_len,
        ..base
    }
}

#[inline]
fn build_partition_plan(sample_rate: u32, host_max_buffer: usize, guard_profile: i32) -> PartitionPlan {
    // Clamp to the minimum practical host size so tiny transient reports (e.g. 8) do not
    // distort partition planning. The processing path still handles the actual callback size.
    let host = host_max_buffer.max(64);

    // Partition table (48k family validated):
    // - <=128  : H=192,  B=64,  T=512,  Kt=256
    // - <=512  : H=256,  B=128, T=512,  Kt=256   (mid-buffer anti-underrun profile)
    // - <=1024 : H=512,  B=128, T=1024, Kt=256
    // - <=2048 : H=1024, B=256, T=2048, Kt=512
    // - >2048  : H=1024, B=256, T=align_up(host, 1024), Kt=512
    let is_supported_rate = matches!(sample_rate, 44100 | 48000 | 88200 | 96000 | 176400 | 192000);

    let base = if is_supported_rate {
        if host <= 128 {
            PartitionPlan {
                tail_block_size: 256,
                head_len: 192,
                body_block_size: 64,
                tail_start: 512,
            }
        } else if host <= 512 {
            PartitionPlan {
                tail_block_size: 256,
                head_len: 256,
                body_block_size: 128,
                tail_start: 512,
            }
        } else if host <= 1024 {
            PartitionPlan {
                tail_block_size: 256,
                head_len: 512,
                body_block_size: 128,
                tail_start: 1024,
            }
        } else if host <= 2048 {
            PartitionPlan {
                tail_block_size: 512,
                head_len: 1024,
                body_block_size: 256,
                tail_start: 2048,
            }
        } else {
            PartitionPlan {
                tail_block_size: 512,
                head_len: 1024,
                body_block_size: 256,
                tail_start: align_up(host, 1024),
            }
        }
    } else {
        // Fallback for uncommon rates: mirror the same host-buffer policy.
        if host <= 128 {
            PartitionPlan {
                tail_block_size: 256,
                head_len: 192,
                body_block_size: 64,
                tail_start: 512,
            }
        } else if host <= 512 {
            PartitionPlan {
                tail_block_size: 256,
                head_len: 256,
                body_block_size: 128,
                tail_start: 512,
            }
        } else if host <= 1024 {
            PartitionPlan {
                tail_block_size: 256,
                head_len: 512,
                body_block_size: 128,
                tail_start: 1024,
            }
        } else if host <= 2048 {
            PartitionPlan {
                tail_block_size: 512,
                head_len: 1024,
                body_block_size: 256,
                tail_start: 2048,
            }
        } else {
            PartitionPlan {
                tail_block_size: 512,
                head_len: 1024,
                body_block_size: 256,
                tail_start: align_up(host, 1024),
            }
        }
    };

    // Guard profile tuning around the current baseline:
    // -2: aggressive2, -1: aggressive1, 0: baseline (default), +1: stable1.
    let profile = GuardProfile::from_i32(guard_profile);
    apply_guard_profile(base, profile)
}

impl ResourceLoader {
    pub fn new(resources_dir: PathBuf) -> Self {
        Self {
            base_dir: resources_dir,
        }
    }

    pub fn load_config(&self) -> Result<Config, Box<dyn Error>> {
        let config_path = self.base_dir.join("config.json");
        let file = File::open(&config_path)?;
        let config = serde_json::from_reader(file)?;
        Ok(config)
    }

    /// Load a multi-channel IR file.
    pub fn load_ir(&self, index: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        // ... (existing implementation)
        let path = self.base_dir.join(format!("cut{}.wav", index));

        let mut reader = WavReader::open(&path)?;
        let spec = reader.spec();
        let channels = spec.channels as usize;

        // Read all samples and normalize integer PCM to f32 when needed.
        // Keep downstream processing in f32.
        let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
            reader.samples::<f32>().collect::<Result<_, _>>()?
        } else {
            let max_val = 2.0f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|x| x as f32 / max_val))
                .collect::<Result<_, _>>()?
        };

        let frames = samples.len() / channels;

        // De-interleave
        let mut deinterleaved = vec![Vec::with_capacity(frames); channels];
        for frame in 0..frames {
            for ch in 0..channels {
                deinterleaved[ch].push(samples[frame * channels + ch]);
            }
        }

        Ok(deinterleaved)
    }

    /// Construct the Matrix Engine from memory buffers.
    /// buffers: List of Files, each containing List of Channels (samples).
    /// Assuming strictly:
    /// - All files have same length? No, code supports variable length pads.
    /// - Total inputs = files.len() ? Or sum of channels?
    /// - MatrixScheduler expects: num_inputs (e.g. 64 channels).
    /// - We assume each "File" in `buffers` corresponds to one Row/Input?
    /// - Or does Plugin flatten it?
    /// - The Plugin `LoadedInfo` has `files: Vec<(Path, Channels, Spectrogram, DATA?)>`.
    /// - Wait, `loader.rs` in Plugin currently DOES NOT load audio data.
    /// - So I need to modify Plugin Loader to load audio data first.
    /// - This function here assumes we have the data.
    pub fn build_engine_from_buffers(
        buffers: &[Vec<Vec<f32>>], // [InputIndex/File][OutputIndex/Channel][Sample]
        sample_rate: u32,
        host_outputs: usize,
        host_max_buffer: usize,
    ) -> Result<(MatrixAudioProcessor, MatrixWorker), Box<dyn Error>> {
        Self::build_engine_from_buffers_with_probe(
            buffers,
            sample_rate,
            host_outputs,
            host_max_buffer,
            None,
        )
    }

    pub fn build_engine_from_buffers_with_probe(
        buffers: &[Vec<Vec<f32>>], // [InputIndex/File][OutputIndex/Channel][Sample]
        sample_rate: u32,
        host_outputs: usize,
        host_max_buffer: usize,
        _tail_probe: Option<TailProbe>,
    ) -> Result<(MatrixAudioProcessor, MatrixWorker), Box<dyn Error>> {
        let (processor, worker, _backend) = Self::build_engine_from_buffers_with_probe_and_backend(
            buffers,
            sample_rate,
            host_outputs,
            host_max_buffer,
            _tail_probe,
            GpuBackend::Auto,
            0,
        )?;
        Ok((processor, worker))
    }

    pub fn build_engine_from_buffers_with_probe_and_backend(
        buffers: &[Vec<Vec<f32>>], // [InputIndex/File][OutputIndex/Channel][Sample]
        sample_rate: u32,
        host_outputs: usize,
        host_max_buffer: usize,
        _tail_probe: Option<TailProbe>,
        backend: GpuBackend,
        guard_profile: i32,
    ) -> Result<(MatrixAudioProcessor, MatrixWorker, GpuBackend), Box<dyn Error>> {
        use crate::dsp::fft::FftContext;
        use crate::framework::DSPComplex;

        if buffers.is_empty() {
            return Err("No IR data provided".into());
        }

        #[cfg(all(
            target_os = "windows",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return Err("AVX2 is required for Windows builds.".into());
            }
        }

        // ------------------------------
        // Topology Strategy (Simple vs Matrix)
        // ------------------------------
        // Simple Mode:
        // - Single file (multi-channel) OR multiple mono files
        // - Treat as non-matrix: each input maps to its own output (diagonal only)
        //
        // Matrix Mode:
        // - Multiple files with multi-channel IRs
        // - Full matrix mapping (Input i -> Output j uses channel j of file i)
        let file_count = buffers.len();
        let max_file_outputs = buffers.iter().map(|f| f.len()).max().unwrap_or(0);
        let all_mono = buffers.iter().all(|f| f.len() <= 1);
        let simple_mode = file_count == 1 || all_mono;

        let capped_host_outputs = host_outputs.min(64);

        let (num_inputs, num_outputs) = if simple_mode {
            let diag_len = if file_count == 1 {
                max_file_outputs
            } else {
                file_count
            };
            let num_outputs = diag_len.min(capped_host_outputs);
            let num_inputs = num_outputs;
            (num_inputs, num_outputs)
        } else {
            let num_outputs = max_file_outputs.min(capped_host_outputs);
            let num_inputs = file_count;
            (num_inputs, num_outputs)
        };

        if num_outputs == 0 || num_inputs == 0 {
            return Err("No valid input/output channels after topology resolution.".into());
        }

        // C profile partition planning with explicit host-buffer tiers.
        // H<=512 keeps legacy 64/448/512 behavior; larger buffers move TailStart to host-aligned ranges.
        let mut max_expected_buffer = host_max_buffer;
        if max_expected_buffer == 0 {
            max_expected_buffer = 2048;
        }

        let partition_plan = build_partition_plan(sample_rate, max_expected_buffer, guard_profile);
        let block_size = partition_plan.tail_block_size;
        let head_len = partition_plan.head_len;
        let body_block_size = partition_plan.body_block_size;
        let body_len = partition_plan.body_len();
        let tail_start = partition_plan.tail_start;

        // 0. Build Matrix View (Row = Input, Col = Output)
        // This is a non-owning view into the original buffers.
        let empty_ir: &[f32] = &[];
        let mut matrix_view: Vec<Vec<&[f32]>> = Vec::with_capacity(num_inputs);
        if simple_mode {
            if file_count == 1 {
                let file = &buffers[0];
                for in_ch in 0..num_inputs {
                    let mut row = vec![empty_ir; num_outputs];
                    if in_ch < file.len() && in_ch < num_outputs {
                        row[in_ch] = file[in_ch].as_slice();
                    }
                    matrix_view.push(row);
                }
            } else {
                // Multiple mono files => diagonal mapping
                for in_ch in 0..num_inputs {
                    let mut row = vec![empty_ir; num_outputs];
                    if in_ch < buffers.len() && !buffers[in_ch].is_empty() && in_ch < num_outputs {
                        row[in_ch] = buffers[in_ch][0].as_slice();
                    }
                    matrix_view.push(row);
                }
            }
        } else {
            for in_ch in 0..num_inputs {
                let file_channels = &buffers[in_ch];
                let mut row = Vec::with_capacity(num_outputs);
                for out_ch in 0..num_outputs {
                    if out_ch < file_channels.len() {
                        row.push(file_channels[out_ch].as_slice());
                    } else {
                        row.push(empty_ir);
                    }
                }
                matrix_view.push(row);
            }
        }

        // 0.1 Build Head Matrix (owning, truncated to head_len)
        let mut head_matrix: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_inputs);
        for in_ch in 0..num_inputs {
            let row = &matrix_view[in_ch];
            let mut out_row: Vec<Vec<f32>> = Vec::with_capacity(num_outputs);
            for out_ch in 0..num_outputs {
                let ir = row[out_ch];
                if ir.is_empty() {
                    out_row.push(Vec::new());
                } else {
                    let len = head_len.min(ir.len());
                    out_row.push(ir[0..len].to_vec());
                }
            }
            head_matrix.push(out_row);
        }

        // 1. Prepare Tail Data (Matrix FDL)
        // Find max length across all channels of all files (after mapping)
        let max_len = matrix_view
            .iter()
            .flat_map(|row| row.iter())
            .map(|ir| ir.len())
            .max()
            .unwrap_or(0);

        let tail_len_time = if max_len > tail_start {
            max_len - tail_start
        } else {
            0
        };

        let partition_size = block_size;
        let num_partitions = if tail_len_time == 0 {
            0
        } else {
            (tail_len_time + partition_size - 1) / partition_size
        };

        let freq_bins = block_size;
        let mut matrix_flat =
            vec![0.0f32; num_partitions * num_inputs * num_outputs * freq_bins * 2];

        if num_partitions > 0 {
            let fft_size = block_size * 2;
            let fft = FftContext::new(fft_size);

            let mut time_domain = vec![0.0f32; fft_size];
            let mut complex_r = vec![0.0f32; block_size];
            let mut complex_i = vec![0.0f32; block_size];

            let stride_freq = 2; // Real/Imag
            let stride_out = freq_bins * stride_freq;
            let stride_in = num_outputs * stride_out;
            let stride_part = num_inputs * stride_in;

            for in_ch in 0..num_inputs {
                let row = &matrix_view[in_ch];

                for out_ch in 0..num_outputs {
                    let ir = row[out_ch];
                    if ir.is_empty() {
                        continue;
                    }

                    for p in 0..num_partitions {
                        let p_start = tail_start + p * block_size;
                        let p_end = p_start + block_size;

                        time_domain.fill(0.0);

                        if p_start < ir.len() {
                            let valid_end = p_end.min(ir.len());
                            let count = valid_end - p_start;
                            time_domain[0..count].copy_from_slice(&ir[p_start..p_start + count]);
                        }

                        let mut split = DSPComplex {
                            realp: complex_r.as_mut_ptr(),
                            imagp: complex_i.as_mut_ptr(),
                        };
                        fft.forward(&time_domain, &mut split);

                        unsafe {
                            let r = std::slice::from_raw_parts(split.realp, block_size);
                            let i = std::slice::from_raw_parts(split.imagp, block_size);

                            // True matrix mapping:
                            // Write ONLY to [p][in_ch][out_ch]
                            let base_idx =
                                p * stride_part + in_ch * stride_in + out_ch * stride_out;
                            for k in 0..block_size {
                                matrix_flat[base_idx + k * 2] = r[k];
                                matrix_flat[base_idx + k * 2 + 1] = i[k];
                            }
                        }
                    }
                }
            }
        }

        // 2. Prepare Body Data (CPU FDL, Optional)
        // Body covers [head_len .. tail_start) using smaller partitions.
        let body_start = head_len;
        let body_len_time = if tail_start > body_start {
            tail_start - body_start
        } else {
            0
        };
        let body_enabled = body_len > 0 && body_block_size > 0 && body_len_time > 0;

        let body_num_partitions = if body_enabled {
            (body_len_time + body_block_size - 1) / body_block_size
        } else {
            0
        };

        let body_freq_bins = body_block_size;
        let mut body_ir_real =
            vec![0.0f32; body_num_partitions * num_inputs * num_outputs * body_freq_bins];
        let mut body_ir_imag =
            vec![0.0f32; body_num_partitions * num_inputs * num_outputs * body_freq_bins];
        let mut body_ir_mask = vec![0u8; body_num_partitions * num_inputs * num_outputs];

        if body_num_partitions > 0 {
            let fft_size = body_block_size * 2;
            let fft = FftContext::new(fft_size);

            let mut time_domain = vec![0.0f32; fft_size];
            let mut complex_r = vec![0.0f32; body_block_size];
            let mut complex_i = vec![0.0f32; body_block_size];

            let stride_out = body_freq_bins;
            let stride_in = num_outputs * stride_out;
            let stride_part = num_inputs * stride_in;
            let mask_stride_in = num_outputs;
            let mask_stride_part = num_inputs * mask_stride_in;

            for in_ch in 0..num_inputs {
                let row = &matrix_view[in_ch];

                for out_ch in 0..num_outputs {
                    let ir = row[out_ch];
                    if ir.is_empty() {
                        continue;
                    }

                    for p in 0..body_num_partitions {
                        let p_start = body_start + p * body_block_size;
                        let p_end = p_start + body_block_size;

                        time_domain.fill(0.0);

                        let mut active = false;
                        if p_start < ir.len() {
                            let valid_end = p_end.min(ir.len());
                            let slice = &ir[p_start..valid_end];
                            for &v in slice {
                                if v != 0.0 {
                                    active = true;
                                    break;
                                }
                            }
                            let count = valid_end - p_start;
                            time_domain[0..count].copy_from_slice(slice);
                        }

                        let mask_idx = p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                        if active {
                            body_ir_mask[mask_idx] = 1;
                        } else {
                            body_ir_mask[mask_idx] = 0;
                            continue;
                        }

                        let mut split = DSPComplex {
                            realp: complex_r.as_mut_ptr(),
                            imagp: complex_i.as_mut_ptr(),
                        };
                        fft.forward(&time_domain, &mut split);

                        unsafe {
                            let r = std::slice::from_raw_parts(split.realp, body_block_size);
                            let i = std::slice::from_raw_parts(split.imagp, body_block_size);

                            let base_idx =
                                p * stride_part + in_ch * stride_in + out_ch * stride_out;
                            body_ir_real[base_idx..base_idx + body_block_size].copy_from_slice(r);
                            body_ir_imag[base_idx..base_idx + body_block_size].copy_from_slice(i);
                        }
                    }
                }
            }
        }

        // 3. Create Backend Context
        let context =
            create_context(backend).ok_or_else(|| "GPU Backend not initialized.".to_string())?;
        let (context, resolved_backend) = context;
        let context = Some(context);

        // 4. Build
        // buffers is `&[Vec<Vec<f32>>]`, which exactly matches `head_matrix` expectation
        let scheduler = MatrixScheduler::new(
            context,
            &matrix_flat,
            body_ir_real,
            body_ir_imag,
            body_ir_mask,
            &head_matrix, // Pass the structured head matrix directly
            num_inputs,
            num_outputs,
            block_size,
            head_len,
            body_block_size,
            body_len,
            tail_start,
            max_expected_buffer,
            if cfg!(target_os = "macos") {
                EngineType::Vdsp
            } else {
                EngineType::Titan
            },
        );

        match scheduler {
            Some(sched) => {
                let (processor, worker) = sched.split_into_async();
                Ok((processor, worker, resolved_backend))
            }
            None => Err("Failed to create MatrixScheduler".into()),
        }
    }
}
