#![cfg(not(target_os = "macos"))]

use crate::framework::cuda::CudaContext;
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
struct SparseCmd {
    partition_idx: u16,
    input_idx: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SparseOffset {
    start_idx: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    num_inputs: u32,
    num_outputs: u32,
    num_partitions: u32,
    freq_bins: u32,
    num_active_inputs: u32,
}

unsafe impl cudarc::driver::DeviceRepr for SparseCmd {}
unsafe impl cudarc::driver::DeviceRepr for SparseOffset {}
unsafe impl cudarc::driver::DeviceRepr for ConvParams {}

unsafe impl ValidAsZeroBits for SparseCmd {}
unsafe impl ValidAsZeroBits for SparseOffset {}
unsafe impl ValidAsZeroBits for ConvParams {}

pub struct CudaTailProcessor {
    context: Arc<CudaContext>,

    // Buffers
    input_buffers: Vec<CudaSlice<f32>>, // Ping Pong FDL (Frequency Delay Line)
    ir_buffer: CudaSlice<f32>,
    output_buffer: CudaSlice<f32>,

    // Sparse Scheduling
    command_buffer: CudaSlice<SparseCmd>,
    offset_buffer: CudaSlice<SparseOffset>,

    // Staging
    _host_input_stage: Vec<f32>,
    host_output_stage: Vec<f32>,

    // Config
    num_outputs: usize,
    freq_bins: usize,
    params: ConvParams,

    // State
    frame_index: usize,

    // Telemetry
    pub last_driver_us: u64,
    pub last_gpu_us: u64,

    // Cache
    input_stride_elems: usize,
    total_input_elems: usize,
    gpu_disabled_due_error: bool,
    runtime_error_logged: bool,
}

impl CudaTailProcessor {
    pub fn new(
        context: Arc<CudaContext>,
        num_inputs: usize,
        num_outputs: usize,
        block_size: usize,
        ir_data: &[f32], // Interleaved [Partition][Input][Output][Freq][Real/Imag]
        active_inputs: &[u32],
    ) -> Option<Self> {
        let device = &context.device;

        let freq_bins = block_size;
        let single_partition_floats = num_inputs * num_outputs * freq_bins * 2;
        if single_partition_floats == 0 {
            return None;
        }

        // Detect num_partitions
        if ir_data.len() % single_partition_floats != 0 {
            println!("Error: IR data size mismatch");
            return None;
        }
        let num_partitions = ir_data.len() / single_partition_floats;

        // --- 1. Build Sparse Schedule (Same logic as Metal) ---
        let mut commands = Vec::new();
        let mut offsets = Vec::with_capacity(num_outputs);

        let stride_freq = 2;
        let stride_out = freq_bins * stride_freq;
        let stride_in = num_outputs * stride_out;
        let stride_part = num_inputs * stride_in;
        // Exact mode: only pure zero blocks are skipped.
        let threshold = 0.0;

        for out_ch in 0..num_outputs {
            let start_idx = commands.len() as u32;
            let mut count = 0;

            for p in 0..num_partitions {
                for &in_ch_real in active_inputs {
                    let base_idx =
                        p * stride_part + (in_ch_real as usize) * stride_in + out_ch * stride_out;
                    let slice = &ir_data[base_idx..base_idx + stride_out];

                    let mut max_val = 0.0f32;
                    for &v in slice {
                        if v.abs() > max_val {
                            max_val = v.abs();
                        }
                    }

                    if max_val > threshold {
                        commands.push(SparseCmd {
                            partition_idx: p as u16,
                            input_idx: in_ch_real as u16,
                        });
                        count += 1;
                    }
                }
            }
            offsets.push(SparseOffset { start_idx, count });
        }

        println!(
            "🚀 [CudaTail] 已生成 {} 条稀疏命令。",
            commands.len()
        );

        // --- 2. GPU Allocations ---
        let input_stride_elems = num_inputs * freq_bins * 2;
        let input_elems_raw = num_partitions * input_stride_elems;
        let input_elems = input_elems_raw.max(input_stride_elems);
        if num_partitions == 0 {
            println!("[CudaTail] No tail partitions (IR shorter than tail_start); running in zero-tail mode.");
        }

        let buf_a = device.alloc_zeros::<f32>(input_elems).ok()?;
        let buf_b = device.alloc_zeros::<f32>(input_elems).ok()?;

        let mut ir_buffer = device.alloc_zeros::<f32>(ir_data.len()).ok()?;
        device.htod_sync_copy_into(ir_data, &mut ir_buffer).ok()?; // Fixed syntax: slice to buffer

        let output_elems = num_outputs * freq_bins * 2;
        let output_buffer = device.alloc_zeros::<f32>(output_elems).ok()?;

        let mut command_buffer = device
            .alloc_zeros::<SparseCmd>(commands.len().max(1))
            .ok()?;
        if !commands.is_empty() {
            device
                .htod_sync_copy_into(&commands, &mut command_buffer)
                .ok()?;
        }

        let mut offset_buffer = device.alloc_zeros::<SparseOffset>(offsets.len()).ok()?;
        device
            .htod_sync_copy_into(&offsets, &mut offset_buffer)
            .ok()?;

        // Host Staging
        let _host_input_stage = vec![0.0f32; input_stride_elems];
        let host_output_stage = vec![0.0f32; output_elems];

        println!("🚀 [CudaTail] 固定内存分配失败，已回退为 Vec。");

        Some(Self {
            context,
            input_buffers: vec![buf_a, buf_b],
            ir_buffer,
            output_buffer,
            command_buffer,
            offset_buffer,
            _host_input_stage,
            host_output_stage,
            num_outputs,
            freq_bins,
            params: ConvParams {
                num_inputs: num_inputs as u32,
                num_outputs: num_outputs as u32,
                num_partitions: num_partitions as u32,
                freq_bins: freq_bins as u32,
                num_active_inputs: 0,
            },
            frame_index: 0,
            last_driver_us: 0,
            last_gpu_us: 0,
            input_stride_elems,
            total_input_elems: input_elems,
            gpu_disabled_due_error: false,
            runtime_error_logged: false,
        })
    }
    fn runtime_fail<E: std::fmt::Display>(&mut self, stage: &str, err: E) -> &[f32] {
        if !self.runtime_error_logged {
            eprintln!("[CudaTail] Disabled GPU tail after {} error: {}", stage, err);
            self.runtime_error_logged = true;
        }
        self.gpu_disabled_due_error = true;
        self.host_output_stage.fill(0.0);
        self.last_driver_us = 0;
        self.last_gpu_us = 0;
        &self.host_output_stage
    }

    pub fn process(&mut self, input_complex: &[f32]) -> &[f32] {
        if self.gpu_disabled_due_error {
            return &self.host_output_stage;
        }

        let device = &self.context.device;
        let t0 = std::time::Instant::now();

        // 0. Write to Host Staging (CPU Copy), padded with zeros if needed.
        let copy_len = input_complex.len().min(self.input_stride_elems);
        self._host_input_stage[..copy_len].copy_from_slice(&input_complex[..copy_len]);
        if copy_len < self.input_stride_elems {
            self._host_input_stage[copy_len..self.input_stride_elems].fill(0.0);
        }

        // 1. Prepare Staging / ping-pong buffers.
        let (first, second) = self.input_buffers.split_at_mut(1);
        let (src_buf, dst_buf) = if self.frame_index % 2 == 0 {
            (&first[0], &mut second[0])
        } else {
            (&second[0], &mut first[0])
        };

        // Shift Old Data: Copy [0..Total-Stride] from Src to [Stride..Total] in Dst.
        let shift_elems = self.total_input_elems.saturating_sub(self.input_stride_elems);
        if shift_elems > 0 {
            let src_view = src_buf.slice(0..shift_elems);
            let mut dst_view = dst_buf.slice_mut(self.input_stride_elems..self.total_input_elems);
            if let Err(err) = device.dtod_copy(&src_view, &mut dst_view) {
                return self.runtime_fail("dtod_copy", err);
            }
        }

        // Upload New Data: Copy input to Dst [0..Stride].
        let mut dst_new_view = dst_buf.slice_mut(0..self.input_stride_elems);
        if let Err(err) = device.htod_sync_copy_into(
            &self._host_input_stage[..self.input_stride_elems],
            &mut dst_new_view,
        ) {
            return self.runtime_fail("htod_sync_copy_into", err);
        }

        // 3. Launch Kernel.
        let func = match device.get_func("irmax", "fused_mimo") {
            Some(func) => func,
            None => return self.runtime_fail("get_func", "kernel fused_mimo not found"),
        };

        let grid_x = (self.freq_bins as u32 + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, self.num_outputs as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let launch_result = unsafe {
            func.launch(
                cfg,
                (
                    &*dst_buf,
                    &self.ir_buffer,
                    &mut self.output_buffer,
                    self.params,
                    &self.command_buffer,
                    &self.offset_buffer,
                ),
            )
        };
        if let Err(err) = launch_result {
            return self.runtime_fail("kernel launch", err);
        }

        let t1 = std::time::Instant::now();

        // 4. Download Result.
        if let Err(err) = device.dtoh_sync_copy_into(&self.output_buffer, &mut self.host_output_stage) {
            return self.runtime_fail("dtoh_sync_copy_into", err);
        }

        let t2 = std::time::Instant::now();
        self.last_driver_us = t1.duration_since(t0).as_micros() as u64;
        self.last_gpu_us = t2.duration_since(t1).as_micros() as u64;

        self.frame_index += 1;

        &self.host_output_stage
    }

    pub fn reset_state(&mut self) {
        let device = &self.context.device;
        self.frame_index = 0;
        self.gpu_disabled_due_error = false;
        self.runtime_error_logged = false;

        let zeros_input = vec![0.0f32; self.total_input_elems];
        for buf in &mut self.input_buffers {
            let _ = device.htod_sync_copy_into(&zeros_input, buf);
        }

        let zeros_output = vec![0.0f32; self.num_outputs * self.freq_bins * 2];
        let _ = device.htod_sync_copy_into(&zeros_output, &mut self.output_buffer);
    }
}
