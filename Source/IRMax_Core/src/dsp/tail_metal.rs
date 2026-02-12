#![cfg(target_os = "macos")]

use metal::*;
use objc::rc::autoreleasepool;
use std::ffi::c_void;
use std::mem;
use std::ptr;

#[repr(C)]
struct SparseCmd {
    partition_idx: u16,
    input_idx: u16,
}

#[repr(C)]
struct SparseOffset {
    start_idx: u32,
    count: u32,
}

#[repr(C)]
struct ConvParams {
    num_inputs: u32,
    num_outputs: u32,
    num_partitions: u32,
    freq_bins: u32,
    num_active_inputs: u32,
}

pub struct GpuTailProcessor {
    context: std::sync::Arc<crate::dsp::metal_context::MetalContext>,

    // Buffers
    input_buffers: Vec<Buffer>, // Ping Pong Buffers
    ir_buffer: Buffer,
    output_buffer: Buffer,

    // Sparse Scheduling Buffers
    command_buffer: Buffer,
    offset_buffer: Buffer,

    params_buffer: Buffer,
    // staging_buffer removed (Zero-Copy)

    // Config
    num_outputs: usize,
    freq_bins: usize,

    // State
    frame_index: usize,

    // Stride cache (bytes)
    input_stride_bytes: u64,
    total_input_bytes: u64,

    // Telemetry (Exposed via getter or direct access if pub)
    pub last_driver_us: u64,
    pub last_gpu_us: u64,
}

impl GpuTailProcessor {
    pub fn new(
        context: std::sync::Arc<crate::dsp::metal_context::MetalContext>,
        num_inputs: usize,
        num_outputs: usize,
        block_size: usize,
        ir_data: &[f32], // Flat [Partition][In][Out][Freq][Real/Imag]
        active_inputs: &[u32],
    ) -> Option<Self> {
        let device = &context.device;

        // Calculcate dimensions
        // vDSP packs Nyquist into Imag[0], so we only need block_size bins.
        let _fft_size = block_size * 2;
        let freq_bins = block_size;

        // Verify IR size
        // Expected: partitions * inputs * outputs * freq_bins * 2
        let single_partition_floats = num_inputs * num_outputs * freq_bins * 2;
        if single_partition_floats == 0 {
            return None;
        }
        let num_partitions = ir_data.len() / single_partition_floats;

        if ir_data.len() % single_partition_floats != 0 {
            println!("Error: IR data size not aligned with params");
            return None;
        }

        // --- 1. Sparse Analysis & Command Generation ---
        let mut commands: Vec<SparseCmd> = Vec::new();
        let mut offsets: Vec<SparseOffset> = Vec::with_capacity(num_outputs);

        let stride_freq = 2; // Real/Imag
        let stride_out = freq_bins * stride_freq;
        let stride_in = num_outputs * stride_out;
        let stride_part = num_inputs * stride_in;

        // Temporary rollback for macOS A/B: keep sparse gating at about -120 dB.
        let threshold = 0.0;

        for out_ch in 0..num_outputs {
            let start_idx = commands.len() as u32;
            let mut count = 0;

            for p in 0..num_partitions {
                for &in_ch_real in active_inputs {
                    // Check energy of this block
                    // Index logic matches Standard Layout: [Partition][Input][Output][Freq]
                    let base_idx =
                        p * stride_part + (in_ch_real as usize) * stride_in + out_ch * stride_out;

                    let slice = &ir_data[base_idx..base_idx + stride_out];

                    // Simple peak check
                    let mut max_val = 0.0f32;
                    // Optimization: Check strided to avoid cache thrashing?
                    // No, slice is contiguous Freq bins.
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
            "🔧 [稀疏调度] GPU 任务优化：已为 {} 个输出生成 {} 条命令。",
            num_outputs,
            commands.len()
        );
        // Calculate sparseness
        let total_possible = num_partitions * active_inputs.len() * num_outputs;
        let usage = commands.len() as f32 / total_possible as f32 * 100.0;
        println!(
            "   👉 稀疏效率：激活 {:.1}%（跳过 {:.1}%）",
            usage,
            100.0 - usage
        );

        // --- Create Buffers ---

        // 1. Input FDL (Zero-Copy: StorageModeShared)
        let input_floats = num_partitions * num_inputs * freq_bins * 2;
        let total_input_bytes = (input_floats * 4) as u64;

        let buf_a = device.new_buffer(total_input_bytes, MTLResourceOptions::StorageModeShared);
        let buf_b = device.new_buffer(total_input_bytes, MTLResourceOptions::StorageModeShared);
        let input_buffers = vec![buf_a, buf_b];

        let input_stride_bytes = (num_inputs * freq_bins * 2 * 4) as u64;

        // 2. IR Buffer
        let ir_bytes = (ir_data.len() * 4) as u64;
        let ir_buffer = device.new_buffer_with_data(
            ir_data.as_ptr() as *const c_void,
            ir_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        // 3. Output Buffer
        let output_floats = num_outputs * freq_bins * 2;
        let output_buffer = device.new_buffer(
            (output_floats * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Always log once per engine build (no env required).
        let out_ptr = output_buffer.contents() as usize;
        let in_ptr = input_buffers[0].contents() as usize;
        let channel_stride_bytes = (freq_bins * 2 * 4) as usize;
        println!(
            "[DiagAlign] tail: freq_bins={} channel_stride_bytes={} input_stride_bytes={} out_ptr_mod16={} in_ptr_mod16={}",
            freq_bins,
            channel_stride_bytes,
            input_stride_bytes as usize,
            out_ptr % 16,
            in_ptr % 16
        );
        println!(
            "[DiagAlign] tail: params num_inputs={} num_outputs={} num_partitions={} freq_bins={}",
            num_inputs, num_outputs, num_partitions, freq_bins
        );

        // 4. Sparse Buffers
        let command_bytes = (commands.len() * mem::size_of::<SparseCmd>()) as u64;
        // Handle empty commands case (silence) without reading from a dangling pointer.
        let command_buffer = if command_bytes == 0 {
            let buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
            unsafe {
                ptr::write_bytes(buf.contents(), 0, 4);
            }
            buf
        } else {
            device.new_buffer_with_data(
                commands.as_ptr() as *const c_void,
                command_bytes,
                MTLResourceOptions::StorageModeShared,
            )
        };

        let offset_bytes = (offsets.len() * mem::size_of::<SparseOffset>()) as u64;
        let offset_buffer = device.new_buffer_with_data(
            offsets.as_ptr() as *const c_void,
            offset_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        // 5. Params
        let params = ConvParams {
            num_inputs: num_inputs as u32,
            num_outputs: num_outputs as u32,
            num_partitions: num_partitions as u32,
            freq_bins: freq_bins as u32,
            num_active_inputs: 0, // Unused in Sparse Mode
        };
        let params_buffer = device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            mem::size_of::<ConvParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // 6. Staging Buffer (Removed for Zero-Copy)

        Some(Self {
            context,
            input_buffers,
            ir_buffer,
            output_buffer,
            command_buffer,
            offset_buffer,
            params_buffer,
            // staging_buffer, // Removed
            num_outputs,
            freq_bins,
            frame_index: 0,
            input_stride_bytes,
            total_input_bytes,
            last_driver_us: 0,
            last_gpu_us: 0,
        })
    }

    /// 执行处理
    pub fn process(&mut self, input_complex: &[f32]) -> &[f32] {
        autoreleasepool(|| {
            let t0 = std::time::Instant::now();
            let cmd_buffer = self.context.queue.new_command_buffer();

            // Ping Pong selection
            let src_idx = self.frame_index % 2;
            let dst_idx = (self.frame_index + 1) % 2;
            let src_buf = &self.input_buffers[src_idx];
            let dst_buf = &self.input_buffers[dst_idx];

            // 1. Shift Input FDL
            let shift_size = self.total_input_bytes - self.input_stride_bytes;
            if shift_size > 0 {
                let blit = cmd_buffer.new_blit_command_encoder();
                blit.copy_from_buffer(src_buf, 0, dst_buf, self.input_stride_bytes, shift_size);
                blit.end_encoding();
            }

            // 2. Upload New Data (Zero-Copy)
            // storage_mode = Shared, so we can write directly to the buffer.
            unsafe {
                let ptr = dst_buf.contents() as *mut f32; // Points to start of buffer
                                                          // We overwrite the first 'input_stride_bytes' which is the "New Block" area.
                                                          // NOTE: For Frequency Domain Delay Line (FDL), the "New Block" is at index 0.
                                                          // Shift moved [0..end-stride] to [stride..end].
                                                          // So we write new data to [0..stride].
                std::ptr::copy_nonoverlapping(input_complex.as_ptr(), ptr, input_complex.len());
            }

            // No Blit needed for upload!

            // 3. Compute (Sparse Kernel)
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.context.pipeline);

            encoder.set_buffer(0, Some(dst_buf), 0);
            encoder.set_buffer(1, Some(&self.ir_buffer), 0);
            encoder.set_buffer(2, Some(&self.output_buffer), 0);
            encoder.set_buffer(3, Some(&self.params_buffer), 0);
            // 4 & 5: Sparse Scheduling
            encoder.set_buffer(4, Some(&self.command_buffer), 0);
            encoder.set_buffer(5, Some(&self.offset_buffer), 0);

            let w = (self.freq_bins as u64 + 31) / 32;
            let h = self.num_outputs as u64;
            encoder.dispatch_thread_groups(MTLSize::new(w, h, 1), MTLSize::new(32, 1, 1));
            encoder.end_encoding();

            cmd_buffer.commit();
            let t1 = std::time::Instant::now(); // Driver Overhead End

            cmd_buffer.wait_until_completed(); // Keep Synchronous for now
            let t2 = std::time::Instant::now(); // GPU Wait End

            self.last_driver_us = t1.duration_since(t0).as_micros() as u64;
            self.last_gpu_us = t2.duration_since(t1).as_micros() as u64;

            self.frame_index += 1;

            let ptr = self.output_buffer.contents() as *const f32;
            let len = self.num_outputs * self.freq_bins * 2;
            unsafe { std::slice::from_raw_parts(ptr, len) }
        })
    }

    pub fn reset_state(&mut self) {
        self.frame_index = 0;

        let input_bytes = self.total_input_bytes as usize;
        for buf in &self.input_buffers {
            unsafe {
                ptr::write_bytes(buf.contents(), 0, input_bytes);
            }
        }

        let output_bytes = self.num_outputs * self.freq_bins * 2 * std::mem::size_of::<f32>();
        unsafe {
            ptr::write_bytes(self.output_buffer.contents(), 0, output_bytes);
        }
    }
}
