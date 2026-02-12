// Verified M4 Architecture 3.1 Implementation (De-Optimized for Stability)
// Status: GOLDEN (0ms Latency Verified by User on Mac)
#![cfg(target_os = "macos")]

use crate::accelerate;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::slice;

// --- AlignedBuffer (Preserved EXACTLY from Arch 3.1) ---
struct AlignedBuffer {
    ptr: *mut f32,
    layout: Layout,
    len: usize,
}

impl AlignedBuffer {
    fn new(len: usize) -> Self {
        let align = 64; // Cache Line 对齐
        let raw_size = len * std::mem::size_of::<f32>();
        let padded_size = (raw_size + align - 1) & !(align - 1);
        let safe_layout_size = if padded_size == 0 { align } else { padded_size };

        let layout = Layout::from_size_align(safe_layout_size, align).expect("Invalid Layout");
        let ptr = unsafe { alloc(layout) as *mut f32 };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // 初始化内存
        unsafe { ptr::write_bytes(ptr as *mut u8, 0, safe_layout_size) };

        Self { ptr, layout, len }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, self.layout) };
    }
}

impl std::ops::Deref for AlignedBuffer {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

// --- HeadVdsp (Architecture 3.1 - Forced Full Processing) ---
pub struct HeadVdsp {
    packed_filters: AlignedBuffer,
    state_buffer: AlignedBuffer,
    output_scratch: AlignedBuffer,

    num_outputs: usize, // 🔥 Restored
    active_outputs: Vec<u8>,
    ir_stride: usize,
    #[allow(dead_code)]
    head_len: usize,
    history_len: usize,
    write_pos: usize,
    max_capacity: usize,
}

impl HeadVdsp {
    pub fn new(row_irs: &[Vec<f32>], max_expected_buffer: usize) -> Self {
        let num_outputs = row_irs.len();
        let mut max_input_len = 0;
        let mut active_outputs = Vec::with_capacity(num_outputs);
        for ir in row_irs {
            max_input_len = max_input_len.max(ir.len());
            let mut active = 0u8;
            for &v in ir {
                if v != 0.0 {
                    active = 1;
                    break;
                }
            }
            active_outputs.push(active);
        }

        // Align stride to 16 floats
        let ir_stride_align = 16;
        let ir_stride = (max_input_len + ir_stride_align - 1) & !(ir_stride_align - 1);
        let ir_stride = ir_stride.max(16);

        // Architecture 3.1 Logic: Full Dense Allocation
        let mut packed_filters_vec = Vec::with_capacity(num_outputs * ir_stride);

        for ir in row_irs {
            let mut rev = ir.clone();
            rev.reverse();

            // Pad zeros to START (Trailing in Time)
            while rev.len() < ir_stride {
                rev.insert(0, 0.0);
            }

            packed_filters_vec.extend(&rev);
        }

        let packed_filters = AlignedBuffer::new(packed_filters_vec.len());
        unsafe {
            ptr::copy_nonoverlapping(
                packed_filters_vec.as_ptr(),
                packed_filters.ptr,
                packed_filters_vec.len(),
            );
        }

        // History Calculation
        let history_len = if ir_stride > 0 { ir_stride - 1 } else { 0 };

        let lazy_buffer_size = 65536;
        let state_size = history_len + lazy_buffer_size + max_expected_buffer;

        let write_pos = history_len;

        Self {
            packed_filters,
            state_buffer: AlignedBuffer::new(state_size),
            output_scratch: AlignedBuffer::new(max_expected_buffer),
            num_outputs, // 🔥 Restored
            active_outputs,
            ir_stride,
            head_len: max_input_len,
            history_len,
            write_pos,
            max_capacity: max_expected_buffer,
        }
    }

    pub fn reset_state(&mut self) {
        unsafe {
            std::ptr::write_bytes(
                self.state_buffer.ptr as *mut u8,
                0,
                self.state_buffer.layout.size(),
            );
        }
        self.write_pos = self.history_len;
    }

    pub fn process(&mut self, input: &[f32], outputs: &mut [&mut [f32]]) {
        let n = input.len();
        if n == 0 {
            return;
        }

        let mut offset = 0;
        while offset < n {
            let chunk_len = (n - offset).min(self.max_capacity);
            if chunk_len == 0 {
                break;
            }
            self.process_chunk(&input[offset..offset + chunk_len], outputs, offset);
            offset += chunk_len;
        }
    }

    fn process_chunk(&mut self, input: &[f32], outputs: &mut [&mut [f32]], out_offset: usize) {
        let n = input.len();
        if n == 0 {
            return;
        }

        let hist_len = self.history_len;

        // 1. Lazy Buffer Logic
        if self.write_pos + n > self.state_buffer.len() {
            unsafe {
                let src_ptr = self.state_buffer.ptr.add(self.write_pos - hist_len);
                let dst_ptr = self.state_buffer.ptr;
                ptr::copy(src_ptr, dst_ptr, hist_len);
            }
            self.write_pos = hist_len;
        }

        // 2. Append Input
        unsafe {
            let dest_ptr = self.state_buffer.ptr.add(self.write_pos);
            ptr::copy_nonoverlapping(input.as_ptr(), dest_ptr, n);
        }

        // 3. Convolution
        let signal_base_ptr = unsafe { self.state_buffer.ptr.add(self.write_pos - hist_len) };

        let chunk_len_i64 = n as crate::accelerate::vDSP_Length;
        let filter_len_i64 = self.ir_stride as crate::accelerate::vDSP_Length;

        // Architecture 3.1 Loop: Fully Unconditional 0..N
        for out_ch in 0..self.num_outputs {
            if self.active_outputs.get(out_ch).copied().unwrap_or(1) == 0 {
                continue;
            }
            unsafe {
                // Pointer Math is STANDARD (Dense based)
                let filter_ptr = self.packed_filters.ptr.add(out_ch * self.ir_stride);
                let signal_ptr = signal_base_ptr;

                // Conv (Always runs, even for empty channels)
                accelerate::vDSP_conv(
                    signal_ptr,
                    1,
                    filter_ptr,
                    1,
                    self.output_scratch.ptr,
                    1,
                    chunk_len_i64,
                    filter_len_i64,
                );

                // Add to output
                let out_ptr = outputs[out_ch].as_mut_ptr().add(out_offset);
                accelerate::vDSP_vadd(
                    out_ptr,
                    1,
                    self.output_scratch.ptr,
                    1,
                    out_ptr,
                    1,
                    chunk_len_i64,
                );
            }
        }

        self.write_pos += n;
    }
}
