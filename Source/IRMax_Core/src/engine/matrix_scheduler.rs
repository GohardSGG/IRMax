use crate::dsp::fft::FftContext;
use crate::framework::DSPComplex;
// 鍒囨崲涓?vDSP 鐗堟湰
use crate::dsp::head_titan::TitanHeadRow;
// use crate::dsp::head_vdsp::HeadProcessorVdsp; // Removed unused
#[cfg(target_os = "windows")]
use crate::dsp::ipp;
use crate::dsp::metal_context::MetalContext;
use crate::dsp::tail_gpu::GpuTailProcessor;
use crossbeam_channel::{Receiver, Sender};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use rayon::prelude::*;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use rayon::ThreadPoolBuilder;
use rtrb::{Consumer, Producer, RingBuffer};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

static ENGINE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
const DEBUG_MUTE_HEAD: bool = false;
const DEBUG_MUTE_TAIL: bool = false;
#[cfg(target_os = "macos")]
const BODY_MAX_THREADS: usize = 4;
#[cfg(target_os = "macos")]
const BODY_OUTPUTS_PER_TASK: usize = 4;
#[cfg(target_os = "windows")]
const BODY_MAX_THREADS: usize = 4;
#[cfg(target_os = "windows")]
const BODY_OUTPUTS_PER_TASK: usize = 4;
#[cfg(target_os = "windows")]
const BODY_PARALLEL_WORK_THRESHOLD: usize = 65536;
#[cfg(target_os = "windows")]
const TAIL_WORKER_IDLE_POLL_US: u64 = 200;
#[cfg(any(target_os = "windows", target_os = "macos"))]
const BODY_WORKER_IDLE_POLL_US: u64 = 200;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const BODY_MAX_THREADS: usize = 1;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const BODY_OUTPUTS_PER_TASK: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EngineType {
    Titan,
    Vdsp,
}

#[derive(Clone)]
pub struct TailProbe {
    reported: Arc<AtomicBool>,
    position: Arc<AtomicU64>,
}

impl TailProbe {
    pub fn new() -> Self {
        Self {
            reported: Arc::new(AtomicBool::new(false)),
            position: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn report(&self, pos: u64) {
        self.position.store(pos, Ordering::Release);
        self.reported.store(true, Ordering::Release);
    }

    pub fn take(&self) -> Option<u64> {
        if self.reported.swap(false, Ordering::AcqRel) {
            Some(self.position.load(Ordering::Acquire))
        } else {
            None
        }
    }
}

/// 鐭╅樀璋冨害鍣細绠＄悊 18x18 鐨勫叏杩炴帴娣峰搷
pub enum HeadEngine {
    Vdsp(Vec<crate::dsp::head_vdsp::HeadVdsp>),
    Titan(Vec<TitanHeadRow>),
}

pub enum TailEngine {
    Gpu(GpuTailProcessor),
}

impl TailEngine {
    pub fn process(&mut self, input: &[f32]) -> &[f32] {
        match self {
            TailEngine::Gpu(gpu) => gpu.process(input),
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            TailEngine::Gpu(gpu) => gpu.reset_state(),
        }
    }
}

/// 鐭╅樀璋冨害鍣細绠＄悊 18x18 鐨勫叏杩炴帴娣峰搷
pub struct MatrixScheduler {
    engine: HeadEngine,
    tail: TailEngine,
    fft: Arc<FftContext>,

    num_inputs: usize,
    num_outputs: usize,
    block_size: usize,
    body_block_size: usize,
    body_len: usize,
    tail_start: usize,

    input_buffers: Vec<Vec<f32>>,
    input_pos: usize,

    overlap_buffers: Vec<Vec<f32>>,
    saved_overlaps: Vec<Vec<f32>>,

    scratch_fft_input: Vec<f32>,
    scratch_complex_r: Vec<f32>,
    scratch_complex_i: Vec<f32>,
    gpu_input_aggregate: Vec<f32>,
    scratch_time_domain: Vec<f32>,

    // Body IR (Freq Domain, Split)
    body_ir_real: Vec<f32>,
    body_ir_imag: Vec<f32>,
    body_ir_mask: Vec<u8>,

    // Telemetry
    pub gpu_execution_time: Arc<std::sync::atomic::AtomicU64>,
    pub gpu_driver_time: Arc<std::sync::atomic::AtomicU64>,
    pub gpu_compute_time: Arc<std::sync::atomic::AtomicU64>,

    pub _head_len: usize,   // 鉁?鏂板瀛楁 (Silence warning)
    pub _tail_start: usize, // 鉁?鐢ㄤ簬 Tail 瀵归綈
}

// --- Async Structures ---

struct BodyPipelineAudio {
    block_size: usize,
    latency_blocks: usize,
    blocks_since_reset: usize,
    blocks_consumed: usize,

    input_buffers: Vec<Vec<f32>>,
    input_pos: usize,

    task_producer: Producer<f32>,
    result_consumer: Consumer<f32>,

    // Flat [out][block]
    overlap_buffers: Vec<f32>,
}

struct BodyPipelineWorker {
    block_size: usize,
    num_partitions: usize,

    task_consumer: Consumer<f32>,
    result_producer: Producer<f32>,

    // IR in split format: [partition][in][out][freq]
    #[cfg(target_os = "macos")]
    ir_real: Vec<f32>,
    #[cfg(target_os = "macos")]
    ir_imag: Vec<f32>,
    #[cfg(target_os = "windows")]
    ir_complex: Vec<ipp::Ipp32fc>,
    // Active mask: [partition][in][out] (1 = active, 0 = zero)
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    active_mask: Vec<u8>,

    // FDL in split format: [partition][in][freq]
    #[cfg(target_os = "macos")]
    fdl_real: Vec<f32>,
    #[cfg(target_os = "macos")]
    fdl_imag: Vec<f32>,
    #[cfg(target_os = "windows")]
    fdl_complex: Vec<ipp::Ipp32fc>,
    fdl_pos: usize,

    fft: FftContext,
    scratch_fft_input: Vec<f32>,
    #[cfg(target_os = "macos")]
    scratch_complex_r: Vec<f32>,
    #[cfg(target_os = "macos")]
    scratch_complex_i: Vec<f32>,

    scratch_task_pop: Vec<f32>,
    scratch_result_push: Vec<f32>,

    // Flat [out][block]
    saved_overlaps: Vec<f32>,

    parallel_outputs_per_task: usize,
    thread_pool: rayon::ThreadPool,
    thread_scratch: ThreadScratchPool,
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
struct BodyWorker {
    num_inputs: usize,
    num_outputs: usize,
    body: BodyPipelineWorker,
    signal_rx: Receiver<()>,
    signal_pending: Arc<AtomicBool>,
    reset_counter: Arc<AtomicU64>,
    reset_ack: Arc<AtomicU64>,
    last_reset_counter: u64,
    stop: Arc<AtomicBool>,
    body_worker_started: Arc<AtomicU64>,
    body_tasks_consumed: Arc<AtomicU64>,
    body_results_pushed: Arc<AtomicU64>,
}

#[cfg(target_os = "windows")]
impl BodyWorker {
    fn wait_for_signal_timeout(&self, timeout: Duration) -> bool {
        // Do not clear pending before blocking: that can suppress a wake edge and delay work.
        let signaled = self.signal_rx.recv_timeout(timeout).is_ok();
        self.signal_pending.store(false, Ordering::Release);
        signaled
    }

    fn handle_reset(&mut self, new_counter: u64) {
        self.last_reset_counter = new_counter;
        self.body
            .fdl_complex
            .fill(ipp::Ipp32fc { re: 0.0, im: 0.0 });
        self.body.saved_overlaps.fill(0.0);
        self.body.fdl_pos = self.body.num_partitions.saturating_sub(1);
        self.drain_body_task_consumer();
        self.reset_ack.store(new_counter, Ordering::Release);
    }

    fn drain_body_task_consumer(&mut self) {
        loop {
            let available = self.body.task_consumer.slots();
            if available == 0 {
                break;
            }
            if let Ok(chunk) = self.body.task_consumer.read_chunk(available) {
                chunk.commit_all();
            } else {
                break;
            }
        }
    }

    fn wait_and_process(&mut self) -> bool {
        let current_reset = self.reset_counter.load(Ordering::Acquire);
        if current_reset != self.last_reset_counter {
            self.handle_reset(current_reset);
        }

        let body_task_size = self.num_inputs * self.body.block_size;
        if self.body.task_consumer.slots() >= body_task_size {
            if let Ok(chunk) = self.body.task_consumer.read_chunk(body_task_size) {
                let (s1, s2) = chunk.as_slices();

                if s1.len() > 0 {
                    self.body.scratch_task_pop[0..s1.len()].copy_from_slice(s1);
                }
                if s2.len() > 0 {
                    self.body.scratch_task_pop[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
                }
                chunk.commit_all();
            }

            self.body_tasks_consumed.fetch_add(1, Ordering::Relaxed);
            self.process_body();
            return true;
        }

        false
    }

    fn run(&mut self) {
        self.body_worker_started.fetch_add(1, Ordering::Release);
        unsafe {
            use windows_sys::Win32::System::Threading::{
                GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
            };
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST as i32);
        }

        loop {
            if self.stop.load(Ordering::Acquire) {
                break;
            }

            if !self.wait_and_process() {
                // Double-check once to narrow the enqueue->sleep race window.
                if !self.wait_and_process() {
                    let _ = self
                        .wait_for_signal_timeout(Duration::from_micros(BODY_WORKER_IDLE_POLL_US));
                }
            }
        }
    }

    fn process_body(&mut self) {
        if self.body.num_partitions == 0 {
            self.body.scratch_result_push.fill(0.0);
            return;
        }

        let freq_bins = self.body.block_size;
        let fdl_stride_in = freq_bins;
        let fdl_stride_part = self.num_inputs * fdl_stride_in;

        self.body.fdl_pos = (self.body.fdl_pos + 1) % self.body.num_partitions;

        for ch in 0..self.num_inputs {
            let offset = ch * self.body.block_size;
            self.body.scratch_fft_input.fill(0.0);
            self.body.scratch_fft_input[0..self.body.block_size].copy_from_slice(
                &self.body.scratch_task_pop[offset..offset + self.body.block_size],
            );

            let base = self.body.fdl_pos * fdl_stride_part + ch * fdl_stride_in;
            let dst = &mut self.body.fdl_complex[base..base + self.body.block_size];
            self.body
                .fft
                .forward_complex_half(&self.body.scratch_fft_input, dst);
        }

        let stride_out = freq_bins;
        let stride_in = self.num_outputs * stride_out;
        let stride_part = self.num_inputs * stride_in;

        let ir_complex = &self.body.ir_complex;
        let fdl_complex = &self.body.fdl_complex;
        let active_mask = &self.body.active_mask;
        let fdl_pos = self.body.fdl_pos;
        let num_parts = self.body.num_partitions;
        let num_inputs = self.num_inputs;
        let num_outputs = self.num_outputs;
        let mask_stride_in = num_outputs;
        let mask_stride_part = num_inputs * mask_stride_in;

        let gain_comp = 1.0f32;

        let run_parallel =
            MatrixWorker::body_parallel_enabled(num_outputs, num_inputs, num_parts, freq_bins);

        if !run_parallel {
            let scratch_cell = self.body.thread_scratch.get(0);
            let scratch = unsafe { &mut *scratch_cell.get() };

            for out_ch in 0..num_outputs {
                for v in scratch.accum.iter_mut() {
                    v.re = 0.0;
                    v.im = 0.0;
                }

                for in_ch in 0..num_inputs {
                    for p in 0..num_parts {
                        let mask_idx = p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                        if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                            continue;
                        }

                        let ir_base = p * stride_part + in_ch * stride_in + out_ch * stride_out;
                        let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                        let fdl_base = fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                        unsafe {
                            // Packed real spectrum: bin0 stores (DC, Nyquist) in (re, im).
                            // Keep bin0 as independent real products while fusing MAC for bins 1..N-1.
                            let ar0 = (*ir_complex.as_ptr().add(ir_base)).re;
                            let ai0 = (*ir_complex.as_ptr().add(ir_base)).im;
                            let br0 = (*fdl_complex.as_ptr().add(fdl_base)).re;
                            let bi0 = (*fdl_complex.as_ptr().add(fdl_base)).im;
                            let prev0 = *scratch.accum.get_unchecked(0);
                            let _ = ipp::ippsAddProduct_32fc(
                                ir_complex.as_ptr().add(ir_base),
                                fdl_complex.as_ptr().add(fdl_base),
                                scratch.accum.as_mut_ptr(),
                                freq_bins as i32,
                            );
                            (*scratch.accum.get_unchecked_mut(0)).re = prev0.re + ar0 * br0;
                            (*scratch.accum.get_unchecked_mut(0)).im = prev0.im + ai0 * bi0;
                        }
                    }
                }

                scratch
                    .fft
                    .inverse_complex_half_to_real(&scratch.accum, &mut scratch.time);

                let out_base = out_ch * freq_bins;
                for k in 0..freq_bins {
                    let val = scratch.time[k] * gain_comp;
                    self.body.scratch_result_push[out_base + k] =
                        val + self.body.saved_overlaps[out_base + k];
                    self.body.saved_overlaps[out_base + k] =
                        scratch.time[k + freq_bins] * gain_comp;
                }
            }
        } else {
            let outputs_per_task = self.body.parallel_outputs_per_task.max(1);
            let group_size = outputs_per_task * freq_bins;
            let thread_scratch = &self.body.thread_scratch;

            self.body.thread_pool.install(|| {
                self.body
                    .scratch_result_push
                    .par_chunks_mut(group_size)
                    .zip(self.body.saved_overlaps.par_chunks_mut(group_size))
                    .enumerate()
                    .for_each(|(group_idx, (out_group, overlap_group))| {
                        let thread_index = rayon::current_thread_index().unwrap_or(0);
                        let scratch_cell = thread_scratch.get(thread_index);
                        let scratch = unsafe { &mut *scratch_cell.get() };

                        let base_out = group_idx * outputs_per_task;
                        let group_outputs = out_group.len() / freq_bins;

                        for g in 0..group_outputs {
                            let out_ch = base_out + g;
                            let out_offset = g * freq_bins;
                            let out_slice = &mut out_group[out_offset..out_offset + freq_bins];
                            let overlap_slice =
                                &mut overlap_group[out_offset..out_offset + freq_bins];

                            for v in scratch.accum.iter_mut() {
                                v.re = 0.0;
                                v.im = 0.0;
                            }

                            for in_ch in 0..num_inputs {
                                for p in 0..num_parts {
                                    let mask_idx =
                                        p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                                    if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                                        continue;
                                    }

                                    let ir_base =
                                        p * stride_part + in_ch * stride_in + out_ch * stride_out;
                                    let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                                    let fdl_base =
                                        fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                                    unsafe {
                                        // Packed real spectrum: bin0 stores (DC, Nyquist) in (re, im).
                                        // Keep bin0 as independent real products while fusing MAC for bins 1..N-1.
                                        let ar0 = (*ir_complex.as_ptr().add(ir_base)).re;
                                        let ai0 = (*ir_complex.as_ptr().add(ir_base)).im;
                                        let br0 = (*fdl_complex.as_ptr().add(fdl_base)).re;
                                        let bi0 = (*fdl_complex.as_ptr().add(fdl_base)).im;
                                        let prev0 = *scratch.accum.get_unchecked(0);
                                        let _ = ipp::ippsAddProduct_32fc(
                                            ir_complex.as_ptr().add(ir_base),
                                            fdl_complex.as_ptr().add(fdl_base),
                                            scratch.accum.as_mut_ptr(),
                                            freq_bins as i32,
                                        );
                                        (*scratch.accum.get_unchecked_mut(0)).re = prev0.re + ar0 * br0;
                                        (*scratch.accum.get_unchecked_mut(0)).im = prev0.im + ai0 * bi0;
                                    }
                                }
                            }

                            scratch
                                .fft
                                .inverse_complex_half_to_real(&scratch.accum, &mut scratch.time);

                            for k in 0..freq_bins {
                                let val = scratch.time[k] * gain_comp;
                                out_slice[k] = val + overlap_slice[k];
                                overlap_slice[k] = scratch.time[k + freq_bins] * gain_comp;
                            }
                        }
                    });
            });
        }

        let res_size = self.num_outputs * self.body.block_size;
        let mut pushed = false;
        if self.body.result_producer.slots() >= res_size {
            if let Ok(mut chunk) = self.body.result_producer.write_chunk_uninit(res_size) {
                let (s1, s2) = chunk.as_mut_slices();
                let src = &self.body.scratch_result_push;

                if s1.len() > 0 {
                    unsafe {
                        let dst = s1.as_mut_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, s1.len());
                    }
                }
                if s2.len() > 0 {
                    unsafe {
                        let dst = s2.as_mut_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src[s1.len()..].as_ptr(), dst, s2.len());
                    }
                }
                unsafe {
                    chunk.commit_all();
                }
                pushed = true;
            }
        }
        if pushed {
            self.body_results_pushed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(target_os = "macos")]
impl BodyWorker {
    fn wait_for_signal_timeout(&self, timeout: Duration) -> bool {
        // Do not clear pending before blocking: that can suppress a wake edge and delay work.
        let signaled = self.signal_rx.recv_timeout(timeout).is_ok();
        self.signal_pending.store(false, Ordering::Release);
        signaled
    }

    fn handle_reset(&mut self, new_counter: u64) {
        self.last_reset_counter = new_counter;
        self.body.fdl_real.fill(0.0);
        self.body.fdl_imag.fill(0.0);
        self.body.saved_overlaps.fill(0.0);
        self.body.fdl_pos = self.body.num_partitions.saturating_sub(1);
        self.drain_body_task_consumer();
        self.reset_ack.store(new_counter, Ordering::Release);
    }

    fn drain_body_task_consumer(&mut self) {
        loop {
            let available = self.body.task_consumer.slots();
            if available == 0 {
                break;
            }
            if let Ok(chunk) = self.body.task_consumer.read_chunk(available) {
                chunk.commit_all();
            } else {
                break;
            }
        }
    }

    fn wait_and_process(&mut self) -> bool {
        let current_reset = self.reset_counter.load(Ordering::Acquire);
        if current_reset != self.last_reset_counter {
            self.handle_reset(current_reset);
        }

        let body_task_size = self.num_inputs * self.body.block_size;
        if self.body.task_consumer.slots() >= body_task_size {
            if let Ok(chunk) = self.body.task_consumer.read_chunk(body_task_size) {
                let (s1, s2) = chunk.as_slices();
                if !s1.is_empty() {
                    self.body.scratch_task_pop[0..s1.len()].copy_from_slice(s1);
                }
                if !s2.is_empty() {
                    self.body.scratch_task_pop[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
                }
                chunk.commit_all();
            }

            self.body_tasks_consumed.fetch_add(1, Ordering::Relaxed);
            self.process_body();
            return true;
        }

        false
    }

    fn run(&mut self) {
        self.body_worker_started.fetch_add(1, Ordering::Release);
        unsafe {
            // Restore high QoS so Body worker avoids being pushed to efficiency cores under load.
            let qos_class = libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE;
            libc::pthread_set_qos_class_self_np(qos_class, 0);
        }

        loop {
            if self.stop.load(Ordering::Acquire) {
                break;
            }

            if !self.wait_and_process() {
                // Double-check once to narrow the enqueue->sleep race window.
                if !self.wait_and_process() {
                    let _ = self
                        .wait_for_signal_timeout(Duration::from_micros(BODY_WORKER_IDLE_POLL_US));
                }
            }
        }
    }

    fn process_body(&mut self) {
        if self.body.num_partitions == 0 {
            self.body.scratch_result_push.fill(0.0);
            return;
        }

        let body = &mut self.body;
        let freq_bins = body.block_size;
        let fdl_stride_in = freq_bins;
        let fdl_stride_part = self.num_inputs * fdl_stride_in;

        body.fdl_pos = (body.fdl_pos + 1) % body.num_partitions;

        for ch in 0..self.num_inputs {
            let offset = ch * body.block_size;
            body.scratch_fft_input.fill(0.0);
            body.scratch_fft_input[0..body.block_size]
                .copy_from_slice(&body.scratch_task_pop[offset..offset + body.block_size]);

            let base = body.fdl_pos * fdl_stride_part + ch * fdl_stride_in;
            let mut split = DSPComplex {
                realp: body.scratch_complex_r.as_mut_ptr(),
                imagp: body.scratch_complex_i.as_mut_ptr(),
            };
            body.fft.forward(&body.scratch_fft_input, &mut split);

            unsafe {
                let r = std::slice::from_raw_parts(split.realp, body.block_size);
                let i = std::slice::from_raw_parts(split.imagp, body.block_size);
                body.fdl_real[base..base + body.block_size].copy_from_slice(r);
                body.fdl_imag[base..base + body.block_size].copy_from_slice(i);
            }
        }

        use crate::accelerate;

        let stride_out = freq_bins;
        let stride_in = self.num_outputs * stride_out;
        let stride_part = self.num_inputs * stride_in;

        let ir_real = &body.ir_real;
        let ir_imag = &body.ir_imag;
        let fdl_real = &body.fdl_real;
        let fdl_imag = &body.fdl_imag;
        let active_mask = &body.active_mask;
        let fdl_pos = body.fdl_pos;
        let num_parts = body.num_partitions;
        let num_inputs = self.num_inputs;
        let num_outputs = self.num_outputs;
        let mask_stride_in = num_outputs;
        let mask_stride_part = num_inputs * mask_stride_in;

        let gain_comp = 0.5f32;
        let outputs_per_task = body.parallel_outputs_per_task.max(1);
        let group_size = outputs_per_task * freq_bins;
        let thread_scratch = &body.thread_scratch;
        let run_parallel = body.parallel_outputs_per_task > 0;

        if !run_parallel {
            let scratch_cell = thread_scratch.get(0);
            let scratch = unsafe { &mut *scratch_cell.get() };

            for out_ch in 0..num_outputs {
                let out_base = out_ch * freq_bins;
                let out_slice = &mut body.scratch_result_push[out_base..out_base + freq_bins];
                let overlap_slice = &mut body.saved_overlaps[out_base..out_base + freq_bins];

                scratch.accum_r.fill(0.0);
                scratch.accum_i.fill(0.0);

                for in_ch in 0..num_inputs {
                    for p in 0..num_parts {
                        let mask_idx = p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                        if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                            continue;
                        }
                        let ir_base = p * stride_part + in_ch * stride_in + out_ch * stride_out;
                        let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                        let fdl_base = fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                        unsafe {
                            let ir_split = DSPComplex {
                                realp: ir_real.as_ptr().add(ir_base) as *mut f32,
                                imagp: ir_imag.as_ptr().add(ir_base) as *mut f32,
                            };
                            let x_split = DSPComplex {
                                realp: fdl_real.as_ptr().add(fdl_base) as *mut f32,
                                imagp: fdl_imag.as_ptr().add(fdl_base) as *mut f32,
                            };
                            let accum_split = DSPComplex {
                                realp: scratch.accum_r.as_mut_ptr(),
                                imagp: scratch.accum_i.as_mut_ptr(),
                            };
                            // Packed real spectrum: bin0 stores (DC, Nyquist) in (re, im).
                            // Keep bin0 as independent real products while fusing MAC for bins 1..N-1.
                            let ar0 = *ir_split.realp;
                            let ai0 = *ir_split.imagp;
                            let br0 = *x_split.realp;
                            let bi0 = *x_split.imagp;
                            let prev_re0 = *scratch.accum_r.get_unchecked(0);
                            let prev_im0 = *scratch.accum_i.get_unchecked(0);
                            accelerate::vDSP_zvma(
                                &ir_split,
                                1,
                                &x_split,
                                1,
                                &accum_split,
                                1,
                                &accum_split,
                                1,
                                freq_bins as crate::accelerate::vDSP_Length,
                            );
                            *scratch.accum_r.get_unchecked_mut(0) = prev_re0 + ar0 * br0;
                            *scratch.accum_i.get_unchecked_mut(0) = prev_im0 + ai0 * bi0;
                        }
                    }
                }

                let mut split = DSPComplex {
                    realp: scratch.accum_r.as_mut_ptr(),
                    imagp: scratch.accum_i.as_mut_ptr(),
                };
                scratch.fft.inverse(&mut split, &mut scratch.time);

                for k in 0..freq_bins {
                    let val = scratch.time[k] * gain_comp;
                    out_slice[k] = val + overlap_slice[k];
                    overlap_slice[k] = scratch.time[k + freq_bins] * gain_comp;
                }
            }
        } else {
            body.thread_pool.install(|| {
                body.scratch_result_push
                    .par_chunks_mut(group_size)
                    .zip(body.saved_overlaps.par_chunks_mut(group_size))
                    .enumerate()
                    .for_each(|(group_idx, (out_group, overlap_group))| {
                        let thread_index = rayon::current_thread_index().unwrap_or(0);
                        let scratch_cell = thread_scratch.get(thread_index);
                        let scratch = unsafe { &mut *scratch_cell.get() };

                        let base_out = group_idx * outputs_per_task;
                        let group_outputs = out_group.len() / freq_bins;

                        for g in 0..group_outputs {
                            let out_ch = base_out + g;
                            let out_offset = g * freq_bins;
                            let out_slice = &mut out_group[out_offset..out_offset + freq_bins];
                            let overlap_slice =
                                &mut overlap_group[out_offset..out_offset + freq_bins];

                            scratch.accum_r.fill(0.0);
                            scratch.accum_i.fill(0.0);

                            for in_ch in 0..num_inputs {
                                for p in 0..num_parts {
                                    let mask_idx =
                                        p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                                    if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                                        continue;
                                    }
                                    let ir_base =
                                        p * stride_part + in_ch * stride_in + out_ch * stride_out;
                                    let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                                    let fdl_base =
                                        fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                                    unsafe {
                                        let ir_split = DSPComplex {
                                            realp: ir_real.as_ptr().add(ir_base) as *mut f32,
                                            imagp: ir_imag.as_ptr().add(ir_base) as *mut f32,
                                        };
                                        let x_split = DSPComplex {
                                            realp: fdl_real.as_ptr().add(fdl_base) as *mut f32,
                                            imagp: fdl_imag.as_ptr().add(fdl_base) as *mut f32,
                                        };
                                        let accum_split = DSPComplex {
                                            realp: scratch.accum_r.as_mut_ptr(),
                                            imagp: scratch.accum_i.as_mut_ptr(),
                                        };
                                        // Packed real spectrum: bin0 stores (DC, Nyquist) in (re, im).
                                        // Keep bin0 as independent real products while fusing MAC for bins 1..N-1.
                                        let ar0 = *ir_split.realp;
                                        let ai0 = *ir_split.imagp;
                                        let br0 = *x_split.realp;
                                        let bi0 = *x_split.imagp;
                                        let prev_re0 = *scratch.accum_r.get_unchecked(0);
                                        let prev_im0 = *scratch.accum_i.get_unchecked(0);
                                        accelerate::vDSP_zvma(
                                            &ir_split,
                                            1,
                                            &x_split,
                                            1,
                                            &accum_split,
                                            1,
                                            &accum_split,
                                            1,
                                            freq_bins as crate::accelerate::vDSP_Length,
                                        );
                                        *scratch.accum_r.get_unchecked_mut(0) = prev_re0 + ar0 * br0;
                                        *scratch.accum_i.get_unchecked_mut(0) = prev_im0 + ai0 * bi0;
                                    }
                                }
                            }

                            let mut split = DSPComplex {
                                realp: scratch.accum_r.as_mut_ptr(),
                                imagp: scratch.accum_i.as_mut_ptr(),
                            };
                            scratch.fft.inverse(&mut split, &mut scratch.time);

                            for k in 0..freq_bins {
                                let val = scratch.time[k] * gain_comp;
                                out_slice[k] = val + overlap_slice[k];
                                overlap_slice[k] = scratch.time[k + freq_bins] * gain_comp;
                            }
                        }
                    });
            });
        }

        let res_size = self.num_outputs * body.block_size;
        let mut pushed = false;
        if body.result_producer.slots() >= res_size {
            if let Ok(mut chunk) = body.result_producer.write_chunk_uninit(res_size) {
                let (s1, s2) = chunk.as_mut_slices();
                let src = &body.scratch_result_push;

                if !s1.is_empty() {
                    unsafe {
                        let dst = s1.as_mut_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, s1.len());
                    }
                }
                if !s2.is_empty() {
                    unsafe {
                        let dst = s2.as_mut_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src[s1.len()..].as_ptr(), dst, s2.len());
                    }
                }
                unsafe {
                    chunk.commit_all();
                }
                pushed = true;
            }
        }
        if pushed {
            self.body_results_pushed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
struct BodyWorkerHandle {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

#[cfg(target_os = "windows")]
impl BodyWorkerHandle {
    fn new(
        num_inputs: usize,
        num_outputs: usize,
        body: BodyPipelineWorker,
        signal_rx: Receiver<()>,
        signal_pending: Arc<AtomicBool>,
        reset_counter: Arc<AtomicU64>,
        reset_ack: Arc<AtomicU64>,
        body_worker_started: Arc<AtomicU64>,
        body_tasks_consumed: Arc<AtomicU64>,
        body_results_pushed: Arc<AtomicU64>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let mut worker = BodyWorker {
            num_inputs,
            num_outputs,
            body,
            signal_rx,
            signal_pending,
            reset_counter,
            reset_ack,
            last_reset_counter: 0,
            stop: stop_thread,
            body_worker_started,
            body_tasks_consumed,
            body_results_pushed,
        };
        let join = std::thread::spawn(move || {
            worker.run();
        });
        Self {
            stop,
            join: Some(join),
        }
    }
}

#[cfg(target_os = "macos")]
impl BodyWorkerHandle {
    fn new(
        num_inputs: usize,
        num_outputs: usize,
        body: BodyPipelineWorker,
        signal_rx: Receiver<()>,
        signal_pending: Arc<AtomicBool>,
        reset_counter: Arc<AtomicU64>,
        reset_ack: Arc<AtomicU64>,
        body_worker_started: Arc<AtomicU64>,
        body_tasks_consumed: Arc<AtomicU64>,
        body_results_pushed: Arc<AtomicU64>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let mut worker = BodyWorker {
            num_inputs,
            num_outputs,
            body,
            signal_rx,
            signal_pending,
            reset_counter,
            reset_ack,
            last_reset_counter: 0,
            stop: stop_thread,
            body_worker_started,
            body_tasks_consumed,
            body_results_pushed,
        };
        let join = std::thread::spawn(move || {
            worker.run();
        });
        Self {
            stop,
            join: Some(join),
        }
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
impl Drop for BodyWorkerHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub struct MatrixAudioProcessor {
    engine_id: u64,
    engine: HeadEngine,
    num_inputs: usize,
    num_outputs: usize,
    block_size: usize,
    pub _head_len: usize, // Unused but kept for structure alignment or debug
    pub _tail_start: usize,
    tail_latency_blocks: usize,
    blocks_since_reset: usize,
    tail_blocks_consumed: usize,

    input_buffers: Vec<Vec<f32>>,
    input_pos: usize,

    task_producer: Producer<f32>,
    result_consumer: Consumer<f32>,

    overlap_buffers: Vec<Vec<f32>>,

    body: Option<BodyPipelineAudio>,
    body_parallel_threads: usize,
    body_parallel_outputs_per_task: usize,

    // Telemetry (Read Only)
    pub gpu_execution_time: Arc<std::sync::atomic::AtomicU64>,
    pub gpu_driver_time: Arc<std::sync::atomic::AtomicU64>,
    pub gpu_compute_time: Arc<std::sync::atomic::AtomicU64>,

    // Architecture Fixes
    signal_tx: Sender<()>,
    signal_pending: Arc<AtomicBool>,

    reset_counter: Arc<AtomicU64>,
    reset_ack: Arc<AtomicU64>,
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    body_signal_tx: Sender<()>,
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    body_signal_pending: Arc<AtomicBool>,
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    body_reset_ack: Arc<AtomicU64>,

    // Tail probe
    tail_probe: Option<TailProbe>,
    processed_samples: u64,
    tail_probe_reported: bool,

    // Underrun diagnostics (throttled logs)
    tail_underrun_count: u64,
    body_underrun_count: u64,
    last_tail_underrun_log_samples: u64,
    last_body_underrun_log_samples: u64,
    tail_task_drop_count: u64,
    body_task_drop_count: u64,
    // Cross-thread counters (all platforms)
    tail_tasks_pushed: Arc<AtomicU64>,
    tail_tasks_consumed: Arc<AtomicU64>,
    tail_results_pushed: Arc<AtomicU64>,
    body_tasks_pushed: Arc<AtomicU64>,
    body_tasks_consumed: Arc<AtomicU64>,
    body_results_pushed: Arc<AtomicU64>,
    body_worker_started: Arc<AtomicU64>,
}

pub struct MatrixWorker {
    tail: TailEngine,
    fft: Arc<FftContext>,

    num_inputs: usize,
    num_outputs: usize,
    block_size: usize,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    #[allow(dead_code)]
    body_block_size: usize,
    #[allow(dead_code)]
    body_num_partitions: usize,

    task_consumer: Consumer<f32>,
    result_producer: Producer<f32>,

    _scratch_fft_input: Vec<f32>,
    _scratch_complex_r: Vec<f32>,
    _scratch_complex_i: Vec<f32>,
    scratch_time: Vec<f32>,
    gpu_input_aggregate: Vec<f32>,

    scratch_task_pop: Vec<f32>,
    scratch_result_push: Vec<f32>,

    saved_overlaps: Vec<Vec<f32>>,

    // Cross-thread counters
    tail_tasks_consumed: Arc<AtomicU64>,
    tail_results_pushed: Arc<AtomicU64>,
    body_tasks_consumed: Arc<AtomicU64>,
    body_results_pushed: Arc<AtomicU64>,

    body: Option<BodyPipelineWorker>,
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    _body_thread: Option<BodyWorkerHandle>,

    // Telemetry (Write)
    gpu_execution_time: Arc<std::sync::atomic::AtomicU64>,
    gpu_driver_time: Arc<std::sync::atomic::AtomicU64>,
    gpu_compute_time: Arc<std::sync::atomic::AtomicU64>,

    signal_rx: Receiver<()>,
    signal_pending: Arc<AtomicBool>,

    reset_counter: Arc<AtomicU64>,
    reset_ack: Arc<AtomicU64>,
    last_reset_counter: u64,
}

impl MatrixScheduler {
    fn resolve_body_parallel(num_outputs: usize) -> (usize, usize) {
        if num_outputs == 0 {
            return (1, 1);
        }

        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut threads = available.saturating_sub(1).max(1);

        threads = threads.min(num_outputs);
        threads = threads.min(BODY_MAX_THREADS).max(1);

        #[cfg(target_os = "windows")]
        let mut outputs_per_task = BODY_OUTPUTS_PER_TASK.min(num_outputs).max(1);
        #[cfg(not(target_os = "windows"))]
        let outputs_per_task = BODY_OUTPUTS_PER_TASK.min(num_outputs).max(1);

        let tasks = (num_outputs + outputs_per_task - 1) / outputs_per_task;
        threads = threads.min(tasks).max(1);

        #[cfg(target_os = "windows")]
        {
            outputs_per_task = (num_outputs + threads - 1) / threads;
        }

        (threads, outputs_per_task)
    }

    pub fn new(
        context: Option<Arc<MetalContext>>,
        matrix_ir_data: &[f32],
        body_ir_real: Vec<f32>,
        body_ir_imag: Vec<f32>,
        body_ir_mask: Vec<u8>,
        head_matrix: &[Vec<Vec<f32>>], // List of Rows. Row = Input. Element = Output IR.
        num_inputs: usize,
        num_outputs: usize,
        block_size: usize,
        head_len: usize,
        body_block_size: usize,
        body_len: usize,
        tail_start: usize,
        max_expected_buffer: usize,
        engine_type: EngineType,
    ) -> Option<Self> {
        #[cfg(all(
            target_os = "windows",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if !std::arch::is_x86_feature_detected!("avx2") {
                eprintln!("[MatrixScheduler] AVX2 is required on Windows.");
                return None;
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = max_expected_buffer;
        let fft_size = block_size * 2;
        let fft = Arc::new(FftContext::new(fft_size));

        // Dual Engine Selection
        let (head_engine, tail_engine) = match engine_type {
            EngineType::Titan => {
                println!("馃殌 [鐭╅樀璋冨害] 鍒濆鍖?Titan 娣峰悎寮曟搸锛堢湡鐭╅樀锛?..");
                let mut rows = Vec::with_capacity(num_inputs);

                for (_in_ch, row_irs) in head_matrix.iter().enumerate() {
                    // row_irs is Vec<Vec<f32>> (The IRs for this input to all outputs)
                    let mut ir_slices = Vec::with_capacity(num_outputs);

                    for out_ch in 0..num_outputs {
                        // Safety: Handle if row_irs has fewer outputs than num_outputs
                        let head_slice = if out_ch < row_irs.len() {
                            let src_ir = &row_irs[out_ch];
                            let actual_len = head_len.min(src_ir.len());
                            &src_ir[0..actual_len]
                        } else {
                            &[] // Empty slice for missing channels implies silence in TitanHeadRow logic?
                                // TitanHeadRow::new expects list of slices.
                                // We should provide a slice to a zero buffer if empty?
                                // Actually TitanHeadRow handles short slices by zero padding in its internal buffer.
                                // An empty slice is valid.
                        };
                        ir_slices.push(head_slice);
                    }
                    #[cfg(target_os = "windows")]
                    {
                        rows.push(TitanHeadRow::new(&ir_slices, head_len, max_expected_buffer));
                    }
                    #[cfg(not(target_os = "windows"))]
                    {
                        rows.push(TitanHeadRow::new(&ir_slices, head_len));
                    }
                }

                let active_inputs: Vec<u32> = (0..num_inputs as u32).collect();
                let tail = GpuTailProcessor::new(
                    context
                        .as_ref()
                        .expect("Titan Engine requires GPU Context!")
                        .clone(),
                    num_inputs,
                    num_outputs,
                    block_size,
                    matrix_ir_data,
                    &active_inputs,
                )?;
                (HeadEngine::Titan(rows), TailEngine::Gpu(tail))
            }
            EngineType::Vdsp => {
                println!("馃殌 [MatrixScheduler] Initializing vDSP Engine (True Matrix)...");

                #[cfg(target_os = "macos")]
                {
                    let mut rows = Vec::with_capacity(num_inputs);
                    for (_in_ch, row_irs) in head_matrix.iter().enumerate() {
                        let mut processed_row = Vec::with_capacity(num_outputs);
                        for out_ch in 0..num_outputs {
                            if out_ch < row_irs.len() {
                                let src_ir = &row_irs[out_ch];
                                let actual_len = head_len.min(src_ir.len());
                                processed_row.push(src_ir[0..actual_len].to_vec());
                            } else {
                                processed_row.push(vec![0.0; 0]);
                            }
                        }
                        rows.push(crate::dsp::head_vdsp::HeadVdsp::new(
                            &processed_row,
                            max_expected_buffer,
                        ));
                    }

                    let active_inputs: Vec<u32> = (0..num_inputs as u32).collect();
                    let tail = GpuTailProcessor::new(
                        context
                            .as_ref()
                            .expect("vDSP Engine requires GPU Context!")
                            .clone(),
                        num_inputs,
                        num_outputs,
                        block_size,
                        matrix_ir_data,
                        &active_inputs,
                    )?;
                    (HeadEngine::Vdsp(rows), TailEngine::Gpu(tail))
                }

                #[cfg(not(target_os = "macos"))]
                {
                    panic!("vDSP Engine is not supported on this platform!");
                }
            }
        };

        let input_buffers = vec![vec![0.0; block_size]; num_inputs];
        let overlap_buffers = vec![vec![0.0; block_size]; num_outputs];
        let saved_overlaps = vec![vec![0.0; block_size]; num_outputs];

        Some(Self {
            engine: head_engine,
            tail: tail_engine,
            fft,
            num_inputs,
            num_outputs,
            block_size,
            body_block_size,
            body_len,
            tail_start,
            input_buffers,
            input_pos: 0,
            overlap_buffers,
            saved_overlaps,
            scratch_fft_input: vec![0.0; fft_size],
            scratch_complex_r: vec![0.0; block_size],
            scratch_complex_i: vec![0.0; block_size],
            gpu_input_aggregate: vec![0.0f32; num_inputs * block_size * 2],
            scratch_time_domain: vec![0.0f32; fft_size],
            body_ir_real,
            body_ir_imag,
            body_ir_mask,
            gpu_execution_time: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            gpu_driver_time: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            gpu_compute_time: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            _head_len: head_len,
            _tail_start: tail_start,
        })
    }

    /// 鍚屾鎿嶄綔 (鐢ㄤ簬鏃ф祴璇?
    pub fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) {
        let n_samples = inputs[0].len();

        // Clear Outputs
        for out in outputs.iter_mut() {
            out.fill(0.0);
        }

        // 1. Head
        match &mut self.engine {
            HeadEngine::Titan(rows) => {
                // Titan Mode: Row-based broadcasting
                for (in_ch, row) in rows.iter_mut().enumerate() {
                    if in_ch < inputs.len() {
                        row.process(inputs[in_ch], outputs);
                    }
                }
            }
            HeadEngine::Vdsp(rows) => {
                // vDSP Row Mode: One call per Input
                for (in_ch, row) in rows.iter_mut().enumerate() {
                    if in_ch < inputs.len() {
                        row.process(inputs[in_ch], outputs);
                    }
                }
            }
        }

        // 2. Tail Logic
        for s in 0..n_samples {
            for out_ch in 0..self.num_outputs {
                outputs[out_ch][s] += self.overlap_buffers[out_ch][self.input_pos];
            }
            for in_ch in 0..self.num_inputs {
                self.input_buffers[in_ch][self.input_pos] = inputs[in_ch][s];
            }
            self.input_pos += 1;

            if self.input_pos == self.block_size {
                self.process_matrix_block_sync();
                self.input_pos = 0;
            }
        }
    }

    fn process_matrix_block_sync(&mut self) {
        // [Same Sync Logic as before]
        let mut split = DSPComplex {
            realp: self.scratch_complex_r.as_mut_ptr(),
            imagp: self.scratch_complex_i.as_mut_ptr(),
        };

        for ch in 0..self.num_inputs {
            self.scratch_fft_input.fill(0.0);
            self.scratch_fft_input[0..self.block_size].copy_from_slice(&self.input_buffers[ch]);
            self.fft.forward(&self.scratch_fft_input, &mut split);

            let offset = ch * self.block_size * 2;
            unsafe {
                let r = std::slice::from_raw_parts(split.realp, self.block_size);
                let i = std::slice::from_raw_parts(split.imagp, self.block_size);
                for k in 0..self.block_size {
                    self.gpu_input_aggregate[offset + k * 2] = r[k];
                    self.gpu_input_aggregate[offset + k * 2 + 1] = i[k];
                }
            }
        }

        let gpu_output_slice = self.tail.process(&self.gpu_input_aggregate);

        #[cfg(target_os = "macos")]
        let gain_comp = 0.5;
        #[cfg(not(target_os = "macos"))]
        let gain_comp = 1.0;
        for ch in 0..self.num_outputs {
            let offset = ch * self.block_size * 2;
            unsafe {
                let r = std::slice::from_raw_parts_mut(split.realp, self.block_size);
                let i = std::slice::from_raw_parts_mut(split.imagp, self.block_size);
                for k in 0..self.block_size {
                    r[k] = gpu_output_slice[offset + k * 2];
                    i[k] = gpu_output_slice[offset + k * 2 + 1];
                }
            }
            self.fft.inverse(&mut split, &mut self.scratch_time_domain);
            for k in 0..self.block_size {
                self.overlap_buffers[ch][k] =
                    self.scratch_time_domain[k] * gain_comp + self.saved_overlaps[ch][k];
                self.saved_overlaps[ch][k] =
                    self.scratch_time_domain[k + self.block_size] * gain_comp;
            }
        }
    }

    /// 馃敟 杞负寮傛妯″紡 (Consumed self to split)
    pub fn split_into_async(self) -> (MatrixAudioProcessor, MatrixWorker) {
        self.split_into_async_with_probe(None)
    }

    pub fn split_into_async_with_probe(
        self,
        tail_probe: Option<TailProbe>,
    ) -> (MatrixAudioProcessor, MatrixWorker) {
        // Create RingBuffers (Capacity scales with max(inputs, outputs), with a safety floor)
        let capacity_channels = self.num_inputs.max(self.num_outputs);
        let base_capacity = 16 * capacity_channels * self.block_size;
        let min_capacity = 65536; // floats (safety floor for low-channel cases)
        let capacity = base_capacity.max(min_capacity);
        // We need capacity to store FLATTENED chunks.
        // Task Chunk = num_inputs * block_size
        // Result Chunk = num_outputs * block_size
        // Let's use large enough buffer.
        let (task_prod, task_cons) = RingBuffer::new(capacity * 4);
        let (res_prod, res_cons) = RingBuffer::new(capacity * 4);

        println!(
            "馃敡 [璋冨害鍣╙ 鍒涘缓鐜舰缂撳啿: 瀹归噺={} (鍩哄噯={})",
            capacity * 4,
            capacity
        );

        // 鉁?Tail 寤惰繜鍧楁暟 = TailStart / BlockSize
        let tail_latency_blocks = if self.block_size > 0 {
            self.tail_start / self.block_size
        } else {
            0
        };

        // 鉁?Body (Optional)
        let body_enabled =
            self.body_len > 0 && self.body_block_size > 0 && !self.body_ir_real.is_empty();

        #[cfg(target_os = "windows")]
        let (
            body_audio,
            mut body_worker,
            body_block_size,
            body_num_partitions,
            body_parallel_threads,
            body_parallel_outputs_per_task,
        ) = if body_enabled {
            let body_capacity_channels = self.num_inputs.max(self.num_outputs);
            let body_base_capacity = 16 * body_capacity_channels * self.body_block_size;
            let body_min_capacity = 65536; // floats
            let body_capacity = body_base_capacity.max(body_min_capacity);
            let (body_task_prod, body_task_cons) = RingBuffer::new(body_capacity * 4);
            let (body_res_prod, body_res_cons) = RingBuffer::new(body_capacity * 4);

            let body_latency_blocks = if self.body_block_size > 0 {
                self._head_len / self.body_block_size
            } else {
                0
            };

            let body_num_partitions =
                (self.body_len + self.body_block_size - 1) / self.body_block_size;

            let body_ir_real = self.body_ir_real;
            let body_ir_imag = self.body_ir_imag;
            let body_ir_mask = self.body_ir_mask;

            let audio = BodyPipelineAudio {
                block_size: self.body_block_size,
                latency_blocks: body_latency_blocks,
                blocks_since_reset: 0,
                blocks_consumed: 0,
                input_buffers: vec![vec![0.0; self.body_block_size]; self.num_inputs],
                input_pos: 0,
                task_producer: body_task_prod,
                result_consumer: body_res_cons,
                overlap_buffers: vec![0.0; self.num_outputs * self.body_block_size],
            };

            let (threads, outputs_per_task) = Self::resolve_body_parallel(self.num_outputs);
            let worker = BodyPipelineWorker {
                block_size: self.body_block_size,
                num_partitions: body_num_partitions,
                task_consumer: body_task_cons,
                result_producer: body_res_prod,
                #[cfg(target_os = "macos")]
                ir_real: body_ir_real,
                #[cfg(target_os = "macos")]
                ir_imag: body_ir_imag,
                #[cfg(target_os = "windows")]
                ir_complex: {
                    let mut v = Vec::with_capacity(body_ir_real.len());
                    for i in 0..body_ir_real.len() {
                        v.push(ipp::Ipp32fc {
                            re: body_ir_real[i],
                            im: body_ir_imag[i],
                        });
                    }
                    v
                },
                active_mask: body_ir_mask,
                #[cfg(target_os = "macos")]
                fdl_real: vec![0.0; body_num_partitions * self.num_inputs * self.body_block_size],
                #[cfg(target_os = "macos")]
                fdl_imag: vec![0.0; body_num_partitions * self.num_inputs * self.body_block_size],
                #[cfg(target_os = "windows")]
                fdl_complex: vec![
                    ipp::Ipp32fc { re: 0.0, im: 0.0 };
                    body_num_partitions * self.num_inputs * self.body_block_size
                ],
                fdl_pos: body_num_partitions.saturating_sub(1),
                fft: FftContext::new(self.body_block_size * 2),
                scratch_fft_input: vec![0.0; self.body_block_size * 2],
                #[cfg(target_os = "macos")]
                scratch_complex_r: vec![0.0; self.body_block_size],
                #[cfg(target_os = "macos")]
                scratch_complex_i: vec![0.0; self.body_block_size],
                scratch_task_pop: vec![0.0; self.num_inputs * self.body_block_size],
                scratch_result_push: vec![0.0; self.num_outputs * self.body_block_size],
                saved_overlaps: vec![0.0; self.num_outputs * self.body_block_size],
                parallel_outputs_per_task: outputs_per_task,
                thread_pool: ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap(),
                thread_scratch: ThreadScratchPool::new(threads, self.body_block_size),
            };

            (
                Some(audio),
                Some(worker),
                self.body_block_size,
                body_num_partitions,
                threads,
                outputs_per_task,
            )
        } else {
            (None, None, 0, 0, 0, 0)
        };
        #[cfg(not(target_os = "windows"))]
        let (
            body_audio,
            mut body_worker,
            body_block_size,
            body_num_partitions,
            body_parallel_threads,
            body_parallel_outputs_per_task,
        ) = if body_enabled {
            let body_capacity_channels = self.num_inputs.max(self.num_outputs);
            let body_base_capacity = 16 * body_capacity_channels * self.body_block_size;
            let body_min_capacity = 65536; // floats
            let body_capacity = body_base_capacity.max(body_min_capacity);
            let (body_task_prod, body_task_cons) = RingBuffer::new(body_capacity * 4);
            let (body_res_prod, body_res_cons) = RingBuffer::new(body_capacity * 4);

            let body_latency_blocks = if self.body_block_size > 0 {
                self._head_len / self.body_block_size
            } else {
                0
            };

            let body_num_partitions =
                (self.body_len + self.body_block_size - 1) / self.body_block_size;

            let body_ir_real = self.body_ir_real;
            let body_ir_imag = self.body_ir_imag;
            let body_ir_mask = self.body_ir_mask;

            let audio = BodyPipelineAudio {
                block_size: self.body_block_size,
                latency_blocks: body_latency_blocks,
                blocks_since_reset: 0,
                blocks_consumed: 0,
                input_buffers: vec![vec![0.0; self.body_block_size]; self.num_inputs],
                input_pos: 0,
                task_producer: body_task_prod,
                result_consumer: body_res_cons,
                overlap_buffers: vec![0.0; self.num_outputs * self.body_block_size],
            };

            let (threads, outputs_per_task) = Self::resolve_body_parallel(self.num_outputs);
            let worker = BodyPipelineWorker {
                block_size: self.body_block_size,
                num_partitions: body_num_partitions,
                task_consumer: body_task_cons,
                result_producer: body_res_prod,
                #[cfg(target_os = "macos")]
                ir_real: body_ir_real,
                #[cfg(target_os = "macos")]
                ir_imag: body_ir_imag,
                #[cfg(target_os = "windows")]
                ir_complex: {
                    let mut v = Vec::with_capacity(body_ir_real.len());
                    for i in 0..body_ir_real.len() {
                        v.push(ipp::Ipp32fc {
                            re: body_ir_real[i],
                            im: body_ir_imag[i],
                        });
                    }
                    v
                },
                active_mask: body_ir_mask,
                #[cfg(target_os = "macos")]
                fdl_real: vec![0.0; body_num_partitions * self.num_inputs * self.body_block_size],
                #[cfg(target_os = "macos")]
                fdl_imag: vec![0.0; body_num_partitions * self.num_inputs * self.body_block_size],
                #[cfg(target_os = "windows")]
                fdl_complex: vec![
                    ipp::Ipp32fc { re: 0.0, im: 0.0 };
                    body_num_partitions * self.num_inputs * self.body_block_size
                ],
                fdl_pos: body_num_partitions.saturating_sub(1),
                fft: FftContext::new(self.body_block_size * 2),
                scratch_fft_input: vec![0.0; self.body_block_size * 2],
                #[cfg(target_os = "macos")]
                scratch_complex_r: vec![0.0; self.body_block_size],
                #[cfg(target_os = "macos")]
                scratch_complex_i: vec![0.0; self.body_block_size],
                scratch_task_pop: vec![0.0; self.num_inputs * self.body_block_size],
                scratch_result_push: vec![0.0; self.num_outputs * self.body_block_size],
                saved_overlaps: vec![0.0; self.num_outputs * self.body_block_size],
                parallel_outputs_per_task: outputs_per_task,
                thread_pool: ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap(),
                thread_scratch: ThreadScratchPool::new(threads, self.body_block_size),
            };

            (
                Some(audio),
                Some(worker),
                self.body_block_size,
                body_num_partitions,
                threads,
                outputs_per_task,
            )
        } else {
            (None, None, 0, 0, 0, 0)
        };

        // 馃敡 Engine identity + counters (shared across threads)
        let engine_id = ENGINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tail_tasks_pushed = Arc::new(AtomicU64::new(0));
        let tail_tasks_consumed = Arc::new(AtomicU64::new(0));
        let tail_results_pushed = Arc::new(AtomicU64::new(0));
        let body_tasks_pushed = Arc::new(AtomicU64::new(0));
        let body_tasks_consumed = Arc::new(AtomicU64::new(0));
        let body_results_pushed = Arc::new(AtomicU64::new(0));
        let body_worker_started = Arc::new(AtomicU64::new(0));

        // 馃敡 Wait-Free Signal Channel (Bounded 1)
        let (sig_tx, sig_rx) = crossbeam_channel::bounded(1);
        let signal_pending = Arc::new(AtomicBool::new(false));
        let reset_counter = Arc::new(AtomicU64::new(0));
        let reset_ack = Arc::new(AtomicU64::new(0));
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        let (body_sig_tx, body_sig_rx) = crossbeam_channel::bounded(1);
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        let body_signal_pending = Arc::new(AtomicBool::new(false));
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        let body_reset_ack = Arc::new(AtomicU64::new(0));

        let tail_probe_reported = tail_probe.is_none();
        let audio = MatrixAudioProcessor {
            engine_id,
            engine: self.engine,
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            block_size: self.block_size,
            _head_len: self._head_len,
            _tail_start: self._tail_start,
            tail_latency_blocks,
            blocks_since_reset: 0,
            tail_blocks_consumed: 0,
            input_buffers: self.input_buffers,
            input_pos: 0,
            task_producer: task_prod,
            result_consumer: res_cons,
            overlap_buffers: self.overlap_buffers,
            body: body_audio,
            body_parallel_threads,
            body_parallel_outputs_per_task,
            gpu_execution_time: self.gpu_execution_time.clone(),
            gpu_driver_time: self.gpu_driver_time.clone(),
            gpu_compute_time: self.gpu_compute_time.clone(),
            signal_tx: sig_tx,
            signal_pending: signal_pending.clone(),
            reset_counter: reset_counter.clone(),
            reset_ack: reset_ack.clone(),
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            body_signal_tx: body_sig_tx,
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            body_signal_pending: body_signal_pending.clone(),
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            body_reset_ack: body_reset_ack.clone(),
            tail_probe,
            processed_samples: 0,
            tail_probe_reported,
            tail_underrun_count: 0,
            body_underrun_count: 0,
            last_tail_underrun_log_samples: 0,
            last_body_underrun_log_samples: 0,
            tail_task_drop_count: 0,
            body_task_drop_count: 0,
            tail_tasks_pushed: tail_tasks_pushed.clone(),
            tail_tasks_consumed: tail_tasks_consumed.clone(),
            tail_results_pushed: tail_results_pushed.clone(),
            body_tasks_pushed: body_tasks_pushed.clone(),
            body_tasks_consumed: body_tasks_consumed.clone(),
            body_results_pushed: body_results_pushed.clone(),
            body_worker_started: body_worker_started.clone(),
        };

        #[cfg(any(target_os = "windows", target_os = "macos"))]
        let body_thread = if let Some(body_worker) = body_worker.take() {
            Some(BodyWorkerHandle::new(
                self.num_inputs,
                self.num_outputs,
                body_worker,
                body_sig_rx,
                body_signal_pending.clone(),
                reset_counter.clone(),
                body_reset_ack.clone(),
                body_worker_started.clone(),
                body_tasks_consumed.clone(),
                body_results_pushed.clone(),
            ))
        } else {
            None
        };

        let worker = MatrixWorker {
            tail: self.tail,
            fft: self.fft,
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            block_size: self.block_size,
            body_block_size,
            body_num_partitions,
            task_consumer: task_cons,
            result_producer: res_prod,
            _scratch_fft_input: self.scratch_fft_input,
            _scratch_complex_r: self.scratch_complex_r,
            _scratch_complex_i: self.scratch_complex_i,
            scratch_time: self.scratch_time_domain,
            gpu_input_aggregate: self.gpu_input_aggregate,
            saved_overlaps: self.saved_overlaps,
            tail_tasks_consumed: tail_tasks_consumed.clone(),
            tail_results_pushed: tail_results_pushed.clone(),
            body_tasks_consumed: body_tasks_consumed.clone(),
            body_results_pushed: body_results_pushed.clone(),
            body: body_worker,
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            _body_thread: body_thread,
            // New scratches
            scratch_task_pop: vec![0.0; self.num_inputs * self.block_size],
            scratch_result_push: vec![0.0; self.num_outputs * self.block_size],
            gpu_execution_time: self.gpu_execution_time,
            gpu_driver_time: self.gpu_driver_time,
            gpu_compute_time: self.gpu_compute_time,
            signal_rx: sig_rx,
            signal_pending,
            reset_counter,
            reset_ack,
            last_reset_counter: 0,
        };

        (audio, worker)
    }
}

// --- Implementation of Async Components ---

impl MatrixAudioProcessor {
    pub fn diag_engine_id(&self) -> u64 {
        self.engine_id
    }
    pub fn diag_counts(&self) -> (u64, u64, u64, u64) {
        (
            self.tail_underrun_count,
            self.body_underrun_count,
            self.tail_task_drop_count,
            self.body_task_drop_count,
        )
    }
    pub fn diag_body_slots(&self) -> Option<(usize, usize)> {
        self.body
            .as_ref()
            .map(|body| (body.task_producer.slots(), body.result_consumer.slots()))
    }
    pub fn diag_body_parallel_profile(&self) -> (usize, usize) {
        (
            self.body_parallel_threads,
            self.body_parallel_outputs_per_task,
        )
    }
    pub fn diag_tail_slots(&self) -> (usize, usize) {
        (self.task_producer.slots(), self.result_consumer.slots())
    }
    pub fn diag_gpu_times_us(&self) -> (u64, u64, u64) {
        (
            self.gpu_execution_time.load(Ordering::Relaxed),
            self.gpu_driver_time.load(Ordering::Relaxed),
            self.gpu_compute_time.load(Ordering::Relaxed),
        )
    }
    pub fn diag_pipeline_layout(&self) -> (usize, usize, usize, usize, usize, usize) {
        let (body_block_size, body_latency_blocks) = self
            .body
            .as_ref()
            .map(|body| (body.block_size, body.latency_blocks))
            .unwrap_or((0, 0));
        (
            self.block_size,
            self.tail_latency_blocks,
            self._head_len,
            self._tail_start,
            body_block_size,
            body_latency_blocks,
        )
    }
    pub fn diag_task_counts(&self) -> (u64, u64, u64, u64, u64, u64) {
        (
            self.tail_tasks_pushed.load(Ordering::Relaxed),
            self.tail_tasks_consumed.load(Ordering::Relaxed),
            self.tail_results_pushed.load(Ordering::Relaxed),
            self.body_tasks_pushed.load(Ordering::Relaxed),
            self.body_tasks_consumed.load(Ordering::Relaxed),
            self.body_results_pushed.load(Ordering::Relaxed),
        )
    }
    pub fn diag_body_worker(&self) -> (u64, u64, u64, u64) {
        (
            self.body_worker_started.load(Ordering::Relaxed),
            self.body_tasks_pushed.load(Ordering::Relaxed),
            self.body_tasks_consumed.load(Ordering::Relaxed),
            self.body_results_pushed.load(Ordering::Relaxed),
        )
    }
    #[inline(always)]
    fn log_tail_underrun(&mut self, needed: usize, read_count: usize) {
        if needed <= read_count {
            return;
        }
        self.tail_underrun_count = self.tail_underrun_count.saturating_add(1);
        // Throttle: log at most once per ~256 blocks
        let throttle = (self.block_size as u64).saturating_mul(256);
        if self
            .processed_samples
            .saturating_sub(self.last_tail_underrun_log_samples)
            >= throttle
        {
            println!(
                "鈿狅笍 [TailUnderrun] needed={} read={} count={}",
                needed, read_count, self.tail_underrun_count
            );
            self.last_tail_underrun_log_samples = self.processed_samples;
        }
    }

    #[inline(always)]
    fn log_body_underrun(
        &mut self,
        needed: usize,
        read_count: usize,
        body_task_slots: usize,
        body_result_slots: usize,
    ) {
        if needed <= read_count {
            return;
        }
        self.body_underrun_count = self.body_underrun_count.saturating_add(1);
        // Throttle: log at most once per ~256 blocks
        let throttle = (self.block_size as u64).saturating_mul(256);
        if self
            .processed_samples
            .saturating_sub(self.last_body_underrun_log_samples)
            >= throttle
        {
            println!(
                "鈿狅笍 [BodyUnderrun] needed={} read={} count={} task_slots={} result_slots={} task_drop={}",
                needed,
                read_count,
                self.body_underrun_count,
                body_task_slots,
                body_result_slots,
                self.body_task_drop_count
            );
            self.last_body_underrun_log_samples = self.processed_samples;
        }
    }
    #[inline(always)]
    fn signal_worker(&self) {
        if !self.signal_pending.swap(true, Ordering::AcqRel) {
            let _ = self.signal_tx.try_send(());
        }
    }

    #[inline(always)]
    fn signal_worker_force(&self) {
        self.signal_pending.store(true, Ordering::Release);
        let _ = self.signal_tx.try_send(());
    }

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    #[inline(always)]
    fn signal_body_worker_force(&self) {
        self.body_signal_pending.store(true, Ordering::Release);
        let _ = self.body_signal_tx.try_send(());
    }
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    pub fn reset_state(&mut self) {
        self.input_pos = 0;
        self.blocks_since_reset = 0;
        self.tail_blocks_consumed = 0;

        for buf in &mut self.input_buffers {
            buf.fill(0.0);
        }
        for buf in &mut self.overlap_buffers {
            buf.fill(0.0);
        }

        if let Some(body) = &mut self.body {
            body.input_pos = 0;
            body.blocks_since_reset = 0;
            body.blocks_consumed = 0;
            for buf in &mut body.input_buffers {
                buf.fill(0.0);
            }
            body.overlap_buffers.fill(0.0);
        }

        match &mut self.engine {
            HeadEngine::Titan(rows) => {
                for row in rows.iter_mut() {
                    row.reset_state();
                }
            }
            HeadEngine::Vdsp(rows) => {
                for row in rows.iter_mut() {
                    row.reset_state();
                }
            }
        }

        self.reset_counter.fetch_add(1, Ordering::SeqCst);
        self.signal_worker_force();
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        if self.body.is_some() {
            self.signal_body_worker_force();
        }

        self.drain_result_consumer();
        self.drain_body_result_consumer();
        self.processed_samples = 0;
        self.tail_probe_reported = self.tail_probe.is_none();
        self.tail_underrun_count = 0;
        self.body_underrun_count = 0;
        self.tail_task_drop_count = 0;
        self.body_task_drop_count = 0;
        self.last_tail_underrun_log_samples = 0;
        self.last_body_underrun_log_samples = 0;
        #[cfg(target_os = "windows")]
        {}
    }

    fn drain_result_consumer(&mut self) {
        loop {
            let available = self.result_consumer.slots();
            if available == 0 {
                break;
            }
            if let Ok(chunk) = self.result_consumer.read_chunk(available) {
                chunk.commit_all();
            } else {
                break;
            }
        }
    }

    fn drain_body_result_consumer(&mut self) {
        if let Some(body) = &mut self.body {
            loop {
                let available = body.result_consumer.slots();
                if available == 0 {
                    break;
                }
                if let Ok(chunk) = body.result_consumer.read_chunk(available) {
                    chunk.commit_all();
                } else {
                    break;
                }
            }
        }
    }

    fn fetch_tail_result(&mut self, needed: usize) -> usize {
        let res_chunk_size = self.num_outputs * self.block_size;
        if needed == 0 {
            for buf in &mut self.overlap_buffers {
                buf.fill(0.0);
            }
            return 0;
        }

        let mut read_count = 0;
        for _ in 0..needed {
            if let Ok(chunk) = self.result_consumer.read_chunk(res_chunk_size) {
                read_count += 1;

                let (s1, s2) = chunk.as_slices();
                let mut src_offset = 0;

                for out_ch in 0..self.num_outputs {
                    let dst = &mut self.overlap_buffers[out_ch];
                    let mut dst_offset = 0;
                    let mut to_copy = self.block_size;

                    // Copy from s1
                    let s1_remaining = s1.len().saturating_sub(src_offset);
                    if s1_remaining > 0 {
                        let chunk_size = to_copy.min(s1_remaining);
                        dst[dst_offset..dst_offset + chunk_size]
                            .copy_from_slice(&s1[src_offset..src_offset + chunk_size]);

                        src_offset += chunk_size;
                        dst_offset += chunk_size;
                        to_copy -= chunk_size;
                    }

                    // Copy from s2
                    if to_copy > 0 {
                        let s2_offset_local = src_offset - s1.len();
                        let s2_remaining = s2.len().saturating_sub(s2_offset_local);
                        let chunk_size = to_copy.min(s2_remaining);
                        if chunk_size > 0 {
                            dst[dst_offset..dst_offset + chunk_size].copy_from_slice(
                                &s2[s2_offset_local..s2_offset_local + chunk_size],
                            );
                            src_offset += chunk_size;
                        }
                    }
                }
                chunk.commit_all();
            } else {
                break;
            }
        }

        if read_count == 0 {
            for buf in &mut self.overlap_buffers {
                buf.fill(0.0);
            }
        }
        read_count
    }

    fn fetch_body_result_from(
        body: &mut BodyPipelineAudio,
        num_outputs: usize,
        needed: usize,
    ) -> usize {
        let res_chunk_size = num_outputs * body.block_size;
        if needed == 0 {
            body.overlap_buffers.fill(0.0);
            return 0;
        }

        let mut read_count = 0;
        for _ in 0..needed {
            if let Ok(chunk) = body.result_consumer.read_chunk(res_chunk_size) {
                read_count += 1;
                let (s1, s2) = chunk.as_slices();
                let total = res_chunk_size;

                if s1.len() > 0 {
                    let count = s1.len().min(total);
                    body.overlap_buffers[0..count].copy_from_slice(&s1[0..count]);
                }
                if s2.len() > 0 {
                    let offset = s1.len();
                    let count = s2.len().min(total.saturating_sub(offset));
                    if count > 0 {
                        body.overlap_buffers[offset..offset + count].copy_from_slice(&s2[0..count]);
                    }
                }

                chunk.commit_all();
            } else {
                break;
            }
        }

        if read_count == 0 {
            body.overlap_buffers.fill(0.0);
        }
        read_count
    }

    #[inline(always)]
    fn trigger_worker_task(&mut self) -> bool {
        let task_size = self.num_inputs * self.block_size;
        if self.task_producer.slots() >= task_size {
            if let Ok(mut chunk) = self.task_producer.write_chunk_uninit(task_size) {
                let (s1, s2) = chunk.as_mut_slices();
                let mut rb_offset = 0;

                for buf in self.input_buffers.iter().take(self.num_inputs) {
                    let src = &buf[0..self.block_size];
                    let mut src_offset = 0;
                    let mut remaining = self.block_size;

                    // Write to s1
                    let s1_avail = s1.len().saturating_sub(rb_offset);
                    if s1_avail > 0 {
                        let count = remaining.min(s1_avail);
                        unsafe {
                            let dst = s1.as_mut_ptr().add(rb_offset) as *mut f32; // Explicit Cast
                            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, count);
                        }
                        rb_offset += count;
                        src_offset += count;
                        remaining -= count;
                    }

                    // Write to s2
                    if remaining > 0 {
                        let s2_rb_offset = rb_offset - s1.len();
                        let s2_avail = s2.len().saturating_sub(s2_rb_offset);
                        let count = remaining.min(s2_avail);
                        if count > 0 {
                            unsafe {
                                let dst = s2.as_mut_ptr().add(s2_rb_offset) as *mut f32; // Explicit Cast
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr().add(src_offset),
                                    dst,
                                    count,
                                );
                            }
                            rb_offset += count;
                        }
                    }
                }
                unsafe {
                    chunk.commit_all();
                }
            }

            self.tail_tasks_pushed.fetch_add(1, Ordering::Relaxed);

            // 馃殌 Signal Worker (Wait-Free, Debounced)
            self.signal_worker();
            return true;
        }
        false
    }

    #[inline(always)]
    fn trigger_body_task_for(
        body: &mut BodyPipelineAudio,
        num_inputs: usize,
        body_tasks_pushed: &AtomicU64,
        signal_tx: &Sender<()>,
        signal_pending: &AtomicBool,
    ) -> bool {
        let task_size = num_inputs * body.block_size;
        if body.task_producer.slots() >= task_size {
            if let Ok(mut chunk) = body.task_producer.write_chunk_uninit(task_size) {
                let (s1, s2) = chunk.as_mut_slices();
                let mut rb_offset = 0;

                for buf in body.input_buffers.iter().take(num_inputs) {
                    let src = &buf[0..body.block_size];
                    let mut src_offset = 0;
                    let mut remaining = body.block_size;

                    let s1_avail = s1.len().saturating_sub(rb_offset);
                    if s1_avail > 0 {
                        let count = remaining.min(s1_avail);
                        unsafe {
                            let dst = s1.as_mut_ptr().add(rb_offset) as *mut f32;
                            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, count);
                        }
                        rb_offset += count;
                        src_offset += count;
                        remaining -= count;
                    }

                    if remaining > 0 {
                        let s2_rb_offset = rb_offset - s1.len();
                        let s2_avail = s2.len().saturating_sub(s2_rb_offset);
                        let count = remaining.min(s2_avail);
                        if count > 0 {
                            unsafe {
                                let dst = s2.as_mut_ptr().add(s2_rb_offset) as *mut f32;
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr().add(src_offset),
                                    dst,
                                    count,
                                );
                            }
                            rb_offset += count;
                        }
                    }
                }

                unsafe {
                    chunk.commit_all();
                }
            }

            body_tasks_pushed.fetch_add(1, Ordering::Relaxed);

            // Always attempt to signal after enqueue. The bounded channel naturally de-bounces,
            // and this avoids stale pending state suppressing a needed wake-up.
            signal_pending.store(true, Ordering::Release);
            let _ = signal_tx.try_send(());
            return true;
        }
        false
    }

    pub fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) {
        let n_frames = inputs[0].len();
        if n_frames == 0 {
            return;
        }

        let reset_pending = {
            let reset_counter = self.reset_counter.load(Ordering::Acquire);
            let tail_pending = self.reset_ack.load(Ordering::Acquire) != reset_counter;
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            let body_pending =
                self.body.is_some() && self.body_reset_ack.load(Ordering::Acquire) != reset_counter;
            #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
            let body_pending = false;
            tail_pending || body_pending
        };

        // Safety: Dimension Check
        let actual_inputs = inputs.len();
        let actual_outputs = outputs.len();
        let safe_out_limit = self.num_outputs.min(actual_outputs);
        let safe_in_limit = self.num_inputs.min(actual_inputs);

        // 1. Clear Outputs
        for out in outputs.iter_mut().take(safe_out_limit) {
            out.fill(0.0);
        }

        let mut update_input_pos = true;
        let mut update_body_pos = true;

        // ========================================================================
        // Phase 1: Input Processing & Latency Hiding (Vectorized)
        // ========================================================================
        // Immediate Fire: Process inputs and trigger GPU ASAP to mask Head CPU cost.
        if !reset_pending {
            if let Some(mut body) = self.body.take() {
                #[cfg(any(target_os = "windows", target_os = "macos"))]
                let body_signal_tx = self.body_signal_tx.clone();
                #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
                let body_signal_tx = self.signal_tx.clone();
                #[cfg(any(target_os = "windows", target_os = "macos"))]
                let body_signal_pending = self.body_signal_pending.clone();
                #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
                let body_signal_pending = self.signal_pending.clone();
                let num_inputs = self.num_inputs;
                let mut cursor = 0;
                let mut tail_pos = self.input_pos;
                let mut body_pos = body.input_pos;

                let body_signal_tx_ref = &body_signal_tx;
                let body_tasks_pushed = self.body_tasks_pushed.clone();

                while cursor < n_frames {
                    let remaining_tail = self.block_size - tail_pos;
                    let remaining_body = body.block_size - body_pos;
                    let chunk_len = remaining_tail.min(remaining_body).min(n_frames - cursor);

                    // Vectorized Input Copy (Tail + Body)
                    for in_ch in 0..safe_in_limit {
                        unsafe {
                            let src_ptr = inputs[in_ch].as_ptr().add(cursor);
                            let tail_dst = self.input_buffers[in_ch].as_mut_ptr().add(tail_pos);
                            let body_dst = body.input_buffers[in_ch].as_mut_ptr().add(body_pos);
                            std::ptr::copy_nonoverlapping(src_ptr, tail_dst, chunk_len);
                            std::ptr::copy_nonoverlapping(src_ptr, body_dst, chunk_len);
                        }
                    }
                    // Zero-fill unused input channels if any
                    if self.num_inputs > safe_in_limit {
                        for in_ch in safe_in_limit..self.num_inputs {
                            unsafe {
                                let tail_dst = self.input_buffers[in_ch].as_mut_ptr().add(tail_pos);
                                let body_dst = body.input_buffers[in_ch].as_mut_ptr().add(body_pos);
                                std::ptr::write_bytes(tail_dst, 0, chunk_len);
                                std::ptr::write_bytes(body_dst, 0, chunk_len);
                            }
                        }
                    }

                    tail_pos += chunk_len;
                    body_pos += chunk_len;
                    cursor += chunk_len;

                    if tail_pos == self.block_size {
                        if !self.trigger_worker_task() {
                            self.tail_task_drop_count = self.tail_task_drop_count.saturating_add(1);
                        }
                        tail_pos = 0;
                    }
                    if body_pos == body.block_size {
                        let body_signal_pending_ref = body_signal_pending.as_ref();
                        if !Self::trigger_body_task_for(
                            &mut body,
                            num_inputs,
                            body_tasks_pushed.as_ref(),
                            body_signal_tx_ref,
                            body_signal_pending_ref,
                        ) {
                            self.body_task_drop_count = self.body_task_drop_count.saturating_add(1);
                        }
                        body_pos = 0;
                    }
                }

                self.body = Some(body);
            } else {
                let mut cursor = 0;
                let mut current_pos = self.input_pos;

                while cursor < n_frames {
                    let remaining_in_block = self.block_size - current_pos;
                    let chunk_len = remaining_in_block.min(n_frames - cursor);

                    // Vectorized Input Copy
                    for in_ch in 0..safe_in_limit {
                        // src = inputs[in_ch][cursor .. cursor + chunk_len]
                        // dst = self.input_buffers[in_ch][current_pos .. current_pos + chunk_len]
                        unsafe {
                            let src_ptr = inputs[in_ch].as_ptr().add(cursor);
                            let dst_ptr = self.input_buffers[in_ch].as_mut_ptr().add(current_pos);
                            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, chunk_len);
                        }
                    }
                    // Zero-fill unused input channels if any
                    if self.num_inputs > safe_in_limit {
                        for in_ch in safe_in_limit..self.num_inputs {
                            unsafe {
                                let dst_ptr =
                                    self.input_buffers[in_ch].as_mut_ptr().add(current_pos);
                                std::ptr::write_bytes(dst_ptr, 0, chunk_len);
                            }
                        }
                    }

                    current_pos += chunk_len;
                    cursor += chunk_len;

                    // Fire Worker if Block Full
                    if current_pos == self.block_size {
                        if !self.trigger_worker_task() {
                            self.tail_task_drop_count = self.tail_task_drop_count.saturating_add(1);
                        }
                        current_pos = 0;
                    }
                }
            }
        } else {
            // During reset, keep tail/body idle to avoid block misalignment.
            update_input_pos = false;
            update_body_pos = false;
        }

        // ========================================================================
        // Phase 2: Head Computation (CPU Bound)
        // ========================================================================
        // While this runs, the GPU worker (awakened above) is processing/submitting.
        if !DEBUG_MUTE_HEAD {
            match &mut self.engine {
                HeadEngine::Titan(rows) => {
                    for (in_ch, row) in rows.iter_mut().enumerate() {
                        if in_ch < actual_inputs {
                            row.process(inputs[in_ch], outputs);
                        }
                    }
                }
                HeadEngine::Vdsp(rows) => {
                    for (in_ch, row) in rows.iter_mut().enumerate() {
                        if in_ch < actual_inputs {
                            row.process(inputs[in_ch], outputs);
                        }
                    }
                }
            }
        }

        // ========================================================================
        // Phase 3: Tail Mixing (Vectorized)
        // ========================================================================
        {
            if let Some(mut body) = self.body.take() {
                let mut cursor = 0;
                let mut tail_pos = self.input_pos;
                let mut body_pos = body.input_pos;

                while cursor < n_frames {
                    if tail_pos == 0 {
                        if reset_pending {
                            for buf in &mut self.overlap_buffers {
                                buf.fill(0.0);
                            }
                        } else {
                            self.blocks_since_reset = self.blocks_since_reset.saturating_add(1);
                            let target_tail = self
                                .blocks_since_reset
                                .saturating_sub(self.tail_latency_blocks);
                            let needed = target_tail.saturating_sub(self.tail_blocks_consumed);
                            if needed == 0 {
                                for buf in &mut self.overlap_buffers {
                                    buf.fill(0.0);
                                }
                            } else {
                                let read_count = self.fetch_tail_result(needed);
                                self.log_tail_underrun(needed, read_count);
                                self.tail_blocks_consumed += read_count;
                            }
                        }
                    }

                    if body_pos == 0 {
                        if reset_pending {
                            body.overlap_buffers.fill(0.0);
                        } else {
                            body.blocks_since_reset = body.blocks_since_reset.saturating_add(1);
                            let target_body =
                                body.blocks_since_reset.saturating_sub(body.latency_blocks);
                            let needed = target_body.saturating_sub(body.blocks_consumed);
                            if needed == 0 {
                                body.overlap_buffers.fill(0.0);
                            } else {
                                let read_count = Self::fetch_body_result_from(
                                    &mut body,
                                    self.num_outputs,
                                    needed,
                                );
                                let body_task_slots = body.task_producer.slots();
                                let body_result_slots = body.result_consumer.slots();
                                self.log_body_underrun(
                                    needed,
                                    read_count,
                                    body_task_slots,
                                    body_result_slots,
                                );
                                body.blocks_consumed += read_count;
                            }
                        }
                    }

                    let remaining_tail = self.block_size - tail_pos;
                    let remaining_body = body.block_size - body_pos;
                    let chunk_len = remaining_tail.min(remaining_body).min(n_frames - cursor);

                    if !DEBUG_MUTE_TAIL {
                        if !self.tail_probe_reported {
                            let mut first_idx: Option<usize> = None;
                            for out_ch in 0..safe_out_limit {
                                let overlap_slice =
                                    &self.overlap_buffers[out_ch][tail_pos..tail_pos + chunk_len];
                                for (i, &v) in overlap_slice.iter().enumerate() {
                                    if v.abs() > 1.0e-6 {
                                        first_idx = Some(i);
                                        break;
                                    }
                                }
                                if first_idx.is_some() {
                                    break;
                                }
                            }
                            if let Some(i) = first_idx {
                                let global_pos = self.processed_samples + cursor as u64 + i as u64;
                                if let Some(probe) = &self.tail_probe {
                                    probe.report(global_pos);
                                }
                                self.tail_probe_reported = true;
                            }
                        }
                    }

                    // Vectorized Mix: Output += Tail + Body
                    #[cfg(target_os = "macos")]
                    {
                        use crate::accelerate;
                        for out_ch in 0..safe_out_limit {
                            unsafe {
                                let out_ptr = outputs[out_ch].as_mut_ptr().add(cursor);
                                if !DEBUG_MUTE_TAIL {
                                    let tail_ptr =
                                        self.overlap_buffers[out_ch].as_ptr().add(tail_pos);
                                    accelerate::vDSP_vadd(
                                        tail_ptr,
                                        1,
                                        out_ptr,
                                        1,
                                        out_ptr,
                                        1,
                                        chunk_len as crate::accelerate::vDSP_Length,
                                    );
                                }

                                let body_base = out_ch * body.block_size + body_pos;
                                let body_ptr = body.overlap_buffers.as_ptr().add(body_base);
                                accelerate::vDSP_vadd(
                                    body_ptr,
                                    1,
                                    out_ptr,
                                    1,
                                    out_ptr,
                                    1,
                                    chunk_len as crate::accelerate::vDSP_Length,
                                );
                            }
                        }
                    }

                    #[cfg(target_os = "windows")]
                    {
                        for out_ch in 0..safe_out_limit {
                            let out = &mut outputs[out_ch][cursor..cursor + chunk_len];
                            unsafe {
                                if !DEBUG_MUTE_TAIL {
                                    let tail_overlap = &self.overlap_buffers[out_ch]
                                        [tail_pos..tail_pos + chunk_len];
                                    let _ = ipp::ippsAdd_32f_I(
                                        tail_overlap.as_ptr(),
                                        out.as_mut_ptr(),
                                        chunk_len as i32,
                                    );
                                }
                                let body_base = out_ch * body.block_size + body_pos;
                                let body_overlap =
                                    &body.overlap_buffers[body_base..body_base + chunk_len];
                                let _ = ipp::ippsAdd_32f_I(
                                    body_overlap.as_ptr(),
                                    out.as_mut_ptr(),
                                    chunk_len as i32,
                                );
                            }
                        }
                    }

                    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
                    {
                        for out_ch in 0..safe_out_limit {
                            let out = &mut outputs[out_ch][cursor..cursor + chunk_len];
                            if !DEBUG_MUTE_TAIL {
                                let tail_overlap =
                                    &self.overlap_buffers[out_ch][tail_pos..tail_pos + chunk_len];
                                for i in 0..chunk_len {
                                    out[i] += tail_overlap[i];
                                }
                            }
                            let body_base = out_ch * body.block_size + body_pos;
                            let body_overlap =
                                &body.overlap_buffers[body_base..body_base + chunk_len];
                            for i in 0..chunk_len {
                                out[i] += body_overlap[i];
                            }
                        }
                    }

                    tail_pos += chunk_len;
                    body_pos += chunk_len;
                    cursor += chunk_len;

                    if tail_pos == self.block_size {
                        tail_pos = 0;
                    }
                    if body_pos == body.block_size {
                        body_pos = 0;
                    }
                }

                if update_input_pos {
                    self.input_pos = tail_pos;
                }
                if update_body_pos {
                    body.input_pos = body_pos;
                }
                self.body = Some(body);
            } else {
                let mut cursor = 0;
                let mut current_pos = self.input_pos;

                while cursor < n_frames {
                    // If at start of block, fetch results (for *next* block logic essentially)
                    if current_pos == 0 {
                        if reset_pending {
                            // Wait for worker reset to complete
                            for buf in &mut self.overlap_buffers {
                                buf.fill(0.0);
                            }
                        } else {
                            self.blocks_since_reset = self.blocks_since_reset.saturating_add(1);
                            let target_tail = self
                                .blocks_since_reset
                                .saturating_sub(self.tail_latency_blocks);
                            let needed = target_tail.saturating_sub(self.tail_blocks_consumed);
                            if needed == 0 {
                                for buf in &mut self.overlap_buffers {
                                    buf.fill(0.0);
                                }
                            } else {
                                let read_count = self.fetch_tail_result(needed);
                                self.log_tail_underrun(needed, read_count);
                                self.tail_blocks_consumed += read_count;
                            }
                        }
                    }

                    let remaining_in_block = self.block_size - current_pos;
                    let chunk_len = remaining_in_block.min(n_frames - cursor);

                    if !DEBUG_MUTE_TAIL {
                        if !self.tail_probe_reported {
                            let mut first_idx: Option<usize> = None;
                            for out_ch in 0..safe_out_limit {
                                let overlap_slice = &self.overlap_buffers[out_ch]
                                    [current_pos..current_pos + chunk_len];
                                for (i, &v) in overlap_slice.iter().enumerate() {
                                    if v.abs() > 1.0e-6 {
                                        first_idx = Some(i);
                                        break;
                                    }
                                }
                                if first_idx.is_some() {
                                    break;
                                }
                            }
                            if let Some(i) = first_idx {
                                let global_pos = self.processed_samples + cursor as u64 + i as u64;
                                if let Some(probe) = &self.tail_probe {
                                    probe.report(global_pos);
                                }
                                self.tail_probe_reported = true;
                            }
                        }

                        // Vectorized Mix: Output += Overlap
                        #[cfg(target_os = "macos")]
                        {
                            use crate::accelerate;
                            for out_ch in 0..safe_out_limit {
                                unsafe {
                                    let overlap_ptr =
                                        self.overlap_buffers[out_ch].as_ptr().add(current_pos);
                                    let out_ptr = outputs[out_ch].as_mut_ptr().add(cursor);

                                    accelerate::vDSP_vadd(
                                        overlap_ptr,
                                        1,
                                        out_ptr,
                                        1,
                                        out_ptr,
                                        1,
                                        chunk_len as crate::accelerate::vDSP_Length,
                                    );
                                }
                            }
                        }

                        #[cfg(target_os = "windows")]
                        {
                            for out_ch in 0..safe_out_limit {
                                unsafe {
                                    let overlap = &self.overlap_buffers[out_ch]
                                        [current_pos..current_pos + chunk_len];
                                    let out = &mut outputs[out_ch][cursor..cursor + chunk_len];
                                    let _ = ipp::ippsAdd_32f_I(
                                        overlap.as_ptr(),
                                        out.as_mut_ptr(),
                                        chunk_len as i32,
                                    );
                                }
                            }
                        }

                        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
                        {
                            for out_ch in 0..safe_out_limit {
                                let overlap = &self.overlap_buffers[out_ch]
                                    [current_pos..current_pos + chunk_len];
                                let out = &mut outputs[out_ch][cursor..cursor + chunk_len];
                                for i in 0..chunk_len {
                                    out[i] += overlap[i];
                                }
                            }
                        }
                    }

                    current_pos += chunk_len;
                    cursor += chunk_len;

                    if current_pos == self.block_size {
                        current_pos = 0;
                    }
                }

                // Final State Update
                if update_input_pos {
                    self.input_pos = current_pos;
                }
            }
        }

        self.processed_samples += n_frames as u64;
    }
}

#[cfg(target_os = "macos")]
struct BodyThreadScratch {
    fft: FftContext,
    accum_r: Vec<f32>,
    accum_i: Vec<f32>,
    temp_r: Vec<f32>,
    temp_i: Vec<f32>,
    time: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl BodyThreadScratch {
    fn new(block_size: usize) -> Self {
        Self {
            fft: FftContext::new(block_size * 2),
            accum_r: vec![0.0f32; block_size],
            accum_i: vec![0.0f32; block_size],
            temp_r: vec![0.0f32; block_size],
            temp_i: vec![0.0f32; block_size],
            time: vec![0.0f32; block_size * 2],
        }
    }
}

#[cfg(target_os = "windows")]
struct BodyThreadScratch {
    fft: FftContext,
    accum: Vec<ipp::Ipp32fc>,
    time: Vec<f32>,
}

#[cfg(target_os = "windows")]
impl BodyThreadScratch {
    fn new(block_size: usize) -> Self {
        Self {
            fft: FftContext::new(block_size * 2),
            accum: vec![ipp::Ipp32fc { re: 0.0, im: 0.0 }; block_size],
            time: vec![0.0f32; block_size * 2],
        }
    }
}

struct ThreadScratchPool {
    inner: Vec<UnsafeCell<BodyThreadScratch>>,
}

unsafe impl Sync for ThreadScratchPool {}

impl ThreadScratchPool {
    fn new(thread_count: usize, block_size: usize) -> Self {
        let count = thread_count.max(1);
        let mut inner = Vec::with_capacity(count);
        for _ in 0..count {
            inner.push(UnsafeCell::new(BodyThreadScratch::new(block_size)));
        }
        Self { inner }
    }

    fn get(&self, index: usize) -> &UnsafeCell<BodyThreadScratch> {
        if self.inner.is_empty() {
            panic!("ThreadScratchPool is empty");
        }
        let idx = index.min(self.inner.len() - 1);
        &self.inner[idx]
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn complex_mul_add_avx2(
    ar: *const f32,
    ai: *const f32,
    br: *const f32,
    bi: *const f32,
    accum_r: *mut f32,
    accum_i: *mut f32,
    len: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut k = 0;
    while k + 8 <= len {
        let ar_v = _mm256_loadu_ps(ar.add(k));
        let ai_v = _mm256_loadu_ps(ai.add(k));
        let br_v = _mm256_loadu_ps(br.add(k));
        let bi_v = _mm256_loadu_ps(bi.add(k));

        let acc_r_v = _mm256_loadu_ps(accum_r.add(k));
        let acc_i_v = _mm256_loadu_ps(accum_i.add(k));

        // accum_r += ar*br - ai*bi
        let ar_br = _mm256_mul_ps(ar_v, br_v);
        let ai_bi = _mm256_mul_ps(ai_v, bi_v);
        let res_r = _mm256_add_ps(acc_r_v, _mm256_sub_ps(ar_br, ai_bi));

        // accum_i += ar*bi + ai*br
        let ar_bi = _mm256_mul_ps(ar_v, bi_v);
        let ai_br = _mm256_mul_ps(ai_v, br_v);
        let res_i = _mm256_add_ps(acc_i_v, _mm256_add_ps(ar_bi, ai_br));

        _mm256_storeu_ps(accum_r.add(k), res_r);
        _mm256_storeu_ps(accum_i.add(k), res_i);

        k += 8;
    }

    while k < len {
        let ar = *ar.add(k);
        let ai = *ai.add(k);
        let br = *br.add(k);
        let bi = *bi.add(k);

        *accum_r.add(k) += ar * br - ai * bi;
        *accum_i.add(k) += ar * bi + ai * br;

        k += 1;
    }
}

impl MatrixWorker {
    #[inline(always)]
    #[cfg(target_os = "windows")]
    fn body_parallel_enabled(
        num_outputs: usize,
        num_inputs: usize,
        num_parts: usize,
        block_size: usize,
    ) -> bool {
        if num_outputs <= 1 {
            return false;
        }
        let work = num_outputs
            .saturating_mul(num_inputs)
            .saturating_mul(num_parts)
            .saturating_mul(block_size);
        work >= BODY_PARALLEL_WORK_THRESHOLD
    }

    pub fn wait_for_signal_timeout(&self, timeout: Duration) -> bool {
        // Clear pending flag so audio thread can signal again
        self.signal_pending.store(false, Ordering::Release);
        self.signal_rx.recv_timeout(timeout).is_ok()
    }

    fn handle_reset(&mut self, new_counter: u64) {
        self.last_reset_counter = new_counter;
        self.tail.reset_state();
        for buf in &mut self.saved_overlaps {
            buf.fill(0.0);
        }
        if let Some(body) = &mut self.body {
            #[cfg(target_os = "macos")]
            {
                body.fdl_real.fill(0.0);
                body.fdl_imag.fill(0.0);
            }
            #[cfg(target_os = "windows")]
            {
                body.fdl_complex.fill(ipp::Ipp32fc { re: 0.0, im: 0.0 });
            }
            body.saved_overlaps.fill(0.0);
            body.fdl_pos = body.num_partitions.saturating_sub(1);
            self.drain_body_task_consumer();
        }
        self.drain_task_consumer();
        self.reset_ack.store(new_counter, Ordering::Release);
    }

    fn drain_task_consumer(&mut self) {
        loop {
            let available = self.task_consumer.slots();
            if available == 0 {
                break;
            }
            if let Ok(chunk) = self.task_consumer.read_chunk(available) {
                chunk.commit_all();
            } else {
                break;
            }
        }
    }

    fn drain_body_task_consumer(&mut self) {
        if let Some(body) = &mut self.body {
            loop {
                let available = body.task_consumer.slots();
                if available == 0 {
                    break;
                }
                if let Ok(chunk) = body.task_consumer.read_chunk(available) {
                    chunk.commit_all();
                } else {
                    break;
                }
            }
        }
    }

    pub fn wait_and_process(&mut self) -> bool {
        let task_size = self.num_inputs * self.block_size;

        let current_reset = self.reset_counter.load(Ordering::Acquire);
        if current_reset != self.last_reset_counter {
            self.handle_reset(current_reset);
        }

        let mut did_work = false;

        // 1) Body (Higher Frequency, CPU)
        if let Some(body) = &mut self.body {
            let body_task_size = self.num_inputs * body.block_size;
            if body.task_consumer.slots() >= body_task_size {
                if let Ok(chunk) = body.task_consumer.read_chunk(body_task_size) {
                    let (s1, s2) = chunk.as_slices();
                    if s1.len() > 0 {
                        body.scratch_task_pop[0..s1.len()].copy_from_slice(s1);
                    }
                    if s2.len() > 0 {
                        body.scratch_task_pop[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
                    }
                    chunk.commit_all();
                }

                self.body_tasks_consumed.fetch_add(1, Ordering::Relaxed);
                self.process_body();
                did_work = true;

                let current_reset = self.reset_counter.load(Ordering::Acquire);
                if current_reset != self.last_reset_counter {
                    self.handle_reset(current_reset);
                    return true;
                }
            }
        }

        // 2) Tail (GPU)
        if self.task_consumer.slots() >= task_size {
            if let Ok(chunk) = self.task_consumer.read_chunk(task_size) {
                let (s1, s2) = chunk.as_slices();

                if s1.len() > 0 {
                    self.scratch_task_pop[0..s1.len()].copy_from_slice(s1);
                }
                if s2.len() > 0 {
                    self.scratch_task_pop[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
                }
                chunk.commit_all();
            }

            self.tail_tasks_consumed.fetch_add(1, Ordering::Relaxed);
            self.process_gpu();
            did_work = true;

            let current_reset = self.reset_counter.load(Ordering::Acquire);
            if current_reset != self.last_reset_counter {
                self.handle_reset(current_reset);
                return true;
            }

            // Write Tail Result
            let res_size = self.num_outputs * self.block_size;
            let slots_before = self.result_producer.slots();
            let mut pushed = false;
            if slots_before >= res_size {
                if let Ok(mut chunk) = self.result_producer.write_chunk_uninit(res_size) {
                    let (s1, s2) = chunk.as_mut_slices();
                    let src = &self.scratch_result_push;

                    if s1.len() > 0 {
                        unsafe {
                            let dst = s1.as_mut_ptr() as *mut f32;
                            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, s1.len());
                        }
                    }
                    if s2.len() > 0 {
                        unsafe {
                            let dst = s2.as_mut_ptr() as *mut f32;
                            std::ptr::copy_nonoverlapping(src[s1.len()..].as_ptr(), dst, s2.len());
                        }
                    }
                    unsafe {
                        chunk.commit_all();
                    }
                    pushed = true;
                }
            }
            if pushed {
                self.tail_results_pushed.fetch_add(1, Ordering::Relaxed);
            }
        }

        did_work
    }

    fn process_body(&mut self) {
        let Some(body) = &mut self.body else {
            return;
        };

        if body.num_partitions == 0 {
            body.scratch_result_push.fill(0.0);
            return;
        }

        let freq_bins = body.block_size;
        let fdl_stride_in = freq_bins;
        let fdl_stride_part = self.num_inputs * fdl_stride_in;

        // Move FDL write head first (so fdl_pos always points to latest)
        body.fdl_pos = (body.fdl_pos + 1) % body.num_partitions;

        // Input FFT -> FDL
        for ch in 0..self.num_inputs {
            let offset = ch * body.block_size;
            body.scratch_fft_input.fill(0.0);
            body.scratch_fft_input[0..body.block_size]
                .copy_from_slice(&body.scratch_task_pop[offset..offset + body.block_size]);

            let base = body.fdl_pos * fdl_stride_part + ch * fdl_stride_in;

            #[cfg(target_os = "macos")]
            {
                let mut split = DSPComplex {
                    realp: body.scratch_complex_r.as_mut_ptr(),
                    imagp: body.scratch_complex_i.as_mut_ptr(),
                };
                body.fft.forward(&body.scratch_fft_input, &mut split);

                unsafe {
                    let r = std::slice::from_raw_parts(split.realp, body.block_size);
                    let i = std::slice::from_raw_parts(split.imagp, body.block_size);
                    body.fdl_real[base..base + body.block_size].copy_from_slice(r);
                    body.fdl_imag[base..base + body.block_size].copy_from_slice(i);
                }
            }

            #[cfg(target_os = "windows")]
            {
                let dst = &mut body.fdl_complex[base..base + body.block_size];
                body.fft.forward_complex_half(&body.scratch_fft_input, dst);
            }
        }

        // Output Accumulation (Parallel)
        #[cfg(target_os = "macos")]
        {
            use crate::accelerate;
            let stride_out = freq_bins;
            let stride_in = self.num_outputs * stride_out;
            let stride_part = self.num_inputs * stride_in;

            let ir_real = &body.ir_real;
            let ir_imag = &body.ir_imag;
            let fdl_real = &body.fdl_real;
            let fdl_imag = &body.fdl_imag;
            let active_mask = &body.active_mask;
            let fdl_pos = body.fdl_pos;
            let num_parts = body.num_partitions;
            let num_inputs = self.num_inputs;
            let num_outputs = self.num_outputs;
            let mask_stride_in = num_outputs;
            let mask_stride_part = num_inputs * mask_stride_in;

            let gain_comp = 0.5;
            let outputs_per_task = body.parallel_outputs_per_task.max(1);
            let group_size = outputs_per_task * freq_bins;
            let thread_scratch = &body.thread_scratch;

            body.thread_pool.install(|| {
                body.scratch_result_push
                    .par_chunks_mut(group_size)
                    .zip(body.saved_overlaps.par_chunks_mut(group_size))
                    .enumerate()
                    .for_each(|(group_idx, (out_group, overlap_group))| {
                        let thread_index = rayon::current_thread_index().unwrap_or(0);
                        let scratch_cell = thread_scratch.get(thread_index);
                        let scratch = unsafe { &mut *scratch_cell.get() };

                        let base_out = group_idx * outputs_per_task;
                        let group_outputs = out_group.len() / freq_bins;

                        for g in 0..group_outputs {
                            let out_ch = base_out + g;
                            let out_offset = g * freq_bins;
                            let out_slice = &mut out_group[out_offset..out_offset + freq_bins];
                            let overlap_slice =
                                &mut overlap_group[out_offset..out_offset + freq_bins];

                            scratch.accum_r.fill(0.0);
                            scratch.accum_i.fill(0.0);

                            for in_ch in 0..num_inputs {
                                for p in 0..num_parts {
                                    let mask_idx =
                                        p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                                    if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                                        continue;
                                    }
                                    let ir_base =
                                        p * stride_part + in_ch * stride_in + out_ch * stride_out;
                                    let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                                    let fdl_base =
                                        fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                                    unsafe {
                                        let ir_split = DSPComplex {
                                            realp: ir_real.as_ptr().add(ir_base) as *mut f32,
                                            imagp: ir_imag.as_ptr().add(ir_base) as *mut f32,
                                        };
                                        let x_split = DSPComplex {
                                            realp: fdl_real.as_ptr().add(fdl_base) as *mut f32,
                                            imagp: fdl_imag.as_ptr().add(fdl_base) as *mut f32,
                                        };
                                        let accum_split = DSPComplex {
                                            realp: scratch.accum_r.as_mut_ptr(),
                                            imagp: scratch.accum_i.as_mut_ptr(),
                                        };
                                        // Packed real spectrum: bin0 stores (DC, Nyquist) in (re, im).
                                        // Keep bin0 as independent real products while fusing MAC for bins 1..N-1.
                                        let ar0 = *ir_split.realp;
                                        let ai0 = *ir_split.imagp;
                                        let br0 = *x_split.realp;
                                        let bi0 = *x_split.imagp;
                                        let prev_re0 = *scratch.accum_r.get_unchecked(0);
                                        let prev_im0 = *scratch.accum_i.get_unchecked(0);
                                        accelerate::vDSP_zvma(
                                            &ir_split,
                                            1,
                                            &x_split,
                                            1,
                                            &accum_split,
                                            1,
                                            &accum_split,
                                            1,
                                            freq_bins as crate::accelerate::vDSP_Length,
                                        );
                                        *scratch.accum_r.get_unchecked_mut(0) = prev_re0 + ar0 * br0;
                                        *scratch.accum_i.get_unchecked_mut(0) = prev_im0 + ai0 * bi0;
                                    }
                                }
                            }

                            let mut split = DSPComplex {
                                realp: scratch.accum_r.as_mut_ptr(),
                                imagp: scratch.accum_i.as_mut_ptr(),
                            };
                            scratch.fft.inverse(&mut split, &mut scratch.time);

                            for k in 0..freq_bins {
                                let val = scratch.time[k] * gain_comp;
                                out_slice[k] = val + overlap_slice[k];
                                overlap_slice[k] = scratch.time[k + freq_bins] * gain_comp;
                            }
                        }
                    });
            });
        }

        #[cfg(target_os = "windows")]
        {
            let stride_out = freq_bins;
            let stride_in = self.num_outputs * stride_out;
            let stride_part = self.num_inputs * stride_in;

            let ir_complex = &body.ir_complex;
            let fdl_complex = &body.fdl_complex;
            let active_mask = &body.active_mask;
            let fdl_pos = body.fdl_pos;
            let num_parts = body.num_partitions;
            let num_inputs = self.num_inputs;
            let num_outputs = self.num_outputs;
            let mask_stride_in = num_outputs;
            let mask_stride_part = num_inputs * mask_stride_in;

            let gain_comp = 1.0f32;
            if !Self::body_parallel_enabled(num_outputs, num_inputs, num_parts, freq_bins) {
                let scratch_cell = body.thread_scratch.get(0);
                let scratch = unsafe { &mut *scratch_cell.get() };

                for out_ch in 0..num_outputs {
                    for v in scratch.accum.iter_mut() {
                        v.re = 0.0;
                        v.im = 0.0;
                    }

                    for in_ch in 0..num_inputs {
                        for p in 0..num_parts {
                            let mask_idx = p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                            if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                                continue;
                            }

                            let ir_base = p * stride_part + in_ch * stride_in + out_ch * stride_out;
                            let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                            let fdl_base = fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                            unsafe {
                                // Packed real spectrum: bin0 stores (DC, Nyquist) in (re, im).
                                // Keep bin0 as independent real products while fusing MAC for bins 1..N-1.
                                let ar0 = (*ir_complex.as_ptr().add(ir_base)).re;
                                let ai0 = (*ir_complex.as_ptr().add(ir_base)).im;
                                let br0 = (*fdl_complex.as_ptr().add(fdl_base)).re;
                                let bi0 = (*fdl_complex.as_ptr().add(fdl_base)).im;
                                let prev0 = *scratch.accum.get_unchecked(0);
                                let _ = ipp::ippsAddProduct_32fc(
                                    ir_complex.as_ptr().add(ir_base),
                                    fdl_complex.as_ptr().add(fdl_base),
                                    scratch.accum.as_mut_ptr(),
                                    freq_bins as i32,
                                );
                                (*scratch.accum.get_unchecked_mut(0)).re = prev0.re + ar0 * br0;
                                (*scratch.accum.get_unchecked_mut(0)).im = prev0.im + ai0 * bi0;
                            }
                        }
                    }

                    scratch
                        .fft
                        .inverse_complex_half_to_real(&scratch.accum, &mut scratch.time);

                    let out_base = out_ch * freq_bins;
                    for k in 0..freq_bins {
                        let val = scratch.time[k] * gain_comp;
                        body.scratch_result_push[out_base + k] =
                            val + body.saved_overlaps[out_base + k];
                        body.saved_overlaps[out_base + k] = scratch.time[k + freq_bins] * gain_comp;
                    }
                }
            } else {
                let outputs_per_task = body.parallel_outputs_per_task.max(1);
                let group_size = outputs_per_task * freq_bins;
                let thread_scratch = &body.thread_scratch;

                body.thread_pool.install(|| {
                    body.scratch_result_push
                        .par_chunks_mut(group_size)
                        .zip(body.saved_overlaps.par_chunks_mut(group_size))
                        .enumerate()
                        .for_each(|(group_idx, (out_group, overlap_group))| {
                            let thread_index = rayon::current_thread_index().unwrap_or(0);
                            let scratch_cell = thread_scratch.get(thread_index);
                            let scratch = unsafe { &mut *scratch_cell.get() };

                            let base_out = group_idx * outputs_per_task;
                            let group_outputs = out_group.len() / freq_bins;

                            for g in 0..group_outputs {
                                let out_ch = base_out + g;
                                let out_offset = g * freq_bins;
                                let out_slice = &mut out_group[out_offset..out_offset + freq_bins];
                                let overlap_slice =
                                    &mut overlap_group[out_offset..out_offset + freq_bins];

                                for v in scratch.accum.iter_mut() {
                                    v.re = 0.0;
                                    v.im = 0.0;
                                }

                                for in_ch in 0..num_inputs {
                                    for p in 0..num_parts {
                                        let mask_idx =
                                            p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                                        if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                                            continue;
                                        }

                                        let ir_base = p * stride_part
                                            + in_ch * stride_in
                                            + out_ch * stride_out;
                                        let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                                        let fdl_base =
                                            fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                                        unsafe {
                                            // Packed real spectrum: bin0 stores (DC, Nyquist) in (re, im).
                                            // Keep bin0 as independent real products while fusing MAC for bins 1..N-1.
                                            let ar0 = (*ir_complex.as_ptr().add(ir_base)).re;
                                            let ai0 = (*ir_complex.as_ptr().add(ir_base)).im;
                                            let br0 = (*fdl_complex.as_ptr().add(fdl_base)).re;
                                            let bi0 = (*fdl_complex.as_ptr().add(fdl_base)).im;
                                            let prev0 = *scratch.accum.get_unchecked(0);
                                            let _ = ipp::ippsAddProduct_32fc(
                                                ir_complex.as_ptr().add(ir_base),
                                                fdl_complex.as_ptr().add(fdl_base),
                                                scratch.accum.as_mut_ptr(),
                                                freq_bins as i32,
                                            );
                                            (*scratch.accum.get_unchecked_mut(0)).re = prev0.re + ar0 * br0;
                                            (*scratch.accum.get_unchecked_mut(0)).im = prev0.im + ai0 * bi0;
                                        }
                                    }
                                }

                                scratch.fft.inverse_complex_half_to_real(
                                    &scratch.accum,
                                    &mut scratch.time,
                                );

                                for k in 0..freq_bins {
                                    let val = scratch.time[k] * gain_comp;
                                    out_slice[k] = val + overlap_slice[k];
                                    overlap_slice[k] = scratch.time[k + freq_bins] * gain_comp;
                                }
                            }
                        });
                });
            }
        }

        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            let stride_out = freq_bins;
            let stride_in = self.num_outputs * stride_out;
            let stride_part = self.num_inputs * stride_in;

            let ir_real = &body.ir_real;
            let ir_imag = &body.ir_imag;
            let fdl_real = &body.fdl_real;
            let fdl_imag = &body.fdl_imag;
            let active_mask = &body.active_mask;
            let fdl_pos = body.fdl_pos;
            let num_parts = body.num_partitions;
            let num_inputs = self.num_inputs;
            let num_outputs = self.num_outputs;
            let mask_stride_in = num_outputs;
            let mask_stride_part = num_inputs * mask_stride_in;

            let gain_comp = 1.0f32;

            let accum_r = &mut body.scratch_complex_r;
            let accum_i = &mut body.scratch_complex_i;
            let time = &mut body.scratch_fft_input;

            for out_ch in 0..num_outputs {
                accum_r.fill(0.0);
                accum_i.fill(0.0);

                for in_ch in 0..num_inputs {
                    for p in 0..num_parts {
                        let mask_idx = p * mask_stride_part + in_ch * mask_stride_in + out_ch;
                        if active_mask.get(mask_idx).copied().unwrap_or(1) == 0 {
                            continue;
                        }

                        let ir_base = p * stride_part + in_ch * stride_in + out_ch * stride_out;
                        let fdl_idx = (fdl_pos + num_parts - p) % num_parts;
                        let fdl_base = fdl_idx * fdl_stride_part + in_ch * fdl_stride_in;

                        for k in 0..freq_bins {
                            let ar = ir_real[ir_base + k];
                            let ai = ir_imag[ir_base + k];
                            let br = fdl_real[fdl_base + k];
                            let bi = fdl_imag[fdl_base + k];
                            accum_r[k] += ar * br - ai * bi;
                            accum_i[k] += ar * bi + ai * br;
                        }
                    }
                }

                let mut split = DSPComplex {
                    realp: accum_r.as_mut_ptr(),
                    imagp: accum_i.as_mut_ptr(),
                };
                body.fft.inverse(&mut split, time);

                let out_base = out_ch * freq_bins;
                for k in 0..freq_bins {
                    let val = time[k] * gain_comp;
                    body.scratch_result_push[out_base + k] =
                        val + body.saved_overlaps[out_base + k];
                    body.saved_overlaps[out_base + k] = time[k + freq_bins] * gain_comp;
                }
            }
        }

        // Write Body Result
        let res_size = self.num_outputs * body.block_size;
        let mut pushed = false;
        if body.result_producer.slots() >= res_size {
            if let Ok(mut chunk) = body.result_producer.write_chunk_uninit(res_size) {
                let (s1, s2) = chunk.as_mut_slices();
                let src = &body.scratch_result_push;

                if s1.len() > 0 {
                    unsafe {
                        let dst = s1.as_mut_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, s1.len());
                    }
                }
                if s2.len() > 0 {
                    unsafe {
                        let dst = s2.as_mut_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src[s1.len()..].as_ptr(), dst, s2.len());
                    }
                }
                unsafe {
                    chunk.commit_all();
                }
                pushed = true;
            }
        }
        if pushed {
            self.body_results_pushed.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn work_loop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::{
                GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
            };
            // Keep worker priority high enough for throughput but below DAW RT-critical scheduling.
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST as i32);
        }

        // 鉁?MacOS Priority Fix
        // 鉁?MacOS Priority Fix
        #[cfg(target_os = "macos")]
        unsafe {
            // "Professional Thinking": Use the correct Enum Variant provided by libc bindings for Apple.
            // Restore worker QoS to high-interactive profile for macOS scheduling consistency.
            let qos_class = libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE;
            libc::pthread_set_qos_class_self_np(qos_class, 0);
        }

        loop {
            // 1. Unlocked check (Fast path)
            if self.wait_and_process() {
                continue;
            }

            #[cfg(target_os = "windows")]
            {
                // Avoid long park->wake jitter on Windows when queue toggles near empty.
                let _ =
                    self.wait_for_signal_timeout(Duration::from_micros(TAIL_WORKER_IDLE_POLL_US));
                continue;
            }

            // 2. Wait for signal (Blocking)
            #[cfg(not(target_os = "windows"))]
            {
                self.signal_pending.store(false, Ordering::Release);
                if let Ok(_) = self.signal_rx.recv() {
                    // Signaled. Loop back to check tasks.
                    continue;
                } else {
                    // Channel disconnected (Audio thread dropped)
                    break;
                }
            }
        }
    }

    fn process_gpu(&mut self) {
        #[cfg(target_os = "macos")]
        let mut split = DSPComplex {
            realp: self._scratch_complex_r.as_mut_ptr(),
            imagp: self._scratch_complex_i.as_mut_ptr(),
        };

        #[cfg(target_os = "macos")]
        {
            for ch in 0..self.num_inputs {
                let offset = ch * self.block_size;
                self._scratch_fft_input.fill(0.0);
                self._scratch_fft_input[0..self.block_size]
                    .copy_from_slice(&self.scratch_task_pop[offset..offset + self.block_size]);

                self.fft.forward(&self._scratch_fft_input, &mut split);

                let gpu_offset = ch * self.block_size * 2;
                unsafe {
                    let r = std::slice::from_raw_parts(split.realp, self.block_size);
                    let i = std::slice::from_raw_parts(split.imagp, self.block_size);
                    for k in 0..self.block_size {
                        self.gpu_input_aggregate[gpu_offset + k * 2] = r[k];
                        self.gpu_input_aggregate[gpu_offset + k * 2 + 1] = i[k];
                    }
                }
            }
        }

        // ================= Windows / Titan 鏋侀€熻矾寰?=================
        #[cfg(not(target_os = "macos"))]
        {
            for ch in 0..self.num_inputs {
                let off = ch * self.block_size;
                let gpu_off = ch * self.block_size * 2;

                let input_slice = &self.scratch_task_pop[off..off + self.block_size];
                // 馃洝锔?Ensure destination slice matches what forward_to_interleaved expects (2 * block_size)
                let gpu_slice =
                    &mut self.gpu_input_aggregate[gpu_off..gpu_off + self.block_size * 2];

                // 馃敟 R2C + Direct Write
                self.fft.forward_to_interleaved(input_slice, gpu_slice);

                // Trace GPU Input
                if ch == 0 && gpu_slice[0] != 0.0 {
                    println!(
                        "鈿?[宸ヤ綔绾跨▼] GPU 杈撳叆灏辩华. Re[0]={}, Im[0]={}",
                        gpu_slice[0], gpu_slice[1]
                    );
                }
            }
        }

        let t0 = std::time::Instant::now();
        let gpu_out = self.tail.process(&self.gpu_input_aggregate);

        // Trace GPU Output
        // if gpu_out.len() > 0 && gpu_out[0].abs() > 0.0001 {
        //    println!("馃摛 [Worker] GPU Output[0]={}", gpu_out[0]);
        // } else if gpu_out.len() > 0 {
        //    // println!("馃摛 [Worker] GPU Output Zero");
        // }
        let total_dt = t0.elapsed().as_micros() as u64;

        self.gpu_execution_time
            .store(total_dt, std::sync::atomic::Ordering::Relaxed);

        #[cfg(target_os = "macos")]
        let gain_comp = 0.5;
        #[cfg(not(target_os = "macos"))]
        let gain_comp = 1.0;

        #[cfg(target_os = "macos")]
        {
            for ch in 0..self.num_outputs {
                let gpu_offset = ch * self.block_size * 2;
                unsafe {
                    let r = std::slice::from_raw_parts_mut(split.realp, self.block_size);
                    let i = std::slice::from_raw_parts_mut(split.imagp, self.block_size);
                    for k in 0..self.block_size {
                        r[k] = gpu_out[gpu_offset + k * 2];
                        i[k] = gpu_out[gpu_offset + k * 2 + 1];
                    }
                }
                self.fft.inverse(&mut split, &mut self.scratch_time);

                // Overlap Logic -> Output to scratch_result_push
                let push_base = ch * self.block_size;
                for k in 0..self.block_size {
                    let val = self.scratch_time[k] * gain_comp;
                    self.scratch_result_push[push_base + k] = val + self.saved_overlaps[ch][k];
                    self.saved_overlaps[ch][k] = self.scratch_time[k + self.block_size] * gain_comp;
                }
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            for ch in 0..self.num_outputs {
                let gpu_off = ch * self.block_size * 2;
                let gpu_slice = &gpu_out[gpu_off..gpu_off + self.block_size * 2];

                // 馃敟 Direct Read + IFFT
                self.fft
                    .inverse_from_interleaved(gpu_slice, &mut self.scratch_time);

                // Trace IFFT
                if ch == 0 && self.scratch_time[0] != 0.0 {
                    println!(
                        "馃搲 [宸ヤ綔绾跨▼] IFFT 閲嶅缓瀹屾垚. Time[0]={}",
                        self.scratch_time[0]
                    );
                }

                // Overlap-Add
                let push_base = ch * self.block_size;
                for k in 0..self.block_size {
                    self.scratch_result_push[push_base + k] =
                        self.scratch_time[k] * gain_comp + self.saved_overlaps[ch][k];
                    self.saved_overlaps[ch][k] = self.scratch_time[k + self.block_size] * gain_comp;
                }
            }
        }

        // 馃殌 Fix: Read Telemetry AFTER gpu_out borrow ends
        // Telemetry: Try to get detailed breakdown if Engine supports it
        let (driver_us, wait_us) = match &self.tail {
            TailEngine::Gpu(g) => (g.last_driver_us, g.last_gpu_us),
        };
        self.gpu_driver_time
            .store(driver_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_compute_time
            .store(wait_us, std::sync::atomic::Ordering::Relaxed);
    }
}


