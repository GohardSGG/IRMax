#![cfg(target_os = "windows")]

use crate::dsp::ipp;
use std::ptr;

struct IppFir {
    taps_len: usize,
    spec_mem: *mut ipp::Ipp8u,
    spec: *mut ipp::IppsFIRSpec_32f,
    buf: *mut ipp::Ipp8u,
    delay: Vec<f32>,
    temp: Vec<f32>,
}

impl IppFir {
    fn new(taps: &[f32], max_block: usize) -> Option<Self> {
        if taps.is_empty() {
            return None;
        }

        ipp::init_once();

        let taps_len = taps.len();
        let mut spec_size = 0;
        let mut buf_size = 0;
        unsafe {
            let _ = ipp::ippsFIRSRGetSize(
                taps_len as i32,
                ipp::IPP_DATATYPE_32F,
                &mut spec_size,
                &mut buf_size,
            );
        }

        let spec_mem = unsafe { ipp::ippsMalloc_8u(spec_size) };
        let spec = spec_mem as *mut ipp::IppsFIRSpec_32f;
        let buf = if buf_size > 0 {
            unsafe { ipp::ippsMalloc_8u(buf_size) }
        } else {
            ptr::null_mut()
        };

        unsafe {
            let _ = ipp::ippsFIRSRInit_32f(
                taps.as_ptr(),
                taps_len as i32,
                ipp::IPP_ALG_DIRECT,
                spec,
            );
        }

        let delay_len = taps_len.saturating_sub(1);
        let delay = vec![0.0f32; delay_len];
        let temp = vec![0.0f32; max_block.max(1)];

        Some(Self {
            taps_len,
            spec_mem,
            spec,
            buf,
            delay,
            temp,
        })
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        if len == 0 {
            return;
        }
        if self.taps_len == 0 {
            return;
        }

        if self.temp.len() < len {
            self.temp.resize(len, 0.0);
        }

        let dly_src = if self.delay.is_empty() {
            ptr::null()
        } else {
            self.delay.as_ptr()
        };
        let dly_dst = if self.delay.is_empty() {
            ptr::null_mut()
        } else {
            self.delay.as_mut_ptr()
        };

        unsafe {
            let _ = ipp::ippsFIRSR_32f(
                input.as_ptr(),
                self.temp.as_mut_ptr(),
                len as i32,
                self.spec,
                dly_src,
                dly_dst,
                self.buf,
            );
            let _ = ipp::ippsAdd_32f_I(self.temp.as_ptr(), output.as_mut_ptr(), len as i32);
        }
    }

    fn reset_state(&mut self) {
        self.delay.fill(0.0);
    }
}

impl Drop for IppFir {
    fn drop(&mut self) {
        unsafe {
            if !self.spec_mem.is_null() {
                ipp::ippsFree(self.spec_mem as *mut _);
            }
            if !self.buf.is_null() {
                ipp::ippsFree(self.buf as *mut _);
            }
        }
    }
}

pub struct HeadRowIpp {
    filters: Vec<Option<IppFir>>,
}

unsafe impl Send for HeadRowIpp {}
unsafe impl Sync for HeadRowIpp {}

impl HeadRowIpp {
    pub fn new(row_irs: &[&[f32]], head_len: usize, max_block: usize) -> Self {
        let mut filters = Vec::with_capacity(row_irs.len());
        for ir in row_irs.iter() {
            if ir.is_empty() || head_len == 0 {
                filters.push(None);
                continue;
            }
            let len = head_len.min(ir.len());
            let taps = &ir[0..len];
            filters.push(IppFir::new(taps, max_block));
        }
        Self { filters }
    }

    pub fn process(&mut self, input: &[f32], all_outputs: &mut [&mut [f32]]) {
        for (out_idx, filter) in self.filters.iter_mut().enumerate() {
            if let Some(filter) = filter {
                if out_idx < all_outputs.len() {
                    filter.process(input, all_outputs[out_idx]);
                }
            }
        }
    }

    pub fn reset_state(&mut self) {
        for filter in self.filters.iter_mut() {
            if let Some(filter) = filter {
                filter.reset_state();
            }
        }
    }
}
