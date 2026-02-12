#![cfg(target_os = "windows")]

use crate::framework::DSPComplex;
use crate::dsp::ipp;
use std::cell::RefCell;
use std::ptr;

pub struct FftContext {
    len: usize,
    _order: i32,
    spec: *mut ipp::IppsFFTSpec_C_32fc,
    spec_mem: *mut ipp::Ipp8u,
    spec_buf: *mut ipp::Ipp8u,
    work_buf: *mut ipp::Ipp8u,
    scratch: RefCell<Vec<ipp::Ipp32fc>>,
    half: RefCell<Vec<ipp::Ipp32fc>>,
}

unsafe impl Send for FftContext {}
unsafe impl Sync for FftContext {}

impl FftContext {
    pub fn new(size: usize) -> Self {
        ipp::init_once();
        assert!(size.is_power_of_two(), "FFT size must be power of 2");
        let order = size.trailing_zeros() as i32;

        let mut spec_size = 0;
        let mut spec_buf_size = 0;
        let mut buf_size = 0;
        unsafe {
            let _ = ipp::ippsFFTGetSize_C_32fc(
                order,
                ipp::IPP_FFT_DIV_INV_BY_N,
                ipp::IPP_ALG_HINT_FAST,
                &mut spec_size,
                &mut spec_buf_size,
                &mut buf_size,
            );
        }

        let spec_mem = unsafe { ipp::ippsMalloc_8u(spec_size) };
        let spec_buf = if spec_buf_size > 0 {
            unsafe { ipp::ippsMalloc_8u(spec_buf_size) }
        } else {
            ptr::null_mut()
        };
        let work_buf = if buf_size > 0 {
            unsafe { ipp::ippsMalloc_8u(buf_size) }
        } else {
            ptr::null_mut()
        };

        let mut spec: *mut ipp::IppsFFTSpec_C_32fc = ptr::null_mut();
        unsafe {
            let _ = ipp::ippsFFTInit_C_32fc(
                &mut spec,
                order,
                ipp::IPP_FFT_DIV_INV_BY_N,
                ipp::IPP_ALG_HINT_FAST,
                spec_mem,
                spec_buf,
            );
        }

        Self {
            len: size,
            _order: order,
            spec,
            spec_mem,
            spec_buf,
            work_buf,
            scratch: RefCell::new(vec![ipp::Ipp32fc { re: 0.0, im: 0.0 }; size]),
            half: RefCell::new(vec![ipp::Ipp32fc { re: 0.0, im: 0.0 }; size / 2]),
        }
    }

    #[inline]
    fn forward_complex_full_inplace(&self, buf: &mut [ipp::Ipp32fc]) {
        unsafe {
            let ptr = buf.as_mut_ptr();
            let _ = ipp::ippsFFTFwd_CToC_32fc(ptr, ptr, self.spec, self.work_buf);
        }
    }

    #[inline]
    fn inverse_complex_full_inplace(&self, buf: &mut [ipp::Ipp32fc]) {
        unsafe {
            let ptr = buf.as_mut_ptr();
            let _ = ipp::ippsFFTInv_CToC_32fc(ptr, ptr, self.spec, self.work_buf);
        }
    }

    pub fn forward_complex_half(&self, input: &[f32], output: &mut [ipp::Ipp32fc]) {
        let mut scratch = self.scratch.borrow_mut();
        let len = self.len.min(input.len());
        for i in 0..len {
            scratch[i].re = input[i];
            scratch[i].im = 0.0;
        }
        if len < self.len {
            for i in len..self.len {
                scratch[i].re = 0.0;
                scratch[i].im = 0.0;
            }
        }

        self.forward_complex_full_inplace(&mut scratch);

        let half = output.len().min(self.len / 2);
        if half == 0 {
            return;
        }
        // Pack real FFT as vDSP-style: output[0].re = DC, output[0].im = Nyquist
        output[0].re = scratch[0].re;
        output[0].im = if self.len >= 2 { scratch[self.len / 2].re } else { 0.0 };
        for k in 1..half {
            output[k] = scratch[k];
        }
    }

    pub fn inverse_complex_half_to_real(&self, input: &[ipp::Ipp32fc], output: &mut [f32]) {
        let mut scratch = self.scratch.borrow_mut();
        let half = input.len().min(self.len / 2);

        if half == 0 {
            return;
        }

        // Unpack vDSP-style packed real FFT: input[0].re = DC, input[0].im = Nyquist
        scratch[0].re = input[0].re;
        scratch[0].im = 0.0;
        if self.len >= 2 {
            let nyquist = self.len / 2;
            scratch[nyquist].re = input[0].im;
            scratch[nyquist].im = 0.0;
        }
        for k in 1..half {
            let v = input[k];
            scratch[k].re = v.re;
            scratch[k].im = v.im;
            let mirror = self.len - k;
            scratch[mirror].re = v.re;
            scratch[mirror].im = -v.im;
        }
        if half < self.len / 2 {
            for k in half..self.len / 2 {
                scratch[k].re = 0.0;
                scratch[k].im = 0.0;
                let mirror = self.len - k;
                if mirror < self.len {
                    scratch[mirror].re = 0.0;
                    scratch[mirror].im = 0.0;
                }
            }
        }

        self.inverse_complex_full_inplace(&mut scratch);

        let count = output.len().min(self.len);
        for i in 0..count {
            output[i] = scratch[i].re;
        }
    }

    pub fn forward(&self, input: &[f32], output_split: &mut DSPComplex) {
        let len = self.len / 2;
        let mut tmp = self.half.borrow_mut();
        self.forward_complex_half(input, &mut tmp);

        unsafe {
            let r_ptr = output_split.realp;
            let i_ptr = output_split.imagp;
            for k in 0..len {
                let c = tmp[k];
                *r_ptr.add(k) = c.re;
                *i_ptr.add(k) = c.im;
            }
        }
    }

    pub fn inverse(&self, input_split: &mut DSPComplex, output: &mut [f32]) {
        let len = self.len / 2;
        let mut tmp = self.half.borrow_mut();
        unsafe {
            let r_ptr = input_split.realp;
            let i_ptr = input_split.imagp;
            for k in 0..len {
                tmp[k].re = *r_ptr.add(k);
                tmp[k].im = *i_ptr.add(k);
            }
        }
        self.inverse_complex_half_to_real(&tmp, output);
    }

    pub fn forward_to_interleaved(&self, input: &[f32], output_interleaved: &mut [f32]) {
        let count = output_interleaved.len() / 2;
        let out = unsafe {
            std::slice::from_raw_parts_mut(output_interleaved.as_mut_ptr() as *mut ipp::Ipp32fc, count)
        };
        self.forward_complex_half(input, out);
    }

    pub fn inverse_from_interleaved(&self, input_interleaved: &[f32], output: &mut [f32]) {
        let count = input_interleaved.len() / 2;
        let input = unsafe {
            std::slice::from_raw_parts(
                input_interleaved.as_ptr() as *const ipp::Ipp32fc,
                count,
            )
        };
        self.inverse_complex_half_to_real(input, output);
    }
}

impl Drop for FftContext {
    fn drop(&mut self) {
        unsafe {
            if !self.spec_mem.is_null() {
                ipp::ippsFree(self.spec_mem as *mut _);
            }
            if !self.spec_buf.is_null() {
                ipp::ippsFree(self.spec_buf as *mut _);
            }
            if !self.work_buf.is_null() {
                ipp::ippsFree(self.work_buf as *mut _);
            }
        }
    }
}
