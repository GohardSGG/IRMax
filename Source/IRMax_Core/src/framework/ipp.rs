#![cfg(target_os = "windows")]

use std::os::raw::{c_int, c_void};
use std::sync::Once;

pub type Ipp8u = u8;
pub type Ipp32f = f32;
pub type IppStatus = c_int;
pub type IppHintAlgorithm = c_int;
pub type IppAlgType = c_int;
pub type IppDataType = c_int;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ipp32fc {
    pub re: f32,
    pub im: f32,
}

#[repr(C)]
pub struct IppsFFTSpec_C_32fc {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IppsFIRSpec_32f {
    _private: [u8; 0],
}

pub const IPP_FFT_DIV_INV_BY_N: c_int = 2;
pub const IPP_FFT_NODIV_BY_ANY: c_int = 8;
pub const IPP_ALG_HINT_FAST: c_int = 1;
pub const IPP_ALG_DIRECT: c_int = 1;
pub const IPP_DATATYPE_32F: c_int = 13;

#[link(name = "ippcore")]
#[link(name = "ipps")]
#[link(name = "ippvm")]
extern "C" {
    pub fn ippInit() -> IppStatus;
    pub fn ippSetNumThreads(n: c_int) -> IppStatus;

    pub fn ippsMalloc_8u(len: c_int) -> *mut Ipp8u;
    pub fn ippsMalloc_32f(len: c_int) -> *mut Ipp32f;
    pub fn ippsMalloc_32fc(len: c_int) -> *mut Ipp32fc;
    pub fn ippsFree(ptr: *mut c_void);

    pub fn ippsFFTGetSize_C_32fc(
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        spec_size: *mut c_int,
        spec_buf_size: *mut c_int,
        buf_size: *mut c_int,
    ) -> IppStatus;
    pub fn ippsFFTInit_C_32fc(
        spec: *mut *mut IppsFFTSpec_C_32fc,
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        spec_mem: *mut Ipp8u,
        spec_buf: *mut Ipp8u,
    ) -> IppStatus;
    pub fn ippsFFTFwd_CToC_32fc(
        src: *const Ipp32fc,
        dst: *mut Ipp32fc,
        spec: *const IppsFFTSpec_C_32fc,
        buf: *mut Ipp8u,
    ) -> IppStatus;
    pub fn ippsFFTInv_CToC_32fc(
        src: *const Ipp32fc,
        dst: *mut Ipp32fc,
        spec: *const IppsFFTSpec_C_32fc,
        buf: *mut Ipp8u,
    ) -> IppStatus;

    pub fn ippsAdd_32f(
        src1: *const Ipp32f,
        src2: *const Ipp32f,
        dst: *mut Ipp32f,
        len: c_int,
    ) -> IppStatus;
    pub fn ippsAdd_32f_I(src: *const Ipp32f, src_dst: *mut Ipp32f, len: c_int) -> IppStatus;
    pub fn ippsAdd_32fc(
        src1: *const Ipp32fc,
        src2: *const Ipp32fc,
        dst: *mut Ipp32fc,
        len: c_int,
    ) -> IppStatus;
    pub fn ippsAdd_32fc_I(src: *const Ipp32fc, src_dst: *mut Ipp32fc, len: c_int)
        -> IppStatus;
    pub fn ippsMul_32fc(
        src1: *const Ipp32fc,
        src2: *const Ipp32fc,
        dst: *mut Ipp32fc,
        len: c_int,
    ) -> IppStatus;
    pub fn ippsAddProduct_32fc(
        src1: *const Ipp32fc,
        src2: *const Ipp32fc,
        src_dst: *mut Ipp32fc,
        len: c_int,
    ) -> IppStatus;

    pub fn ippsFIRSRGetSize(
        taps_len: c_int,
        taps_type: IppDataType,
        spec_size: *mut c_int,
        buf_size: *mut c_int,
    ) -> IppStatus;
    pub fn ippsFIRSRInit_32f(
        taps: *const Ipp32f,
        taps_len: c_int,
        alg_type: IppAlgType,
        spec: *mut IppsFIRSpec_32f,
    ) -> IppStatus;
    pub fn ippsFIRSR_32f(
        src: *const Ipp32f,
        dst: *mut Ipp32f,
        num_iters: c_int,
        spec: *mut IppsFIRSpec_32f,
        dly_src: *const Ipp32f,
        dly_dst: *mut Ipp32f,
        buf: *mut Ipp8u,
    ) -> IppStatus;
}

static INIT: Once = Once::new();

pub fn init_once() {
    INIT.call_once(|| unsafe {
        let _ = ippInit();
        let _ = ippSetNumThreads(1);
    });
}
