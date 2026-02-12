// Basic bindings for Apple Accelerate vDSP
// Needed because we don't have a dedicated crate for it yet.
// These names match Apple's C API, so we suppress Rust naming warnings.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(dead_code)]

#[cfg(target_os = "macos")]
use libc::c_float;
use libc::{c_int, c_long, c_void};
pub use super::r#type::{DSPComplex, DSMComplex};

pub type FFTSetup = *mut c_void;
pub type vDSP_Length = c_long;
pub type vDSP_Stride = c_long;

// Constants from vDSP.h
pub const kFFTRadix2: c_int = 0;
pub const kFFTDirection_Forward: c_int = 1;
pub const kFFTDirection_Inverse: c_int = -1;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    pub fn vDSP_create_fftsetup(__Log2n: vDSP_Length, __Radix: c_int) -> FFTSetup;

    pub fn vDSP_destroy_fftsetup(__Setup: FFTSetup);

    // ZIP = Z (Split Complex) In-Place
    pub fn vDSP_fft_zrip(
        __Setup: FFTSetup,
        __C: *const DSMComplex,
        __Stride: vDSP_Stride,
        __Log2n: vDSP_Length,
        __Direction: c_int,
    );

    // vDSP_ctoz: Interleaved -> Split
    pub fn vDSP_ctoz(
        __C: *const c_float, // Actually interleaved float*
        __Stride: vDSP_Stride,
        __Z: *const DSMComplex,
        __ZStride: vDSP_Stride,
        __N: vDSP_Length,
    );

    // vDSP_ztoc: Split -> Interleaved
    pub fn vDSP_ztoc(
        __Z: *const DSMComplex,
        __ZStride: vDSP_Stride,
        __C: *mut DSMComplex,
        __Stride: vDSP_Stride,
        __N: vDSP_Length,
    );

    // vDSP_zvmul: Split Complex Vector Multiply
    pub fn vDSP_zvmul(
        __A: *const DSMComplex,
        __I: vDSP_Stride,
        __B: *const DSMComplex,
        __J: vDSP_Stride,
        __C: *const DSMComplex,
        __K: vDSP_Stride,
        __N: vDSP_Length,
        __U: c_int, // usually 0 or 1 for conjugate? No, zvmul is just mul.
                    // Wait, check signature.
                    // void vDSP_zvmul(const DSPSplitComplex *__A, vDSP_Stride __I, const DSPSplitComplex *__B, vDSP_Stride __J, const DSPSplitComplex *__C, vDSP_Stride __K, vDSP_Length __N, int __U);
                    // __U: +1 for conjugate A, -1 for conjugate C ?
                    // Usually 1 means normal multiplication.
    );

    pub fn vDSP_zvma(
        __A: *const DSMComplex,
        __I: vDSP_Stride,
        __B: *const DSMComplex,
        __J: vDSP_Stride,
        __C: *const DSMComplex,
        __K: vDSP_Stride,
        __D: *const DSMComplex,
        __L: vDSP_Stride,
        __N: vDSP_Length,
    );

    // vDSP_conv: Correlation / Convolution
    pub fn vDSP_conv(
        __A: *const c_float,
        __I: vDSP_Stride,
        __F: *const c_float, // Filter
        __J: vDSP_Stride,
        __C: *mut c_float, // Result
        __K: vDSP_Stride,
        __N: vDSP_Length, // Result Length (SignalLen - FilterLen + 1)
        __P: vDSP_Length, // Filter Length
    );

    // vDSP_vsmul: Vector Scalar Multiply
    pub fn vDSP_vsmul(
        __A: *const c_float,
        __I: vDSP_Stride,
        __B: *const c_float, // Scalar
        __C: *mut c_float,
        __K: vDSP_Stride,
        __N: vDSP_Length,
    );

    // vDSP_vadd: Vector Add
    pub fn vDSP_vadd(
        __A: *const c_float,
        __I: vDSP_Stride,
        __B: *const c_float,
        __J: vDSP_Stride,
        __C: *mut c_float,
        __K: vDSP_Stride,
        __N: vDSP_Length,
    );

    // vDSP_mmov: Matrix Move (Strided Copy)
    pub fn vDSP_mmov(
        __A: *const c_float, // Source
        __C: *mut c_float,   // Dest
        __M: vDSP_Length,    // Input/Output Columns (Chunk Size)
        __N: vDSP_Length,    // Input Rows (Frames)
        __P: vDSP_Length,    // Source Columns Count (Stride)
        __Q: vDSP_Length,    // Dest Columns Count (Stride)
    );

    // vDSP_mmovD: Double precision version? No, vDSP_mmov is float.

    // vDSP_vlint: Linear Interpolation (Vector)
    // C[i] = A[i] + B * (A[i+1] - A[i]) ?
    // No, vDSP_vlint(input, input_stride, index_vector, index_stride, output, output_stride, length, interpolation_length)
    // Wait, we need linear crossfade.
    // Use vDSP_vintb maybe?
    // Actually simpler: Input Copy doesn't need vlint. Mixing might.
    // Let's stick to vmmov first.
}

pub fn FFTRadix(r: c_int) -> c_int {
    r
}
pub fn FFTDirection(d: c_int) -> c_int {
    d
}
