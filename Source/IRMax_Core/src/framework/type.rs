#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use libc::c_float;

// OS-neutral split-complex type (used by FFT + loader)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DSMComplex {
    pub realp: *mut c_float,
    pub imagp: *mut c_float,
}

pub type DSPComplex = DSMComplex;
