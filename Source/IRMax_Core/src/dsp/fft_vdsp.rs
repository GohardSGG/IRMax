#![cfg(target_os = "macos")]

use crate::accelerate::*;

/// 封装 vDSP FFT 上下文
/// 包含 Setup 复用、复数转换逻辑和正确的缩放处理
pub struct FftContext {
    setup: FFTSetup,
    log2n: vDSP_Length,
    n: usize,
    half_n: usize,
    // Workspaces for format conversion if needed,
    // but typically we let caller manage buffers for max efficiency.
    // We only hold the setup.
}

unsafe impl Send for FftContext {}
unsafe impl Sync for FftContext {}

impl FftContext {
    /// 创建 FFT 上下文
    /// size: FFT 点数 (必须是 2 的幂, e.g. 512)
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "FFT size must be power of 2");

        let log2n = (size as f32).log2() as vDSP_Length;
        let setup = unsafe { vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) };

        if setup.is_null() {
            panic!("vDSP_create_fftsetup failed (OOM?)");
        }

        Self {
            setup,
            log2n,
            n: size,
            half_n: size / 2,
        }
    }

    /// 执行 R2C (Real to Complex) 变换
    /// input: 时域数据 (长度 N)
    /// output_split: 频域数据 (Split Complex 格式)
    ///   - output_split.realp 与 imagp 长度应为 N/2
    ///   - 注意: vDSP 的 R2C 结果遵从 Packing 规则 (DC在 imagp[0]...)
    pub fn forward(&self, input: &[f32], output_split: &mut DSPComplex) {
        assert_eq!(input.len(), self.n);
        // vDSP_fft_zrip 要求 Input 是经过 Pack 的 (ctoz)
        // 或者我们手动做 ctoz

        unsafe {
            // 1. Interleaved (f32) -> Split Complex
            // Treat input as interleaved complex (stride 2) of length N/2
            // output_split is destination
            vDSP_ctoz(
                input.as_ptr(), // No cast needed, takes *const c_float
                2,
                output_split,
                1,
                self.half_n as vDSP_Length,
            );

            // 2. FFT In-Place (on split buffer)
            vDSP_fft_zrip(
                self.setup,
                output_split,
                1,
                self.log2n,
                FFTDirection(kFFTDirection_Forward),
            );
        }
    }

    /// 执行 C2R (Complex to Real) 变换
    /// 包含 1/(2N) 缩放!
    pub fn inverse(&self, input_split: &mut DSPComplex, output: &mut [f32]) {
        assert_eq!(output.len(), self.n);

        unsafe {
            // 1. IFFT (Result is scaled by 2N)
            vDSP_fft_zrip(
                self.setup,
                input_split,
                1,
                self.log2n,
                FFTDirection(kFFTDirection_Inverse),
            );

            // 2. Split Complex -> Interleaved
            vDSP_ztoc(
                input_split,
                1,
                output.as_mut_ptr() as *mut DSPComplex,
                2,
                self.half_n as vDSP_Length,
            );

            // 3. Scaling (1/2N)
            // vDSP FFT ZRIP Forward + Inverse = 2N * Original
            // Factor = 0.5 / N
            let scale_factor = 0.5 / (self.n as f32);
            vDSP_vsmul(
                output.as_ptr(),
                1,
                &scale_factor,
                output.as_mut_ptr(),
                1,
                self.n as vDSP_Length,
            );
        }
    }
}

impl Drop for FftContext {
    fn drop(&mut self) {
        unsafe {
            vDSP_destroy_fftsetup(self.setup);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_fft_roundtrip_scaling() {
        let n = 256;
        let fft = FftContext::new(n);

        // Signal: Sine wave
        let mut input: Vec<f32> = (0..n)
            .map(|i| (i as f32 / n as f32 * 2.0 * PI).sin())
            .collect();
        let original = input.clone();

        // Buffers
        let mut real = vec![0.0f32; n / 2];
        let mut imag = vec![0.0f32; n / 2];
        let mut split = DSPComplex {
            realp: real.as_mut_ptr(),
            imagp: imag.as_mut_ptr(),
        };

        // Forward
        // input gets modified by ctoz conceptually if we passed it as split,
        // but here we pass &mut input. Wait, ctoz reads from Input and writes to Output.
        // Input is technically const in ctoz, but we often modify it for temp usage.
        // My API takes &mut input for forward? Let's check impl.
        // forward(&mut input, ...) -> ctoz input as const.
        // So input is preserved?
        // Ah, ctoz reads interleaved.
        // Wait, does ctoz destroy input? No, const.
        // But I used &mut [f32]. It should be & [f32] for forward if it's strictly read.
        // Let's verify accelerate signature.
        // `input.as_ptr() as *const DSPComplex` -> It's const.
        // So safe to take &[f32].

        fft.forward(&mut input, &mut split); // Using mut to be safe, but data is copied

        // Inverse
        let mut output = vec![0.0f32; n];
        fft.inverse(&mut split, &mut output);

        // Check
        let mut max_diff = 0.0f32;
        for i in 0..n {
            let diff = (output[i] - original[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        println!("FFT Roundtrip Max Diff: {:.8}", max_diff);
        assert!(max_diff < 1e-6, "FFT scaling or logic incorrect");
    }
}
