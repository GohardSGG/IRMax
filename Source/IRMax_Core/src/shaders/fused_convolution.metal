// Filename: fused_convolution.metal
// Plan A: Safe & Correct Implementation
// - One thread per (FreqBin, OutputChannel)
// - No Shared Memory (Avoids indexing bugs)
// - No Register Tiling (Low Register Pressure)
// - Correct Logic verified by Math Suite

#include <metal_stdlib>
using namespace metal;

typedef float2 Complex;

struct ConvParams {
    uint num_inputs;      
    uint num_outputs;     
    uint num_partitions;  
    uint freq_bins;       
    uint num_active_inputs;
};

// Complex Multiply-Add: acc += a * b
inline void complex_fma(thread Complex& acc, Complex a, Complex b) {
    acc.x = fma(a.x, b.x, acc.x);
    acc.x = fma(-a.y, b.y, acc.x);
    acc.y = fma(a.x, b.y, acc.y);
    acc.y = fma(a.y, b.x, acc.y);
}

// ------------------------------------------------------------------
// KERNEL: Standard 1-to-1 Mapping
// Grid: [freq_bins, num_outputs, 1]
// ------------------------------------------------------------------
struct SparseCmd {
    ushort partition_idx;
    ushort input_idx;
};

struct SparseOffset {
    uint start_idx;
    uint count;
};

// ------------------------------------------------------------------
// KERNEL: Sparse Scheduling (Zero-Skipping)
// Grid: [freq_bins, num_outputs, 1]
// ------------------------------------------------------------------
kernel void fused_mimo_convolution(
    device const Complex* input_fd [[buffer(0)]],     
    device const Complex* ir_fd [[buffer(1)]],        
    device Complex* output_fd [[buffer(2)]],          
    constant ConvParams& params [[buffer(3)]],
    device const SparseCmd* command_buffer [[buffer(4)]],   
    device const SparseOffset* offset_buffer [[buffer(5)]],
    
    uint2 gid [[thread_position_in_grid]] // (freq_bin, output_ch)
) {
    uint freq_bin = gid.x;
    uint out_ch = gid.y;

    // Boundary check
    if (freq_bin >= params.freq_bins || out_ch >= params.num_outputs) {
        return;
    }
    
    Complex acc = float2(0.0f, 0.0f);
    
    // Pre-calculate strides
    uint stride_in_ch = params.freq_bins;
    uint stride_in_part = params.num_inputs * params.freq_bins;
    
    uint stride_ir_out = params.freq_bins;
    uint stride_ir_in = params.num_outputs * params.freq_bins;
    uint stride_ir_part = params.num_inputs * stride_ir_in;

    // Retrieve Sparse Task List for this Output Channel
    SparseOffset offset = offset_buffer[out_ch];
    uint start = offset.start_idx;
    uint end = start + offset.count;
    
    // Loop only Active Tasks (Skipping Zeros)
    for (uint k = start; k < end; k++) {
        SparseCmd cmd = command_buffer[k];
        uint p = (uint)cmd.partition_idx;
        uint in_real = (uint)cmd.input_idx;

        // 1. Load Input
        // Input Layout: [Partition][Input][Freq]
        uint in_idx = p * stride_in_part + in_real * stride_in_ch + freq_bin;
        Complex in_val = input_fd[in_idx];
            
        // 2. Load IR
        // IR Layout: [Partition][Input][Output][Freq] (Wait, check Rust side)
        // Rust side said: [Partition][In][Out][Freq] (Input Major??)
        // Let's re-verify `tail_gpu.rs`:
        // base_idx = p * stride_part + in_ch * stride_in + out_ch * stride_out;
        // Yes, Rust calc confirms: Partition -> Input -> Output -> Freq.
        
        uint ir_idx = p * stride_ir_part 
                    + in_real * stride_ir_in 
                    + out_ch * stride_ir_out 
                    + freq_bin;
            
        Complex ir_val = ir_fd[ir_idx];
            
        // 3. Accumulate
        if (freq_bin == 0) {
            // DC/Nyquist Mode (Packed Real)
            acc.x = fma(in_val.x, ir_val.x, acc.x);
            acc.y = fma(in_val.y, ir_val.y, acc.y);
        } else {
            // Standard Complex Mode
            complex_fma(acc, in_val, ir_val);
        }
    }
    // Write Output
    uint out_idx = out_ch * params.freq_bins + freq_bin;
    output_fd[out_idx] = acc;
}
