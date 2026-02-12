#include <cuda_runtime.h>

typedef float2 Complex;

__device__ inline void complex_fma(Complex a, Complex b, Complex& acc) {
    acc.x = fmaf(a.x, b.x, acc.x);
    acc.x = fmaf(-a.y, b.y, acc.x);
    acc.y = fmaf(a.x, b.y, acc.y);
    acc.y = fmaf(a.y, b.x, acc.y);
}

struct SparseCmd {
    unsigned short partition_idx;
    unsigned short input_idx;
};

struct SparseOffset {
    unsigned int start_idx;
    unsigned int count;
};

struct ConvParams {
    unsigned int num_inputs;
    unsigned int num_outputs;
    unsigned int num_partitions;
    unsigned int freq_bins;
    unsigned int num_active_inputs;
};

extern "C" __global__ void fused_mimo(
    const float* __restrict__ input_fdl,      // [Partition][Input][Freq][Complex]
    const float* __restrict__ ir_buf,         // [Partition][Input][Output][Freq][Complex]
    float* __restrict__ output_buf,           // [Output][Freq][Complex]
    ConvParams params,
    const SparseCmd* __restrict__ commands,
    const SparseOffset* __restrict__ offsets
) {
    // 1. Calculate Frequency Bin Index
    unsigned int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= params.freq_bins) return;

    // 2. Identify Output Channel
    unsigned int out_ch = blockIdx.y;
    if (out_ch >= params.num_outputs) return;

    // 3. Setup Accumulator
    Complex sum = make_float2(0.0f, 0.0f);

    // 4. Retrieve Sparse Commands for this Output
    SparseOffset offset = offsets[out_ch];
    unsigned int start = offset.start_idx;
    unsigned int end = start + offset.count;

    // 5. Iterate through contributing impulse responses
    for (unsigned int i = start; i < end; ++i) {
        SparseCmd cmd = commands[i];
        unsigned int p = cmd.partition_idx;
        unsigned int in_ch = cmd.input_idx;

        // Input Index: [p][in_ch][f]
        // Stride Freq = freq_bins * 2 (Complex)
        // Stride Input = Stride Freq * num_inputs ?? 
        // Let's verify layout.
        // tail_cuda.rs: dst_buf holds the FDL.
        // FDL Layout: [Partition][Input][Freq][Complex] or [Partition][Input][Freq] ?
        // tail_cuda.rs L163: input_elems = num_partitions * num_inputs * freq_bins * 2;
        // So it is contiguous. Flat index logic:
        // idx = p * (num_inputs * params.freq_bins * 2) + in_ch * (params.freq_bins * 2) + f * 2;
        
        unsigned long long input_idx = 
            (unsigned long long)p * params.num_inputs * params.freq_bins * 2 + 
            (unsigned long long)in_ch * params.freq_bins * 2 + 
            f * 2;

        float val_r = input_fdl[input_idx];
        float val_i = input_fdl[input_idx + 1];
        Complex input_val = make_float2(val_r, val_i);

        // IR Index: [p][in_ch][out_ch][f]
        // tail_cuda.rs L135: p * stride_part + in * stride_in + out * stride_out
        // stride_out = freq_bins * 2
        // stride_in = num_outputs * stride_out
        // stride_part = num_inputs * stride_in
        
        unsigned long long ir_idx = 
            (unsigned long long)p * params.num_inputs * params.num_outputs * params.freq_bins * 2 +
            (unsigned long long)in_ch * params.num_outputs * params.freq_bins * 2 +
            (unsigned long long)out_ch * params.freq_bins * 2 +
            f * 2;

        float ir_r = ir_buf[ir_idx];
        float ir_i = ir_buf[ir_idx + 1];
        Complex ir_val = make_float2(ir_r, ir_i);

        // MAC (packed real: bin0 holds DC/Nyquist in re/imag)
        if (f == 0) {
            sum.x = fmaf(input_val.x, ir_val.x, sum.x);
            sum.y = fmaf(input_val.y, ir_val.y, sum.y);
        } else {
            complex_fma(input_val, ir_val, sum);
        }
    }

    // 6. Write Output
    // Output Layout: [Output][Freq][Complex]
    // idx = out_ch * freq_bins * 2 + f * 2
    unsigned long long out_idx = (unsigned long long)out_ch * params.freq_bins * 2 + f * 2;
    output_buf[out_idx] = sum.x;
    output_buf[out_idx + 1] = sum.y;
}
