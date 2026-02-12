#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Structures matching Rust/C++
struct SparseCmd {
    uint partition_idx; // u16 in Rust, aligned to u32 in std430 often, but let's check packing. 
                        // Actually standard alignment for struct is base alignment of largest member.
                        // u16 is not standard GLSL type. Using uint. 
                        // We will need to unpack or use bitfield if we want to save space, 
                        // but for simplicity we assume the buffer is packed as u16 and we read uint?
                        // No, GLSL std430 layout aligns to 4 bytes. 
                        // Let's assume we pass data as u32 for simplicity or pack them.
                        // Rust struct: { partition_idx: u16, input_idx: u16 } -> 4 bytes total.
                        // We can read this as a single uint and unpack.
    uint input_idx; 
};
// Wait, to match `u16, u16` packed in 4 bytes:
// GLSL `uint` is 4 bytes.
// We should declare the buffer as `uint[]` and unpack manually.

struct SparseOffset {
    uint start_idx;
    uint count;
};

struct ConvParams {
    uint num_inputs;
    uint num_outputs;
    uint num_partitions;
    uint freq_bins;
    uint num_active_inputs;
};

layout(set = 0, binding = 0) buffer InputBuffer {
    float input_data[]; // [Partition][Input][Freq * 2]
};

layout(set = 0, binding = 1) buffer IrBuffer {
    float ir_data[]; // [Partition][Input][Output][Freq * 2]
};

layout(set = 0, binding = 2) buffer OutputBuffer {
    float output_data[]; // [Output][Freq * 2]
};

layout(set = 0, binding = 3) buffer CommandBuffer {
    uint commands_packed[]; // Packed {u16 partition, u16 input}
};

layout(set = 0, binding = 4) buffer OffsetBuffer {
    SparseOffset offsets[];
};

layout(push_constant) uniform PushConstants {
    ConvParams params;
} pc;

void main() {
    uint out_ch = gl_GlobalInvocationID.y;
    uint f = gl_GlobalInvocationID.x;

    if (out_ch >= pc.params.num_outputs || f >= pc.params.freq_bins) {
        return;
    }

    float sum_r = 0.0;
    float sum_i = 0.0;

    SparseOffset offset = offsets[out_ch];
    uint start = offset.start_idx;
    uint end = start + offset.count;

    for (uint i = start; i < end; i++) {
        uint packed_cmd = commands_packed[i];
        uint p = packed_cmd & 0xFFFF;
        uint in_ch = (packed_cmd >> 16) & 0xFFFF;

        // Calculate Input Index
        // input_idx = p * (num_inputs * freq_bins * 2) + in_ch * (freq_bins * 2) + f * 2
        uint input_idx = p * pc.params.num_inputs * pc.params.freq_bins * 2 +
                         in_ch * pc.params.freq_bins * 2 +
                         f * 2;

        float in_r = input_data[input_idx];
        float in_i = input_data[input_idx + 1];

        // Calculate IR Index
        // ir_idx = p * (num_inputs * num_outputs * freq_bins * 2) + 
        //          in_ch * (num_outputs * freq_bins * 2) + 
        //          out_ch * (freq_bins * 2) + 
        //          f * 2
        uint ir_idx = p * pc.params.num_inputs * pc.params.num_outputs * pc.params.freq_bins * 2 +
                      in_ch * pc.params.num_outputs * pc.params.freq_bins * 2 +
                      out_ch * pc.params.freq_bins * 2 +
                      f * 2;
        
        float ir_r = ir_data[ir_idx];
        float ir_i = ir_data[ir_idx + 1];

        // Complex Multiply (packed real: bin0 holds DC/Nyquist in re/imag)
        if (f == 0) {
            sum_r += in_r * ir_r;
            sum_i += in_i * ir_i;
        } else {
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            sum_r += in_r * ir_r - in_i * ir_i;
            sum_i += in_r * ir_i + in_i * ir_r;
        }
    }

    uint out_idx = out_ch * pc.params.freq_bins * 2 + f * 2;
    output_data[out_idx] = sum_r;
    output_data[out_idx + 1] = sum_i;
}
