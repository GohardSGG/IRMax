#![cfg(not(target_os = "macos"))]

use crate::framework::vulkan::{VkBuffer, VulkanContext};
use ash::vk;
use std::ffi::CString;
use std::fs;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy)]
struct SparseOffset {
    start_idx: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    num_inputs: u32,
    num_outputs: u32,
    num_partitions: u32,
    freq_bins: u32,
    num_active_inputs: u32,
}

pub struct VulkanTailProcessor {
    context: Arc<VulkanContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; 2],
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    input_buffers: [VkBuffer; 2],
    ir_buffer: VkBuffer,
    output_buffer: VkBuffer,
    command_buffer_buf: VkBuffer,
    offset_buffer: VkBuffer,

    input_upload_buffers: [VkBuffer; 2],
    output_readback_buffer: VkBuffer,
    mapped_input_ptrs: [usize; 2],
    mapped_output_ptr: usize,

    host_input_stage: Vec<f32>,
    host_output_stage: Vec<f32>,

    num_outputs: usize,
    freq_bins: usize,
    params: ConvParams,

    frame_index: usize,
    input_stride_elems: usize,
    total_input_elems: usize,

    pub last_driver_us: u64,
    pub last_gpu_us: u64,
}

impl VulkanTailProcessor {
    pub fn new(
        context: Arc<VulkanContext>,
        num_inputs: usize,
        num_outputs: usize,
        block_size: usize,
        ir_data: &[f32],
        active_inputs: &[u32],
    ) -> Option<Self> {
        let freq_bins = block_size;
        let single_partition_floats = num_inputs * num_outputs * freq_bins * 2;
        if single_partition_floats == 0 {
            return None;
        }

        if ir_data.len() % single_partition_floats != 0 {
            println!("Error: IR data size mismatch");
            return None;
        }
        let num_partitions = ir_data.len() / single_partition_floats;

        // --- 1. Build Sparse Schedule ---
        let mut commands_packed: Vec<u32> = Vec::new();
        let mut offsets: Vec<SparseOffset> = Vec::with_capacity(num_outputs);

        let stride_freq = 2;
        let stride_out = freq_bins * stride_freq;
        let stride_in = num_outputs * stride_out;
        let stride_part = num_inputs * stride_in;
        // Exact mode: only pure zero blocks are skipped.
        let threshold = 0.0;

        for out_ch in 0..num_outputs {
            let start_idx = commands_packed.len() as u32;
            let mut count = 0u32;

            for p in 0..num_partitions {
                for &in_ch_real in active_inputs {
                    let base_idx =
                        p * stride_part + (in_ch_real as usize) * stride_in + out_ch * stride_out;
                    let slice = &ir_data[base_idx..base_idx + stride_out];
                    let mut max_val = 0.0f32;
                    for &v in slice {
                        let av = v.abs();
                        if av > max_val {
                            max_val = av;
                        }
                    }

                    if max_val > threshold {
                        let packed = ((in_ch_real as u32) << 16) | (p as u32);
                        commands_packed.push(packed);
                        count += 1;
                    }
                }
            }

            offsets.push(SparseOffset { start_idx, count });
        }

        println!(
            "🔧 [VulkanTail] 已生成 {} 条稀疏命令。",
            commands_packed.len()
        );

        // --- 2. Create Buffers ---
        let input_stride_elems = num_inputs * freq_bins * 2;
        let input_elems_raw = num_partitions * input_stride_elems;
        let input_elems = input_elems_raw.max(input_stride_elems);
        if num_partitions == 0 {
            println!("[VulkanTail] No tail partitions (IR shorter than tail_start); running in zero-tail mode.");
        }
        let input_bytes = (input_elems * 4) as vk::DeviceSize;
        let input_stride_bytes = (input_stride_elems * 4) as vk::DeviceSize;

        let output_elems = num_outputs * freq_bins * 2;
        let output_bytes = (output_elems * 4) as vk::DeviceSize;
        let ir_bytes = (ir_data.len() * 4) as vk::DeviceSize;

        let cmd_len = commands_packed.len().max(1);
        let cmd_bytes = (cmd_len * 4) as vk::DeviceSize;
        let offset_bytes = (offsets.len() * std::mem::size_of::<SparseOffset>()) as vk::DeviceSize;

        let device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let host_props =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let input_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let static_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let output_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let input_staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let output_staging_usage = vk::BufferUsageFlags::TRANSFER_DST;

        let buf_a = context.create_buffer(input_bytes, input_usage, device_local)?;
        let buf_b = context.create_buffer(input_bytes, input_usage, device_local)?;
        let ir_buf = context.create_buffer(ir_bytes, static_usage, device_local)?;
        let output_buf = context.create_buffer(output_bytes, output_usage, device_local)?;
        let cmd_buf = context.create_buffer(cmd_bytes, static_usage, device_local)?;
        let offset_buf = context.create_buffer(offset_bytes, static_usage, device_local)?;

        let upload_a = context.create_buffer(input_stride_bytes, input_staging_usage, host_props)?;
        let upload_b = context.create_buffer(input_stride_bytes, input_staging_usage, host_props)?;
        let output_readback = context.create_buffer(output_bytes, output_staging_usage, host_props)?;

        // --- 3. Build pipeline skeleton and initialize resources ---
        let processor = Self::build_skeleton(
            context,
            buf_a,
            buf_b,
            ir_buf,
            output_buf,
            cmd_buf,
            offset_buf,
            upload_a,
            upload_b,
            output_readback,
            num_inputs,
            num_outputs,
            num_partitions,
            freq_bins,
            active_inputs.len(),
        )?;

        processor.upload_f32_to_device(&processor.ir_buffer, ir_data, true)?;
        if !commands_packed.is_empty() {
            processor.upload_u32_to_device(&processor.command_buffer_buf, &commands_packed, true)?;
        } else {
            processor.upload_u32_to_device(&processor.command_buffer_buf, &[0u32], true)?;
        }
        processor.upload_structs_to_device(&processor.offset_buffer, &offsets, true)?;

        // Zero FDL/output device buffers to keep initial math deterministic.
        processor.zero_device_buffer(&processor.input_buffers[0])?;
        processor.zero_device_buffer(&processor.input_buffers[1])?;
        processor.zero_device_buffer(&processor.output_buffer)?;

        Some(processor)
    }

    fn build_skeleton(
        context: Arc<VulkanContext>,
        buf_a: VkBuffer,
        buf_b: VkBuffer,
        ir_buf: VkBuffer,
        output_buf: VkBuffer,
        cmd_buf: VkBuffer,
        offset_buf: VkBuffer,
        upload_a: VkBuffer,
        upload_b: VkBuffer,
        output_readback: VkBuffer,
        num_inputs: usize,
        num_outputs: usize,
        num_partitions: usize,
        freq_bins: usize,
        num_active_inputs: usize,
    ) -> Option<Self> {
        let device = &context.device;

        let shader_bytes = load_spv_bytes()?;
        let shader_module = create_shader_module(device, &shader_bytes)?;

        let entry = CString::new("main").ok()?;
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry);

        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];

        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None).ok()? };

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ConvParams>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).ok()? };

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .ok()?
        };
        let pipeline = pipelines[0];

        unsafe { device.destroy_shader_module(shader_module, None) };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 10,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(2);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).ok()? };

        let set_layouts = [descriptor_set_layout, descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).ok()? };
        let descriptor_sets = [sets[0], sets[1]];

        // Update descriptor sets for each input buffer
        update_descriptor_set(device, descriptor_sets[0], &buf_a, &ir_buf, &output_buf, &cmd_buf, &offset_buf);
        update_descriptor_set(device, descriptor_sets[1], &buf_b, &ir_buf, &output_buf, &cmd_buf, &offset_buf);

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(context.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None).ok()? };

        let alloc_cmd_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_cmd_info).ok()? }[0];

        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_info, None).ok()? };

        let output_elems = num_outputs * freq_bins * 2;
        let input_stride = num_inputs * freq_bins * 2;

        let mapped_input_a = unsafe {
            device
                .map_memory(upload_a.memory, 0, upload_a.size, vk::MemoryMapFlags::empty())
                .ok()?
        } as usize;

        let mapped_input_b = match unsafe {
            device.map_memory(upload_b.memory, 0, upload_b.size, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => ptr as usize,
            Err(_) => {
                unsafe {
                    device.unmap_memory(upload_a.memory);
                }
                return None;
            }
        };

        let mapped_output_ptr = match unsafe {
            device.map_memory(
                output_readback.memory,
                0,
                output_readback.size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(ptr) => ptr as usize,
            Err(_) => {
                unsafe {
                    device.unmap_memory(upload_a.memory);
                    device.unmap_memory(upload_b.memory);
                }
                return None;
            }
        };

        Some(Self {
            context,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            command_pool,
            command_buffer,
            fence,
            input_buffers: [buf_a, buf_b],
            ir_buffer: ir_buf,
            output_buffer: output_buf,
            command_buffer_buf: cmd_buf,
            offset_buffer: offset_buf,
            input_upload_buffers: [upload_a, upload_b],
            output_readback_buffer: output_readback,
            mapped_input_ptrs: [mapped_input_a, mapped_input_b],
            mapped_output_ptr,
            host_input_stage: vec![0.0; input_stride],
            host_output_stage: vec![0.0; output_elems],
            num_outputs,
            freq_bins,
            params: ConvParams {
                num_inputs: num_inputs as u32,
                num_outputs: num_outputs as u32,
                num_partitions: num_partitions as u32,
                freq_bins: freq_bins as u32,
                num_active_inputs: num_active_inputs as u32,
            },
            frame_index: 0,
            input_stride_elems: num_inputs * freq_bins * 2,
            total_input_elems: num_partitions * num_inputs * freq_bins * 2,
            last_driver_us: 0,
            last_gpu_us: 0,
        })
    }

    pub fn process(&mut self, input_complex: &[f32]) -> &[f32] {
        let device = &self.context.device;
        let t0 = Instant::now();

        let dst_index = if self.frame_index % 2 == 0 { 1 } else { 0 };
        let src_index = if self.frame_index % 2 == 0 { 0 } else { 1 };

        // 0) Prepare host input staging for current frame.
        let stride = self.input_stride_elems;
        let copy_len = input_complex.len().min(stride);
        self.host_input_stage[..copy_len].copy_from_slice(&input_complex[..copy_len]);
        if copy_len < stride {
            self.host_input_stage[copy_len..].fill(0.0);
        }

        unsafe {
            ptr::copy_nonoverlapping(
                self.host_input_stage.as_ptr(),
                self.mapped_input_ptrs[dst_index] as *mut f32,
                stride,
            );
        }

        if !self.input_upload_buffers[dst_index].coherent {
            let range = vk::MappedMemoryRange::builder()
                .memory(self.input_upload_buffers[dst_index].memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                device
                    .flush_mapped_memory_ranges(std::slice::from_ref(&range))
                    .ok();
            }
        }

        // 1) Record commands.
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }

        let begin_info =
            vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .unwrap();
        }

        // Host writes -> transfer reads from input upload buffer.
        let barrier_host = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier_host),
                &[],
                &[],
            );
        }

        // Shift FDL history: src[0..shift] -> dst[stride..]
        let shift_elems = self.total_input_elems.saturating_sub(self.input_stride_elems);
        if shift_elems > 0 {
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: (self.input_stride_elems * 4) as u64,
                size: (shift_elems * 4) as u64,
            };
            unsafe {
                device.cmd_copy_buffer(
                    self.command_buffer,
                    self.input_buffers[src_index].buffer,
                    self.input_buffers[dst_index].buffer,
                    std::slice::from_ref(&copy_region),
                );
            }
        }

        // Upload current block to dst FDL head.
        let upload_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (self.input_stride_elems * 4) as u64,
        };
        unsafe {
            device.cmd_copy_buffer(
                self.command_buffer,
                self.input_upload_buffers[dst_index].buffer,
                self.input_buffers[dst_index].buffer,
                std::slice::from_ref(&upload_region),
            );
        }

        // Transfer writes -> compute shader reads.
        let barrier_compute = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier_compute),
                &[],
                &[],
            );
        }

        // Compute dispatch.
        unsafe {
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_sets[dst_index]),
                &[],
            );
            let pc_bytes = as_u8_slice(&self.params);
            device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            let group_x = (self.freq_bins as u32).div_ceil(256);
            device.cmd_dispatch(self.command_buffer, group_x, self.num_outputs as u32, 1);
        }

        // Compute writes -> transfer reads.
        let barrier_output_copy = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier_output_copy),
                &[],
                &[],
            );
        }

        // Copy device output -> host-visible readback buffer.
        let output_copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: self.output_readback_buffer.size,
        };
        unsafe {
            device.cmd_copy_buffer(
                self.command_buffer,
                self.output_buffer.buffer,
                self.output_readback_buffer.buffer,
                std::slice::from_ref(&output_copy),
            );
        }

        // Transfer writes -> host reads.
        let barrier_host_read = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier_host_read),
                &[],
                &[],
            );
            device.end_command_buffer(self.command_buffer).unwrap();
        }

        // Submit and wait for completion (keeps existing math latency unchanged).
        unsafe {
            device
                .reset_fences(std::slice::from_ref(&self.fence))
                .unwrap();
            let submit_info =
                vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&self.command_buffer));
            device
                .queue_submit(self.context.queue, std::slice::from_ref(&submit_info), self.fence)
                .unwrap();
        }

        let t1 = Instant::now();

        unsafe {
            device
                .wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)
                .unwrap();
        }

        let t2 = Instant::now();

        if !self.output_readback_buffer.coherent {
            let range = vk::MappedMemoryRange::builder()
                .memory(self.output_readback_buffer.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                device
                    .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))
                    .ok();
            }
        }

        unsafe {
            ptr::copy_nonoverlapping(
                self.mapped_output_ptr as *const f32,
                self.host_output_stage.as_mut_ptr(),
                self.host_output_stage.len(),
            );
        }

        self.last_driver_us = t1.duration_since(t0).as_micros() as u64;
        self.last_gpu_us = t2.duration_since(t1).as_micros() as u64;
        self.frame_index += 1;

        &self.host_output_stage
    }

    pub fn reset_state(&mut self) {
        let _ = self.zero_device_buffer(&self.input_buffers[0]);
        let _ = self.zero_device_buffer(&self.input_buffers[1]);
        let _ = self.zero_device_buffer(&self.output_buffer);
        self.host_output_stage.fill(0.0);
        self.frame_index = 0;
    }

    fn upload_f32_to_device(&self, buffer: &VkBuffer, data: &[f32], zero_tail: bool) -> Option<()> {
        let src_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.upload_bytes_to_device(buffer, src_bytes, zero_tail)
    }

    fn upload_u32_to_device(&self, buffer: &VkBuffer, data: &[u32], zero_tail: bool) -> Option<()> {
        let src_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.upload_bytes_to_device(buffer, src_bytes, zero_tail)
    }

    fn upload_structs_to_device<T: Copy>(
        &self,
        buffer: &VkBuffer,
        data: &[T],
        zero_tail: bool,
    ) -> Option<()> {
        let src_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.upload_bytes_to_device(buffer, src_bytes, zero_tail)
    }

    fn upload_bytes_to_device(
        &self,
        buffer: &VkBuffer,
        src_bytes: &[u8],
        zero_tail: bool,
    ) -> Option<()> {
        let device = &self.context.device;
        let dst_size = buffer.size as usize;
        let bytes_to_copy = src_bytes.len().min(dst_size);
        let upload_size = if zero_tail { dst_size } else { bytes_to_copy };
        if upload_size == 0 {
            return Some(());
        }

        let staging = self.context.create_buffer(
            upload_size as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let ptr_raw = device
                .map_memory(staging.memory, 0, staging.size, vk::MemoryMapFlags::empty())
                .ok()?;
            let dst = ptr_raw as *mut u8;
            if bytes_to_copy > 0 {
                ptr::copy_nonoverlapping(src_bytes.as_ptr(), dst, bytes_to_copy);
            }
            if zero_tail && bytes_to_copy < upload_size {
                ptr::write_bytes(dst.add(bytes_to_copy), 0, upload_size - bytes_to_copy);
            }
            if !staging.coherent {
                let range = vk::MappedMemoryRange::builder()
                    .memory(staging.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                device
                    .flush_mapped_memory_ranges(std::slice::from_ref(&range))
                    .ok();
            }
            device.unmap_memory(staging.memory);
        }

        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: upload_size as vk::DeviceSize,
        };
        let result = self.execute_transfer(|device, command_buffer| unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                staging.buffer,
                buffer.buffer,
                std::slice::from_ref(&copy_region),
            );
        });

        destroy_buffer(device, &staging);
        result
    }

    fn zero_device_buffer(&self, buffer: &VkBuffer) -> Option<()> {
        self.execute_transfer(|device, command_buffer| unsafe {
            device.cmd_fill_buffer(command_buffer, buffer.buffer, 0, buffer.size, 0);
        })
    }

    fn execute_transfer<F>(&self, record: F) -> Option<()>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        let device = &self.context.device;

        unsafe {
            device
                .wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)
                .ok()?;
            device.reset_fences(std::slice::from_ref(&self.fence)).ok()?;
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .ok()?;
        }

        let begin_info =
            vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .ok()?;
        }

        record(device, self.command_buffer);

        unsafe {
            device.end_command_buffer(self.command_buffer).ok()?;
            let submit_info =
                vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&self.command_buffer));
            device
                .queue_submit(self.context.queue, std::slice::from_ref(&submit_info), self.fence)
                .ok()?;
            device
                .wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)
                .ok()?;
        }

        Some(())
    }
}

impl Drop for VulkanTailProcessor {
    fn drop(&mut self) {
        unsafe {
            let device = &self.context.device;
            let _ = device.device_wait_idle();

            device.unmap_memory(self.input_upload_buffers[0].memory);
            device.unmap_memory(self.input_upload_buffers[1].memory);
            device.unmap_memory(self.output_readback_buffer.memory);

            device.destroy_fence(self.fence, None);
            device.free_command_buffers(
                self.command_pool,
                std::slice::from_ref(&self.command_buffer),
            );
            device.destroy_command_pool(self.command_pool, None);

            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);

            destroy_buffer(device, &self.input_buffers[0]);
            destroy_buffer(device, &self.input_buffers[1]);
            destroy_buffer(device, &self.ir_buffer);
            destroy_buffer(device, &self.output_buffer);
            destroy_buffer(device, &self.command_buffer_buf);
            destroy_buffer(device, &self.offset_buffer);
            destroy_buffer(device, &self.input_upload_buffers[0]);
            destroy_buffer(device, &self.input_upload_buffers[1]);
            destroy_buffer(device, &self.output_readback_buffer);
        }
    }
}

fn destroy_buffer(device: &ash::Device, buffer: &VkBuffer) {
    unsafe {
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}

fn update_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    input_buf: &VkBuffer,
    ir_buf: &VkBuffer,
    output_buf: &VkBuffer,
    cmd_buf: &VkBuffer,
    offset_buf: &VkBuffer,
) {
    let input_info = vk::DescriptorBufferInfo {
        buffer: input_buf.buffer,
        offset: 0,
        range: vk::WHOLE_SIZE,
    };
    let ir_info = vk::DescriptorBufferInfo {
        buffer: ir_buf.buffer,
        offset: 0,
        range: vk::WHOLE_SIZE,
    };
    let out_info = vk::DescriptorBufferInfo {
        buffer: output_buf.buffer,
        offset: 0,
        range: vk::WHOLE_SIZE,
    };
    let cmd_info = vk::DescriptorBufferInfo {
        buffer: cmd_buf.buffer,
        offset: 0,
        range: vk::WHOLE_SIZE,
    };
    let offset_info = vk::DescriptorBufferInfo {
        buffer: offset_buf.buffer,
        offset: 0,
        range: vk::WHOLE_SIZE,
    };

    let writes = [
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&input_info))
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&ir_info))
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&out_info))
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&cmd_info))
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&offset_info))
            .build(),
    ];

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

fn load_spv_bytes() -> Option<Vec<u8>> {
    if let Ok(path) = std::env::var("IRMAX_VULKAN_SPV") {
        if let Ok(bytes) = fs::read(path) {
            return Some(bytes);
        }
    }

    // Primary path for packaged builds: embed SPIR-V directly into the binary.
    let embedded = include_bytes!("../shaders/lib/fused_convolution.spv");
    if !embedded.is_empty() {
        return Some(embedded.to_vec());
    }

    // Development fallback: source-tree shader file.
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("shaders");
    path.push("lib");
    path.push("fused_convolution.spv");
    fs::read(path).ok()
}

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> Option<vk::ShaderModule> {
    let mut words: Vec<u32> = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let create_info = vk::ShaderModuleCreateInfo::builder().code(&words);
    unsafe { device.create_shader_module(&create_info, None).ok() }
}

fn as_u8_slice<T: Sized>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((val as *const T) as *const u8, std::mem::size_of::<T>()) }
}
