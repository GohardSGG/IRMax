#![cfg(not(target_os = "macos"))]

use ash::{vk, Entry};
use std::ffi::CString;
use std::sync::Arc;

pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub coherent: bool,
}

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

pub fn probe_vulkan_available() -> bool {
    let entry = unsafe { Entry::load() };
    let Ok(entry) = entry else {
        return false;
    };

    let app_name = CString::new("IRMaxProbe").ok();
    let app_info = app_name.as_ref().map(|name| {
        vk::ApplicationInfo::builder()
            .application_name(name)
            .engine_name(name)
            .api_version(vk::make_api_version(0, 1, 1, 0))
    });

    let mut create_info = vk::InstanceCreateInfo::builder();
    if let Some(info) = app_info.as_ref() {
        create_info = create_info.application_info(info);
    }

    let instance = unsafe { entry.create_instance(&create_info, None) };
    let Ok(instance) = instance else {
        return false;
    };

    let physical_devices = unsafe { instance.enumerate_physical_devices() };
    let Ok(physical_devices) = physical_devices else {
        unsafe { instance.destroy_instance(None) };
        return false;
    };

    let mut ok = false;
    for device in physical_devices {
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        if queue_families
            .iter()
            .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        {
            ok = true;
            break;
        }
    }

    unsafe { instance.destroy_instance(None) };
    ok
}

impl VulkanContext {
    pub fn new() -> Option<Arc<Self>> {
        let entry = unsafe { Entry::load().ok()? };
        let app_name = CString::new("IRMax").ok()?;

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&app_name)
            .api_version(vk::make_api_version(0, 1, 1, 0));

        let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None).ok()? };

        let physical_devices = unsafe { instance.enumerate_physical_devices().ok()? };
        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return None;
        }

        let mut best: Option<(vk::PhysicalDevice, u32)> = None;
        for device in physical_devices {
            let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };
            let queue_family_index = queue_families
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|i| i as u32);

            let Some(queue_family_index) = queue_family_index else {
                continue;
            };

            let props = unsafe { instance.get_physical_device_properties(device) };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                _ => 4,
            };

            let replace = match best {
                None => true,
                Some((best_dev, _)) => {
                    let best_props = unsafe { instance.get_physical_device_properties(best_dev) };
                    let best_score = match best_props.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                        vk::PhysicalDeviceType::CPU => 3,
                        _ => 4,
                    };
                    score < best_score
                }
            };

            if replace {
                best = Some((device, queue_family_index));
            }
        }

        let Some((physical_device, queue_family_index)) = best else {
            unsafe { instance.destroy_instance(None) };
            return None;
        };

        // Keep compute queue priority moderate to avoid starving desktop graphics work.
        let queue_priorities = [0.5f32];
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&queue_info));
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None).ok()? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None).ok()? };

        Some(Arc::new(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            memory_properties,
        }))
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<VkBuffer> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None).ok()? };

        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let mut requested = properties;
        let mut found = self.find_memory_type(mem_requirements.memory_type_bits, requested);
        if found.is_none() && requested.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            requested = requested & !vk::MemoryPropertyFlags::HOST_COHERENT;
            found = self.find_memory_type(mem_requirements.memory_type_bits, requested);
        }

        let (memory_type_index, coherent) = found?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None).ok()? };

        unsafe {
            self.device.bind_buffer_memory(buffer, memory, 0).ok()?;
        }

        Some(VkBuffer {
            buffer,
            memory,
            size: mem_requirements.size,
            coherent,
        })
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<(u32, bool)> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) == 0 {
                continue;
            }
            let mem_type = self.memory_properties.memory_types[i as usize];
            if mem_type.property_flags.contains(properties) {
                let coherent = mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                return Some((i, coherent));
            }
        }
        None
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
