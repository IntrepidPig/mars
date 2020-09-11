use thiserror::Error;

use rk::{
	ash::extensions,
	command::CommandPool,
	device::{Device, Queue},
	instance::Instance,
	PhysicalDevice, PhysicalDeviceChooser,
};

// Look at all these leaks
pub use rk;
pub use rk::ash;
pub use rk::ash::vk;

pub mod buffer;
pub mod function;
pub mod image;
pub mod math;
pub mod pass;
pub mod render;
pub mod target;
pub mod window;

pub type MarsResult<T> = rk::VkResult<T>;

pub struct Context {
	pub(crate) instance: Instance,
	pub(crate) physical_device: PhysicalDevice,
	pub(crate) device: Device,
	pub(crate) queue: Queue,
	pub(crate) command_pool: CommandPool,
}

impl Context {
	pub fn create<C: PhysicalDeviceChooser>(app_name: &str, chooser: C) -> Result<Self, ContextCreateError> {
		let instance = create_instance(app_name)?;
		let physical_device =
			rk::PhysicalDevice::choose(&instance, chooser).map_err(|_| ContextCreateError::NoDevice)?;
		let (mut device, queue) = create_device(&physical_device)?;
		let command_pool = device.create_command_pool()?;

		Ok(Self {
			instance,
			physical_device,
			device,
			queue,
			command_pool,
		})
	}
}

#[derive(Debug, Error)]
pub enum ContextCreateError {
	#[error(transparent)]
	InstanceError(#[from] ash::InstanceError),
	#[error("No suitable graphics device was found")]
	NoDevice,
	#[error("No queue supporting graphics and transfer operations was found on the selected device")]
	NoQueue,
	#[error("Vulkan error: {0}")]
	VulkanError(#[from] vk::Result),
}

fn create_instance(app_name: &str) -> Result<Instance, ContextCreateError> {
	let entry = rk::create_entry().expect("Failed to load Vulkan entry");

	let mut extensions = Instance::new_extensions_list();
	extensions.add_extension::<extensions::ext::DebugUtils>();
	extensions.add_extension::<extensions::khr::Surface>();
	extensions.add_extension::<extensions::khr::XlibSurface>();
	extensions.add_extension::<extensions::khr::WaylandSurface>();

	let instance = Instance::create(
		&entry,
		app_name,
		(0, 1, 0),
		"mars",
		(0, 1, 0),
		(1, 2, 0),
		vec![String::from("VK_LAYER_KHRONOS_validation")],
		&extensions,
	)?;

	/* let _ = rk::create_debug_report_callback(&instance, vk::DebugUtilsMessageSeverityFlagsEXT::all(), vk::DebugUtilsMessageTypeFlagsEXT::all(), None)
	.map_err(|_| log::warn!("Failed to create debug report callback")); */

	Ok(instance)
}

fn create_device(physical_device: &PhysicalDevice) -> Result<(Device, Queue), ContextCreateError> {
	let queue_family_index = physical_device
		.find_queue_family_index(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER)
		.ok_or(ContextCreateError::NoQueue)?;
	let mut device_extensions = Device::new_extensions_list();
	device_extensions.add_extension::<extensions::khr::Swapchain>();
	let (device, queue) = Device::create(
		physical_device,
		queue_family_index,
		vec![String::from("VK_LAYER_KHRONOS_validation")],
		&device_extensions,
	)?;
	Ok((device, queue))
}
