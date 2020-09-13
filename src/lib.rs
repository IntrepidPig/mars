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
	pub(crate) physical_device: PhysicalDevice,
	pub(crate) device: Device,
	pub(crate) queue: Queue,
	pub(crate) command_pool: CommandPool,
	#[allow(unused)]
	pub(crate) debug_messenger: Option<rk::DebugUtilsMessengerInner>,
}

impl Context {
	pub fn create<C: PhysicalDeviceChooser>(app_name: &str, chooser: C) -> Result<Self, ContextCreateError> {
		let instance = create_instance(app_name)?;

		let debug_messenger = rk::create_debug_report_callback(
			&instance,
			vk::DebugUtilsMessageSeverityFlagsEXT::all(),
			vk::DebugUtilsMessageTypeFlagsEXT::all(),
			None,
		)
		.map_err(|_| log::warn!("Failed to create debug report callback"))
		.ok();

		let physical_device =
			rk::PhysicalDevice::choose(&instance, chooser).map_err(|_| ContextCreateError::NoDevice)?;
		let (device, queue) = create_device(&physical_device)?;
		let command_pool = CommandPool::create(&device)?;

		Ok(Self {
			physical_device,
			device,
			queue,
			command_pool,
			debug_messenger,
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
