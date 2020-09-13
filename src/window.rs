use raw_window_handle::HasRawWindowHandle;

use rk::{vk, wsi::{PresentationEngine, Surface}};

use crate::{
	render::RenderEngine,
	pass::{RenderPass},
	target::Target,
	Context, MarsResult,
};

pub struct WindowEngine {
	pub render: RenderEngine,
	pub(crate) presentation_engine: PresentationEngine,
	pub(crate) current_extent: vk::Extent2D,
}

impl WindowEngine
{
	pub fn new<W: HasRawWindowHandle>(
		context: &Context,
		window: &W,
	) -> MarsResult<Self> {
		let handle = window.raw_window_handle();
		let surface = unsafe { Surface::create_from_raw_handle(&context.physical_device, handle).unwrap() };
		let surface_info = unsafe { surface.get_info()? };
		let surface_format = surface_info.formats[0];
		let swapchain = context
			.device
			.create_swapchain(&surface, vk::ImageUsageFlags::TRANSFER_DST, surface_format, None)
			.unwrap();
		let surface_size = swapchain.current_extent();
		let presentation_engine = unsafe { PresentationEngine::new(swapchain).unwrap() };

		let render_pass = RenderPass::create(context)?;
		let render = RenderEngine::new(context, render_pass)?;

		Ok(Self {
			render,
			presentation_engine,
			current_extent: surface_size,
		})
	}

	pub fn create_target(&mut self, context: &mut Context, render_pass: &mut RenderPass) -> MarsResult<Target> {
		Target::create(context, render_pass, self.current_extent)
	}

	pub fn present(&mut self, context: &mut Context, target: &mut Target) -> MarsResult<()> {
		let new_extent_opt = context.queue.with_lock(|| {
			unsafe { self.presentation_engine.present(
				&context.queue,
				&target.color_image,
			) }
		})?;
		if let Some(new_extent) = new_extent_opt {
			target.resize(context, &mut self.render.render_pass, new_extent)?;
			self.current_extent = new_extent;
		}

		Ok(())
	}

	pub fn current_extent(&self) -> vk::Extent2D {
		self.current_extent
	}
}
