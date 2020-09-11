use raw_window_handle::HasRawWindowHandle;

use rk::{vk, wsi::PresentationEngine};

use crate::{
	function::{FunctionDef, FunctionShader},
	render::RenderEngine,
	target::Target,
	Context, MarsResult,
};

pub struct WindowEngine<F: FunctionDef> {
	pub render: RenderEngine<F>,
	pub(crate) presentation_engine: PresentationEngine,
}

impl<F> WindowEngine<F>
where
	F: FunctionDef,
{
	pub fn new<W: HasRawWindowHandle>(
		context: &mut Context,
		window: &W,
		shader: FunctionShader<F>,
	) -> MarsResult<Self> {
		let handle = window.raw_window_handle();
		let surface = context.instance.create_surface_from_raw_window_handle(handle).unwrap();
		let surface_info = context
			.instance
			.get_surface_info(&context.physical_device, &surface)
			.unwrap();
		let surface_format = surface_info.formats[0];
		let swapchain = context
			.device
			.create_swapchain(&surface, vk::ImageUsageFlags::TRANSFER_DST, surface_format, None)
			.unwrap();
		let surface_size = swapchain.current_extent();
		let presentation_engine = PresentationEngine::new(&mut context.device, swapchain).unwrap();

		let target = Target::create(context, surface_size, shader)?;
		let render = RenderEngine::new(context, target)?;

		Ok(Self {
			render,
			presentation_engine,
		})
	}

	pub fn present(&mut self, context: &mut Context) -> MarsResult<()> {
		if let Some(new_extent) = self.presentation_engine.present(
			&mut context.device,
			&mut context.queue,
			&self.render.target.color_image,
		)? {
			self.render.target.resize(context, new_extent)?;
		}

		Ok(())
	}

	pub fn current_extent(&self) -> vk::Extent2D {
		self.render.target.extent
	}
}
