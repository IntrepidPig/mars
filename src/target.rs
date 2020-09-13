use rk::{
	device::Device,
	image::{Image as RkImage, ImageView as RkImageView},
	pass::{Framebuffer},
	vk,
};

use crate::{
	Context, MarsResult,
	pass::{RenderPass},
	math::*,
};

pub struct Target {
	pub(crate) extent: vk::Extent2D,
	pub(crate) color_image: RkImage,
	pub(crate) depth_image: RkImage,
	pub(crate) framebuffer: Framebuffer,
}

impl Target {
	pub fn create(context: &mut Context, render_pass: &mut RenderPass, extent: vk::Extent2D) -> MarsResult<Self> {
		let initialization = Self::initialize(&mut context.device, render_pass, extent)?;
		Ok(Self {
			extent,
			color_image: initialization.color_image,
			depth_image: initialization.depth_image,
			framebuffer: initialization.framebuffer,
		})
	}

	fn initialize(
		device: &mut Device,
		render_pass: &mut RenderPass,
		extent: vk::Extent2D,
	) -> MarsResult<Initialization> {
		let (color_image, depth_image, color_image_view, depth_image_view) = create_attachments(device, extent)?;
		let framebuffer = device.create_framebuffer(
			&render_pass.render_pass,
			vec![color_image_view, depth_image_view],
			extent.width,
			extent.height,
			1,
		)?;
		Ok(Initialization {
			color_image,
			depth_image,
			framebuffer,
		})
	}

	pub fn resize(&mut self, context: &mut Context, render_pass: &mut RenderPass, new_extent: vk::Extent2D) -> MarsResult<()> {
		let initialization = Self::initialize(&mut context.device, render_pass, new_extent)?;
		self.color_image = initialization.color_image;
		self.depth_image = initialization.depth_image;
		self.framebuffer = initialization.framebuffer;
		self.extent = new_extent;
		Ok(())
	}

	pub fn clear(&mut self, context: &mut Context, color: Vec4) -> MarsResult<()> {
		let float32 = [color.x, color.y, color.z, color.w];
		let pending = unsafe {
			context.queue.with_lock(|| {
				context.queue.quick_submit(&context.command_pool, &[], &[], |command_buffer| {
					command_buffer.clear_color_image(
						&self.color_image,
						vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
						vk::ClearColorValue { float32 },
					);
					command_buffer.clear_depth_image(
						&self.depth_image,
						vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
						vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
					);
				})
			})?
		};
		pending.wait()?;
		Ok(())
	}
}

struct Initialization {
	color_image: RkImage,
	depth_image: RkImage,
	//color_image_view: ImageView,
	//depth_image_view: ImageView,
	framebuffer: Framebuffer,
}

fn create_attachments(device: &Device, extent: vk::Extent2D) -> MarsResult<(RkImage, RkImage, RkImageView, RkImageView)> {
	unsafe {
		let color_image = RkImage::create(
			device,
			vk::Format::B8G8R8A8_UNORM,
			vk::Extent3D {
				width: extent.width,
				height: extent.height,
				depth: 1,
			},
			vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
			vk::ImageLayout::UNDEFINED,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		)?;
		let depth_image = RkImage::create(
			device,
			vk::Format::D32_SFLOAT,
			vk::Extent3D {
				width: extent.width,
				height: extent.height,
				depth: 1,
			},
			vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
			vk::ImageLayout::UNDEFINED,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		)?;
		let color_image_view = RkImageView::create(&color_image, vk::ImageAspectFlags::COLOR)?;
		let depth_image_view = RkImageView::create(&depth_image, vk::ImageAspectFlags::DEPTH)?;
		Ok((color_image, depth_image, color_image_view, depth_image_view))
	}
}
