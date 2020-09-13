use rk::{pass::Framebuffer, vk};

use crate::{
	pass::{Attachments, RenderPass, SubpassGraph},
	Context, MarsResult,
};

pub struct Target<G: SubpassGraph> {
	pub(crate) extent: vk::Extent2D,
	#[allow(unused)]
	pub(crate) attachments: G::Attachments,
	pub(crate) framebuffer: Framebuffer,
}

impl<G: SubpassGraph> Target<G> {
	pub fn create(context: &Context, render_pass: &RenderPass<G>, attachments: G::Attachments) -> MarsResult<Self> {
		let extent = attachments.extent();
		let framebuffer = context.device.create_framebuffer(
			&render_pass.render_pass,
			attachments.as_raw(),
			extent.width,
			extent.height,
			1,
		)?;
		Ok(Self {
			extent,
			attachments,
			framebuffer: framebuffer,
		})
	}

	pub fn attachments(&self) -> &G::Attachments {
		&self.attachments
	}
}
