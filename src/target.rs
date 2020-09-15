use rk::{pass::Framebuffer};

use crate::{
	pass::{RenderPass, RenderPassPrototype, Attachments},
	Context, MarsResult,
};

pub struct Target<G: RenderPassPrototype> {
	pub(crate) render_pass: RenderPass<G>,
	pub(crate) attachments: Attachments<G>,
	pub(crate) framebuffer: Framebuffer,
}

impl<G: RenderPassPrototype> Target<G> {
	pub fn create(context: &Context, render_pass: RenderPass<G>, attachments: Attachments<G>) -> MarsResult<Self> {
		let framebuffer = Self::create_framebuffer(context, &render_pass, &attachments)?;
		Ok(Self {
			render_pass,
			attachments,
			framebuffer,
		})
	}

	pub fn change_attachments(&mut self, context: &Context, attachments: Attachments<G>) -> MarsResult<()> {
		self.framebuffer = Self::create_framebuffer(context, &self.render_pass, &attachments)?;
		self.attachments = attachments;
		Ok(())
	}

	pub fn render_pass(&self) -> &RenderPass<G> {
		&self.render_pass
	}

	pub fn attachments(&self) -> &Attachments<G> {
		&self.attachments
	}

	pub fn color_attachments(&self) -> &G::ColorAttachments {
		&self.attachments.color_attachments
	}

	fn create_framebuffer(context: &Context, render_pass: &RenderPass<G>, attachments: &Attachments<G>) -> MarsResult<Framebuffer> {
		let extent = attachments.extent();
		context.device.create_framebuffer(
			&render_pass.render_pass,
			attachments.as_raw(),
			extent.width,
			extent.height,
			1,
		)
	}
}
