use std::sync::Arc;

use rk::pass::{Framebuffer, RenderPassInner};

use crate::{
	pass::{Attachments, RenderPass, RenderPassPrototype},
	Context, MarsResult,
};

pub struct Target<G: RenderPassPrototype> {
	pub(crate) render_pass: Arc<RenderPassInner>,
	pub(crate) attachments: Attachments<G>,
	pub(crate) framebuffer: Framebuffer,
}

impl<G: RenderPassPrototype> Target<G> {
	pub fn create(context: &Context, render_pass: &RenderPass<G>, attachments: Attachments<G>) -> MarsResult<Self> {
		let render_pass = render_pass.render_pass.clone();
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

	pub fn attachments(&self) -> &Attachments<G> {
		&self.attachments
	}

	pub fn color_attachments(&self) -> &G::ColorAttachments {
		&self.attachments.color_attachments
	}

	fn create_framebuffer(
		context: &Context,
		render_pass: &Arc<RenderPassInner>,
		attachments: &Attachments<G>,
	) -> MarsResult<Framebuffer> {
		let extent = attachments.extent();
		context
			.device
			.create_framebuffer(render_pass, attachments.as_raw(), extent.width, extent.height, 1)
	}
}
