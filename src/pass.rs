use rk::{
	vk,
	device::{Device},
	pass::{self, RenderPass as RkRenderPass},
};

use crate::{
	Context, MarsResult,
	image::{Image, FormatType},
};

pub struct RenderPass {
	pub(crate) render_pass: RkRenderPass,
}

impl RenderPass {
	pub fn create(context: &Context) -> MarsResult<Self> {
		let render_pass = create_render_pass(&context.device);
		Ok(Self {
			render_pass,
		})
	}
}

pub unsafe trait InputAttachments {

}

pub unsafe trait ColorAttachments {

}

pub unsafe trait DepthAttachment {

}

pub struct NoDepthAttachment;

unsafe impl DepthAttachment for NoDepthAttachment {

}

pub struct SomeDepthAttachment;

pub trait SubpassPrototype {
	type InputAttachments: InputAttachments;
	type ColorAttachments: ColorAttachments;
	type DepthAttachment: DepthAttachment;
}

pub struct Subpass {

}

pub struct ColorAttachment {
	pub load_op: vk::AttachmentLoadOp,
	pub store_op: vk::AttachmentStoreOp,
	
}

fn create_render_pass(device: &Device) -> RkRenderPass {
	let input_attachments = vec![];
	let color_attachments = vec![pass::ColorAttachment {
		color: pass::AttachmentRef {
			attachment: 0,
			layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
		},
		resolve: None,
	}];
	let depth_attachment = pass::AttachmentRef {
		attachment: 1,
		layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	let subpass = device.create_subpass(input_attachments, color_attachments, Some(depth_attachment));

	let color_attachment = pass::Attachment {
		format: vk::Format::B8G8R8A8_UNORM,
		samples: vk::SampleCountFlags::TYPE_1,
		load_op: vk::AttachmentLoadOp::LOAD,
		store_op: vk::AttachmentStoreOp::STORE,
		stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
		stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
		initial_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
		final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
	};
	let depth_attachment = pass::Attachment {
		format: vk::Format::D32_SFLOAT,
		samples: vk::SampleCountFlags::TYPE_1,
		load_op: vk::AttachmentLoadOp::LOAD,
		store_op: vk::AttachmentStoreOp::STORE,
		stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
		stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
		initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	let render_pass = device
		.create_render_pass(vec![color_attachment, depth_attachment], vec![subpass], vec![])
		.expect("Failed to create render pass");

	render_pass
}
