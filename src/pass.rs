use std::{marker::PhantomData, sync::Arc};

use rk::{
	image::{ImageLayoutTransition, ImageViewInner as RkImageViewInner},
	pass::{self, RenderPass as RkRenderPass},
	vk,
};

use crate::{
	image::{usage, DynImageUsage, FormatType, Image, ImageView},
	math::*,
	Context, MarsResult,
};

pub struct RenderPass<G: SubpassGraph> {
	pub(crate) render_pass: RkRenderPass,
	_phantom: PhantomData<G>,
}

impl<G> RenderPass<G>
where
	G: SubpassGraph,
{
	pub fn create(context: &Context, graph: &G) -> MarsResult<Self> {
		let (attachments, subpasses, dependencies) = graph.desc();
		let render_pass = unsafe {
			context
				.device
				.create_render_pass(attachments, subpasses, dependencies)?
		};
		Ok(Self {
			render_pass,
			_phantom: PhantomData,
		})
	}
}

pub unsafe trait InputAttachments {
	fn desc() -> Vec<pass::Attachment>;

	fn as_raw(&self) -> Vec<Arc<RkImageViewInner>>;

	fn clears(&self, color: Vec4, depth: f32) -> Vec<vk::ClearValue>;
}

unsafe impl InputAttachments for () {
	fn desc() -> Vec<pass::Attachment> {
		Vec::new()
	}

	fn as_raw(&self) -> Vec<Arc<RkImageViewInner>> {
		Vec::new()
	}

	fn clears(&self, _color: Vec4, _depth: f32) -> Vec<vk::ClearValue> {
		Vec::new()
	}
}

fn collapse_resolve<T>(vec: Vec<(T, Option<T>)>) -> Vec<T> {
	let mut buf = Vec::new();
	for item in vec {
		buf.push(item.0);
		if let Some(item) = item.1 {
			buf.push(item);
		}
	}
	buf
}

pub unsafe trait ColorAttachmentType {
	fn desc() -> (pass::Attachment, Option<pass::Attachment>);

	fn as_raw(&self) -> (Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>);

	fn clear(&self, color: Vec4) -> (vk::ClearValue, Option<vk::ClearValue>);
}

// TODO: use a subtrait that ensures the format is a color format
pub struct ColorAttachment<F: FormatType> {
	// TODO: make not pub and add getters instead
	pub image: Image<usage::ColorAttachment, F>,
	pub view: ImageView<usage::ColorAttachment, F>,
}

impl<F> ColorAttachment<F>
where
	F: FormatType,
{
	pub fn new(image: Image<usage::ColorAttachment, F>, view: ImageView<usage::ColorAttachment, F>) -> Self {
		Self { image, view }
	}

	pub fn create(context: &Context, usage: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self> {
		let mut image = Image::create(context, usage | DynImageUsage::COLOR_ATTACHMENT, extent)?;
		image.transition(
			context,
			&ImageLayoutTransition {
				aspect: vk::ImageAspectFlags::COLOR,
				src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
				dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
				src_access_mask: vk::AccessFlags::empty(),
				dst_access_mask: vk::AccessFlags::MEMORY_READ,
				old_layout: vk::ImageLayout::UNDEFINED,
				new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			},
		)?;
		let image = image.cast_usage(usage::ColorAttachment).map_err(|_| ()).unwrap();
		let view = ImageView::create(&image)?;
		Ok(Self::new(image, view))
	}
}

unsafe impl<F> ColorAttachmentType for ColorAttachment<F>
where
	F: FormatType,
{
	fn desc() -> (pass::Attachment, Option<pass::Attachment>) {
		assert!(F::aspect().contains(vk::ImageAspectFlags::COLOR));

		(
			pass::Attachment {
				format: F::as_raw(),
				samples: vk::SampleCountFlags::TYPE_1,
				load_op: vk::AttachmentLoadOp::LOAD,
				store_op: vk::AttachmentStoreOp::STORE,
				stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
				stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
				initial_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			},
			None,
		)
	}

	fn as_raw(&self) -> (Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>) {
		(self.view.image_view.clone(), None)
	}

	fn clear(&self, color: Vec4) -> (vk::ClearValue, Option<vk::ClearValue>) {
		(
			vk::ClearValue {
				color: vk::ClearColorValue {
					float32: [color.x, color.y, color.z, color.w],
				},
			},
			None,
		)
	}
}

pub unsafe trait ColorAttachments {
	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)>;

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)>;

	fn clears(&self, color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)>;
}

unsafe impl ColorAttachments for () {
	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)> {
		Vec::new()
	}

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)> {
		Vec::new()
	}

	fn clears(&self, _color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)> {
		Vec::new()
	}
}

unsafe impl<A> ColorAttachments for (A,)
where
	A: ColorAttachmentType,
{
	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)> {
		vec![A::desc()]
	}

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)> {
		vec![self.0.as_raw()]
	}

	fn clears(&self, color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)> {
		vec![self.0.clear(color)]
	}
}

unsafe impl<A, B> ColorAttachments for (A, B)
where
	A: ColorAttachmentType,
	B: ColorAttachmentType,
{
	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)> {
		vec![A::desc(), B::desc()]
	}

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)> {
		vec![self.0.as_raw(), self.1.as_raw()]
	}

	fn clears(&self, color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)> {
		vec![self.0.clear(color), self.1.clear(color)]
	}
}

pub unsafe trait DepthAttachmentType {
	fn desc() -> Option<pass::Attachment>;

	fn as_raw(&self) -> Option<Arc<RkImageViewInner>>;

	fn clear(&self, depth: f32) -> Option<vk::ClearValue>;
}

pub struct NoDepthAttachment;

unsafe impl DepthAttachmentType for NoDepthAttachment {
	fn desc() -> Option<pass::Attachment> {
		None
	}

	fn as_raw(&self) -> Option<Arc<RkImageViewInner>> {
		None
	}

	fn clear(&self, _depth: f32) -> Option<vk::ClearValue> {
		None
	}
}

pub struct DepthAttachment<F: FormatType> {
	pub image: Image<usage::DepthStencilAttachment, F>,
	pub view: ImageView<usage::DepthStencilAttachment, F>,
}

impl<F> DepthAttachment<F>
where
	F: FormatType,
{
	pub fn new(
		image: Image<usage::DepthStencilAttachment, F>,
		view: ImageView<usage::DepthStencilAttachment, F>,
	) -> Self {
		Self { image, view }
	}

	pub fn create(context: &Context, usage: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self> {
		let mut image = Image::create(context, usage | DynImageUsage::DEPTH_STENCIL_ATTACHMENT, extent)?;
		image.transition(
			context,
			&ImageLayoutTransition {
				aspect: vk::ImageAspectFlags::DEPTH,
				src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
				dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
				src_access_mask: vk::AccessFlags::empty(),
				dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
					| vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
				old_layout: vk::ImageLayout::UNDEFINED,
				new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			},
		)?;
		let image = image.cast_usage(usage::DepthStencilAttachment).map_err(|_| ()).unwrap();
		let view = ImageView::create(&image)?;
		Ok(Self::new(image, view))
	}
}

unsafe impl<F> DepthAttachmentType for DepthAttachment<F>
where
	F: FormatType,
{
	fn desc() -> Option<pass::Attachment> {
		assert!(F::aspect().contains(vk::ImageAspectFlags::DEPTH));

		Some(pass::Attachment {
			format: F::as_raw(),
			samples: vk::SampleCountFlags::TYPE_1,
			load_op: vk::AttachmentLoadOp::LOAD,
			store_op: vk::AttachmentStoreOp::STORE,
			stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
			stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
			initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		})
	}

	fn as_raw(&self) -> Option<Arc<RkImageViewInner>> {
		Some(self.view.image_view.clone())
	}

	fn clear(&self, depth: f32) -> Option<vk::ClearValue> {
		Some(vk::ClearValue {
			depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
		})
	}
}

pub unsafe trait Attachments {
	fn as_raw(&self) -> Vec<Arc<RkImageViewInner>>;

	fn extent(&self) -> vk::Extent2D;

	fn clears(&self, color: Vec4, depth: f32) -> Vec<vk::ClearAttachment>;
}

pub unsafe trait SubpassGraph {
	type Attachments: Attachments;

	fn desc(&self) -> (Vec<pass::Attachment>, Vec<pass::Subpass>, Vec<pass::Dependency>);
}

pub trait SubpassPrototype {
	type InputAttachments: InputAttachments;
	type ColorAttachments: ColorAttachments;
	type DepthAttachment: DepthAttachmentType;
}

pub mod graphs {
	use super::*;

	pub struct SingleSubpassAttachments<S: SubpassPrototype> {
		pub input_attachments: S::InputAttachments,
		pub color_attachments: S::ColorAttachments,
		pub depth_attachment: S::DepthAttachment,
		extent: vk::Extent2D,
	}

	impl<S: SubpassPrototype> SingleSubpassAttachments<S> {
		pub(crate) fn new(
			input_attachments: S::InputAttachments,
			color_attachments: S::ColorAttachments,
			depth_attachment: S::DepthAttachment,
			extent: vk::Extent2D,
		) -> Self {
			Self {
				input_attachments,
				color_attachments,
				depth_attachment,
				extent,
			}
		}
	}

	unsafe impl<S: SubpassPrototype> Attachments for SingleSubpassAttachments<S> {
		fn as_raw(&self) -> Vec<Arc<RkImageViewInner>> {
			let mut attachments = Vec::new();
			attachments.append(&mut self.input_attachments.as_raw());
			for (color, resolve) in self.color_attachments.as_raw() {
				attachments.push(color);
				if let Some(resolve) = resolve {
					attachments.push(resolve);
				}
			}
			if let Some(depth) = self.depth_attachment.as_raw() {
				attachments.push(depth)
			}
			attachments
		}

		fn extent(&self) -> vk::Extent2D {
			self.extent
		}

		fn clears(&self, color: Vec4, depth: f32) -> Vec<vk::ClearAttachment> {
			let mut clears = Vec::new();
			let mut attachment = 0;
			clears.extend(self.input_attachments.clears(color, depth).into_iter().map(|clear| {
				let color_attachment = attachment;
				attachment += 1;
				#[allow(unreachable_code)]
				vk::ClearAttachment {
					aspect_mask: vk::ImageAspectFlags::COLOR,
					color_attachment,
					clear_value: clear,
				}
			}));
			clears.extend(
				collapse_resolve(self.color_attachments.clears(color))
					.into_iter()
					.map(|clear| {
						let color_attachment = attachment;
						attachment += 1;
						#[allow(unreachable_code)]
						vk::ClearAttachment {
							aspect_mask: vk::ImageAspectFlags::COLOR,
							color_attachment,
							clear_value: clear,
						}
					}),
			);
			self.depth_attachment.clear(depth).map(|clear| {
				clears.push(vk::ClearAttachment {
					aspect_mask: vk::ImageAspectFlags::DEPTH,
					color_attachment: attachment,
					clear_value: clear,
				})
			});

			clears
		}
	}

	unsafe impl<S: SubpassPrototype> SubpassGraph for S {
		type Attachments = SingleSubpassAttachments<S>;

		fn desc(&self) -> (Vec<pass::Attachment>, Vec<pass::Subpass>, Vec<pass::Dependency>) {
			let mut attachments = Vec::new();
			let mut input_refs = Vec::new();
			let mut color_refs = Vec::new();
			let mut depth_ref = None;
			let mut index = 0;

			let inputs = S::InputAttachments::desc();
			for input in &inputs {
				attachments.push(*input);
				input_refs.push(pass::AttachmentRef {
					attachment: index,
					layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
				});
				index += 1;
			}
			let colors = S::ColorAttachments::desc();
			for (color, resolve) in &colors {
				attachments.push(*color);
				color_refs.push(pass::ColorAttachment {
					color: pass::AttachmentRef {
						attachment: index,
						layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
					},
					resolve: if resolve.is_some() {
						index += 1;
						Some(pass::AttachmentRef {
							attachment: index,
							layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
						})
					} else {
						None
					},
				});
				index += 1;
			}
			let depth = S::DepthAttachment::desc();
			if let Some(depth) = &depth {
				attachments.push(*depth);
				depth_ref = Some(pass::AttachmentRef {
					attachment: index,
					layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				})
			}

			let subpass = pass::Subpass {
				input_attachments: input_refs,
				color_attachments: color_refs,
				depth_stencil_attachment: depth_ref,
			};

			(attachments, vec![subpass], Vec::new())
		}
	}
}

pub mod subpasses {
	use super::graphs::*;
	use super::*;

	pub struct SimpleSubpassPrototype<C: FormatType, D: FormatType>(PhantomData<(C, D)>);

	impl<C, D> SimpleSubpassPrototype<C, D>
	where
		C: FormatType,
		D: FormatType,
	{
		pub fn new() -> Self {
			Self(PhantomData)
		}

		pub fn create_attachments(
			context: &Context,
			usage: DynImageUsage,
			extent: vk::Extent2D,
		) -> MarsResult<SingleSubpassAttachments<Self>> {
			let color_attachment = ColorAttachment::create(context, usage, extent)?;
			let depth_attachment = DepthAttachment::create(context, DynImageUsage::empty(), extent)?;
			let attachments = SingleSubpassAttachments::new((), (color_attachment,), depth_attachment, extent);
			Ok(attachments)
		}
	}

	impl<C, D> SubpassPrototype for SimpleSubpassPrototype<C, D>
	where
		C: FormatType,
		D: FormatType,
	{
		type InputAttachments = ();
		type ColorAttachments = (ColorAttachment<C>,);
		type DepthAttachment = DepthAttachment<D>;
	}
}
