use std::{marker::PhantomData, sync::Arc};

use rk::{
	image::{ImageLayoutTransition, ImageViewInner as RkImageViewInner},
	pass::{self, RenderPass as RkRenderPass},
	vk,
};

use crate::{
	image::{usage, samples::{SampleCount1}, SampleCountType, MultiSampleCountType, DynImageUsage, FormatType, Image, ImageView},
	math::*,
	Context, MarsResult,
};

pub trait RenderPassPrototype {
	type SampleCount: SampleCountType;
	type InputAttachments: InputAttachments;
	type ColorAttachments: ColorAttachments<Self::SampleCount>;
	type DepthAttachment: DepthAttachmentType<Self::SampleCount>;
}

pub struct RenderPass<G: RenderPassPrototype> {
	pub(crate) render_pass: RkRenderPass,
	_phantom: PhantomData<G>,
}

impl<G> RenderPass<G>
where
	G: RenderPassPrototype,
{
	pub fn create(context: &Context) -> MarsResult<Self> {
		let (attachments, subpasses, dependencies) = get_render_pass_desc::<G>();
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

fn get_render_pass_desc<G: RenderPassPrototype>() -> (Vec<pass::Attachment>, Vec<pass::Subpass>, Vec<pass::Dependency>)
{
	let mut attachments = Vec::new();
	let mut input_refs = Vec::new();
	let mut color_refs = Vec::new();
	let mut depth_ref = None;
	let mut index = 0;

	let inputs = G::InputAttachments::desc();
	for input in &inputs {
		attachments.push(*input);
		input_refs.push(pass::AttachmentRef {
			attachment: index,
			layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
		});
		index += 1;
	}
	let colors = G::ColorAttachments::desc();
	for (color, resolve) in &colors {
		attachments.push(*color);
		if let Some(resolve) = resolve {
			attachments.push(*resolve);
		}
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
	let depth = G::DepthAttachment::desc();
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

pub struct Attachments<G: RenderPassPrototype> {
	pub(crate) extent: vk::Extent2D,
	pub(crate) input_attachments: G::InputAttachments,
	pub(crate) color_attachments: G::ColorAttachments,
	pub(crate) depth_attachment: G::DepthAttachment,
}

impl<G> Attachments<G>
where
	G: RenderPassPrototype,
{
	// TODO: allow more granular specification of usages
	pub fn create(context: &Context, extent: vk::Extent2D, color_usages: DynImageUsage) -> MarsResult<Self> {
		let input_attachments = G::InputAttachments::create(context, DynImageUsage::empty(), extent)?;
		let color_attachments = G::ColorAttachments::create(context, color_usages, extent)?;
		let depth_attachment = G::DepthAttachment::create(context, DynImageUsage::empty(), extent)?;
		Ok(Self {
			extent,
			input_attachments,
			color_attachments,
			depth_attachment,
		})
	}

	pub fn extent(&self) -> vk::Extent2D {
		self.extent
	}

	pub fn input_attachments(&self) -> &G::InputAttachments {
		&self.input_attachments
	}

	pub fn color_attachments(&self) -> &G::ColorAttachments {
		&self.color_attachments
	}

	pub fn depth_attachment(&self) -> &G::DepthAttachment {
		&self.depth_attachment
	}

	pub(crate) fn as_raw(&self) -> Vec<Arc<RkImageViewInner>> {
		self.input_attachments
			.as_raw()
			.into_iter()
			.chain(
				self.color_attachments
					.as_raw()
					.into_iter()
					.map(|(color, resolve)| {
						if let Some(resolve) = resolve {
							vec![color, resolve]
						} else {
							vec![color]
						}
					})
					.flatten(),
			)
			.chain(self.depth_attachment().as_raw().into_iter())
			.collect()
	}

	pub(crate) fn clears(
		&self,
		colors: <G::ColorAttachments as ColorAttachments<G::SampleCount>>::ClearValues,
		depth: <G::DepthAttachment as DepthAttachmentType<G::SampleCount>>::ClearValue,
	) -> Vec<vk::ClearAttachment> {
		let mut clear_attachments = Vec::new();
		let input_attachments_count = G::InputAttachments::desc().len() as u32;
		for color in colors.as_raw() {
			let idx = clear_attachments.len() as u32 + input_attachments_count;
			clear_attachments.push(vk::ClearAttachment {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				color_attachment: idx,
				clear_value: vk::ClearValue { color },
			})
		}
		if let Some(depth_stencil) = depth.as_raw() {
			clear_attachments.push(vk::ClearAttachment {
				aspect_mask: vk::ImageAspectFlags::DEPTH,
				color_attachment: vk::ATTACHMENT_UNUSED,
				clear_value: vk::ClearValue { depth_stencil },
			})
		}
		clear_attachments
	}
}

pub unsafe trait InputAttachments: Sized {
	fn desc() -> Vec<pass::Attachment>;

	fn as_raw(&self) -> Vec<Arc<RkImageViewInner>>;

	fn clears(&self, color: Vec4, depth: f32) -> Vec<vk::ClearValue>;

	fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self>;
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

	fn create(_context: &Context, _usages: DynImageUsage, _extent: vk::Extent2D) -> MarsResult<Self> {
		Ok(())
	}
}

/* fn collapse_resolve<T>(vec: Vec<(T, Option<T>)>) -> Vec<T> {
	let mut buf = Vec::new();
	for item in vec {
		buf.push(item.0);
		if let Some(item) = item.1 {
			buf.push(item);
		}
	}
	buf
} */

pub unsafe trait ColorAttachmentType<S: SampleCountType>: Sized {
	type ClearValue: ColorClearValue;

	fn desc() -> (pass::Attachment, Option<pass::Attachment>);

	fn as_raw(&self) -> (Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>);

	fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self>;
}

// TODO: use a subtrait that ensures the format is a color format
pub struct ColorAttachment<F: FormatType> {
	// TODO: make not pub and add getters instead
	pub image: Image<usage::ColorAttachment, F, SampleCount1>,
	pub view: ImageView<usage::ColorAttachment, F, SampleCount1>,
}

impl<F> ColorAttachment<F>
where
	F: FormatType,
{
	pub(crate) fn new(image: Image<usage::ColorAttachment, F, SampleCount1>, view: ImageView<usage::ColorAttachment, F, SampleCount1>) -> Self {
		Self { image, view }
	}
}

unsafe impl<F> ColorAttachmentType<SampleCount1> for ColorAttachment<F>
where
	F: FormatType,
	F::Pixel: ColorClearValue,
{
	type ClearValue = F::Pixel;

	fn desc() -> (pass::Attachment, Option<pass::Attachment>) {
		// TODO: implement subtype traits for formats and image usages to avoid these asserts
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

	/* fn clear(&self, color: Vec4) -> (vk::ClearValue, Option<vk::ClearValue>) {
		(
			vk::ClearValue {
				color: vk::ClearColorValue {
					float32: [color.x, color.y, color.z, color.w],
				},
			},
			None,
		)
	} */

	fn create(context: &Context, usage: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self> {
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

pub struct MultisampledColorAttachment<F: FormatType, S: MultiSampleCountType> {
	// TODO: fields not pub for fear of user changing them to wrongly-sized images
	#[allow(unused)]
	pub color_image: Image<usage::ColorAttachment, F, S>,
	pub color_image_view: ImageView<usage::ColorAttachment, F, S>,
	#[allow(unused)]
	pub resolve_image: Image<usage::ColorAttachment, F, SampleCount1>,
	pub resolve_image_view: ImageView<usage::ColorAttachment, F, SampleCount1>,
}

unsafe impl<F, S> ColorAttachmentType<S> for MultisampledColorAttachment<F, S> where F: FormatType, F::Pixel: ColorClearValue, S: MultiSampleCountType {
    type ClearValue = F::Pixel;

    fn desc() -> (pass::Attachment, Option<pass::Attachment>) {
		assert!(F::aspect().contains(vk::ImageAspectFlags::COLOR));

		(
			pass::Attachment {
				format: F::as_raw(),
				samples: S::as_raw(),
				load_op: vk::AttachmentLoadOp::LOAD,
				store_op: vk::AttachmentStoreOp::STORE,
				stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
				stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
				initial_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			},
			Some(pass::Attachment {
				format: F::as_raw(),
				samples: vk::SampleCountFlags::TYPE_1,
				load_op: vk::AttachmentLoadOp::LOAD,
				store_op: vk::AttachmentStoreOp::STORE,
				stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
				stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
				initial_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			}),
		)
    }

    fn as_raw(&self) -> (Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>) {
        (self.color_image_view.image_view.clone(), Some(self.resolve_image_view.image_view.clone()))
    }

    fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self> {
        let mut color_image = Image::create(context, usages | DynImageUsage::COLOR_ATTACHMENT, extent)?;
		color_image.transition(
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
		let color_image = color_image.cast_usage(usage::ColorAttachment).map_err(|_| ()).unwrap();
		let color_image_view = ImageView::create(&color_image)?;
		let mut resolve_image = Image::create(context, usages | DynImageUsage::COLOR_ATTACHMENT, extent)?;
		resolve_image.transition(
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
		let resolve_image = resolve_image.cast_usage(usage::ColorAttachment).map_err(|_| ()).unwrap();
		let resolve_image_view = ImageView::create(&resolve_image)?;
		Ok(Self {
			color_image,
			color_image_view,
			resolve_image,
			resolve_image_view,
		})
    }
}

pub unsafe trait ColorAttachments<S: SampleCountType>: Sized {
	type ClearValues: ColorClearValues;

	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)>;

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)>;

	//fn clears(&self, color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)>;

	fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self>;
}

unsafe impl<S: SampleCountType> ColorAttachments<S> for () {
	type ClearValues = ();

	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)> {
		Vec::new()
	}

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)> {
		Vec::new()
	}

	/* fn clears(&self, _color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)> {
		Vec::new()
	} */

	fn create(_context: &Context, _usages: DynImageUsage, _extent: vk::Extent2D) -> MarsResult<Self> {
		Ok(())
	}
}

unsafe impl<S, A> ColorAttachments<S> for (A,)
where
	S: SampleCountType,
	A: ColorAttachmentType<S>,
{
	type ClearValues = (A::ClearValue,);

	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)> {
		vec![A::desc()]
	}

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)> {
		vec![self.0.as_raw()]
	}

	/* fn clears(&self, color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)> {
		vec![self.0.clear(color)]
	} */

	fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self> {
		Ok((A::create(context, usages, extent)?,))
	}
}

unsafe impl<S, A, B> ColorAttachments<S> for (A, B)
where
	S: SampleCountType,
	A: ColorAttachmentType<S>,
	B: ColorAttachmentType<S>,
{
	type ClearValues = (A::ClearValue, B::ClearValue);

	fn desc() -> Vec<(pass::Attachment, Option<pass::Attachment>)> {
		vec![A::desc(), B::desc()]
	}

	fn as_raw(&self) -> Vec<(Arc<RkImageViewInner>, Option<Arc<RkImageViewInner>>)> {
		vec![self.0.as_raw(), self.1.as_raw()]
	}

	/* fn clears(&self, color: Vec4) -> Vec<(vk::ClearValue, Option<vk::ClearValue>)> {
		vec![self.0.clear(color), self.1.clear(color)]
	} */

	fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self> {
		Ok((A::create(context, usages, extent)?, B::create(context, usages, extent)?))
	}
}

pub unsafe trait DepthAttachmentType<S: SampleCountType>: Sized {
	type ClearValue: DepthClearValue;

	fn desc() -> Option<pass::Attachment>;

	fn as_raw(&self) -> Option<Arc<RkImageViewInner>>;

	fn clear(&self, depth: f32) -> Option<vk::ClearValue>;

	fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self>;
}

pub struct NoDepthAttachment;

unsafe impl<S> DepthAttachmentType<S> for NoDepthAttachment where S: SampleCountType {
	type ClearValue = ();

	fn desc() -> Option<pass::Attachment> {
		None
	}

	fn as_raw(&self) -> Option<Arc<RkImageViewInner>> {
		None
	}

	fn clear(&self, _depth: f32) -> Option<vk::ClearValue> {
		None
	}

	fn create(_context: &Context, _usages: DynImageUsage, _extent: vk::Extent2D) -> MarsResult<Self> {
		Ok(NoDepthAttachment)
	}
}

pub struct DepthAttachment<F: FormatType, S: SampleCountType> {
	pub image: Image<usage::DepthStencilAttachment, F, S>,
	pub view: ImageView<usage::DepthStencilAttachment, F, S>,
}

impl<F, S> DepthAttachment<F, S>
where
	F: FormatType,
	S: SampleCountType
{
	pub(crate) fn new(
		image: Image<usage::DepthStencilAttachment, F, S>,
		view: ImageView<usage::DepthStencilAttachment, F, S>,
	) -> Self {
		Self { image, view }
	}
}

unsafe impl<F, S> DepthAttachmentType<S> for DepthAttachment<F, S>
where
	F: FormatType,
	F::Pixel: DepthClearValue,
	S: SampleCountType,
{
	type ClearValue = F::Pixel;

	fn desc() -> Option<pass::Attachment> {
		assert!(F::aspect().contains(vk::ImageAspectFlags::DEPTH));

		Some(pass::Attachment {
			format: F::as_raw(),
			samples: S::as_raw(),
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

	fn create(context: &Context, usages: DynImageUsage, extent: vk::Extent2D) -> MarsResult<Self> {
		let mut image = Image::create(context, usages | DynImageUsage::DEPTH_STENCIL_ATTACHMENT, extent)?;
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

pub trait ColorClearValue {
	fn as_raw(&self) -> vk::ClearColorValue;
}

impl ColorClearValue for Vec4 {
	fn as_raw(&self) -> vk::ClearColorValue {
		vk::ClearColorValue {
			float32: [self.x, self.y, self.z, self.w],
		}
	}
}

pub trait ColorClearValues {
	fn as_raw(&self) -> Vec<vk::ClearColorValue>;
}

impl ColorClearValues for () {
	fn as_raw(&self) -> Vec<vk::ClearColorValue> {
		Vec::new()
	}
}

impl<A> ColorClearValues for (A,)
where
	A: ColorClearValue,
{
	fn as_raw(&self) -> Vec<vk::ClearColorValue> {
		vec![self.0.as_raw()]
	}
}

impl<A, B> ColorClearValues for (A, B)
where
	A: ColorClearValue,
	B: ColorClearValue,
{
	fn as_raw(&self) -> Vec<vk::ClearColorValue> {
		vec![self.0.as_raw(), self.1.as_raw()]
	}
}

pub trait DepthClearValue {
	fn as_raw(&self) -> Option<vk::ClearDepthStencilValue>;
}

impl DepthClearValue for () {
	fn as_raw(&self) -> Option<vk::ClearDepthStencilValue> {
		None
	}
}

impl DepthClearValue for f32 {
	fn as_raw(&self) -> Option<vk::ClearDepthStencilValue> {
		Some(vk::ClearDepthStencilValue {
			depth: *self,
			stencil: 0,
		})
	}
}
