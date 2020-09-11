use std::marker::PhantomData;

use rk::{
	image::{Image as RkImage, ImageLayoutTransition, ImageView as RkImageView, Sampler as RkSampler},
	vk,
};

use crate::{
	buffer::{Buffer, TransferSrcBufferUsage},
	Context, MarsResult,
};

pub struct Image<U: ImageUsageType, F: FormatType> {
	pub(crate) image: RkImage,
	pub(crate) layout: vk::ImageLayout,
	_phantom: PhantomData<(U, F)>,
}

impl<U, F> Image<U, F>
where
	U: ImageUsageType,
	F: FormatType,
{
	pub(crate) fn create_raw(
		context: &mut Context,
		extent: vk::Extent2D,
		usage: vk::ImageUsageFlags,
		format: vk::Format,
	) -> MarsResult<Self> {
		let extent = vk::Extent3D {
			width: extent.width,
			height: extent.height,
			depth: 1,
		};

		let image = context.device.create_image(
			format,
			extent,
			usage,
			vk::ImageLayout::UNDEFINED,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		)?;

		Ok(Self {
			image,
			layout: vk::ImageLayout::UNDEFINED,
			_phantom: PhantomData,
		})
	}

	pub fn create(context: &mut Context, extent: vk::Extent2D, other_usages: vk::ImageUsageFlags) -> MarsResult<Self> {
		Self::create_raw(context, extent, U::as_raw() | other_usages, F::as_raw())
	}

	pub fn make_image(context: &mut Context, extent: vk::Extent2D, data: &[u8]) -> MarsResult<Self> {
		let mut image = Self::create(context, extent, vk::ImageUsageFlags::TRANSFER_DST)?;
		image.transition(
			context,
			&ImageLayoutTransition {
				aspect: F::aspect(),
				src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
				dst_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
				src_access_mask: vk::AccessFlags::empty(),
				dst_access_mask: vk::AccessFlags::MEMORY_READ,
				old_layout: vk::ImageLayout::UNDEFINED,
				new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			},
		)?;

		let staging_buffer = Buffer::<TransferSrcBufferUsage, _>::make_buffer(context, data)?;

		unsafe {
			context.device.copy_buffer_to_image(
				&mut context.queue,
				&mut context.command_pool,
				&staging_buffer.buffer,
				&image.image,
				extent,
				F::aspect(),
			)?;
			image.layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
		}

		Ok(image)
	}

	pub(crate) fn transition(&mut self, context: &mut Context, transition: &ImageLayoutTransition) -> MarsResult<()> {
		unsafe {
			rk::image::transition_image_layout(
				&mut context.device,
				&mut context.queue,
				&mut context.command_pool,
				&mut self.image,
				transition,
			)?
		};
		self.layout = transition.new_layout;
		Ok(())
	}
}

pub struct ImageView {
	pub(crate) image_view: RkImageView,
}

impl ImageView {
	pub fn create<U: ImageUsageType, F: FormatType>(context: &mut Context, image: &Image<U, F>) -> MarsResult<Self> {
		let image_view = context.device.create_image_view(&image.image, F::aspect())?;
		Ok(Self { image_view })
	}
}

pub struct Sampler {
	pub(crate) sampler: RkSampler,
}

impl Sampler {
	pub fn create(context: &mut Context) -> MarsResult<Self> {
		let sampler = context.device.create_sampler()?;
		Ok(Self { sampler })
	}
}

pub struct SampledImage<F: FormatType> {
	pub image: Image<SampledImageUsage, F>,
	pub image_view: ImageView,
	pub sampler: Sampler,
}

impl<F> SampledImage<F>
where
	F: FormatType,
{
	pub fn new(image: Image<SampledImageUsage, F>, image_view: ImageView, sampler: Sampler) -> Self {
		Self {
			image,
			image_view,
			sampler,
		}
	}

	pub fn create(context: &mut Context, mut image: Image<SampledImageUsage, F>) -> MarsResult<Self> {
		if image.layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
			let transition = ImageLayoutTransition {
				aspect: F::aspect(),
				src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
				dst_stage_mask: vk::PipelineStageFlags::ALL_GRAPHICS,
				src_access_mask: vk::AccessFlags::empty(),
				dst_access_mask: vk::AccessFlags::SHADER_READ,
				old_layout: image.layout,
				new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
			};
			image.transition(context, &transition)?;
		}
		let image_view = ImageView::create(context, &image)?;
		let sampler = Sampler::create(context)?;
		Ok(Self::new(image, image_view, sampler))
	}
}

pub trait ImageUsageType {
	fn as_raw() -> vk::ImageUsageFlags;
}

pub trait FormatType {
	fn as_raw() -> vk::Format;

	fn aspect() -> vk::ImageAspectFlags;
}

macro_rules! image_usage {
	($name:ident, $usage:ident) => {
		pub struct $name;

		impl ImageUsageType for $name {
			fn as_raw() -> vk::ImageUsageFlags {
				vk::ImageUsageFlags::$usage
			}
		}
	};
}

image_usage!(SampledImageUsage, SAMPLED);

macro_rules! format {
	($name:ident, $raw:ident, $aspect:ident) => {
		pub struct $name;

		impl FormatType for $name {
			fn as_raw() -> vk::Format {
				vk::Format::$raw
			}

			fn aspect() -> vk::ImageAspectFlags {
				vk::ImageAspectFlags::$aspect
			}
		}
	};
}

format!(B8G8R8A8UnormFormat, B8G8R8A8_UNORM, COLOR);

format!(R8G8B8A8UnormFormat, R8G8B8A8_UNORM, COLOR);
format!(R8G8B8A8SrgbFormat, R8G8B8A8_SRGB, COLOR);
