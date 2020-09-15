use std::marker::PhantomData;

use rk::{
	image::{Image as RkImage, ImageLayoutTransition, ImageView as RkImageView, Sampler as RkSampler},
	vk,
};

use crate::{
	buffer::{Buffer, TransferSrcBufferUsage},
	Context, MarsResult,
};

pub use self::{
	format::FormatType,
	usage::{DynImageUsage, ImageUsageType},
};

/// An unique handle to an image stored on the GPU
///
/// ## Image Usage Flags
///
/// The usage flags of this image are specified by the `U: ImageUsageType` type parameter.
/// `ImageUsageType` may be either a specific image usage type such as `SampledImageUsage`, or it
/// may be `DynImageUsage` to indicate that the usages of this image are not known at compile time.
///
/// The usage flags of an image are immutable, and can only be specified at the creation of an
/// image.
///
/// Additionally, a concrete image usage type paramater such as `SampledImageUsage` does not
/// necessarily indicate the full range of usages the image supports. To convert the `U` type
/// paramater of the current image to a more useful type, use the `cast_usage*` method family. For
/// example, if you create an image to be used both as a sampled image and a transfer source, you
/// would create the image with a `DynImageUsage` parameter that includes the `SAMPLED_IMAGE` and
/// `TRANSFER_SRC` flags. Then, when passing it to a function that requires a specific specific
/// usage, for example `SampledImageUsage`, then you would cast it using the appropriate member of
/// the `cast_usage*` methods.
///
/// ## Image Formats
///
/// Like image usage flags, image formats are immutable and must be specified only once at creation
/// time. Unlike image usage flags, an image can only have one image format, so there is no method
/// for casting formats at this time, and the exact type of the format is always known (that is,
/// there is no `DynImageUsage` analog for image formats). However, this may change in the future,
/// depending on API requirements.
pub struct Image<U: ImageUsageType, F: FormatType> {
	pub(crate) image: RkImage,
	pub(crate) layout: vk::ImageLayout,
	pub(crate) extent: vk::Extent2D,
	pub(crate) usage: DynImageUsage,
	_phantom: PhantomData<(U, F)>,
}

impl<U, F> Image<U, F>
where
	U: ImageUsageType,
	F: FormatType,
{
	pub(crate) unsafe fn create_raw(
		context: &Context,
		usage: DynImageUsage,
		format: vk::Format,
		extent: vk::Extent2D,
	) -> MarsResult<Self> {
		let extent3d = vk::Extent3D {
			width: extent.width,
			height: extent.height,
			depth: 1,
		};

		let image = RkImage::create(
			&context.device,
			format,
			extent3d,
			usage.as_raw(),
			vk::ImageLayout::UNDEFINED,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		)?;

		Ok(Self {
			image,
			layout: vk::ImageLayout::UNDEFINED,
			extent,
			usage,
			_phantom: PhantomData,
		})
	}

	pub fn create(context: &Context, usage: U, extent: vk::Extent2D) -> MarsResult<Self> {
		unsafe { Self::create_raw(context, usage.as_dyn(), F::as_raw(), extent) }
	}

	pub fn make_image(context: &Context, usage: U, extent: vk::Extent2D, data: &[u8]) -> MarsResult<Self> {
		let mut image = unsafe {
			Self::create_raw(
				context,
				usage.as_dyn() | DynImageUsage::TRANSFER_DST,
				F::as_raw(),
				extent,
			)?
		};
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

		let staging_buffer = Buffer::<TransferSrcBufferUsage, _>::make_array_buffer(context, data)?;

		unsafe {
			context.device.copy_buffer_to_image(
				&context.queue,
				&context.command_pool,
				&staging_buffer.buffer,
				&image.image,
				extent,
				F::aspect(),
			)?;
			image.layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
		}

		Ok(image)
	}

	/// Returns all of the usages this image supports. (This may be more than the usage type
	/// parameter indicates).
	pub fn usage(&self) -> DynImageUsage {
		self.usage
	}

	pub fn extent(&self) -> vk::Extent2D {
		self.extent
	}

	pub fn cast_usage<U2: ImageUsageType>(self, usage: U2) -> Result<Image<U2, F>, Self> {
		if self.usage.as_dyn().contains(usage.as_dyn()) {
			Ok(unsafe { self.cast_unchecked() })
		} else {
			Err(self)
		}
	}

	pub fn cast_usage_ref<U2: ImageUsageType>(&self, usage: U2) -> Option<&Image<U2, F>> {
		if self.usage.as_dyn().contains(usage.as_dyn()) {
			Some(unsafe { self.cast_unchecked_ref() })
		} else {
			None
		}
	}

	pub fn cast_usage_mut<U2: ImageUsageType>(&mut self, usage: U2) -> Option<&mut Image<U2, F>> {
		if self.usage.as_dyn().contains(usage.as_dyn()) {
			Some(unsafe { self.cast_unchecked_mut() })
		} else {
			None
		}
	}

	pub(crate) unsafe fn cast_unchecked<U2: ImageUsageType, F2: FormatType>(self) -> Image<U2, F2> {
		let Image {
			image,
			layout,
			extent,
			usage,
			_phantom,
		} = self;
		Image {
			image,
			layout,
			extent,
			usage,
			_phantom: PhantomData,
		}
	}

	pub(crate) unsafe fn cast_unchecked_ref<U2: ImageUsageType, F2: FormatType>(&self) -> &Image<U2, F2> {
		&*(self as *const Self as *const Image<U2, F2>)
	}

	pub(crate) unsafe fn cast_unchecked_mut<U2: ImageUsageType, F2: FormatType>(&mut self) -> &mut Image<U2, F2> {
		&mut *(self as *mut Self as *mut Image<U2, F2>)
	}

	// TODO: worry about image synchronization
	pub(crate) fn transition(&mut self, context: &Context, transition: &ImageLayoutTransition) -> MarsResult<()> {
		unsafe {
			context.queue.with_lock(|| {
				rk::image::transition_image_layout(&context.queue, &context.command_pool, &mut self.image, transition)
			})?;
		};
		self.layout = transition.new_layout;
		Ok(())
	}
}

pub struct ImageView<U: ImageUsageType, F: FormatType> {
	pub(crate) image_view: RkImageView,
	pub(crate) usage: DynImageUsage,
	_phantom: PhantomData<(U, F)>,
}

impl<U: ImageUsageType, F: FormatType> ImageView<U, F> {
	pub fn create(image: &Image<U, F>) -> MarsResult<Self> {
		let image_view = unsafe { RkImageView::create(&image.image, F::aspect())? };
		Ok(Self {
			image_view,
			usage: image.usage,
			_phantom: PhantomData,
		})
	}

	/// Returns all of the usages this image supports. (This may be more than the usage type
	/// parameter indicates).
	pub fn usage(&self) -> DynImageUsage {
		self.usage
	}

	pub fn cast_usage<U2: ImageUsageType>(self, usage: U2) -> Result<ImageView<U2, F>, Self> {
		if self.usage.as_dyn().contains(usage.as_dyn()) {
			Ok(unsafe { self.cast_unchecked() })
		} else {
			Err(self)
		}
	}

	pub fn cast_usage_ref<U2: ImageUsageType>(&self, usage: U2) -> Option<&ImageView<U2, F>> {
		if self.usage.as_dyn().contains(usage.as_dyn()) {
			Some(unsafe { self.cast_unchecked_ref() })
		} else {
			None
		}
	}

	pub fn cast_usage_mut<U2: ImageUsageType>(&mut self, usage: U2) -> Option<&mut ImageView<U2, F>> {
		if self.usage.as_dyn().contains(usage.as_dyn()) {
			Some(unsafe { self.cast_unchecked_mut() })
		} else {
			None
		}
	}

	pub(crate) unsafe fn cast_unchecked<U2: ImageUsageType, F2: FormatType>(self) -> ImageView<U2, F2> {
		let Self {
			image_view,
			usage,
			_phantom,
		} = self;
		ImageView {
			image_view,
			usage,
			_phantom: PhantomData,
		}
	}

	pub(crate) unsafe fn cast_unchecked_ref<U2: ImageUsageType, F2: FormatType>(&self) -> &ImageView<U2, F2> {
		&*(self as *const Self as *const ImageView<U2, F2>)
	}

	pub(crate) unsafe fn cast_unchecked_mut<U2: ImageUsageType, F2: FormatType>(&mut self) -> &mut ImageView<U2, F2> {
		&mut *(self as *mut Self as *mut ImageView<U2, F2>)
	}
}

pub struct Sampler {
	pub(crate) sampler: RkSampler,
}

impl Sampler {
	pub fn create(context: &Context) -> MarsResult<Self> {
		let sampler = context.device.create_sampler()?;
		Ok(Self { sampler })
	}
}

pub struct SampledImage<F: FormatType> {
	pub image: Image<usage::SampledImage, F>,
	pub image_view: ImageView<usage::SampledImage, F>,
	pub sampler: Sampler,
}

impl<F> SampledImage<F>
where
	F: FormatType,
{
	pub fn new(
		image: Image<usage::SampledImage, F>,
		image_view: ImageView<usage::SampledImage, F>,
		sampler: Sampler,
	) -> Self {
		Self {
			image,
			image_view,
			sampler,
		}
	}

	pub fn create(context: &Context, mut image: Image<usage::SampledImage, F>) -> MarsResult<Self> {
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
		let image_view = ImageView::create(&image)?;
		let sampler = Sampler::create(context)?;
		Ok(Self::new(image, image_view, sampler))
	}
}

pub mod usage {
	use rk::vk;

	pub unsafe trait ImageUsageType: Copy {
		fn as_dyn(self) -> DynImageUsage;
		fn as_raw(self) -> vk::ImageUsageFlags;
	}

	bitflags::bitflags! {
		pub struct DynImageUsage: u32 {
			const TRANSFER_SRC = vk::ImageUsageFlags::TRANSFER_SRC.as_raw();
			const TRANSFER_DST = vk::ImageUsageFlags::TRANSFER_DST.as_raw();
			const SAMPLED = vk::ImageUsageFlags::SAMPLED.as_raw();
			const STORAGE = vk::ImageUsageFlags::STORAGE.as_raw();
			const COLOR_ATTACHMENT = vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw();
			const DEPTH_STENCIL_ATTACHMENT = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw();
			const INPUT_ATTACHMENT = vk::ImageUsageFlags::INPUT_ATTACHMENT.as_raw();
		}
	}

	unsafe impl ImageUsageType for DynImageUsage {
		fn as_dyn(self) -> DynImageUsage {
			self
		}

		fn as_raw(self) -> vk::ImageUsageFlags {
			vk::ImageUsageFlags::from_raw(self.bits())
		}
	}

	macro_rules! image_usage {
		($name:ident, $usage:ident) => {
			#[derive(Debug, Copy, Clone)]
			pub struct $name;

			unsafe impl ImageUsageType for $name {
				fn as_dyn(self) -> DynImageUsage {
					DynImageUsage::$usage
				}

				fn as_raw(self) -> vk::ImageUsageFlags {
					vk::ImageUsageFlags::$usage
				}
			}
		};
	}

	image_usage!(TransferSrc, TRANSFER_SRC);
	image_usage!(TransferDst, TRANSFER_DST);
	image_usage!(SampledImage, SAMPLED);
	image_usage!(Storage, STORAGE);
	image_usage!(ColorAttachment, COLOR_ATTACHMENT);
	image_usage!(DepthStencilAttachment, DEPTH_STENCIL_ATTACHMENT);
	image_usage!(InputAttachment, INPUT_ATTACHMENT);
}

pub mod format {
	use crate::math::*;
	use rk::vk;

	pub unsafe trait FormatType {
		type Pixel;

		fn as_raw() -> vk::Format;

		fn aspect() -> vk::ImageAspectFlags;
	}

	macro_rules! format {
		($name:ident, $raw:ident, $aspect:ident, $pixel:ty) => {
			pub struct $name;

			unsafe impl FormatType for $name {
				type Pixel = $pixel;

				fn as_raw() -> vk::Format {
					vk::Format::$raw
				}

				fn aspect() -> vk::ImageAspectFlags {
					vk::ImageAspectFlags::$aspect
				}
			}
		};
	}

	format!(B8G8R8A8Unorm, B8G8R8A8_UNORM, COLOR, Vec4);

	format!(R8G8B8A8Unorm, R8G8B8A8_UNORM, COLOR, Vec4);
	format!(R8G8B8A8Srgb, R8G8B8A8_SRGB, COLOR, Vec4);

	format!(D32Sfloat, D32_SFLOAT, DEPTH, f32);
}
