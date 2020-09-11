use std::marker::PhantomData;

use rk::{
	descriptor::DescriptorPool,
	device::Device,
	image::{Image, ImageView},
	pass::{self, Framebuffer, RenderPass},
	pipe::{DescriptorSetLayout, Pipeline, PipelineLayout},
	shader::ShaderModule,
	vk,
};

use crate::{
	function::{self, BindingDesc, Bindings, FunctionDef, FunctionShader, Parameter, ParameterDesc},
	Context, MarsResult,
};

pub struct Target<F: FunctionDef> {
	pub(crate) extent: vk::Extent2D,
	//pub(crate) color_image_view: ImageView,
	//pub(crate) depth_image_view: ImageView,
	pub(crate) render_pass: RenderPass,
	pub(crate) descriptor_pool: DescriptorPool,
	pub(crate) descriptor_set_layout: DescriptorSetLayout,
	pub(crate) pipeline: Pipeline,
	pub(crate) pipeline_layout: PipelineLayout,
	pub(crate) color_image: Image,
	pub(crate) depth_image: Image,
	pub(crate) framebuffer: Framebuffer,
	//pub(crate) function_impl: FunctionImpl<F>,
	pub(crate) _phantom: PhantomData<F>,
}

impl<F> Target<F>
where
	F: FunctionDef,
{
	pub fn create(context: &mut Context, extent: vk::Extent2D, shader: FunctionShader<F>) -> MarsResult<Self> {
		let mut render_pass = create_render_pass(&mut context.device);
		//let parameters = F::VertexInputs::parameters();
		let parameters = vec![ParameterDesc {
			attributes: F::VertexInput::attributes(),
		}];
		let (vertex_bindings, vertex_attributes) = function::parameter_descs_to_raw(&parameters);
		let bindings = F::Bindings::descriptions();
		let descriptor_pool = create_descriptor_pool(&mut context.device, &bindings)?;
		let descriptor_bindings = function::bindings_descs_to_raw(&bindings);
		let (pipeline, pipeline_layout, descriptor_set_layout) = create_pipeline(
			&mut context.device,
			&render_pass,
			vertex_bindings,
			vertex_attributes,
			descriptor_bindings,
			&shader.vert,
			&shader.frag,
		)?;
		let initialization = Self::initialize(&mut context.device, &mut render_pass, extent)?;
		Ok(Self {
			extent,
			render_pass,
			descriptor_pool,
			descriptor_set_layout,
			pipeline,
			pipeline_layout,
			color_image: initialization.color_image,
			depth_image: initialization.depth_image,
			framebuffer: initialization.framebuffer,
			_phantom: PhantomData,
		})
	}

	fn initialize(
		device: &mut Device,
		render_pass: &mut RenderPass,
		extent: vk::Extent2D,
	) -> MarsResult<Initialization> {
		let (color_image, depth_image, color_image_view, depth_image_view) = create_attachments(device, extent)?;
		let framebuffer = device.create_framebuffer(
			render_pass,
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

	pub fn resize(&mut self, context: &mut Context, new_extent: vk::Extent2D) -> MarsResult<()> {
		let initialization = Self::initialize(&mut context.device, &mut self.render_pass, new_extent)?;
		self.color_image = initialization.color_image;
		self.depth_image = initialization.depth_image;
		self.framebuffer = initialization.framebuffer;
		self.extent = new_extent;
		Ok(())
	}
}

struct Initialization {
	color_image: Image,
	depth_image: Image,
	//color_image_view: ImageView,
	//depth_image_view: ImageView,
	framebuffer: Framebuffer,
}

fn create_attachments(device: &mut Device, extent: vk::Extent2D) -> MarsResult<(Image, Image, ImageView, ImageView)> {
	let color_image = device.create_image(
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
	let depth_image = device.create_image(
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
	let color_image_view = device.create_image_view(&color_image, vk::ImageAspectFlags::COLOR)?;
	let depth_image_view = device.create_image_view(&depth_image, vk::ImageAspectFlags::DEPTH)?;
	Ok((color_image, depth_image, color_image_view, depth_image_view))
}

fn create_render_pass(device: &mut Device) -> RenderPass {
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

/* fn compile_shader(source: &str, filename: &str, kind: shaderc::ShaderKind) -> Vec<u32> {
	let mut compiler = shaderc::Compiler::new().expect("Failed to initialize compiler");
	let artifact = compiler.compile_into_spirv(source, kind, filename, "main", None)
		.expect("Failed to compile shader");
	artifact.as_binary().to_owned()
} */

fn create_shader_module(device: &Device, spirv: &[u32]) -> ShaderModule {
	device
		.create_shader_module_from_spirv(spirv)
		.expect("Failed to create shader_module")
}

fn create_descriptor_pool(device: &mut Device, binding_descs: &[BindingDesc]) -> MarsResult<DescriptorPool> {
	const MAX_SETS: u32 = 1024;
	const PER_BINDING: u32 = 128;
	let mut pool_sizes = binding_descs
		.iter()
		.map(|b| vk::DescriptorPoolSize {
			ty: b.binding_type.into(),
			descriptor_count: PER_BINDING,
		})
		.collect::<Vec<_>>();
	if pool_sizes.is_empty() {
		// Workaround for when there are no bindings because pool_sizes must have at least one element
		pool_sizes.push(vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 1,
		})
	}

	let pool = device.create_descriptor_pool(MAX_SETS, &pool_sizes)?;
	Ok(pool)
}

fn create_pipeline(
	device: &mut Device,
	render_pass: &RenderPass,
	vertex_binding_descs: Vec<vk::VertexInputBindingDescription>,
	vertex_attribute_descs: Vec<vk::VertexInputAttributeDescription>,
	binding_descs: Vec<vk::DescriptorSetLayoutBinding>,
	vert_spirv: &[u32],
	frag_spirv: &[u32],
) -> MarsResult<(Pipeline, PipelineLayout, DescriptorSetLayout)> {
	let vertex_shader = create_shader_module(device, &vert_spirv);
	let fragment_shader = create_shader_module(device, &frag_spirv);
	let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
		.logic_op_enable(false)
		.attachments(&[vk::PipelineColorBlendAttachmentState::builder()
			.blend_enable(true)
			.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.color_blend_op(vk::BlendOp::ADD)
			.src_alpha_blend_factor(vk::BlendFactor::ONE)
			.dst_alpha_blend_factor(vk::BlendFactor::ZERO)
			.alpha_blend_op(vk::BlendOp::ADD)
			.color_write_mask(vk::ColorComponentFlags::all())
			.build()])
		.blend_constants([1.0, 1.0, 1.0, 1.0])
		.build();
	let descriptor_set_layout = device.create_descriptor_set_layout(&binding_descs)?;
	let pipeline_layout = device.create_pipeline_layout(&descriptor_set_layout)?;
	let pipeline = device.create_pipeline(
		&vertex_shader,
		&vertex_binding_descs,
		&vertex_attribute_descs,
		&fragment_shader,
		&color_blend_state,
		&pipeline_layout,
		render_pass,
		0,
	)?;

	Ok((pipeline, pipeline_layout, descriptor_set_layout))
}
