use std::{marker::PhantomData, sync::Arc};

use rk::{
	descriptor::{DescriptorPool, DescriptorSet},
	device::Device,
	pass::RenderPass as RkRenderPass,
	pipe::{DescriptorSetLayout, Pipeline, PipelineLayout},
	shader::ShaderModule,
	vk,
};

use crate::{
	buffer::{Buffer, UniformBufferUsage, UntypedBuffer},
	image::{FormatType, SampledImage, SampleCountType},
	pass::{RenderPass, RenderPassPrototype, ColorAttachments},
	Context, MarsResult,
};

pub trait FunctionPrototype {
	type RenderPass: RenderPassPrototype;
	type VertexInput: Parameter;
	type Bindings: Bindings;
}

pub struct FunctionImpl<F: FunctionPrototype> {
	pub(crate) vert: Vec<u32>,
	pub(crate) frag: Vec<u32>,
	pub(crate) _phantom: PhantomData<F>,
}

impl<F> FunctionImpl<F>
where
	F: FunctionPrototype,
{
	pub unsafe fn from_raw(vert: Vec<u32>, frag: Vec<u32>) -> Self {
		Self {
			vert,
			frag,
			_phantom: PhantomData,
		}
	}
}

pub struct FunctionDef<F: FunctionPrototype> {
	pub(crate) descriptor_pool: DescriptorPool,
	pub(crate) descriptor_set_layout: DescriptorSetLayout,
	pub(crate) pipeline: Pipeline,
	pub(crate) pipeline_layout: PipelineLayout,
	_phantom: PhantomData<F>,
}

impl<F> FunctionDef<F>
where
	F: FunctionPrototype,
{
	pub fn create(
		context: &Context,
		render_pass: &RenderPass<F::RenderPass>,
		function_impl: FunctionImpl<F>,
	) -> MarsResult<Self> {
		//let parameters = F::VertexInputs::parameters(); // TODO: multiple vertex bindings
		let parameters = vec![ParameterDesc {
			attributes: F::VertexInput::attributes(),
		}];
		let (vertex_bindings, vertex_attributes) = parameter_descs_to_raw(&parameters);
		let bindings = F::Bindings::descriptions();
		let descriptor_pool = create_descriptor_pool(&context.device, &bindings)?;
		let descriptor_bindings = bindings_descs_to_raw(&bindings);
		let color_blend_states = create_blend_states::<F::RenderPass>();
		let multisample_state = create_multisample_state::<F::RenderPass>();
		let (pipeline, pipeline_layout, descriptor_set_layout) = create_pipeline(
			&context.device,
			&render_pass.render_pass,
			vertex_bindings,
			vertex_attributes,
			descriptor_bindings,
			&color_blend_states,
			&multisample_state,
			&function_impl.vert,
			&function_impl.frag,
		)?;
		Ok(Self {
			descriptor_pool,
			descriptor_set_layout,
			pipeline,
			pipeline_layout,
			_phantom: PhantomData,
		})
	}

	pub fn make_arguments(
		&mut self,
		context: &Context,
		arguments: <F::Bindings as Bindings>::Arguments,
	) -> MarsResult<ArgumentsContainer<F>> {
		let descriptor_set = context
			.device
			.allocate_descriptor_set(&self.descriptor_pool, &self.descriptor_set_layout)?;
		let writes = arguments.as_writes();
		let (raw_writes, _backing) = writes_to_raw(***descriptor_set, &writes);
		unsafe { context.device.write_descriptor_set(&raw_writes)? };
		Ok(ArgumentsContainer {
			arguments,
			descriptor_set,
		})
	}
}

pub struct ArgumentsContainer<F: FunctionPrototype> {
	pub arguments: <F::Bindings as Bindings>::Arguments,
	pub(crate) descriptor_set: DescriptorSet,
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

fn create_descriptor_pool(device: &Device, binding_descs: &[BindingDesc]) -> MarsResult<DescriptorPool> {
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

// TODO: make blend states customizable
fn create_blend_states<G: RenderPassPrototype>() -> Vec<vk::PipelineColorBlendAttachmentState> {
	let default = vk::PipelineColorBlendAttachmentState::builder()
		.blend_enable(true)
		.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
		.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
		.color_blend_op(vk::BlendOp::ADD)
		.src_alpha_blend_factor(vk::BlendFactor::ONE)
		.dst_alpha_blend_factor(vk::BlendFactor::ZERO)
		.alpha_blend_op(vk::BlendOp::ADD)
		.color_write_mask(vk::ColorComponentFlags::all())
		.build();
	let amount = <G::ColorAttachments as ColorAttachments<G::SampleCount>>::desc().len();
	vec![default; amount]
}

fn create_multisample_state<G: RenderPassPrototype>() -> vk::PipelineMultisampleStateCreateInfo {
	vk::PipelineMultisampleStateCreateInfo::builder()
		.rasterization_samples(G::SampleCount::as_raw())
		.sample_shading_enable(false)
		.alpha_to_coverage_enable(false)
		.alpha_to_one_enable(false)
		.build()
}

fn create_pipeline(
	device: &Device,
	render_pass: &RkRenderPass,
	vertex_binding_descs: Vec<vk::VertexInputBindingDescription>,
	vertex_attribute_descs: Vec<vk::VertexInputAttributeDescription>,
	binding_descs: Vec<vk::DescriptorSetLayoutBinding>,
	color_blend_attachment_states: &[vk::PipelineColorBlendAttachmentState],
	multisample_state: &vk::PipelineMultisampleStateCreateInfo,
	vert_spirv: &[u32],
	frag_spirv: &[u32],
) -> MarsResult<(Pipeline, PipelineLayout, DescriptorSetLayout)> {
	let vertex_shader = create_shader_module(device, &vert_spirv);
	let fragment_shader = create_shader_module(device, &frag_spirv);
	let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
		.logic_op_enable(false)
		.attachments(color_blend_attachment_states)
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
		multisample_state,
		&pipeline_layout,
		render_pass,
		0,
	)?;

	Ok((pipeline, pipeline_layout, descriptor_set_layout))
}

pub struct ParameterDesc {
	pub attributes: Vec<AttributeDesc>,
}

pub struct AttributeDesc {
	format: AttributeFormat,
}

#[derive(Debug, Copy, Clone)]
pub enum AttributeFormat {
	Vec2F,
	Vec3F,
	Vec4F,
}

impl AttributeFormat {
	fn size(self) -> u32 {
		match self {
			AttributeFormat::Vec2F => 4 * 2,
			AttributeFormat::Vec3F => 4 * 3,
			AttributeFormat::Vec4F => 4 * 4,
		}
	}
}

impl From<AttributeFormat> for vk::Format {
	fn from(t: AttributeFormat) -> Self {
		match t {
			AttributeFormat::Vec2F => vk::Format::R32G32_SFLOAT,
			AttributeFormat::Vec3F => vk::Format::R32G32B32_SFLOAT,
			AttributeFormat::Vec4F => vk::Format::R32G32B32A32_SFLOAT,
		}
	}
}

pub unsafe trait Parameter: Copy {
	fn attributes() -> Vec<AttributeDesc>;
}

unsafe impl<A, B> Parameter for (A, B)
where
	A: Parameter,
	B: Parameter,
{
	fn attributes() -> Vec<AttributeDesc> {
		let mut a = A::attributes();
		let mut b = B::attributes();
		let mut buf = Vec::with_capacity(a.len() + b.len());
		buf.append(&mut a);
		buf.append(&mut b);
		buf
	}
}

pub unsafe trait Parameters: Copy {
	fn parameters() -> Vec<ParameterDesc>;
}

unsafe impl<A> Parameters for (A,)
where
	A: Parameter,
{
	fn parameters() -> Vec<ParameterDesc> {
		vec![ParameterDesc {
			attributes: A::attributes(),
		}]
	}
}

unsafe impl<A, B> Parameters for (A, B)
where
	A: Parameter,
	B: Parameter,
{
	fn parameters() -> Vec<ParameterDesc> {
		vec![
			ParameterDesc {
				attributes: A::attributes(),
			},
			ParameterDesc {
				attributes: B::attributes(),
			},
		]
	}
}

unsafe impl<A, B, C> Parameters for (A, B, C)
where
	A: Parameter,
	B: Parameter,
	C: Parameter,
{
	fn parameters() -> Vec<ParameterDesc> {
		vec![
			ParameterDesc {
				attributes: A::attributes(),
			},
			ParameterDesc {
				attributes: B::attributes(),
			},
			ParameterDesc {
				attributes: C::attributes(),
			},
		]
	}
}

#[derive(Debug, Copy, Clone)]
pub enum BindingType {
	Uniform,
	SampledImage,
}

impl From<BindingType> for vk::DescriptorType {
	fn from(t: BindingType) -> Self {
		match t {
			BindingType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
			BindingType::SampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		}
	}
}

pub struct BindingDesc {
	pub binding_type: BindingType,
	pub count: u32,
}

pub unsafe trait Binding {
	type Argument: Argument;

	fn description() -> BindingDesc;
}

unsafe impl<F: FormatType> Binding for SampledImage<F> {
	type Argument = Self;

	fn description() -> BindingDesc {
		BindingDesc {
			binding_type: BindingType::SampledImage,
			count: 1,
		}
	}
}

pub unsafe trait Bindings {
	type Arguments: Arguments;

	fn descriptions() -> Vec<BindingDesc>;
}

unsafe impl Bindings for () {
	type Arguments = ();

	fn descriptions() -> Vec<BindingDesc> {
		Vec::new()
	}
}

unsafe impl<A> Bindings for (A,)
where
	A: Binding,
{
	type Arguments = (A::Argument,);

	fn descriptions() -> Vec<BindingDesc> {
		vec![A::description()]
	}
}

unsafe impl<A, B> Bindings for (A, B)
where
	A: Binding,
	B: Binding,
{
	type Arguments = (A::Argument, B::Argument);

	fn descriptions() -> Vec<BindingDesc> {
		vec![A::description(), B::description()]
	}
}

unsafe impl<A, B, C> Bindings for (A, B, C)
where
	A: Binding,
	B: Binding,
	C: Binding,
{
	type Arguments = (A::Argument, B::Argument, C::Argument);

	fn descriptions() -> Vec<BindingDesc> {
		vec![A::description(), B::description(), C::description()]
	}
}

pub trait Argument {
	fn as_write(&self) -> WriteArgument;
}

impl<T> Argument for Buffer<UniformBufferUsage, T>
where
	T: Copy,
{
	fn as_write(&self) -> WriteArgument {
		WriteArgument::Uniform(WriteUniformArgument {
			buffer: self.as_untyped(),
		})
	}
}

impl<F> Argument for SampledImage<F>
where
	F: FormatType,
{
	fn as_write(&self) -> WriteArgument {
		WriteArgument::SampledImage(WriteSampledImageArgument {
			sampler: self.sampler.sampler.clone(),
			image_view: self.image_view.image_view.clone(),
			image_layout: self.image.layout,
		})
	}
}

pub trait Arguments {
	fn as_writes(&self) -> Vec<WriteArgument>;
}

impl Arguments for () {
	fn as_writes(&self) -> Vec<WriteArgument> {
		Vec::new()
	}
}

impl<A> Arguments for (A,)
where
	A: Argument,
{
	fn as_writes(&self) -> Vec<WriteArgument> {
		vec![self.0.as_write()]
	}
}

impl<A, B> Arguments for (A, B)
where
	A: Argument,
	B: Argument,
{
	fn as_writes(&self) -> Vec<WriteArgument> {
		vec![self.0.as_write(), self.1.as_write()]
	}
}

impl<A, B, C> Arguments for (A, B, C)
where
	A: Argument,
	B: Argument,
	C: Argument,
{
	fn as_writes(&self) -> Vec<WriteArgument> {
		vec![self.0.as_write(), self.1.as_write(), self.2.as_write()]
	}
}

pub enum WriteArgument<'a> {
	Uniform(WriteUniformArgument<'a>),
	SampledImage(WriteSampledImageArgument),
}

impl<'a> WriteArgument<'a> {
	fn descriptor_type(&self) -> vk::DescriptorType {
		match *self {
			WriteArgument::Uniform(_) => vk::DescriptorType::UNIFORM_BUFFER,
			WriteArgument::SampledImage(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		}
	}
}

pub struct WriteUniformArgument<'a> {
	buffer: UntypedBuffer<'a, UniformBufferUsage>,
}

pub struct WriteSampledImageArgument {
	sampler: Arc<rk::image::SamplerInner>,
	image_view: Arc<rk::image::ImageViewInner>,
	image_layout: vk::ImageLayout,
}

pub(crate) fn parameter_descs_to_raw(
	parameters: &[ParameterDesc],
) -> (
	Vec<vk::VertexInputBindingDescription>,
	Vec<vk::VertexInputAttributeDescription>,
) {
	let mut bindings = Vec::new();
	let mut attributes = Vec::new();

	let mut location = 0;
	for (i, parameter) in parameters.iter().enumerate() {
		bindings.push(vk::VertexInputBindingDescription {
			binding: i as u32,
			stride: parameter.attributes.iter().map(|p| p.format.size()).sum(),
			input_rate: vk::VertexInputRate::VERTEX,
		});
		let mut offset = 0;
		for attribute in &parameter.attributes {
			attributes.push(vk::VertexInputAttributeDescription {
				location,
				binding: i as u32,
				format: attribute.format.into(),
				offset,
			});
			location += 1;
			offset += attribute.format.size();
		}
	}

	(bindings, attributes)
}

pub(crate) fn bindings_descs_to_raw(bindings: &[BindingDesc]) -> Vec<vk::DescriptorSetLayoutBinding> {
	let mut raw_bindings = Vec::new();

	for (i, binding) in bindings.iter().enumerate() {
		raw_bindings.push(
			vk::DescriptorSetLayoutBinding::builder()
				.binding(i as u32)
				.descriptor_type(binding.binding_type.into())
				.descriptor_count(binding.count)
				.stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
				.build(),
		);
	}

	raw_bindings
}

pub enum WriteBacking {
	Buffer(Vec<vk::DescriptorBufferInfo>),
	Image(Vec<vk::DescriptorImageInfo>),
}

pub(crate) fn writes_to_raw(
	set: vk::DescriptorSet,
	writes: &[WriteArgument],
) -> (Vec<vk::WriteDescriptorSet>, Vec<WriteBacking>) {
	let mut raw_writes = Vec::new();
	let mut backing = Vec::new();

	for (i, write) in writes.iter().enumerate() {
		let builder = vk::WriteDescriptorSet::builder()
			.dst_set(set)
			.dst_binding(i as u32)
			.dst_array_element(0)
			.descriptor_type(write.descriptor_type());
		let builder = match write {
			WriteArgument::Uniform(write) => {
				let buffer_info = vk::DescriptorBufferInfo {
					buffer: ***write.buffer.buffer.buffer,
					offset: 0,
					range: write.buffer.buffer.size as u64,
				};
				backing.push(WriteBacking::Buffer(vec![buffer_info]));
				builder.buffer_info(if let WriteBacking::Buffer(buffer) = backing.last().unwrap() {
					&buffer
				} else {
					unreachable!()
				})
			}
			WriteArgument::SampledImage(write) => {
				let image_info = vk::DescriptorImageInfo {
					sampler: **write.sampler,
					image_view: **write.image_view,
					image_layout: write.image_layout,
				};
				backing.push(WriteBacking::Image(vec![image_info]));
				builder.image_info(if let WriteBacking::Image(image) = backing.last().unwrap() {
					&image
				} else {
					unreachable!()
				})
			}
		};
		raw_writes.push(builder.build());
	}

	(raw_writes, backing)
}

mod nalgebra {
	use super::*;
	use crate::{
		buffer::{Buffer, UniformBufferUsage},
		math::*,
	};

	unsafe impl Parameter for Vec2 {
		fn attributes() -> Vec<AttributeDesc> {
			vec![AttributeDesc {
				format: AttributeFormat::Vec2F,
			}]
		}
	}

	unsafe impl Parameter for Vec3 {
		fn attributes() -> Vec<AttributeDesc> {
			vec![AttributeDesc {
				format: AttributeFormat::Vec3F,
			}]
		}
	}

	unsafe impl Parameter for Vec4 {
		fn attributes() -> Vec<AttributeDesc> {
			vec![AttributeDesc {
				format: AttributeFormat::Vec4F,
			}]
		}
	}

	unsafe impl Binding for Vec3 {
		type Argument = Buffer<UniformBufferUsage, Vec3>;

		fn description() -> BindingDesc {
			BindingDesc {
				binding_type: BindingType::Uniform,
				count: 1,
			}
		}
	}

	unsafe impl Binding for Mat4 {
		type Argument = Buffer<UniformBufferUsage, Mat4>;

		fn description() -> BindingDesc {
			BindingDesc {
				binding_type: BindingType::Uniform,
				count: 1,
			}
		}
	}

	unsafe impl Binding for Mvp {
		type Argument = Buffer<UniformBufferUsage, Mvp>;

		fn description() -> BindingDesc {
			BindingDesc {
				binding_type: BindingType::Uniform,
				count: 1,
			}
		}
	}
}
