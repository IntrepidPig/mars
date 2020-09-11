use std::marker::PhantomData;

use rk::vk;

use crate::buffer::{Buffer, UniformBufferUsage, UntypedBuffer};

pub trait FunctionDef {
	type VertexInput: Parameter;
	type Bindings: Bindings;
}

pub struct FunctionShader<F: FunctionDef> {
	pub(crate) vert: Vec<u32>,
	pub(crate) frag: Vec<u32>,
	pub(crate) _phantom: PhantomData<F>,
}

impl<F> FunctionShader<F>
where
	F: FunctionDef,
{
	pub unsafe fn from_raw(vert: Vec<u32>, frag: Vec<u32>) -> Self {
		Self {
			vert,
			frag,
			_phantom: PhantomData,
		}
	}
}

pub struct ParameterDesc {
	pub attributes: Vec<AttributeDesc>,
}

pub struct AttributeDesc {
	format: AttributeFormat,
}

#[derive(Debug, Copy, Clone)]
pub enum AttributeFormat {
	Vec4F,
}

impl AttributeFormat {
	fn size(self) -> u32 {
		match self {
			AttributeFormat::Vec4F => 4 * 4,
		}
	}
}

impl From<AttributeFormat> for vk::Format {
	fn from(t: AttributeFormat) -> Self {
		match t {
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
}

impl From<BindingType> for vk::DescriptorType {
	fn from(t: BindingType) -> Self {
		match t {
			BindingType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
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
	//Sampled
}

impl<'a> WriteArgument<'a> {
	fn descriptor_type(&self) -> vk::DescriptorType {
		match *self {
			WriteArgument::Uniform(_) => vk::DescriptorType::UNIFORM_BUFFER,
		}
	}
}

pub struct WriteUniformArgument<'a> {
	buffer: UntypedBuffer<'a, UniformBufferUsage>,
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

	unsafe impl Parameter for Vec4 {
		fn attributes() -> Vec<AttributeDesc> {
			vec![AttributeDesc {
				format: AttributeFormat::Vec4F,
			}]
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
}
