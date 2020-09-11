use rk::{
	command::{CommandBuffer, CommandPool, Recording},
	descriptor::DescriptorSet as RkDescriptorSet,
	vk,
};

use crate::{
	buffer::{Buffer, IndexBufferUsage, VertexBufferUsage},
	function::{self, Arguments, Bindings, FunctionDef},
	math::*,
	target::Target,
	Context, MarsResult,
};

pub struct RenderEngine<F: FunctionDef> {
	pub(crate) target: Target<F>,
	pub(crate) command_pool: CommandPool,
}

impl<F> RenderEngine<F>
where
	F: FunctionDef,
{
	pub fn new(context: &mut Context, target: Target<F>) -> MarsResult<Self> {
		let command_pool = context.device.create_command_pool()?;

		let mut this = Self { target, command_pool };
		this.clear(context, Vec4::new(0.0, 0.0, 0.0, 1.0))?;
		Ok(this)
	}

	pub fn make_descriptor_set(
		&mut self,
		context: &mut Context,
		arguments: <F::Bindings as Bindings>::Arguments,
	) -> MarsResult<DescriptorSet<F::Bindings>> {
		let descriptor_set = context
			.device
			.allocate_descriptor_set(&self.target.descriptor_pool, &self.target.descriptor_set_layout)?;
		let writes = arguments.as_writes();
		let (raw_writes, _backing) = function::writes_to_raw(***descriptor_set, &writes);
		unsafe { context.device.write_descriptor_set(&raw_writes)? };
		Ok(DescriptorSet {
			arguments,
			descriptor_set,
		})
	}

	pub fn draw(
		&mut self,
		context: &mut Context,
		set: &DescriptorSet<F::Bindings>,
		vertices: &Buffer<VertexBufferUsage, [F::VertexInput]>,
		indices: &Buffer<IndexBufferUsage, [u32]>,
	) -> MarsResult<()> {
		self.submit(context, |this, command_buffer| {
			command_buffer.begin_render_pass(
				&mut this.target.render_pass,
				&mut this.target.framebuffer,
				vk::Rect2D {
					offset: vk::Offset2D { x: 0, y: 0 },
					extent: this.target.extent,
				},
				&[
					vk::ClearValue {
						color: vk::ClearColorValue {
							float32: [1.0, 1.0, 1.0, 1.0],
						},
					},
					vk::ClearValue {
						depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
					},
				],
			)?;
			command_buffer.set_viewport(vk::Viewport {
				x: 0.0,
				y: 0.0,
				width: this.target.extent.width as f32,
				height: this.target.extent.height as f32,
				min_depth: 0.0,
				max_depth: 1.0,
			});
			command_buffer.set_scissor(vk::Rect2D {
				offset: vk::Offset2D { x: 0, y: 0 },
				extent: vk::Extent2D {
					width: this.target.extent.width,
					height: this.target.extent.height,
				},
			});
			command_buffer.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &this.target.pipeline);
			command_buffer.bind_vertex_buffers(0, &[&vertices.buffer], &[0]);
			command_buffer.bind_index_buffer(&indices.buffer, 0, vk::IndexType::UINT32);
			command_buffer.bind_descriptor_set(&this.target.pipeline_layout, &set.descriptor_set);
			command_buffer.draw_indexed(indices.len as u32, 1, 0, 0, 0);
			command_buffer.end_render_pass();

			Ok(())
		})
	}

	pub fn clear(&mut self, context: &mut Context, color: Vec4) -> MarsResult<()> {
		let float32 = [color.x, color.y, color.z, color.w];
		self.submit(context, |this, command_buffer| {
			command_buffer.clear_color_image(
				&this.target.color_image,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				vk::ClearColorValue { float32 },
			);
			command_buffer.clear_depth_image(
				&this.target.depth_image,
				vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
			);
			Ok(())
		})
	}

	fn submit<R: FnOnce(&mut Self, &mut CommandBuffer<Recording>) -> MarsResult<()>>(
		&mut self,
		context: &mut Context,
		recording: R,
	) -> MarsResult<()> {
		let command_buffer = context.device.allocate_command_buffer(&self.command_pool)?;
		let mut command_buffer = context.device.begin_command_buffer(command_buffer)?;

		recording(self, &mut command_buffer)?;
		let command_buffer = context.device.end_command_buffer(command_buffer)?;
		let command_buffer = context
			.device
			.queue_submit(&mut context.queue, command_buffer, &[], &[])?;
		context.device.wait_command_buffer(command_buffer)?;

		Ok(())
	}
}

pub struct DescriptorSet<B: Bindings> {
	pub arguments: B::Arguments,
	descriptor_set: RkDescriptorSet,
}
