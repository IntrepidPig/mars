use rk::{
	command::{CommandBuffer, CommandPool, Recording},
	vk,
};

use crate::{
	buffer::{Buffer, IndexBufferUsage, VertexBufferUsage},
	function::{FunctionPrototype, FunctionDef, ArgumentsContainer},
	pass::{RenderPass},
	target::Target,
	Context, MarsResult,
};

pub struct RenderEngine {
	pub(crate) render_pass: RenderPass,
	pub(crate) command_pool: CommandPool,
}

impl RenderEngine {
	pub fn new(context: &Context, render_pass: RenderPass) -> MarsResult<Self> {
		let command_pool = CommandPool::create(&context.device)?;

		let this = Self { render_pass, command_pool };
		
		Ok(this)
	}

	pub fn draw<F: FunctionPrototype>(
		&mut self,
		context: &mut Context,
		target: &mut Target,
		function: &FunctionDef<F>,
		bindings: &ArgumentsContainer<F>,
		vertices: &Buffer<VertexBufferUsage, [F::VertexInput]>,
		indices: &Buffer<IndexBufferUsage, [u32]>,
	) -> MarsResult<()> {
		self.submit(context, |this, command_buffer| {
			unsafe {
				command_buffer.begin_render_pass(
					&mut this.render_pass.render_pass,
					&mut target.framebuffer,
					vk::Rect2D {
						offset: vk::Offset2D { x: 0, y: 0 },
						extent: target.extent,
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
					width: target.extent.width as f32,
					height: target.extent.height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				});
				command_buffer.set_scissor(vk::Rect2D {
					offset: vk::Offset2D { x: 0, y: 0 },
					extent: vk::Extent2D {
						width: target.extent.width,
						height: target.extent.height,
					},
				});
				command_buffer.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &function.pipeline);
				command_buffer.bind_descriptor_set(&function.pipeline_layout, &bindings.descriptor_set);
				command_buffer.bind_vertex_buffers(0, &[&vertices.buffer], &[0]);
				command_buffer.bind_index_buffer(&indices.buffer, 0, vk::IndexType::UINT32);
				command_buffer.draw_indexed(indices.len as u32, 1, 0, 0, 0);
				command_buffer.end_render_pass();
			}

			Ok(())
		})
	}

	fn submit<R: FnOnce(&mut Self, &mut CommandBuffer<Recording>) -> MarsResult<()>>(
		&mut self,
		context: &Context,
		recording: R,
	) -> MarsResult<()> {
		let command_buffer = CommandBuffer::allocate(&self.command_pool)?;
		let mut command_buffer = command_buffer.begin()?;

		recording(self, &mut command_buffer)?;
		let command_buffer = command_buffer.end()?;
		let command_buffer = unsafe {
			context.queue.with_lock(|| context.queue.submit(command_buffer, &[], &[]))?
		};
		command_buffer.wait()?;

		Ok(())
	}
}
