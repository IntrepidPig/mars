use rk::{
	command::{CommandBuffer, CommandPool, Recording},
	vk,
};

use crate::{
	buffer::{Buffer, IndexBufferUsage, VertexBufferUsage},
	function::{ArgumentsContainer, FunctionDef, FunctionPrototype},
	pass::{ColorAttachments, DepthAttachmentType, RenderPassPrototype},
	target::Target,
	Context, MarsResult,
};

pub struct RenderEngine {
	pub(crate) command_pool: CommandPool,
}

impl RenderEngine {
	pub fn new(context: &Context) -> MarsResult<Self> {
		let command_pool = CommandPool::create(&context.device)?;

		let this = Self { command_pool };

		Ok(this)
	}

	pub fn clear<G: RenderPassPrototype>(
		&mut self,
		context: &Context,
		target: &mut Target<G>,
		colors: <G::ColorAttachments as ColorAttachments>::ClearValues,
		depth: <G::DepthAttachment as DepthAttachmentType>::ClearValue,
	) -> MarsResult<()> {
		self.submit(context, |_this, command_buffer| {
			unsafe {
				command_buffer.begin_render_pass(
					&target.render_pass,
					&target.framebuffer,
					vk::Rect2D {
						offset: vk::Offset2D { x: 0, y: 0 },
						extent: target.attachments.extent(),
					},
					&[],
				)?;
				let clear_attachments = target.attachments.clears(colors, depth);
				let clear_rects = vec![
					vk::ClearRect {
						rect: vk::Rect2D {
							offset: vk::Offset2D { x: 0, y: 0 },
							extent: target.attachments.extent(),
						},
						base_array_layer: 0,
						layer_count: 1,
					};
					clear_attachments.len()
				];
				command_buffer.clear_attachments(&clear_attachments, &clear_rects);
				command_buffer.end_render_pass();
			}

			Ok(())
		})
	}

	pub fn pass<F: FunctionPrototype>(
		&mut self,
		context: &Context,
		target: &mut Target<F::RenderPass>,
		function: &FunctionDef<F>,
		bindings: &ArgumentsContainer<F>,
		vertices: &Buffer<VertexBufferUsage, [F::VertexInput]>,
		indices: &Buffer<IndexBufferUsage, [u32]>,
	) -> MarsResult<()> {
		self.submit(context, |_this, command_buffer| {
			unsafe {
				command_buffer.begin_render_pass(
					&target.render_pass,
					&target.framebuffer,
					vk::Rect2D {
						offset: vk::Offset2D { x: 0, y: 0 },
						extent: target.attachments.extent,
					},
					&[],
				)?;
				command_buffer.set_viewport(vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: target.attachments.extent.width as f32,
					height: target.attachments.extent.height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				});
				command_buffer.set_scissor(vk::Rect2D {
					offset: vk::Offset2D { x: 0, y: 0 },
					extent: vk::Extent2D {
						width: target.attachments.extent.width,
						height: target.attachments.extent.height,
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
			context
				.queue
				.with_lock(|| context.queue.submit(command_buffer, &[], &[]))?
		};
		command_buffer.wait()?;

		Ok(())
	}
}
