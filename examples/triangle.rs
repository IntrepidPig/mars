use mars::{
	buffer::Buffer,
	function::{FunctionDef, FunctionImpl, FunctionPrototype},
	image::{format, usage, DynImageUsage},
	math::*,
	pass::{Attachments, ColorAttachment, NoDepthAttachment, RenderPass, RenderPassPrototype},
	target::Target,
	window::WindowEngine,
	Context,
};

use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

const TRIANGLE_VERTEX_SHADER: &str = "
#version 450

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 col;

layout(location = 0) out vec4 vCol;

void main() {
	gl_Position = pos;
	vCol = col;
}
";

const TRIANGLE_FRAGMENT_SHADER: &str = "
#version 450

layout(location = 0) in vec4 vCol;

layout(location = 0) out vec4 fCol;

void main() {
	fCol = vCol;
}
";

struct TrianglePass;

impl RenderPassPrototype for TrianglePass {
	type InputAttachments = ();
	type ColorAttachments = (ColorAttachment<format::B8G8R8A8Unorm>,);
	type DepthAttachment = NoDepthAttachment;
}

struct TriangleFunction;

impl FunctionPrototype for TriangleFunction {
	type RenderPass = TrianglePass;
	type VertexInput = (Vec4, Vec4);
	type Bindings = ();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	simple_logger::SimpleLogger::new().init()?;

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop)?;

	let context = Context::create("mars_triangle_example", rk::FirstPhysicalDeviceChooser)?;

	let mut window_engine = WindowEngine::new(&context, &window)?;

	let render_pass = RenderPass::<TrianglePass>::create(&context)?;
	let attachments = Attachments::create(&context, window_engine.current_extent(), DynImageUsage::TRANSFER_SRC)?;
	let mut target = Target::create(&context, &render_pass, attachments)?;

	let vert_shader = compile_shader(TRIANGLE_VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex)?;
	let frag_shader = compile_shader(TRIANGLE_FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment)?;
	let function_impl = unsafe { FunctionImpl::<TriangleFunction>::from_raw(vert_shader, frag_shader) };
	let mut function_def = FunctionDef::create(&context, &render_pass, function_impl)?;

	let vertices = [
		(Vec4::new(-0.5, 0.5, 0.0, 1.0), Vec4::new(1.0, 0.0, 0.0, 1.0)),
		(Vec4::new(0.0, -0.5, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)),
		(Vec4::new(0.5, 0.5, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)),
	];
	let indices = [0, 1, 2];
	let vertex_buffer = Buffer::make_array_buffer(&context, &vertices)?;
	let index_buffer = Buffer::make_array_buffer(&context, &indices)?;

	let set = function_def.make_arguments(&context, ())?;

	event_loop.run(move |event, _, control_flow| {
		window_engine
			.render
			.clear(&context, &mut target, (Vec4::new(1.0, 1.0, 1.0, 1.0),), ())
			.unwrap();
		window_engine
			.render
			.pass(
				&context,
				&mut target,
				&function_def,
				&set,
				&vertex_buffer,
				&index_buffer,
			)
			.unwrap();
		if let Some(new_extent) = window_engine
			.present(
				&context,
				target
					.color_attachments()
					.0
					.image
					.cast_usage_ref(usage::TransferSrc)
					.unwrap(),
			)
			.unwrap()
		{
			let attachments = Attachments::create(&context, new_extent, DynImageUsage::TRANSFER_SRC).unwrap();
			target.change_attachments(&context, attachments).unwrap();
		}

		match event {
			Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} => *control_flow = ControlFlow::Exit,
			_ => {}
		}
	});
}

fn compile_shader(
	source: &str,
	filename: &str,
	kind: shaderc::ShaderKind,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
	let mut compiler = shaderc::Compiler::new().unwrap();
	let artifact = compiler.compile_into_spirv(source, kind, filename, "main", None)?;
	Ok(artifact.as_binary().to_owned())
}
