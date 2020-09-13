use mars::{
	Context,
	buffer::Buffer,
	function::{FunctionPrototype, FunctionImpl, FunctionDef},
	pass::{RenderPass, subpasses::{SimpleSubpassPrototype}},
	image::{usage, format, DynImageUsage},
	target::{Target},
	math::*,
	window::WindowEngine,
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

struct TriangleFunction;

impl FunctionPrototype for TriangleFunction {
	type VertexInput = (Vec4, Vec4);
	type Bindings = ();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	simple_logger::SimpleLogger::new().init()?;

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop)?;

	let context = Context::create("mars_triangle_example", rk::FirstPhysicalDeviceChooser)?;
	
	let mut window_engine = WindowEngine::new(&context, &window).unwrap();
	
	let simple_subpass = SimpleSubpassPrototype::<format::B8G8R8A8Unorm, format::D32Sfloat>::new();
	let render_pass = RenderPass::create(&context, &simple_subpass)?;
	
	let attachments=  SimpleSubpassPrototype::create_attachments(&context, DynImageUsage::TRANSFER_SRC, window_engine.current_extent())?;
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
			.clear(&context, &mut target, &render_pass, Vec4::new(1.0, 1.0, 1.0, 1.0), 1.0)
			.unwrap();
		window_engine
			.render
			.draw(&context, &mut target, &render_pass, &function_def, &set, &vertex_buffer, &index_buffer)
			.unwrap();
		if let Some(new_extent) = window_engine.present(&context, target.attachments().color_attachments.0.image.cast_usage_ref(usage::TransferSrc).unwrap()).unwrap() {
			let attachments = SimpleSubpassPrototype::create_attachments(&context, DynImageUsage::TRANSFER_SRC, new_extent).unwrap();
			target = Target::create(&context, &render_pass, attachments).unwrap();
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

fn compile_shader(source: &str, filename: &str, kind: shaderc::ShaderKind) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
	let mut compiler = shaderc::Compiler::new().unwrap();
	let artifact = compiler.compile_into_spirv(source, kind, filename, "main", None)?;
	Ok(artifact.as_binary().to_owned())
}
