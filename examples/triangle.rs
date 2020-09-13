use mars::{
	buffer::Buffer,
	function::{FunctionPrototype, FunctionImpl, FunctionDef},
	pass::{RenderPass},
	math::*,
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

struct TriangleFunction;

impl FunctionPrototype for TriangleFunction {
	type VertexInput = (Vec4, Vec4);
	type Bindings = ();
}

fn main() {
	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	let mut context = Context::create("mars_triangle_example", rk::FirstPhysicalDeviceChooser).unwrap();
	
	let mut render_pass = RenderPass::create(&mut context).unwrap();
	let mut window_engine = WindowEngine::new(&mut context, &window).unwrap();
	let mut target = window_engine.create_target(&mut context, &mut render_pass).unwrap();
	
	let vert_shader = compile_shader(TRIANGLE_VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex);
	let frag_shader = compile_shader(TRIANGLE_FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment);
	let function_impl = unsafe { FunctionImpl::<TriangleFunction>::from_raw(vert_shader, frag_shader) };
	let mut function_def = FunctionDef::create(&mut context, &mut render_pass, function_impl).unwrap();

	let vertices = [
		(Vec4::new(-0.5, 0.5, 0.0, 1.0), Vec4::new(1.0, 0.0, 0.0, 1.0)),
		(Vec4::new(0.0, -0.5, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)),
		(Vec4::new(0.5, 0.5, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)),
	];
	let indices = [0, 1, 2];
	let vertex_buffer = Buffer::make_array_buffer(&mut context, &vertices).unwrap();
	let index_buffer = Buffer::make_array_buffer(&mut context, &indices).unwrap();
	let set = function_def.make_arguments(&mut context, ()).unwrap();

	event_loop.run(move |event, _, control_flow| {
		target
			.clear(&mut context, Vec4::new(1.0, 1.0, 1.0, 1.0))
			.unwrap();
		window_engine
			.render
			.draw(&mut context, &mut target, &function_def, &set, &vertex_buffer, &index_buffer)
			.unwrap();
		window_engine.present(&mut context, &mut target).unwrap();

		match event {
			Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} => *control_flow = ControlFlow::Exit,
			_ => {}
		}
	});
}

fn compile_shader(source: &str, filename: &str, kind: shaderc::ShaderKind) -> Vec<u32> {
	let mut compiler = shaderc::Compiler::new().expect("Failed to initialize compiler");
	let artifact = compiler
		.compile_into_spirv(source, kind, filename, "main", None)
		.expect("Failed to compile shader");
	artifact.as_binary().to_owned()
}
