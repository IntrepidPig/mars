use std::time::Instant;

use mars::{
	buffer::{Buffer, UniformBufferUsage},
	function::{FunctionDef, FunctionImpl, FunctionPrototype},
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

const VERTEX_SHADER: &str = "
#version 450

layout(set = 0, binding = 0) uniform Mvp {
	mat4 model;
	mat4 view;
	mat4 proj;
} mvp; 

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 col;

layout(location = 0) out vec4 vCol;

void main() {
	gl_Position = mvp.proj * mvp.view * mvp.model * pos;
	vCol = col;
}
";

const FRAGMENT_SHADER: &str = "
#version 450

layout(location = 0) in vec4 vCol;

layout(location = 0) out vec4 fCol;

void main() {
	fCol = vCol;
}
";

struct UniformFunction;

impl FunctionPrototype for UniformFunction {
	type VertexInput = (Vec4, Vec4);
	type Bindings = (Mvp,);
}

fn main() {
	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	let mut context = Context::create("mars_uniform_example", rk::FirstPhysicalDeviceChooser).unwrap();
	
	let mut render_pass = RenderPass::create(&mut context).unwrap();
	let mut window_engine = WindowEngine::new(&mut context, &window).unwrap();
	let mut target = window_engine.create_target(&mut context, &mut render_pass).unwrap();
	
	let vert_shader = compile_shader(VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex);
	let frag_shader = compile_shader(FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment);
	let function_impl = unsafe { FunctionImpl::<UniformFunction>::from_raw(vert_shader, frag_shader) };
	let mut function_def = FunctionDef::create(&mut context, &mut render_pass, function_impl).unwrap();

	let vertices = [
		(Vec4::new(-0.5, 0.5, 0.0, 1.0), Vec4::new(1.0, 0.0, 0.0, 1.0)),
		(Vec4::new(0.0, -0.5, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)),
		(Vec4::new(0.5, 0.5, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)),
		(Vec4::new(0.0, 0.0, -0.5, 1.0), Vec4::new(1.0, 1.0, 1.0, 1.0)),
	];
	let indices = [0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3];
	let vertex_buffer = Buffer::make_array_buffer(&mut context, &vertices).unwrap();
	let index_buffer = Buffer::make_array_buffer(&mut context, &indices).unwrap();

	let extent = window_engine.current_extent();
	let aspect = extent.width as f32 / extent.height as f32;
	let mvp_buffer_a = Buffer::<UniformBufferUsage, _>::make_item_buffer(
		&mut context,
		create_mvp(
			aspect,
			Point3::new(1.0, -1.5, 0.0),
			Vec3::new(0.0, 0.0, 0.0),
		),
	)
	.unwrap();
	let mvp_buffer_b = Buffer::<UniformBufferUsage, _>::make_item_buffer(
		&mut context,
		create_mvp(aspect, Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 5.0, 0.0)),
	)
	.unwrap();

	let mut set_a = function_def
		.make_arguments(&mut context, (mvp_buffer_a,))
		.unwrap();
	let mut set_b = function_def
		.make_arguments(&mut context, (mvp_buffer_b,))
		.unwrap();

	let start = Instant::now();
	event_loop.run(move |event, _, control_flow| {
		let t = start.elapsed().as_secs_f32();

		let extent = window_engine.current_extent();
		let aspect = extent.width as f32 / extent.height as f32;

		set_a
			.arguments
			.0
			.with_map_mut(|map| *map = create_mvp(aspect, Point3::new(1.0, -1.5, 0.0), Vec3::new(0.0, 0.0, 0.0)))
			.unwrap();
		set_b
			.arguments
			.0
			.with_map_mut(|map| *map = create_mvp(aspect, Point3::new(0.0, 0.0, 0.0), Vec3::new(t, t, t)))
			.unwrap();

		target
			.clear(&mut context, Vec4::new(1.0, 1.0, 1.0, 1.0))
			.unwrap();
		window_engine
			.render
			.draw(&mut context, &mut target, &function_def, &set_a, &vertex_buffer, &index_buffer)
			.unwrap();
		window_engine
			.render
			.draw(&mut context, &mut target, &function_def, &set_b, &vertex_buffer, &index_buffer)
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

fn create_model(position: Point3, rotation: Vec3) -> Mat4 {
	nalgebra::Isometry3::new(position.to_homogeneous().xyz(), rotation).to_homogeneous()
}

fn create_view() -> Mat4 {
	Mat4::look_at_rh(
		&Point3::new(0.0, 0.0, -3.0),
		&Point3::new(0.0, 0.0, 0.0),
		&Vec3::new(0.0, -1.0, 0.0),
	)
}

fn create_proj(aspect: f32) -> Mat4 {
	nalgebra::Perspective3::new(aspect, 3.14 / 2.0, 1.0, 1000.0).to_homogeneous()
}

fn create_mvp(aspect: f32, position: Point3, rotation: Vec3) -> Mvp {
	Mvp::new(create_model(position, rotation), create_view(), create_proj(aspect))
}
