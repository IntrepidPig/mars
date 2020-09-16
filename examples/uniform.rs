use std::time::Instant;

use mars::{
	buffer::{Buffer, UniformBufferUsage},
	function::{FunctionDef, FunctionImpl, FunctionPrototype},
	image::{format, usage, DynImageUsage, SampleCount1},
	math::*,
	pass::{Attachments, DepthAttachment, RenderPass, RenderPassPrototype, ColorAttachment},
	target::Target,
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

struct UniformPass;

impl RenderPassPrototype for UniformPass {
	type SampleCount = SampleCount1;
	type InputAttachments = ();
	type ColorAttachments = (ColorAttachment<format::B8G8R8A8Unorm>,);
	type DepthAttachment = DepthAttachment<format::D32Sfloat, Self::SampleCount>;
}

struct UniformFunction;

impl FunctionPrototype for UniformFunction {
	type RenderPass = UniformPass;
	type VertexInput = (Vec4, Vec4);
	type Bindings = (Mvp,);
}

fn main() {
	simple_logger::SimpleLogger::new().init().unwrap();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	let context = Context::create("mars_uniform_example", rk::FirstPhysicalDeviceChooser).unwrap();

	let mut window_engine = WindowEngine::new(&context, &window).unwrap();

	let render_pass = RenderPass::<UniformPass>::create(&context).unwrap();
	let attachments =
		Attachments::create(&context, window_engine.current_extent(), DynImageUsage::TRANSFER_SRC).unwrap();
	let mut target = Target::create(&context, &render_pass, attachments).unwrap();

	let vert_shader = compile_shader(VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex);
	let frag_shader = compile_shader(FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment);
	let function_impl = unsafe { FunctionImpl::<UniformFunction>::from_raw(vert_shader, frag_shader) };
	let mut function_def = FunctionDef::create(&context, &render_pass, function_impl).unwrap();

	let vertices = [
		(Vec4::new(-0.5, 0.5, 0.0, 1.0), Vec4::new(1.0, 0.0, 0.0, 1.0)),
		(Vec4::new(0.0, -0.5, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)),
		(Vec4::new(0.5, 0.5, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)),
		(Vec4::new(0.0, 0.0, -0.5, 1.0), Vec4::new(1.0, 1.0, 1.0, 1.0)),
	];
	let indices = [0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3];
	let vertex_buffer = Buffer::make_array_buffer(&context, &vertices).unwrap();
	let index_buffer = Buffer::make_array_buffer(&context, &indices).unwrap();

	let extent = window_engine.current_extent();
	let aspect = extent.width as f32 / extent.height as f32;
	let mvp_buffer_a = Buffer::<UniformBufferUsage, _>::make_item_buffer(
		&context,
		create_mvp(aspect, Point3::new(1.0, -1.5, 0.0), Vec3::new(0.0, 0.0, 0.0)),
	)
	.unwrap();
	let mvp_buffer_b = Buffer::<UniformBufferUsage, _>::make_item_buffer(
		&context,
		create_mvp(aspect, Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 5.0, 0.0)),
	)
	.unwrap();

	let mut set_a = function_def.make_arguments(&context, (mvp_buffer_a,)).unwrap();
	let mut set_b = function_def.make_arguments(&context, (mvp_buffer_b,)).unwrap();

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

		window_engine
			.render
			.clear(&context, &mut target, (Vec4::new(1.0, 1.0, 1.0, 1.0),), 1.0)
			.unwrap();
		window_engine
			.render
			.pass(
				&context,
				&mut target,
				&function_def,
				&set_a,
				&vertex_buffer,
				&index_buffer,
			)
			.unwrap();
		window_engine
			.render
			.pass(
				&context,
				&mut target,
				&function_def,
				&set_b,
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
