use std::time::Instant;

use mars::{
	buffer::Buffer,
	function::{FunctionDef, FunctionShader},
	image::{self, Image, SampledImage},
	math::*,
	vk,
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

layout(set = 0, binding = 0) uniform MVP {
	mat4 mvp;
} mvp;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 texCoord;

layout(location = 0) out vec2 texCoordOut;

void main() {
	gl_Position = mvp.mvp * vec4(pos.xy, 0.5, 1.0);
	texCoordOut = texCoord;
}
";

const TRIANGLE_FRAGMENT_SHADER: &str = "
#version 450

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fCol;

void main() {
	fCol = texture(texSampler, texCoord);
}
";

struct TriangleFunction;

impl FunctionDef for TriangleFunction {
	type VertexInput = (Vec2, Vec2);
	type Bindings = (Mat4, SampledImage<image::R8G8B8A8SrgbFormat>);
}

fn main() {
	setup_logging();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	let mut context = Context::create("mars_triangle_example", rk::FirstPhysicalDeviceChooser).unwrap();
	let vert_shader = compile_shader(TRIANGLE_VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex);
	let frag_shader = compile_shader(TRIANGLE_FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment);
	let function_shader = unsafe { FunctionShader::<TriangleFunction>::from_raw(vert_shader, frag_shader) };
	let mut window_engine = WindowEngine::new(&mut context, &window, function_shader).unwrap();

	let vertices = [
		(Vec2::new(-0.5, -0.5), Vec2::new(0.0, 0.0)),
		(Vec2::new(0.5, -0.5), Vec2::new(1.0, 0.0)),
		(Vec2::new(0.5, 0.5), Vec2::new(1.0, 1.0)),
		(Vec2::new(-0.5, 0.5), Vec2::new(0.0, 1.0)),
	];
	let indices = [0, 1, 2, 2, 3, 0];
	let vertex_buffer = Buffer::make_buffer(&mut context, &vertices).unwrap();
	let index_buffer = Buffer::make_buffer(&mut context, &indices).unwrap();

	let mvp = Mat4::identity();
	let mvp_buffer = Buffer::make_buffer(&mut context, &[mvp]).unwrap();

	let texture_data = load_image();
	let image = Image::make_image(
		&mut context,
		vk::Extent2D {
			width: texture_data.width(),
			height: texture_data.height(),
		},
		&texture_data,
	)
	.unwrap();
	let sampled_image = SampledImage::create(&mut context, image).unwrap();

	let mut set = window_engine
		.render
		.make_descriptor_set(&mut context, (mvp_buffer, sampled_image))
		.unwrap();

	let start = Instant::now();
	event_loop.run(move |event, _, control_flow| {
		let t = start.elapsed().as_secs_f32();

		let extent = window_engine.current_extent();
		let aspect = extent.width as f32 / extent.height as f32;

		set.arguments
			.0
			.with_map_mut(&mut context, |map| {
				map[0] = create_mvp(aspect, Point3::new(0.0, 0.0, 0.0), Vec3::new(0.2, t, -0.2));
			})
			.unwrap();

		window_engine
			.render
			.clear(&mut context, Vec4::new(1.0, 1.0, 1.0, 1.0))
			.unwrap();
		window_engine
			.render
			.draw(&mut context, &set, &vertex_buffer, &index_buffer)
			.unwrap();
		window_engine.present(&mut context).unwrap();

		match event {
			Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} => *control_flow = ControlFlow::Exit,
			_ => {}
		}
	});
}

fn create_model(position: Point3, rotation: Vec3) -> Mat4 {
	nalgebra::Isometry3::new(position.to_homogeneous().xyz(), rotation).to_homogeneous()
}

fn create_view() -> Mat4 {
	Mat4::look_at_rh(
		&Point3::new(0.0, 0.0, -1.2),
		&Point3::new(0.0, 0.0, 0.0),
		&Vec3::new(0.0, -1.0, 0.0),
	)
}

fn create_proj(aspect: f32) -> Mat4 {
	nalgebra::Perspective3::new(aspect, 3.14 / 2.0, 0.1, 1000.0).to_homogeneous()
}

fn create_mvp(aspect: f32, position: Point3, rotation: Vec3) -> Mat4 {
	create_proj(aspect) * create_view() * create_model(position, rotation)
}

fn load_image() -> ::image::RgbaImage {
	let dyn_image = ::image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/mars.jpg")).unwrap();
	dyn_image.into_rgba()
}

fn compile_shader(source: &str, filename: &str, kind: shaderc::ShaderKind) -> Vec<u32> {
	let mut compiler = shaderc::Compiler::new().expect("Failed to initialize compiler");
	let artifact = compiler
		.compile_into_spirv(source, kind, filename, "main", None)
		.expect("Failed to compile shader");
	artifact.as_binary().to_owned()
}

fn setup_logging() {
	let colors = Box::new(fern::colors::ColoredLevelConfig::new())
		.info(fern::colors::Color::Blue)
		.warn(fern::colors::Color::Yellow)
		.error(fern::colors::Color::Red)
		.debug(fern::colors::Color::BrightGreen);
	fern::Dispatch::new()
		.format(move |out, message, record| out.finish(format_args!("[{}] {}", colors.color(record.level()), message)))
		.level(log::LevelFilter::Trace)
		.chain(std::io::stderr())
		.apply()
		.expect("Failed to setup logging dispatch");
}
