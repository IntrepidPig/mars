use std::time::Instant;

use mars::{
	buffer::Buffer,
	function::{FunctionDef, FunctionImpl, FunctionPrototype},
	image::{format, samples::SampleCount8, usage, DynImageUsage, Image, SampledImage},
	math::*,
	pass::{Attachments, MultisampledColorAttachment, NoDepthAttachment, RenderPass, RenderPassPrototype},
	target::Target,
	vk,
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

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 texCoord;

layout(location = 0) out vec2 texCoordOut;

void main() {
	gl_Position = mvp.proj * mvp.view * mvp.model * vec4(pos.xy, 0.5, 1.0);
	texCoordOut = texCoord;
}
";

const FRAGMENT_SHADER: &str = "
#version 450

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fCol;

void main() {
	fCol = texture(texSampler, texCoord);
}
";

struct TexturePass;

impl RenderPassPrototype for TexturePass {
	type SampleCount = SampleCount8;
	type InputAttachments = ();
	type ColorAttachments = (MultisampledColorAttachment<format::B8G8R8A8Unorm, Self::SampleCount>,);
	type DepthAttachment = NoDepthAttachment;
}

struct TextureFunction;

impl FunctionPrototype for TextureFunction {
	type RenderPass = TexturePass;
	type VertexInput = (Vec2, Vec2);
	type Bindings = (Mvp, SampledImage<format::R8G8B8A8Srgb>);
}

fn main() {
	simple_logger::SimpleLogger::new().init().unwrap();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	let context = Context::create("mars_texture_example", rk::FirstPhysicalDeviceChooser).unwrap();

	let mut window_engine = WindowEngine::new(&context, &window).unwrap();

	let render_pass = RenderPass::<TexturePass>::create(&context).unwrap();
	let attachments =
		Attachments::create(&context, window_engine.current_extent(), DynImageUsage::TRANSFER_SRC).unwrap();
	let mut target = Target::create(&context, &render_pass, attachments).unwrap();

	let vert_shader = compile_shader(VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex);
	let frag_shader = compile_shader(FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment);
	let function_impl = unsafe { FunctionImpl::<TextureFunction>::from_raw(vert_shader, frag_shader) };
	let mut function_def = FunctionDef::create(&context, &render_pass, function_impl).unwrap();

	let vertices = [
		(Vec2::new(-0.5, -0.5), Vec2::new(0.0, 0.0)),
		(Vec2::new(0.5, -0.5), Vec2::new(1.0, 0.0)),
		(Vec2::new(0.5, 0.5), Vec2::new(1.0, 1.0)),
		(Vec2::new(-0.5, 0.5), Vec2::new(0.0, 1.0)),
	];
	let indices = [0, 1, 2, 2, 3, 0];
	let vertex_buffer = Buffer::make_array_buffer(&context, &vertices).unwrap();
	let index_buffer = Buffer::make_array_buffer(&context, &indices).unwrap();

	let mvp = Mvp::new(Mat4::identity(), Mat4::identity(), Mat4::identity());
	let mvp_buffer = Buffer::make_item_buffer(&context, mvp).unwrap();

	let texture_data = load_image();
	let image = Image::make_image(
		&context,
		usage::SampledImage,
		vk::Extent2D {
			width: texture_data.width(),
			height: texture_data.height(),
		},
		&texture_data,
	)
	.unwrap();
	let sampled_image = SampledImage::create(&context, image).unwrap();

	let mut set = function_def
		.make_arguments(&context, (mvp_buffer, sampled_image))
		.unwrap();

	let start = Instant::now();
	event_loop.run(move |event, _, control_flow| {
		let t = start.elapsed().as_secs_f32();

		let extent = window_engine.current_extent();
		let aspect = extent.width as f32 / extent.height as f32;

		set.arguments
			.0
			.with_map_mut(|map| *map = create_mvp(aspect, Point3::new(0.0, 0.0, 0.0), Vec3::new(0.2, t, -0.2)))
			.unwrap();

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
				[(&set, &vertex_buffer, &index_buffer).into()].iter().copied(),
			)
			.unwrap();

		if let Some(new_extent) = window_engine
			.present(
				&context,
				target
					.color_attachments()
					.0
					.resolve_image
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

fn create_mvp(aspect: f32, position: Point3, rotation: Vec3) -> Mvp {
	Mvp::new(create_model(position, rotation), create_view(), create_proj(aspect))
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
