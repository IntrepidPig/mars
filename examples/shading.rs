use std::time::Instant;

use mars::{
	buffer::Buffer,
	function::{FunctionDef, FunctionImpl, FunctionPrototype},
	image::{usage, DynImageUsage, format},
	pass::{RenderPass, subpasses::SimpleSubpassPrototype},
	target::{Target},
	math::*,
	window::WindowEngine,
	Context,
};

use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

const CUBE_VERTEX_SHADER: &str = "
#version 450

layout(set = 0, binding = 0) uniform Mvp {
	mat4 model;
	mat4 view;
	mat4 proj;
} mvp; 

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNormal;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec3 fNormal;
layout(location = 2) out vec4 fCol;

void main() {
	gl_Position = mvp.proj * mvp.view * mvp.model * vec4(vPos, 1.0);
	fPos = (mvp.model * vec4(vPos, 1.0)).xyz;
	fNormal = mat3(transpose(inverse(mvp.model))) * vNormal;
	fCol = vec4(0.6, 0.3, 0.1, 1.0);
}
";

const CUBE_FRAGMENT_SHADER: &str = "
#version 450

layout(set = 0, binding = 1) uniform Light {
	vec3 pos;
} light;

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec3 fNormal;
layout(location = 2) in vec4 fCol;

layout(location = 0) out vec4 col;

void main() {
	vec3 ambient = vec3(0.2, 0.2, 0.2);
	vec3 fragNorm = normalize(fNormal.xyz);
	vec3 lightNorm = normalize(light.pos - fPos);
	float diff = max(dot(fragNorm, lightNorm), 0.0);
	vec3 diffuse = vec3(diff);
	vec3 result = (diffuse + ambient) * fCol.xyz;
	col = vec4(result, 1.0);
}
";

const LIGHT_VERTEX_SHADER: &str = "
#version 450

layout(set = 0, binding = 0) uniform Mvp {
	mat4 model;
	mat4 view;
	mat4 proj;
} mvp; 

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNormal;

void main() {
	gl_Position = mvp.proj * mvp.view * mvp.model * vec4(vPos, 1.0);
}
";

const LIGHT_FRAGMENT_SHADER: &str = "
#version 450

layout(location = 0) out vec4 col;

void main() {
	col = vec4(1.0, 1.0, 1.0, 1.0);
}
";

struct CubeShadingFunction;

impl FunctionPrototype for CubeShadingFunction {
	type VertexInput = (Vec3, Vec3);
	type Bindings = (Mvp, Vec3);
}

struct LightShadingFunction;

impl FunctionPrototype for LightShadingFunction {
	type VertexInput = (Vec3, Vec3);
	type Bindings = (Mvp,);
}

fn main() {
	simple_logger::SimpleLogger::new().init().unwrap();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	let context = Context::create("mars_shading_example", rk::FirstPhysicalDeviceChooser).unwrap();

	let mut window_engine = WindowEngine::new(&context, &window).unwrap();
	
	let simple_subpass = SimpleSubpassPrototype::<format::B8G8R8A8Unorm, format::D32Sfloat>::new();
	let render_pass = RenderPass::create(&context, &simple_subpass).unwrap();
	
	let attachments=  SimpleSubpassPrototype::create_attachments(&context, DynImageUsage::TRANSFER_SRC, window_engine.current_extent()).unwrap();
	let mut target = Target::create(&context, &render_pass, attachments).unwrap();
	
	let cube_vert_shader = compile_shader(CUBE_VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex);
	let cube_frag_shader = compile_shader(CUBE_FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment);
	let cube_function_impl = unsafe { FunctionImpl::<CubeShadingFunction>::from_raw(cube_vert_shader, cube_frag_shader) };
	let mut cube_function_def = FunctionDef::create(&context, &render_pass, cube_function_impl).unwrap();

	let light_vert_shader = compile_shader(LIGHT_VERTEX_SHADER, "vert.glsl", shaderc::ShaderKind::Vertex);
	let light_frag_shader = compile_shader(LIGHT_FRAGMENT_SHADER, "frag.glsl", shaderc::ShaderKind::Fragment);
	let light_function_impl = unsafe { FunctionImpl::<LightShadingFunction>::from_raw(light_vert_shader, light_frag_shader) };
	let mut light_function_def = FunctionDef::create(&context, &render_pass, light_function_impl).unwrap();

	let vertices: [(Vec3, Vec3); 36] = [
		(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.0,  0.0, -1.0)),
		(Vec3::new( 0.5, -0.5, -0.5), Vec3::new(0.0,  0.0, -1.0)),
		(Vec3::new( 0.5,  0.5, -0.5), Vec3::new(0.0,  0.0, -1.0)),
		(Vec3::new( 0.5,  0.5, -0.5), Vec3::new(0.0,  0.0, -1.0)),
		(Vec3::new(-0.5,  0.5, -0.5), Vec3::new(0.0,  0.0, -1.0)),
		(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.0,  0.0, -1.0)),
	
		(Vec3::new(-0.5, -0.5,  0.5), Vec3::new( 0.0,  0.0,  1.0)),
		(Vec3::new( 0.5, -0.5,  0.5), Vec3::new( 0.0,  0.0,  1.0)),
		(Vec3::new( 0.5,  0.5,  0.5), Vec3::new( 0.0,  0.0,  1.0)),
		(Vec3::new( 0.5,  0.5,  0.5), Vec3::new( 0.0,  0.0,  1.0)),
		(Vec3::new(-0.5,  0.5,  0.5), Vec3::new( 0.0,  0.0,  1.0)),
		(Vec3::new(-0.5, -0.5,  0.5), Vec3::new( 0.0,  0.0,  1.0)),
	
		(Vec3::new(-0.5,  0.5,  0.5),  Vec3::new(-1.0,  0.0,  0.0)),
		(Vec3::new(-0.5,  0.5, -0.5),  Vec3::new(-1.0,  0.0,  0.0)),
		(Vec3::new(-0.5, -0.5, -0.5),  Vec3::new(-1.0,  0.0,  0.0)),
		(Vec3::new(-0.5, -0.5, -0.5),  Vec3::new(-1.0,  0.0,  0.0)),
		(Vec3::new(-0.5, -0.5,  0.5),  Vec3::new(-1.0,  0.0,  0.0)),
		(Vec3::new(-0.5,  0.5,  0.5),  Vec3::new(-1.0,  0.0,  0.0)),
	
		(Vec3::new( 0.5,  0.5,  0.5), Vec3::new( 1.0,  0.0,  0.0)),
		(Vec3::new( 0.5,  0.5, -0.5), Vec3::new( 1.0,  0.0,  0.0)),
		(Vec3::new( 0.5, -0.5, -0.5), Vec3::new( 1.0,  0.0,  0.0)),
		(Vec3::new( 0.5, -0.5, -0.5), Vec3::new( 1.0,  0.0,  0.0)),
		(Vec3::new( 0.5, -0.5,  0.5), Vec3::new( 1.0,  0.0,  0.0)),
		(Vec3::new( 0.5,  0.5,  0.5), Vec3::new( 1.0,  0.0,  0.0)),
	
		(Vec3::new(-0.5, -0.5, -0.5), Vec3::new( 0.0, -1.0,  0.0)),
		(Vec3::new( 0.5, -0.5, -0.5), Vec3::new( 0.0, -1.0,  0.0)),
		(Vec3::new( 0.5, -0.5,  0.5), Vec3::new( 0.0, -1.0,  0.0)),
		(Vec3::new( 0.5, -0.5,  0.5), Vec3::new( 0.0, -1.0,  0.0)),
		(Vec3::new(-0.5, -0.5,  0.5), Vec3::new( 0.0, -1.0,  0.0)),
		(Vec3::new(-0.5, -0.5, -0.5), Vec3::new( 0.0, -1.0,  0.0)),
	
		(Vec3::new(-0.5,  0.5, -0.5), Vec3::new( 0.0,  1.0,  0.0)),
		(Vec3::new( 0.5,  0.5, -0.5), Vec3::new( 0.0,  1.0,  0.0)),
		(Vec3::new( 0.5,  0.5,  0.5), Vec3::new( 0.0,  1.0,  0.0)),
		(Vec3::new( 0.5,  0.5,  0.5), Vec3::new( 0.0,  1.0,  0.0)),
		(Vec3::new(-0.5,  0.5,  0.5), Vec3::new( 0.0,  1.0,  0.0)),
		(Vec3::new(-0.5,  0.5, -0.5), Vec3::new(  0.0,  1.0,  0.0)),
	];
	let indices = (0..36).collect::<Vec<_>>();
	let vertex_buffer = Buffer::make_array_buffer(&context, &vertices).unwrap();
	let index_buffer = Buffer::make_array_buffer(&context, &indices).unwrap();

	let extent = window_engine.current_extent();
	let aspect = extent.width as f32 / extent.height as f32;
	let cube_mvp_buffer = Buffer::make_item_buffer(
		&context,
		create_mvp(
			aspect,
			Point3::new(1.0, -1.5, 0.0),
			Vec3::new(0.0, 0.0, 0.0),
		),
	)
	.unwrap();
	let light_position_buffer = Buffer::make_item_buffer(&context, Vec3::new(0.0, 0.0, 0.0)).unwrap();
	let light_mvp_buffer = Buffer::make_item_buffer(&context, Mvp::new(Mat4::identity(), Mat4::identity(), Mat4::identity())).unwrap();

	let mut cube_arguments = cube_function_def
		.make_arguments(&context, (cube_mvp_buffer, light_position_buffer))
		.unwrap();
	let mut light_arguments = light_function_def
		.make_arguments(&context, (light_mvp_buffer,))
		.unwrap();

	let start = Instant::now();
	event_loop.run(move |event, _, control_flow| {
		let t = start.elapsed().as_secs_f32();

		let extent = window_engine.current_extent();
		let aspect = extent.width as f32 / extent.height as f32;

		let light_pos = Point3::new(t.cos() * 3.0, 1.5, t.sin() * 3.0);

		cube_arguments
			.arguments
			.0
			.with_map_mut(|map| *map = create_mvp(aspect, Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)))
			.unwrap();
		cube_arguments
			.arguments
			.1
			.with_map_mut(|map| *map = Vec3::new(light_pos.x, light_pos.y, light_pos.z))
			.unwrap();
		
		light_arguments
			.arguments
			.0
			.with_map_mut(|map| {
				let model = nalgebra::Isometry3::new(Vec3::new(light_pos.x, light_pos.y, light_pos.z), Vec3::new(0.0, 0.0, 0.0)).to_homogeneous()
					* Mat4::new_scaling(0.3);
				let view = create_view();
				let proj = create_proj(aspect);
				let mvp = Mvp::new(model, view, proj);
				*map = mvp;
			})
			.unwrap();

		window_engine.render
			.clear(&context, &mut target, &render_pass, Vec4::new(0.3, 0.3, 0.3, 0.3), 1.0)
			.unwrap();
		window_engine
			.render
			.draw(&context, &mut target, &render_pass, &cube_function_def, &cube_arguments, &vertex_buffer, &index_buffer)
			.unwrap();
		window_engine
			.render
			.draw(&context, &mut target, &render_pass, &light_function_def, &light_arguments, &vertex_buffer, &index_buffer)
			.unwrap();
		window_engine.present(&context, target.attachments().color_attachments.0.image.cast_usage_ref(usage::TransferSrc).unwrap()).unwrap();

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
		&Point3::new(1.5 * 3.0, 1.0 * 3.0, -1.5 * 3.0),
		&Point3::new(0.0, 0.0, 0.0),
		&Vec3::new(0.0, -1.0, 0.0),
	)
}

fn create_proj(aspect: f32) -> Mat4 {
	nalgebra::Perspective3::new(aspect, 3.14 / 2.5, 0.1, 1000.0).to_homogeneous()
}

fn create_mvp(aspect: f32, position: Point3, rotation: Vec3) -> Mvp {
	Mvp::new(create_model(position, rotation), create_view(), create_proj(aspect))
}
