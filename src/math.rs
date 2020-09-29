pub type Scalar = f32;

pub type Vec2<S = Scalar> = nalgebra::Vector2<S>;
pub type Vec3<S = Scalar> = nalgebra::Vector3<S>;
pub type Vec4<S = Scalar> = nalgebra::Vector4<S>;

pub type Mat4<S = Scalar> = nalgebra::Matrix4<S>;

pub type Point3<S = Scalar> = nalgebra::Point3<S>;

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct Mvp {
	pub model: Mat4,
	pub view: Mat4,
	pub proj: Mat4,
}

impl Mvp {
	pub fn new(model: Mat4, view: Mat4, proj: Mat4) -> Self {
		Self { model, view, proj }
	}

	pub fn identity() -> Self {
		Self::new(Mat4::identity(), Mat4::identity(), Mat4::identity())
	}
}
