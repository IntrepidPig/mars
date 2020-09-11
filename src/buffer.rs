use std::{
	marker::PhantomData,
	ops::{Deref, DerefMut},
	os::raw::c_void,
};

use rk::{buffer::Buffer as RkBuffer, vk};

use crate::{Context, MarsResult};

pub trait BufferUsageType {
	fn as_raw() -> vk::BufferUsageFlags;
}

pub struct Buffer<U: BufferUsageType, T: Copy> {
	pub(crate) buffer: RkBuffer,
	pub(crate) len: usize,
	pub(crate) size: usize,
	pub(crate) _phantom: PhantomData<(U, T)>,
}

impl<U, T> Buffer<U, T>
where
	U: BufferUsageType,
	T: Copy,
{
	pub fn make_buffer(context: &mut Context, data: &[T]) -> MarsResult<Self> {
		let buffer = context.device.make_buffer(data, U::as_raw())?;
		Ok(Self {
			buffer,
			len: data.len(),
			size: data.len() * std::mem::size_of::<T>(),
			_phantom: PhantomData,
		})
	}

	pub fn map<'a>(&'a self, context: &'a mut Context) -> MarsResult<Map<'a, U, T>> {
		unsafe {
			let ptr = context.device.map_buffer(&self.buffer)?;
			Ok(Map {
				context,
				buffer: self,
				ptr,
			})
		}
	}

	pub fn map_mut<'a>(&'a mut self, context: &'a mut Context) -> MarsResult<MapMut<'a, U, T>> {
		unsafe {
			let ptr = context.device.map_buffer(&self.buffer)?;
			Ok(MapMut {
				context,
				buffer: self,
				ptr,
			})
		}
	}

	pub fn with_map<F: FnOnce(&[T])>(&self, context: &mut Context, f: F) -> MarsResult<()> {
		f(&*self.map(context)?);
		Ok(())
	}

	pub fn with_map_mut<F: FnOnce(&mut [T])>(&mut self, context: &mut Context, f: F) -> MarsResult<()> {
		f(&mut *self.map_mut(context)?);
		Ok(())
	}

	pub fn as_untyped(&self) -> UntypedBuffer<U> {
		UntypedBuffer {
			buffer: self.cast_ref::<()>(),
		}
	}

	fn cast_ref<V: Copy>(&self) -> &Buffer<U, V> {
		unsafe { std::mem::transmute(self) }
	}
}

pub struct Map<'a, U: BufferUsageType, T: Copy> {
	context: &'a Context,
	buffer: &'a Buffer<U, T>,
	ptr: *const c_void,
}

impl<'a, U, T> Deref for Map<'a, U, T>
where
	U: BufferUsageType,
	T: Copy,
{
	type Target = [T];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.ptr as *const _, self.buffer.len) }
	}
}

impl<'a, U, T> Drop for Map<'a, U, T>
where
	U: BufferUsageType,
	T: Copy,
{
	fn drop(&mut self) {
		unsafe {
			self.context.device.unmap_buffer(&self.buffer.buffer);
		}
	}
}

pub struct MapMut<'a, U: BufferUsageType, T: Copy> {
	context: &'a Context,
	buffer: &'a Buffer<U, T>,
	ptr: *mut c_void,
}

impl<'a, U, T> Deref for MapMut<'a, U, T>
where
	U: BufferUsageType,
	T: Copy,
{
	type Target = [T];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.ptr as *mut _ as *const _, self.buffer.len) }
	}
}

impl<'a, U, T> DerefMut for MapMut<'a, U, T>
where
	U: BufferUsageType,
	T: Copy,
{
	fn deref_mut(&mut self) -> &mut Self::Target {
		unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut _, self.buffer.len) }
	}
}

impl<'a, U, T> Drop for MapMut<'a, U, T>
where
	U: BufferUsageType,
	T: Copy,
{
	fn drop(&mut self) {
		unsafe {
			self.context.device.unmap_buffer(&self.buffer.buffer);
		}
	}
}

pub struct UntypedBuffer<'a, U: BufferUsageType> {
	pub(crate) buffer: &'a Buffer<U, ()>,
}

macro_rules! buffer_usage {
	($name:ident, $usage:ident) => {
		pub struct $name;

		impl BufferUsageType for $name {
			fn as_raw() -> vk::BufferUsageFlags {
				vk::BufferUsageFlags::$usage
			}
		}
	};
}

buffer_usage!(VertexBufferUsage, VERTEX_BUFFER);
buffer_usage!(IndexBufferUsage, INDEX_BUFFER);
buffer_usage!(UniformBufferUsage, UNIFORM_BUFFER);
buffer_usage!(TransferSrcBufferUsage, TRANSFER_SRC);
