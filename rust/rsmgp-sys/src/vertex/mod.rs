use c_str_macro::c_str;
use std::ffi::CStr;

use crate::context::*;
use crate::mgp::*;
use crate::property::*;
use crate::result::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

pub struct VerticesIterator {
    ptr: *mut mgp_vertices_iterator,
    is_first: bool,
    context: Memgraph,
}

impl Default for VerticesIterator {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            is_first: true,
            context: Memgraph {
                ..Default::default()
            },
        }
    }
}

impl Drop for VerticesIterator {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_vertices_iterator_destroy(self.ptr);
            }
        }
    }
}

impl Iterator for VerticesIterator {
    type Item = Vertex;

    fn next(&mut self) -> Option<Vertex> {
        if self.is_first {
            self.is_first = false;
            unsafe {
                let data = ffi::mgp_vertices_iterator_get(self.ptr);
                if data.is_null() {
                    None
                } else {
                    // TODO(gitbuda): Handle error.
                    let vertex_copy = ffi::mgp_vertex_copy(data, self.context.memory());
                    Some(Vertex {
                        ptr: vertex_copy,
                        context: self.context.clone(),
                    })
                }
            }
        } else {
            unsafe {
                let data = ffi::mgp_vertices_iterator_next(self.ptr);
                if data.is_null() {
                    None
                } else {
                    // TODO(gitbuda): Handle error.
                    let vertex_copy = ffi::mgp_vertex_copy(data, self.context.memory());
                    Some(Vertex {
                        ptr: vertex_copy,
                        context: self.context.clone(),
                    })
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Vertex {
    pub ptr: *mut mgp_vertex,
    pub context: Memgraph,
}

impl Drop for Vertex {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_vertex_destroy(self.ptr);
            }
        }
    }
}

impl Vertex {
    pub fn id(&self) -> i64 {
        unsafe { ffi::mgp_vertex_get_id(self.ptr).as_int }
    }

    pub fn labels_count(&self) -> u64 {
        unsafe {
            // TODO(gitbuda): Figure out why this is not clippy::not_unsafe_ptr_arg_deref.
            ffi::mgp_vertex_labels_count(self.ptr)
        }
    }

    // TODO(gitbuda): Figure out the correct lifetime.
    pub fn label_at(&self, index: u64) -> MgpResult<&CStr> {
        unsafe {
            let c_label = ffi::mgp_vertex_label_at(self.ptr, index);
            if c_label.name.is_null() {
                return Err(MgpError::MgpOutOfBoundLabelIndex);
            }
            Ok(CStr::from_ptr(c_label.name))
        }
    }

    pub fn has_label(&self, name: &CStr) -> bool {
        unsafe {
            let c_mgp_label = mgp_label {
                name: name.as_ptr(),
            };
            ffi::mgp_vertex_has_label(self.ptr, c_mgp_label) != 0
        }
    }

    // TODO(gitbuda): This lifetime is probably problematic.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn property<'a>(&self, name: &'a CStr) -> MgpResult<Property<'a>> {
        unsafe {
            let mgp_value =
                ffi::mgp_vertex_get_property(self.ptr, name.as_ptr(), self.context.memory());
            if mgp_value.is_null() {
                return Err(MgpError::MgpAllocationError);
            }
            Ok(Property {
                name,
                value: mgp_value_to_value(mgp_value, &self.context)?,
            })
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn properties(&self) -> PropertiesIterator {
        unsafe {
            PropertiesIterator {
                // TODO(gitbuda): Handle errors.
                ptr: ffi::mgp_vertex_iter_properties(self.ptr, self.context.memory()),
                context: self.context.clone(),
                ..Default::default()
            }
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_graph_vertices_iterator(context: &Memgraph) -> MgpResult<VerticesIterator> {
    let unable_alloc_iter_msg = c_str!("Unable to allocate vertices iterator.");
    unsafe {
        let iterator: VerticesIterator = VerticesIterator {
            // TODO(gitbuda): Handle error.
            ptr: ffi::mgp_graph_iter_vertices(context.graph(), context.memory()),
            context: context.clone(),
            ..Default::default()
        };
        if iterator.ptr.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_iter_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(iterator)
    }
}

#[cfg(test)]
mod tests;
