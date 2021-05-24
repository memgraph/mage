use c_str_macro::c_str;
use std::ffi::{CStr, CString};

use crate::context::*;
use crate::edge::*;
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
        unsafe {
            let data: *const mgp_vertex;
            if self.is_first {
                self.is_first = false;
                data = ffi::mgp_vertices_iterator_get(self.ptr);
            } else {
                data = ffi::mgp_vertices_iterator_next(self.ptr);
            }

            if data.is_null() {
                None
            } else {
                let copy_ptr = ffi::mgp_vertex_copy(data, self.context.memory());
                if copy_ptr.is_null() {
                    panic!("Unable to allocate new vertex during vertex iteration.");
                }
                Some(Vertex {
                    ptr: copy_ptr,
                    context: self.context.clone(),
                })
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
        unsafe { ffi::mgp_vertex_labels_count(self.ptr) }
    }

    // TODO(gitbuda): Figure out the correct lifetime. CString should be used.
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

    pub fn property(&self, name: &CStr) -> MgpResult<Property> {
        unsafe {
            let mgp_value = MgpValue {
                ptr: ffi::mgp_vertex_get_property(self.ptr, name.as_ptr(), self.context.memory()),
            };
            if mgp_value.ptr.is_null() {
                return Err(MgpError::MgpAllocationError);
            }
            let value = mgp_value_to_value(&mgp_value, &self.context)?;
            match CString::new(name.to_bytes()) {
                Ok(c_string) => Ok(Property {
                    name: c_string,
                    value,
                }),
                Err(_) => Err(MgpError::MgpCreationOfCStringError),
            }
        }
    }

    pub fn properties(&self) -> MgpResult<PropertiesIterator> {
        let unable_alloc_iter_msg = c_str!("Unable to allocate properties iterator on vertex.");
        unsafe {
            let mgp_iterator = ffi::mgp_vertex_iter_properties(self.ptr, self.context.memory());
            if mgp_iterator.is_null() {
                ffi::mgp_result_set_error_msg(
                    self.context.result(),
                    unable_alloc_iter_msg.as_ptr(),
                );
                return Err(MgpError::MgpAllocationError);
            }
            Ok(PropertiesIterator {
                ptr: mgp_iterator,
                context: self.context.clone(),
                ..Default::default()
            })
        }
    }

    pub fn in_edges(&self) -> MgpResult<EdgesIterator> {
        let unable_alloc_iter_msg = c_str!("Unable to allocate in_edges iterator.");
        unsafe {
            let mgp_iterator = ffi::mgp_vertex_iter_in_edges(self.ptr, self.context.memory());
            if mgp_iterator.is_null() {
                ffi::mgp_result_set_error_msg(
                    self.context.result(),
                    unable_alloc_iter_msg.as_ptr(),
                );
                return Err(MgpError::MgpAllocationError);
            }
            Ok(EdgesIterator {
                ptr: mgp_iterator,
                context: self.context.clone(),
                ..Default::default()
            })
        }
    }

    pub fn out_edges(&self) -> MgpResult<EdgesIterator> {
        let unable_alloc_iter_msg = c_str!("Unable to allocate out_edges iterator.");
        unsafe {
            let mgp_iterator = ffi::mgp_vertex_iter_out_edges(self.ptr, self.context.memory());
            if mgp_iterator.is_null() {
                ffi::mgp_result_set_error_msg(
                    self.context.result(),
                    unable_alloc_iter_msg.as_ptr(),
                );
                return Err(MgpError::MgpAllocationError);
            }
            Ok(EdgesIterator {
                ptr: mgp_iterator,
                context: self.context.clone(),
                ..Default::default()
            })
        }
    }
}

pub fn make_graph_vertices_iterator(context: &Memgraph) -> MgpResult<VerticesIterator> {
    let unable_alloc_iter_msg = c_str!("Unable to allocate vertices iterator.");
    unsafe {
        let mgp_iterator = ffi::mgp_graph_iter_vertices(context.graph(), context.memory());
        if mgp_iterator.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_iter_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(VerticesIterator {
            ptr: mgp_iterator,
            context: context.clone(),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests;
