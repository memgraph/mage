use c_str_macro::c_str;
use std::ffi::{CStr, CString};

use crate::context::*;
use crate::mgp::*;
use crate::property::*;
use crate::result::*;
use crate::value::*;
use crate::vertex::Vertex;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

pub struct EdgesIterator {
    pub ptr: *mut mgp_edges_iterator,
    pub is_first: bool,
    pub context: Memgraph,
}

impl Default for EdgesIterator {
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

impl Drop for EdgesIterator {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_edges_iterator_destroy(self.ptr);
            }
        }
    }
}

impl Iterator for EdgesIterator {
    type Item = Edge;

    fn next(&mut self) -> Option<Edge> {
        unsafe {
            let data: *const mgp_edge;
            if self.is_first {
                self.is_first = false;
                data = ffi::mgp_edges_iterator_get(self.ptr);
            } else {
                data = ffi::mgp_edges_iterator_next(self.ptr);
            }

            if data.is_null() {
                None
            } else {
                let copy_ptr = ffi::mgp_edge_copy(data, self.context.memory());
                if copy_ptr.is_null() {
                    panic!("Unable to allocate new vertex during vertex iteration.");
                }
                Some(Edge {
                    ptr: copy_ptr,
                    context: self.context.clone(),
                })
            }
        }
    }
}

#[derive(Debug)]
pub struct Edge {
    pub ptr: *mut mgp_edge,
    pub context: Memgraph,
}

impl Drop for Edge {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_edge_destroy(self.ptr);
            }
        }
    }
}

impl Edge {
    pub fn id(&self) -> i64 {
        unsafe { ffi::mgp_edge_get_id(self.ptr).as_int }
    }

    // TODO(gitbuda): Figure out the correct lifetime. CString should be used.
    pub fn edge_type(&self) -> MgpResult<&CStr> {
        unsafe {
            let mgp_edge_type = ffi::mgp_edge_get_type(self.ptr);
            Ok(CStr::from_ptr(mgp_edge_type.name))
        }
    }

    pub fn from_vertex(&self) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_edge_get_from(self.ptr);
            assert!(!mgp_vertex.is_null(), "Unable to get from vertex.");
            let mgp_vertex_copy = ffi::mgp_vertex_copy(mgp_vertex, self.context.memory());
            if mgp_vertex_copy.is_null() {
                return Err(MgpError::MgpCreationOfVertexError);
            }
            Ok(Vertex {
                ptr: mgp_vertex_copy,
                context: self.context.clone(),
            })
        }
    }

    pub fn to_vertex(&self) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_edge_get_to(self.ptr);
            assert!(!mgp_vertex.is_null(), "Unable to get to vertex.");
            let mgp_vertex_copy = ffi::mgp_vertex_copy(mgp_vertex, self.context.memory());
            if mgp_vertex_copy.is_null() {
                return Err(MgpError::MgpCreationOfVertexError);
            }
            Ok(Vertex {
                ptr: mgp_vertex_copy,
                context: self.context.clone(),
            })
        }
    }

    pub fn property(&self, name: &CStr) -> MgpResult<Property> {
        unsafe {
            let mgp_value = MgpValue {
                ptr: ffi::mgp_edge_get_property(self.ptr, name.as_ptr(), self.context.memory()),
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
        let unable_alloc_iter_msg = c_str!("Unable to allocate properties iterator on edge.");
        unsafe {
            let mgp_iterator = ffi::mgp_edge_iter_properties(self.ptr, self.context.memory());
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
}

#[cfg(test)]
mod tests;
