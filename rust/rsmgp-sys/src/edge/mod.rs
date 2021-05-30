use std::ffi::{CStr, CString};

use crate::context::Memgraph;
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
    ptr: *mut mgp_edges_iterator,
    is_first: bool,
    context: Memgraph,
}

impl EdgesIterator {
    pub fn new(ptr: *mut mgp_edges_iterator, context: &Memgraph) -> EdgesIterator {
        EdgesIterator {
            ptr,
            is_first: true,
            context: context.clone(),
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
    ptr: *mut mgp_edge,
    context: Memgraph,
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
    pub fn new(ptr: *mut mgp_edge, context: &Memgraph) -> Edge {
        Edge {
            ptr,
            context: context.clone(),
        }
    }

    pub(crate) unsafe fn mgp_copy(ptr: *const mgp_edge, context: &Memgraph) -> MgpResult<Edge> {
        // Test passes null ptr because nothing else is possible.
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to make edge copy because edge is null."
        );
        let mgp_copy = ffi::mgp_edge_copy(ptr, context.memory());
        if mgp_copy.is_null() {
            return Err(MgpError::UnableToAllocateEdgeValue);
        }
        Ok(Edge::new(mgp_copy, &context))
    }

    pub fn mgp_ptr(&self) -> *const mgp_edge {
        self.ptr
    }

    pub fn copy(&self) -> MgpResult<Edge> {
        unsafe { Edge::mgp_copy(self.ptr, &self.context) }
    }

    pub fn id(&self) -> i64 {
        unsafe { ffi::mgp_edge_get_id(self.ptr).as_int }
    }

    pub fn edge_type(&self) -> MgpResult<CString> {
        unsafe {
            let mgp_edge_type = ffi::mgp_edge_get_type(self.ptr);
            create_cstring(mgp_edge_type.name)
        }
    }

    pub fn from_vertex(&self) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_edge_get_from(self.ptr);
            let vertex = make_vertex_copy(mgp_vertex, &self.context)?;
            Ok(vertex)
        }
    }

    pub fn to_vertex(&self) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_edge_get_to(self.ptr);
            let vertex = make_vertex_copy(mgp_vertex, &self.context)?;
            Ok(vertex)
        }
    }

    pub fn property(&self, name: &CStr) -> MgpResult<Property> {
        unsafe {
            let mgp_value =
                ffi::mgp_edge_get_property(self.ptr, name.as_ptr(), self.context.memory());
            if mgp_value.is_null() {
                return Err(MgpError::UnableToReturnEdgePropertyValueAllocationError);
            }
            let value = match mgp_value_to_value(&MgpValue { ptr: mgp_value }, &self.context) {
                Ok(v) => v,
                Err(_) => return Err(MgpError::UnableToReturnEdgePropertyValueCreationError),
            };
            match CString::new(name.to_bytes()) {
                Ok(c_string) => Ok(Property {
                    name: c_string,
                    value,
                }),
                Err(_) => Err(MgpError::UnableToReturnEdgePropertyNameAllocationError),
            }
        }
    }

    pub fn properties(&self) -> MgpResult<PropertiesIterator> {
        unsafe {
            let mgp_iterator = ffi::mgp_edge_iter_properties(self.ptr, self.context.memory());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToReturnEdgePropertiesIterator);
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
