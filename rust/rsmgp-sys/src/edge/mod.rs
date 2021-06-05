use std::ffi::{CStr, CString};

use crate::memgraph::Memgraph;
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
    memgraph: Memgraph,
}

impl EdgesIterator {
    pub fn new(ptr: *mut mgp_edges_iterator, memgraph: &Memgraph) -> EdgesIterator {
        EdgesIterator {
            ptr,
            is_first: true,
            memgraph: memgraph.clone(),
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
                Some(match Edge::mgp_copy(data, &self.memgraph) {
                    Ok(v) => v,
                    Err(_) => panic!("Unable to create new edge during edges iteration."),
                })
            }
        }
    }
}

#[derive(Debug)]
pub struct Edge {
    ptr: *mut mgp_edge,
    memgraph: Memgraph,
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
    pub fn new(ptr: *mut mgp_edge, memgraph: &Memgraph) -> Edge {
        Edge {
            ptr,
            memgraph: memgraph.clone(),
        }
    }

    pub(crate) unsafe fn mgp_copy(ptr: *const mgp_edge, memgraph: &Memgraph) -> MgpResult<Edge> {
        // Test passes null ptr because nothing else is possible.
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to make edge copy because edge pointer is null."
        );

        let mgp_copy = ffi::mgp_edge_copy(ptr, memgraph.memory());
        if mgp_copy.is_null() {
            return Err(MgpError::UnableToMakeEdgeCopy);
        }
        Ok(Edge::new(mgp_copy, &memgraph))
    }

    pub fn mgp_ptr(&self) -> *const mgp_edge {
        self.ptr
    }

    pub fn copy(&self) -> MgpResult<Edge> {
        unsafe { Edge::mgp_copy(self.ptr, &self.memgraph) }
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
            Vertex::mgp_copy(mgp_vertex, &self.memgraph)
        }
    }

    pub fn to_vertex(&self) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_edge_get_to(self.ptr);
            Vertex::mgp_copy(mgp_vertex, &self.memgraph)
        }
    }

    pub fn property(&self, name: &CStr) -> MgpResult<Property> {
        unsafe {
            let mgp_value =
                ffi::mgp_edge_get_property(self.ptr, name.as_ptr(), self.memgraph.memory());
            if mgp_value.is_null() {
                return Err(MgpError::UnableToReturnEdgePropertyValueAllocationError);
            }
            let value = match MgpValue::to_value(&MgpValue::new(mgp_value), &self.memgraph) {
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
            let mgp_iterator = ffi::mgp_edge_iter_properties(self.ptr, self.memgraph.memory());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToReturnEdgePropertiesIterator);
            }
            Ok(PropertiesIterator::new(mgp_iterator, &self.memgraph))
        }
    }
}

#[cfg(test)]
mod tests;
