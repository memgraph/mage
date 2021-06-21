// Copyright (c) 2016-2021 Memgraph Ltd. [https://memgraph.com]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//! All vertex (node) related.

use std::ffi::{CStr, CString};

use crate::edge::*;
use crate::memgraph::*;
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
    memgraph: Memgraph,
}

impl VerticesIterator {
    pub(crate) fn new(ptr: *mut mgp_vertices_iterator, memgraph: &Memgraph) -> VerticesIterator {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create vertices iterator because the given pointer is null."
        );

        VerticesIterator {
            ptr,
            is_first: true,
            memgraph: memgraph.clone(),
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
            let data = if self.is_first {
                self.is_first = false;
                ffi::mgp_vertices_iterator_get(self.ptr)
            } else {
                ffi::mgp_vertices_iterator_next(self.ptr)
            };

            if data.is_null() {
                None
            } else {
                Some(match Vertex::mgp_copy(data, &self.memgraph) {
                    Ok(v) => v,
                    Err(_) => panic!("Unable to create new vertex during vertices iteration."),
                })
            }
        }
    }
}

pub struct Vertex {
    ptr: *mut mgp_vertex,
    memgraph: Memgraph,
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
    pub(crate) fn new(ptr: *mut mgp_vertex, memgraph: &Memgraph) -> Vertex {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create vertex because the given pointer is null."
        );

        Vertex {
            ptr,
            memgraph: memgraph.clone(),
        }
    }

    /// Creates a new Vertex based on [mgp_vertex].
    pub(crate) unsafe fn mgp_copy(
        mgp_vertex: *const mgp_vertex,
        memgraph: &Memgraph,
    ) -> MgpResult<Vertex> {
        #[cfg(not(test))]
        assert!(
            !mgp_vertex.is_null(),
            "Unable to make vertex copy because vertex pointer is null."
        );

        let mgp_copy = ffi::mgp_vertex_copy(mgp_vertex, memgraph.memory_ptr());
        if mgp_copy.is_null() {
            return Err(MgpError::UnableToCopyVertex);
        }
        Ok(Vertex::new(mgp_copy, &memgraph))
    }

    pub(crate) fn mgp_ptr(&self) -> *const mgp_vertex {
        self.ptr
    }

    pub fn id(&self) -> i64 {
        unsafe { ffi::mgp_vertex_get_id(self.ptr).as_int }
    }

    pub fn labels_count(&self) -> u64 {
        unsafe { ffi::mgp_vertex_labels_count(self.ptr) }
    }

    pub fn label_at(&self, index: u64) -> MgpResult<CString> {
        unsafe {
            let c_label = ffi::mgp_vertex_label_at(self.ptr, index);
            if c_label.name.is_null() {
                return Err(MgpError::OutOfBoundLabelIndexError);
            }
            create_cstring(c_label.name)
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
            let mgp_value =
                ffi::mgp_vertex_get_property(self.ptr, name.as_ptr(), self.memgraph.memory_ptr());
            if mgp_value.is_null() {
                return Err(MgpError::UnableToGetVertexProperty);
            }
            let value = MgpValue::new(mgp_value, &self.memgraph).to_value()?;
            match CString::new(name.to_bytes()) {
                Ok(c_string) => Ok(Property {
                    name: c_string,
                    value,
                }),
                Err(_) => Err(MgpError::UnableToReturnVertexPropertyMakeNameEror),
            }
        }
    }

    pub fn properties(&self) -> MgpResult<PropertiesIterator> {
        unsafe {
            let mgp_iterator =
                ffi::mgp_vertex_iter_properties(self.ptr, self.memgraph.memory_ptr());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToReturnVertexPropertiesIterator);
            }
            Ok(PropertiesIterator::new(mgp_iterator, &self.memgraph))
        }
    }

    pub fn in_edges(&self) -> MgpResult<EdgesIterator> {
        unsafe {
            let mgp_iterator = ffi::mgp_vertex_iter_in_edges(self.ptr, self.memgraph.memory_ptr());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToReturnVertexInEdgesIterator);
            }
            Ok(EdgesIterator::new(mgp_iterator, &self.memgraph))
        }
    }

    pub fn out_edges(&self) -> MgpResult<EdgesIterator> {
        unsafe {
            let mgp_iterator = ffi::mgp_vertex_iter_out_edges(self.ptr, self.memgraph.memory_ptr());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToReturnVertexOutEdgesIterator);
            }
            Ok(EdgesIterator::new(mgp_iterator, &self.memgraph))
        }
    }
}

#[cfg(test)]
mod tests;
