use crate::context::*;
use crate::edge::*;
use crate::mgp::*;
use crate::result::*;
use crate::value::*;
use crate::vertex::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct Path {
    pub ptr: *mut mgp_path,
    pub context: Memgraph,
}

impl Drop for Path {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_path_destroy(self.ptr);
            }
        }
    }
}

impl Path {
    pub fn size(&self) -> u64 {
        unsafe { ffi::mgp_path_size(self.ptr) }
    }

    pub fn vertex_at(&self, index: u64) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_path_vertex_at(self.ptr, index);
            if mgp_vertex.is_null() {
                return Err(MgpError::OutOfBoundPathVertexIndex);
            }
            make_vertex_copy(mgp_vertex, &self.context)
        }
    }

    pub fn edge_at(&self, index: u64) -> MgpResult<Edge> {
        unsafe {
            let mgp_edge = ffi::mgp_path_edge_at(self.ptr, index);
            if mgp_edge.is_null() {
                return Err(MgpError::OutOfBoundPathEdgeIndex);
            }
            Edge::mgp_copy(mgp_edge, &self.context)
        }
    }
}

#[cfg(test)]
mod tests;
