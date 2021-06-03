use crate::context::*;
use crate::edge::*;
use crate::mgp::*;
use crate::result::*;
use crate::vertex::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct Path {
    ptr: *mut mgp_path,
    context: Memgraph,
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
    pub fn new(ptr: *mut mgp_path, context: &Memgraph) -> Path {
        Path {
            ptr,
            context: context.clone(),
        }
    }

    pub(crate) unsafe fn mgp_copy(
        mgp_path: *const mgp_path,
        context: &Memgraph,
    ) -> MgpResult<Path> {
        // Test passes null ptr because nothing else is possible.
        #[cfg(not(test))]
        assert!(
            !mgp_path.is_null(),
            "Unable to make path copy because path pointer is null."
        );

        let mgp_copy = ffi::mgp_path_copy(mgp_path, context.memory());
        if mgp_copy.is_null() {
            return Err(MgpError::UnableToMakePathCopy);
        }
        Ok(Path::new(mgp_copy, &context))
    }

    pub fn mgp_ptr(&self) -> *const mgp_path {
        self.ptr
    }

    pub fn size(&self) -> u64 {
        unsafe { ffi::mgp_path_size(self.ptr) }
    }

    pub fn make_with_start(vertex: &Vertex, context: &Memgraph) -> MgpResult<Path> {
        unsafe {
            let mgp_path = ffi::mgp_path_make_with_start(vertex.mgp_ptr(), context.memory());
            if mgp_path.is_null() {
                return Err(MgpError::UnableToCreatePathWithStartVertex);
            }
            Ok(Path::new(mgp_path, &context))
        }
    }

    /// Fails if the current last vertex in the path is not part of the given edge or if there is
    /// not memory to expand the path.
    pub fn expand(&self, edge: &Edge) -> MgpResult<()> {
        unsafe {
            let mgp_result = ffi::mgp_path_expand(self.ptr, edge.mgp_ptr());
            if mgp_result == 0 {
                return Err(MgpError::UnableToExpandPath);
            }
            Ok(())
        }
    }

    pub fn vertex_at(&self, index: u64) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_path_vertex_at(self.ptr, index);
            if mgp_vertex.is_null() {
                return Err(MgpError::OutOfBoundPathVertexIndex);
            }
            Vertex::mgp_copy(mgp_vertex, &self.context)
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
