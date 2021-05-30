use std::rc::Rc;

use crate::mgp::*;
use crate::result::*;
use crate::vertex::*;
// Required here, if not present, tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct MgpMemgraph {
    args: *const mgp_list,
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
}

impl Default for MgpMemgraph {
    fn default() -> Self {
        Self {
            args: std::ptr::null(),
            graph: std::ptr::null(),
            result: std::ptr::null_mut(),
            memory: std::ptr::null_mut(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Memgraph {
    pub context: Rc<MgpMemgraph>,
}

impl Default for Memgraph {
    fn default() -> Self {
        Self {
            context: Rc::new(MgpMemgraph {
                ..Default::default()
            }),
        }
    }
}

impl Memgraph {
    pub fn new(
        args: *const mgp_list,
        graph: *const mgp_graph,
        result: *mut mgp_result,
        memory: *mut mgp_memory,
    ) -> Memgraph {
        Memgraph {
            context: Rc::new(MgpMemgraph {
                args,
                graph,
                result,
                memory,
            }),
        }
    }

    pub fn args(&self) -> *const mgp_list {
        self.context.args
    }

    pub fn graph(&self) -> *const mgp_graph {
        self.context.graph
    }

    pub fn result(&self) -> *mut mgp_result {
        self.context.result
    }

    pub fn memory(&self) -> *mut mgp_memory {
        self.context.memory
    }

    pub fn vertices_iter(&self) -> MgpResult<VerticesIterator> {
        unsafe {
            let mgp_iterator = ffi::mgp_graph_iter_vertices(self.graph(), self.memory());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToCreateGraphVerticesIterator);
            }
            Ok(VerticesIterator {
                ptr: mgp_iterator,
                context: self.clone(),
                ..Default::default()
            })
        }
    }

    pub fn vertex_by_id(&self, id: i64) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_graph_get_vertex_by_id(
                self.graph(),
                mgp_vertex_id { as_int: id },
                self.memory(),
            );
            if mgp_vertex.is_null() {
                return Err(MgpError::UnableToFindVertexById);
            }
            Ok(Vertex {
                ptr: mgp_vertex,
                context: self.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests;
