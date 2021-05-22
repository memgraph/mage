use crate::mgp::*;
use std::rc::Rc;

// TODO(gitbuda): Memgraph can't be copied implicitly, consider borrow.
// TODO(gitbuda): Explain why MgpMemgraph and Memgraph with Rc.

#[derive(Debug)]
pub struct MgpMemgraph {
    pub args: *const mgp_list,
    pub graph: *const mgp_graph,
    pub result: *mut mgp_result,
    pub memory: *mut mgp_memory,
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
}
