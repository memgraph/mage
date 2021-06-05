use std::ffi::CStr;
use std::rc::Rc;

use crate::mgp::*;
use crate::result::*;
use crate::rsmgp::*;
use crate::vertex::*;
// Required here, if not present, tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
struct MgpMemgraph {
    args: *const mgp_list,
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
    module: *mut mgp_module,
}

#[derive(Debug, Clone)]
pub struct Memgraph {
    context: Rc<MgpMemgraph>,
}

impl Memgraph {
    pub fn new(
        args: *const mgp_list,
        graph: *const mgp_graph,
        result: *mut mgp_result,
        memory: *mut mgp_memory,
        module: *mut mgp_module,
    ) -> Memgraph {
        Memgraph {
            context: Rc::new(MgpMemgraph {
                args,
                graph,
                result,
                memory,
                module,
            }),
        }
    }

    pub fn new_default() -> Memgraph {
        Memgraph {
            context: Rc::new(MgpMemgraph {
                args: std::ptr::null(),
                graph: std::ptr::null(),
                result: std::ptr::null_mut(),
                memory: std::ptr::null_mut(),
                module: std::ptr::null_mut(),
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

    pub fn module(&self) -> *mut mgp_module {
        self.context.module
    }

    pub fn vertices_iter(&self) -> MgpResult<VerticesIterator> {
        unsafe {
            let mgp_iterator = ffi::mgp_graph_iter_vertices(self.graph(), self.memory());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToCreateGraphVerticesIterator);
            }
            Ok(VerticesIterator::new(mgp_iterator, &self))
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
            Ok(Vertex::new(mgp_vertex, &self))
        }
    }

    pub fn result_record(&self) -> MgpResult<MgpResultRecord> {
        MgpResultRecord::new(&self)
    }

    pub fn add_read_procedure(
        &self,
        proc_ptr: extern "C" fn(
            *const mgp_list,
            *const mgp_graph,
            *mut mgp_result,
            *mut mgp_memory,
        ),
        name: &CStr,
        result_fields: &[ResultFieldType],
    ) -> MgpResult<()> {
        unsafe {
            let procedure =
                ffi::mgp_module_add_read_procedure(self.module(), name.as_ptr(), Some(proc_ptr));
            if procedure.is_null() {
                return Err(MgpError::UnableToRegisterReadProcedure);
            }
            for result_field in result_fields {
                let mut mgp_type: *const mgp_type = std::ptr::null_mut();
                for field_type in result_field.types.iter().rev() {
                    mgp_type = match field_type {
                        FieldType::Any => ffi::mgp_type_any(),
                        FieldType::Bool => ffi::mgp_type_bool(),
                        FieldType::Number => ffi::mgp_type_number(),
                        FieldType::Int => ffi::mgp_type_int(),
                        FieldType::Double => ffi::mgp_type_float(),
                        FieldType::String => ffi::mgp_type_string(),
                        FieldType::Map => ffi::mgp_type_map(),
                        FieldType::Vertex => ffi::mgp_type_node(),
                        FieldType::Edge => ffi::mgp_type_relationship(),
                        FieldType::Path => ffi::mgp_type_path(),
                        FieldType::Nullable => ffi::mgp_type_nullable(mgp_type),
                        FieldType::List => ffi::mgp_type_list(mgp_type),
                    };
                }
                if ffi::mgp_proc_add_result(procedure, result_field.name.as_ptr(), mgp_type) == 0 {
                    return Err(MgpError::AddProcedureParameterTypeError);
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
