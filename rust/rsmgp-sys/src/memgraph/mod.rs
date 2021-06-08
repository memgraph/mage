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

/// Main object to interact with Memgraph instance.
#[derive(Debug, Clone)]
pub struct Memgraph {
    memgraph: Rc<MgpMemgraph>,
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
            memgraph: Rc::new(MgpMemgraph {
                args,
                graph,
                result,
                memory,
                module,
            }),
        }
    }

    /// Creates a new object with all underlying data set to null. Used for the testing purposes.
    pub fn new_default() -> Memgraph {
        Memgraph {
            memgraph: Rc::new(MgpMemgraph {
                args: std::ptr::null(),
                graph: std::ptr::null(),
                result: std::ptr::null_mut(),
                memory: std::ptr::null_mut(),
                module: std::ptr::null_mut(),
            }),
        }
    }

    // TODO(gitbuda): Implement args abstraction.

    /// Returns pointer to the object with all arguments passed to the procedure call.
    pub fn args(&self) -> *const mgp_list {
        self.memgraph.args
    }

    /// Returns pointer to the object with graph data.
    pub fn graph(&self) -> *const mgp_graph {
        self.memgraph.graph
    }

    /// Returns pointer to the object where results could be stored.
    pub fn result(&self) -> *mut mgp_result {
        self.memgraph.result
    }

    /// Returns pointer to the memory object for advanced memory control.
    pub fn memory(&self) -> *mut mgp_memory {
        self.memgraph.memory
    }

    /// Returns pointer to the module object.
    pub fn module(&self) -> *mut mgp_module {
        self.memgraph.module
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

    /// Creates a new result record.
    ///
    /// Keep this object on the stack and add data that will be returned to Memgraph / client
    /// during/after the procedure call.
    pub fn result_record(&self) -> MgpResult<MgpResultRecord> {
        MgpResultRecord::new(self)
    }

    /// Registers a new read procedure.
    ///
    /// * `proc_ptr` - Identifier of the top level C function that represents the procedure.
    /// * `name` - A string that will be registered as a procedure name inside Memgraph instance.
    /// * `result_fields` - An array of all [ResultFieldType]s, each one define by [FieldType].
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
