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
//! Abstraction to interact with Memgraph.

use std::ffi::CStr;

use crate::list::*;
use crate::mgp::*;
use crate::result::*;
use crate::rsmgp::*;
use crate::vertex::*;
// Required here, if not present, tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

fn resolve_mgp_type(types: &[Type]) -> *const mgp_type {
    unsafe {
        let mut mgp_type: *const mgp_type = std::ptr::null_mut();
        for field_type in types.iter().rev() {
            mgp_type = match field_type {
                Type::Any => ffi::mgp_type_any(),
                Type::Bool => ffi::mgp_type_bool(),
                Type::Number => ffi::mgp_type_number(),
                Type::Int => ffi::mgp_type_int(),
                Type::Double => ffi::mgp_type_float(),
                Type::String => ffi::mgp_type_string(),
                Type::Map => ffi::mgp_type_map(),
                Type::Vertex => ffi::mgp_type_node(),
                Type::Edge => ffi::mgp_type_relationship(),
                Type::Path => ffi::mgp_type_path(),
                Type::Nullable => ffi::mgp_type_nullable(mgp_type),
                Type::List => ffi::mgp_type_list(mgp_type),
            };
        }
        mgp_type
    }
}

/// Main object to interact with Memgraph instance.
#[derive(Clone)]
pub struct Memgraph {
    args: *const mgp_list,
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
    module: *mut mgp_module,
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
            args,
            graph,
            result,
            memory,
            module,
        }
    }

    /// Creates a new object with all underlying data set to null. Used for the testing purposes.
    #[cfg(test)]
    pub(crate) fn new_default() -> Memgraph {
        Memgraph {
            args: std::ptr::null(),
            graph: std::ptr::null(),
            result: std::ptr::null_mut(),
            memory: std::ptr::null_mut(),
            module: std::ptr::null_mut(),
        }
    }

    pub fn args(&self) -> MgpResult<List> {
        // TODO(gitbuda): Avoid list copy when accessing args.
        unsafe { List::mgp_copy(self.args_ptr(), &self) }
    }

    /// Returns pointer to the object with all arguments passed to the procedure call.
    pub(crate) fn args_ptr(&self) -> *const mgp_list {
        self.args
    }

    /// Returns pointer to the object with graph data.
    pub(crate) fn graph_ptr(&self) -> *const mgp_graph {
        self.graph
    }

    /// Returns pointer to the object where results could be stored.
    pub(crate) fn result_ptr(&self) -> *mut mgp_result {
        self.result
    }

    /// Returns pointer to the memory object for advanced memory control.
    pub(crate) fn memory_ptr(&self) -> *mut mgp_memory {
        self.memory
    }

    /// Returns pointer to the module object.
    pub fn module_ptr(&self) -> *mut mgp_module {
        self.module
    }

    pub fn vertices_iter(&self) -> MgpResult<VerticesIterator> {
        unsafe {
            let mgp_iterator = ffi::mgp_graph_iter_vertices(self.graph_ptr(), self.memory_ptr());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToCreateGraphVerticesIterator);
            }
            Ok(VerticesIterator::new(mgp_iterator, &self))
        }
    }

    pub fn vertex_by_id(&self, id: i64) -> MgpResult<Vertex> {
        unsafe {
            let mgp_vertex = ffi::mgp_graph_get_vertex_by_id(
                self.graph_ptr(),
                mgp_vertex_id { as_int: id },
                self.memory_ptr(),
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
        MgpResultRecord::create(self)
    }

    /// Registers a new read procedure.
    ///
    /// * `proc_ptr` - Identifier of the top level C function that represents the procedure.
    /// * `name` - A string that will be registered as a procedure name inside Memgraph instance.
    /// * `required_input_types` - An array of all [NamedType]s, each one define by name and an array of [Type]s.
    /// * `optional_input_types` - An array of all [OptionalNamedType]s, each one define by name, an array of [Type]s, and default value.
    /// * `result_field_types` - An array of all [NamedType]s, each one define by name and an array of [Type]s.
    pub fn add_read_procedure(
        &self,
        proc_ptr: extern "C" fn(
            *const mgp_list,
            *const mgp_graph,
            *mut mgp_result,
            *mut mgp_memory,
        ),
        name: &CStr,
        required_input_types: &[NamedType],
        optional_input_types: &[OptionalNamedType],
        result_field_types: &[NamedType],
    ) -> MgpResult<()> {
        unsafe {
            let procedure = ffi::mgp_module_add_read_procedure(
                self.module_ptr(),
                name.as_ptr(),
                Some(proc_ptr),
            );
            if procedure.is_null() {
                return Err(MgpError::UnableToRegisterReadProcedure);
            }

            for required_type in required_input_types {
                let mgp_type = resolve_mgp_type(&required_type.types);
                if ffi::mgp_proc_add_arg(procedure, required_type.name.as_ptr(), mgp_type) == 0 {
                    return Err(MgpError::AddProcedureRequiredInputParameterError);
                }
            }

            for optional_input in optional_input_types {
                let mgp_type = resolve_mgp_type(&optional_input.types);
                if ffi::mgp_proc_add_opt_arg(
                    procedure,
                    optional_input.name.as_ptr(),
                    mgp_type,
                    optional_input.default.mgp_ptr(),
                ) == 0
                {
                    return Err(MgpError::AddProcedureOptionalInputParameterError);
                }
            }

            for result_field in result_field_types {
                let mgp_type = resolve_mgp_type(&result_field.types);
                if result_field.deprecated {
                    if ffi::mgp_proc_add_deprecated_result(
                        procedure,
                        result_field.name.as_ptr(),
                        mgp_type,
                    ) == 0
                    {
                        return Err(MgpError::AddProcedureParameterTypeError);
                    }
                } else if ffi::mgp_proc_add_result(procedure, result_field.name.as_ptr(), mgp_type)
                    == 0
                {
                    return Err(MgpError::AddProcedureParameterTypeError);
                }
            }
            Ok(())
        }
    }

    /// Return `true` if the currently executing procedure should abort as soon as possible.
    pub fn must_abort(&self) -> bool {
        unsafe { ffi::mgp_must_abort(self.graph_ptr()) != 0 }
    }
}

#[cfg(test)]
mod tests;
