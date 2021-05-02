use c_str_macro::c_str;
use std::ffi::{CStr, CString};

// Import all structs, functions are in the mgp::ffi module.
use crate::mgp::*;
use crate::result::*;
use crate::value::*;
// All mgp_ functions are mocked.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

pub struct MgpVerticesIterator {
    ptr: *mut mgp_vertices_iterator,
    is_first: bool,
}
impl Default for MgpVerticesIterator {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            is_first: true,
        }
    }
}
impl Drop for MgpVerticesIterator {
    fn drop(&mut self) {
        unsafe {
            ffi::mgp_vertices_iterator_destroy(self.ptr);
        }
    }
}
impl Iterator for MgpVerticesIterator {
    type Item = MgpVertex;
    fn next(&mut self) -> Option<MgpVertex> {
        if self.is_first {
            self.is_first = false;
            unsafe {
                let data = ffi::mgp_vertices_iterator_get(self.ptr);
                if data.is_null() {
                    None
                } else {
                    Some(MgpVertex { ptr: data })
                }
            }
        } else {
            unsafe {
                let data = ffi::mgp_vertices_iterator_next(self.ptr);
                if data.is_null() {
                    None
                } else {
                    Some(MgpVertex { ptr: data })
                }
            }
        }
    }
}

pub struct MgpVertex {
    ptr: *const mgp_vertex,
}
impl MgpVertex {
    pub fn labels_count(&self) -> u64 {
        unsafe {
            // TODO(gitbuda): Figure out why this is not clippy::not_unsafe_ptr_arg_deref.
            ffi::mgp_vertex_labels_count(self.ptr)
        }
    }

    pub fn has_label(&self, name: &str) -> bool {
        let c_str = CString::new(name).unwrap();
        unsafe {
            let c_mgp_label = mgp_label {
                name: c_str.as_ptr(),
            };
            ffi::mgp_vertex_has_label(self.ptr, c_mgp_label) != 0
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_graph_vertices_iterator(
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> MgpResult<MgpVerticesIterator> {
    let unable_alloc_iter_msg = c_str!("Unable to allocate a vertices iterator.");
    unsafe {
        let iterator: MgpVerticesIterator = MgpVerticesIterator {
            ptr: ffi::mgp_graph_iter_vertices(graph, memory),
            ..Default::default()
        };
        if iterator.ptr.is_null() {
            ffi::mgp_result_set_error_msg(result, unable_alloc_iter_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(iterator)
    }
}

pub struct MgpResultRecord {
    record: *mut mgp_result_record,
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_result_record(result: *mut mgp_result) -> MgpResult<MgpResultRecord> {
    let record_fail_msg = c_str!("Unable to allocate record");
    unsafe {
        let record = ffi::mgp_result_new_record(result);
        if record.is_null() {
            ffi::mgp_result_set_error_msg(result, record_fail_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(MgpResultRecord { record })
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn insert_result_record(
    mgp_record: &MgpResultRecord,
    mgp_name: &CStr,
    mgp_value: &MgpValue,
    result: *mut mgp_result,
) -> MgpResult<()> {
    let name_not_inserted_msg = c_str!("Unable to insert record to the result.");
    unsafe {
        let inserted =
            ffi::mgp_result_record_insert(mgp_record.record, mgp_name.as_ptr(), mgp_value.value);
        if inserted == 0 {
            ffi::mgp_result_set_error_msg(result, name_not_inserted_msg.as_ptr());
            return Err(MgpError::MgpPreparingResultError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_read_procedure(
    proc_ptr: extern "C" fn(*const mgp_list, *const mgp_graph, *mut mgp_result, *mut mgp_memory),
    name: &CStr,
    module: *mut mgp_module,
) -> *mut mgp_proc {
    unsafe { ffi::mgp_module_add_read_procedure(module, name.as_ptr(), Some(proc_ptr)) }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_int_result_type(
    procedure: *mut mgp_proc,
    name: &CStr,
) -> Result<(), MgpAddProcedureParameterTypeError> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_int()) == 0 {
            return Err(MgpAddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_bool_result_type(
    procedure: *mut mgp_proc,
    name: &CStr,
) -> Result<(), MgpAddProcedureParameterTypeError> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_bool()) == 0 {
            return Err(MgpAddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mgp::mock_ffi::*;
}
