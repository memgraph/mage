use std::ffi::CStr;

use crate::context::*;
#[double]
use crate::mgp::ffi;
use crate::mgp::*;
use crate::result::*;
use crate::value::*;
use mockall_double::double;

pub struct MgpResultRecord {
    record: *mut mgp_result_record,
}

pub fn make_result_record(context: &Memgraph) -> MgpResult<MgpResultRecord> {
    unsafe {
        let record = ffi::mgp_result_new_record(context.result());
        if record.is_null() {
            return Err(MgpError::UnableToCreateResultRecord);
        }
        Ok(MgpResultRecord { record })
    }
}

pub fn insert_result_record(
    mgp_record: &MgpResultRecord,
    mgp_name: &CStr,
    mgp_value: &MgpValue,
) -> MgpResult<()> {
    unsafe {
        let inserted =
            ffi::mgp_result_record_insert(mgp_record.record, mgp_name.as_ptr(), mgp_value.ptr);
        if inserted == 0 {
            return Err(MgpError::PreparingResultError);
        }
        Ok(())
    }
}

pub fn set_memgraph_error_msg(msg: &CStr, context: &Memgraph) {
    unsafe {
        let status = ffi::mgp_result_set_error_msg(context.result(), msg.as_ptr());
        if status == 0 {
            panic!("Unable to pass error message to the Memgraph engine.");
        }
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

pub fn get_type_any() -> *const mgp_type {
    unsafe { ffi::mgp_type_any() }
}

pub fn get_type_bool() -> *const mgp_type {
    unsafe { ffi::mgp_type_bool() }
}

pub fn get_type_string() -> *const mgp_type {
    unsafe { ffi::mgp_type_string() }
}

pub fn get_type_int() -> *const mgp_type {
    unsafe { ffi::mgp_type_int() }
}

pub fn get_type_float() -> *const mgp_type {
    unsafe { ffi::mgp_type_float() }
}

pub fn get_type_number() -> *const mgp_type {
    unsafe { ffi::mgp_type_number() }
}

pub fn get_type_map() -> *const mgp_type {
    unsafe { ffi::mgp_type_map() }
}

pub fn get_type_vertex() -> *const mgp_type {
    unsafe { ffi::mgp_type_node() }
}

pub fn get_type_edge() -> *const mgp_type {
    unsafe { ffi::mgp_type_relationship() }
}

pub fn get_type_path() -> *const mgp_type {
    unsafe { ffi::mgp_type_path() }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_int_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_int()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_bool_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_bool()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_string_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_string()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_double_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_float()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_vertex_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_node()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_edge_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_relationship()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_path_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_path()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_map_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_map()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_list_result_type(
    procedure: *mut mgp_proc,
    name: &CStr,
    item_type: *const mgp_type,
) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_list(item_type)) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

// TODO(gitbuda): Add nullable result type.

// TODO(gitbuda): Add transaction management (abort) stuff.

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mgp::mock_ffi::*;
}
