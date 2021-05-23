use c_str_macro::c_str;
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
    let record_fail_msg = c_str!("Unable to allocate record");
    unsafe {
        let record = ffi::mgp_result_new_record(context.result());
        if record.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), record_fail_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(MgpResultRecord { record })
    }
}

pub fn insert_result_record(
    mgp_record: &MgpResultRecord,
    mgp_name: &CStr,
    mgp_value: &MgpValue,
    context: &Memgraph,
) -> MgpResult<()> {
    let name_not_inserted_msg = c_str!("Unable to insert record to the result.");
    unsafe {
        let inserted =
            ffi::mgp_result_record_insert(mgp_record.record, mgp_name.as_ptr(), mgp_value.ptr);
        if inserted == 0 {
            ffi::mgp_result_set_error_msg(context.result(), name_not_inserted_msg.as_ptr());
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

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_string_result_type(
    procedure: *mut mgp_proc,
    name: &CStr,
) -> Result<(), MgpAddProcedureParameterTypeError> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_string()) == 0 {
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
