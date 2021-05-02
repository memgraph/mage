use c_str_macro::c_str;
use std::ffi::CStr;

use crate::mgp::*;
use crate::result::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

pub struct MgpValue {
    pub value: *mut mgp_value,
}
impl Drop for MgpValue {
    fn drop(&mut self) {
        unsafe {
            if !self.value.is_null() {
                ffi::mgp_value_destroy(self.value);
            }
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_bool_value(
    value: bool,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate boolean value.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: ffi::mgp_value_make_bool(if !value { 0 } else { 1 }, memory),
        };
        if mg_value.value.is_null() {
            ffi::mgp_result_set_error_msg(result, unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mg_value)
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_int_value(
    value: i64,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate integer.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: ffi::mgp_value_make_int(value, memory),
        };
        if mg_value.value.is_null() {
            ffi::mgp_result_set_error_msg(result, unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mg_value)
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_string_value(
    value: &CStr,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate string.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: ffi::mgp_value_make_string(value.as_ptr(), memory),
        };
        if mg_value.value.is_null() {
            ffi::mgp_result_set_error_msg(result, unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mg_value)
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_double_value(
    value: f64,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate double.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: ffi::mgp_value_make_double(value, memory),
        };
        if mg_value.value.is_null() {
            ffi::mgp_result_set_error_msg(result, unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mg_value)
    }
}

#[cfg(test)]
mod tests;
