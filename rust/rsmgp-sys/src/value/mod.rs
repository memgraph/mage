use c_str_macro::c_str;
use std::ffi::CStr;

use crate::mgp::*;
use crate::result::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct MgpValue {
    // It's not wise to create a new MgpValue out of the existing value pointer because drop with a
    // valid pointer will be called multiple times -> double free problem.
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

impl MgpValue {
    pub fn is_null(&self) -> bool {
        unsafe { ffi::mgp_value_is_null(self.value) != 0 }
    }

    pub fn is_int(&self) -> bool {
        unsafe { ffi::mgp_value_is_int(self.value) != 0 }
    }

    pub fn is_bool(&self) -> bool {
        unsafe { ffi::mgp_value_is_bool(self.value) != 0 }
    }

    pub fn is_string(&self) -> bool {
        unsafe { ffi::mgp_value_is_string(self.value) != 0 }
    }

    pub fn is_double(&self) -> bool {
        unsafe { ffi::mgp_value_is_double(self.value) != 0 }
    }
}

// TODO(gitbuda): Unify MgpValue and MgpConstValue.
//
// mgp_value used to return data is non-const, owned by the module code. Function to delete
// mgp_value is non-const.
// mgp_value containing data from Memgraph is const.
//
// `make` functions return non-const value that has to be deleted.
// `get_property` functions return non-const copied value that has to be deleted.
// `mgp_property` holds *const mgp_value.
//
// Possible solutions:
//   * An enum containing *mut and *const can work but the implementation would also contain
//   duplicated code.
//   * A generic data type seems complex to implement https://stackoverflow.com/questions/40317860.
//   * Holding a *const mgp_value + an ownership flag + convert to *mut when delete function has to
//   be called.
//   * Hold only *mut mgp_value... as soon as there is *const mgp_value, make a copy (own *mut
//   mgp_value). The same applies for all other data types, e.g. mgp_edge, mgp_vertex.
//   mgp_value_make_vertex accepts *mut mgp_vartex (required to return user data), but data from
//   the graph is all *const T.

#[derive(Debug)]
pub struct MgpConstValue {
    pub value: *const mgp_value,
}

impl MgpConstValue {
    pub fn is_null(&self) -> bool {
        unsafe { ffi::mgp_value_is_null(self.value) != 0 }
    }

    pub fn is_int(&self) -> bool {
        unsafe { ffi::mgp_value_is_int(self.value) != 0 }
    }

    pub fn int(&self) -> Option<i64> {
        if self.is_int() {
            unsafe {
                return Some(ffi::mgp_value_get_int(self.value));
            }
        }
        None
    }

    pub fn is_bool(&self) -> bool {
        unsafe { ffi::mgp_value_is_bool(self.value) != 0 }
    }

    pub fn bool(&self) -> Option<bool> {
        if self.is_bool() {
            unsafe {
                return Some(ffi::mgp_value_get_bool(self.value) != 0);
            }
        }
        None
    }

    pub fn is_string(&self) -> bool {
        unsafe { ffi::mgp_value_is_string(self.value) != 0 }
    }

    pub fn string(&self) -> Option<&CStr> {
        if self.is_string() {
            unsafe {
                return Some(CStr::from_ptr(ffi::mgp_value_get_string(self.value)));
            }
        }
        None
    }

    pub fn is_double(&self) -> bool {
        unsafe { ffi::mgp_value_is_double(self.value) != 0 }
    }

    pub fn double(&self) -> Option<f64> {
        if self.is_double() {
            unsafe {
                return Some(ffi::mgp_value_get_double(self.value));
            }
        }
        None
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_null_value(result: *mut mgp_result, memory: *mut mgp_memory) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate null.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: ffi::mgp_value_make_null(memory),
        };
        if mg_value.value.is_null() {
            ffi::mgp_result_set_error_msg(result, unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mg_value)
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_bool_value(
    value: bool,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate bool.");
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
