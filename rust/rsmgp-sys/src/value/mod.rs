use c_str_macro::c_str;
use std::ffi::CStr;

use crate::context::*;
use crate::mgp::*;
use crate::result::*;
use crate::vertex::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

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
pub struct MgpValue {
    // It's not wise to create a new Vertex out of the existing value pointer because drop with a
    // valid pointer will be called multiple times -> double free problem.
    pub ptr: *mut mgp_value,
}
impl Drop for MgpValue {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_value_destroy(self.ptr);
            }
        }
    }
}

#[derive(Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Vertex(Vertex),
}
impl Value {
    // TODO(gitbuda): Remove to_result_mgp_value dead code.
    #[allow(dead_code)]
    fn to_result_mgp_value(&self, context: &Memgraph) -> MgpResult<MgpValue> {
        match self {
            Value::Null => Ok(make_null_value(context)?),
            Value::Bool(x) => Ok(make_bool_value(*x, context)?),
            Value::Int(x) => Ok(make_int_value(*x, context)?),
            // TODO(gitbuda): Implement float and vertex conversion.
            Value::Float(_) => Ok(make_null_value(context)?),
            Value::Vertex(_) => Ok(make_null_value(context)?),
        }
    }
}

/// # Safety
/// TODO(gitbuda): Write section about safety.
pub unsafe fn mgp_raw_value_to_value(
    value: *const mgp_value,
    context: &Memgraph,
) -> MgpResult<Value> {
    #[allow(non_upper_case_globals)]
    match ffi::mgp_value_get_type(value) {
        mgp_value_type_MGP_VALUE_TYPE_NULL => Ok(Value::Null),
        mgp_value_type_MGP_VALUE_TYPE_BOOL => Ok(Value::Bool(ffi::mgp_value_get_bool(value) == 0)),
        mgp_value_type_MGP_VALUE_TYPE_INT => Ok(Value::Int(ffi::mgp_value_get_int(value))),
        mgp_value_type_MGP_VALUE_TYPE_DOUBLE => Ok(Value::Float(ffi::mgp_value_get_double(value))),
        mgp_value_type_MGP_VALUE_TYPE_VERTEX => {
            let vertex = ffi::mgp_value_get_vertex(value);
            // TODO(gitbuda): Handle error.
            Ok(Value::Vertex(Vertex {
                ptr: ffi::mgp_vertex_copy(vertex, context.memory()),
                context: context.clone(),
            }))
        }
        _ => {
            panic!()
        } // TODO(gitbuda): Handle mgp_value_type unhandeled values.
    }
}

pub fn mgp_value_to_value(value: &MgpValue, context: &Memgraph) -> MgpResult<Value> {
    unsafe { mgp_raw_value_to_value(value.ptr, context) }
}

impl MgpValue {
    pub fn is_null(&self) -> bool {
        unsafe { ffi::mgp_value_is_null(self.ptr) != 0 }
    }

    pub fn is_int(&self) -> bool {
        unsafe { ffi::mgp_value_is_int(self.ptr) != 0 }
    }

    pub fn is_bool(&self) -> bool {
        unsafe { ffi::mgp_value_is_bool(self.ptr) != 0 }
    }

    pub fn is_string(&self) -> bool {
        unsafe { ffi::mgp_value_is_string(self.ptr) != 0 }
    }

    pub fn is_double(&self) -> bool {
        unsafe { ffi::mgp_value_is_double(self.ptr) != 0 }
    }
}

pub fn make_null_value(context: &Memgraph) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate null.");
    unsafe {
        let mgp_value = MgpValue {
            ptr: ffi::mgp_value_make_null(context.memory()),
        };
        if mgp_value.ptr.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mgp_value)
    }
}

pub fn make_bool_value(value: bool, context: &Memgraph) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate bool.");
    unsafe {
        let mgp_value = MgpValue {
            ptr: ffi::mgp_value_make_bool(if !value { 0 } else { 1 }, context.memory()),
        };
        if mgp_value.ptr.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mgp_value)
    }
}

pub fn make_int_value(value: i64, context: &Memgraph) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate integer.");
    unsafe {
        let mgp_value = MgpValue {
            ptr: ffi::mgp_value_make_int(value, context.memory()),
        };
        if mgp_value.ptr.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mgp_value)
    }
}

pub fn make_string_value(value: &CStr, context: &Memgraph) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate string.");
    unsafe {
        let mgp_value = MgpValue {
            ptr: ffi::mgp_value_make_string(value.as_ptr(), context.memory()),
        };
        if mgp_value.ptr.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mgp_value)
    }
}

pub fn make_double_value(value: f64, context: &Memgraph) -> MgpResult<MgpValue> {
    let unable_alloc_value_msg = c_str!("Unable to allocate double.");
    unsafe {
        let mgp_value = MgpValue {
            ptr: ffi::mgp_value_make_double(value, context.memory()),
        };
        if mgp_value.ptr.is_null() {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_value_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(mgp_value)
    }
}

#[cfg(test)]
mod tests;
