use c_str_macro::c_str;
use std::ffi::{CStr, CString};

use crate::context::*;
use crate::edge::*;
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

/// Useful to hold user created `mgp_value` but also to store values coming from Memgraph, e.g.,
/// property `mgp_value`s.
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

    pub fn is_vertex(&self) -> bool {
        unsafe { ffi::mgp_value_is_vertex(self.ptr) != 0 }
    }

    pub fn is_edge(&self) -> bool {
        unsafe { ffi::mgp_value_is_edge(self.ptr) != 0 }
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

pub fn make_vertex_value(vertex: &Vertex, context: &Memgraph) -> MgpResult<MgpValue> {
    unsafe {
        let mgp_copy = ffi::mgp_vertex_copy(vertex.ptr, context.memory());
        if mgp_copy.is_null() {
            return Err(MgpError::MgpResultVertexAllocationError);
        }
        let mgp_value = ffi::mgp_value_make_vertex(mgp_copy);
        if mgp_value.is_null() {
            ffi::mgp_vertex_destroy(mgp_copy);
            return Err(MgpError::MgpResultVertexAllocationError);
        }
        Ok(MgpValue { ptr: mgp_value })
    }
}

pub fn make_edge_value(edge: &Edge, context: &Memgraph) -> MgpResult<MgpValue> {
    unsafe {
        let mgp_copy = ffi::mgp_edge_copy(edge.ptr, context.memory());
        if mgp_copy.is_null() {
            return Err(MgpError::MgpResultVertexAllocationError);
        }
        let mgp_value = ffi::mgp_value_make_edge(mgp_copy);
        if mgp_value.is_null() {
            ffi::mgp_edge_destroy(mgp_copy);
            return Err(MgpError::MgpResultVertexAllocationError);
        }
        Ok(MgpValue { ptr: mgp_value })
    }
}

#[derive(Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(CString),
    Vertex(Vertex),
    Edge(Edge),
}

impl Value {
    // TODO(gitbuda): Remove to_result_mgp_value dead code.
    #[allow(dead_code)]
    fn to_result_mgp_value(&self, context: &Memgraph) -> MgpResult<MgpValue> {
        match self {
            Value::Null => make_null_value(context),
            Value::Bool(x) => make_bool_value(*x, context),
            Value::Int(x) => make_int_value(*x, context),
            Value::String(x) => make_string_value(&*x.as_c_str(), context),
            Value::Float(x) => make_double_value(*x, context),
            Value::Vertex(x) => make_vertex_value(&x, context),
            Value::Edge(x) => make_edge_value(&x, context),
        }
    }
}

/// # Safety
/// TODO(gitbuda): Write section about safety.
pub unsafe fn make_vertex_copy(
    mgp_vertex: *const mgp_vertex,
    context: &Memgraph,
) -> MgpResult<Vertex> {
    assert!(
        !mgp_vertex.is_null(),
        "Unable to make vertex copy because vertex is null."
    );
    let mgp_vertex_copy = ffi::mgp_vertex_copy(mgp_vertex, context.memory());
    if mgp_vertex_copy.is_null() {
        ffi::mgp_result_set_error_msg(
            context.result(),
            c_str!("Unable to make vertex copy.").as_ptr(),
        );
        return Err(MgpError::MgpCreationOfVertexError);
    }
    Ok(Vertex {
        ptr: mgp_vertex_copy,
        context: context.clone(),
    })
}

/// # Safety
/// TODO(gitbuda): Write section about safety.
pub unsafe fn make_edge_copy(mgp_edge: *const mgp_edge, context: &Memgraph) -> MgpResult<Edge> {
    assert!(
        !mgp_edge.is_null(),
        "Unable to make edge copy because edge is null."
    );
    let mgp_edge_copy = ffi::mgp_edge_copy(mgp_edge, context.memory());
    if mgp_edge_copy.is_null() {
        ffi::mgp_result_set_error_msg(
            context.result(),
            c_str!("Unable to make edge copy.").as_ptr(),
        );
        return Err(MgpError::MgpCreationOfEdgeError);
    }
    Ok(Edge {
        ptr: mgp_edge_copy,
        context: context.clone(),
    })
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
        mgp_value_type_MGP_VALUE_TYPE_STRING => {
            let mgp_string = ffi::mgp_value_get_string(value);
            match create_cstring(mgp_string, &context) {
                Ok(value) => Ok(Value::String(value)),
                Err(_) => Err(MgpError::MgpCreationOfCStringError),
            }
        }
        mgp_value_type_MGP_VALUE_TYPE_DOUBLE => Ok(Value::Float(ffi::mgp_value_get_double(value))),
        mgp_value_type_MGP_VALUE_TYPE_VERTEX => {
            let mgp_vertex = ffi::mgp_value_get_vertex(value);
            let vertex = make_vertex_copy(mgp_vertex, &context)?;
            Ok(Value::Vertex(vertex))
        }
        mgp_value_type_MGP_VALUE_TYPE_EDGE => {
            let mgp_edge = ffi::mgp_value_get_edge(value);
            let edge = make_edge_copy(mgp_edge, &context)?;
            Ok(Value::Edge(edge))
        }
        // TODO(gitbuda): Handle mgp_value_type unhandeled values.
        _ => {
            println!("Uncovered mgp_value type!");
            panic!()
        }
    }
}

pub fn mgp_value_to_value(value: &MgpValue, context: &Memgraph) -> MgpResult<Value> {
    unsafe { mgp_raw_value_to_value(value.ptr, context) }
}

/// # Safety
/// TODO(gitbuda): Write section about safety.
pub unsafe fn create_cstring(c_char_ptr: *const i8, context: &Memgraph) -> MgpResult<CString> {
    let unable_alloc_msg = c_str!("Unable to create/allocate new CString.");
    match CString::new(CStr::from_ptr(c_char_ptr).to_bytes()) {
        Ok(v) => Ok(v),
        Err(_) => {
            ffi::mgp_result_set_error_msg(context.result(), unable_alloc_msg.as_ptr());
            Err(MgpError::MgpAllocationError)
        }
    }
}

#[cfg(test)]
mod tests;
