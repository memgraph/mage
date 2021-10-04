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
//! All related to the value (container for any data type).

use std::convert::From;
use std::ffi::{CStr, CString};

use crate::edge::*;
use crate::list::*;
use crate::map::*;
use crate::memgraph::*;
use crate::mgp::*;
use crate::path::*;
use crate::result::*;
use crate::vertex::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

/// Creates a copy of the provided string.
///
/// # Safety
///
/// The caller has provided a pointer that points to a valid C string. More here [CStr::from_ptr].
pub(crate) unsafe fn create_cstring(c_char_ptr: *const i8) -> MgpResult<CString> {
    match CString::new(CStr::from_ptr(c_char_ptr).to_bytes()) {
        Ok(v) => Ok(v),
        Err(_) => Err(MgpError::UnableToCreateCString),
    }
}

// NOTE: on why mutable pointer to mgp_value has to be owned by this code.
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
//
// The decision is to move on with a copy of mgp_value because it's cheap to make the copy.

/// Useful to own `mgp_value` coming from / going into Memgraph as a result.
///
/// Underlying pointer object is going to be automatically deleted.
///
/// NOTE: Implementing From<Value> for MgpValue is not simple because not all Value objects can
/// contain Memgraph object (primitive types).
pub struct MgpValue {
    // It's not wise to create a new MgpValue out of the existing value pointer because drop with a
    // valid pointer will be called multiple times -> double free problem.
    ptr: *mut mgp_value,
    memgraph: Memgraph,
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
    pub(crate) fn new(ptr: *mut mgp_value, memgraph: &Memgraph) -> MgpValue {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create Memgraph value because the given pointer is null."
        );

        MgpValue {
            ptr,
            memgraph: memgraph.clone(),
        }
    }

    pub(crate) fn mgp_ptr(&self) -> *const mgp_value {
        self.ptr
    }

    pub fn to_value(&self) -> MgpResult<Value> {
        unsafe { mgp_raw_value_to_value(self.mgp_ptr(), &self.memgraph) }
    }

    pub fn make_null(memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            let mgp_ptr = ffi::mgp_value_make_null(memgraph.memory_ptr());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToMakeNullValue);
            }
            Ok(MgpValue::new(mgp_ptr, &memgraph))
        }
    }

    pub fn is_null(&self) -> bool {
        unsafe { ffi::mgp_value_is_null(self.ptr) != 0 }
    }

    pub fn make_bool(value: bool, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            let mgp_ptr =
                ffi::mgp_value_make_bool(if !value { 0 } else { 1 }, memgraph.memory_ptr());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToMakeBoolValue);
            }
            Ok(MgpValue::new(mgp_ptr, &memgraph))
        }
    }

    pub fn is_bool(&self) -> bool {
        unsafe { ffi::mgp_value_is_bool(self.ptr) != 0 }
    }

    pub fn make_int(value: i64, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            let mgp_ptr = ffi::mgp_value_make_int(value, memgraph.memory_ptr());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToMakeIntegerValue);
            }
            Ok(MgpValue::new(mgp_ptr, &memgraph))
        }
    }

    pub fn is_int(&self) -> bool {
        unsafe { ffi::mgp_value_is_int(self.ptr) != 0 }
    }

    pub fn make_double(value: f64, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            let mgp_ptr = ffi::mgp_value_make_double(value, memgraph.memory_ptr());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToMakeDoubleValue);
            }
            Ok(MgpValue::new(mgp_ptr, &memgraph))
        }
    }

    pub fn is_double(&self) -> bool {
        unsafe { ffi::mgp_value_is_double(self.ptr) != 0 }
    }

    pub fn make_string(value: &CStr, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            let mgp_ptr = ffi::mgp_value_make_string(value.as_ptr(), memgraph.memory_ptr());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToMakeMemgraphStringValue);
            }
            Ok(MgpValue::new(mgp_ptr, &memgraph))
        }
    }

    pub fn is_string(&self) -> bool {
        unsafe { ffi::mgp_value_is_string(self.ptr) != 0 }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] objects owns
    /// the new object.
    pub fn make_list(list: &List, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            // The new object should be manually destroyed in case something within this function
            // fails.
            let mgp_list = ffi::mgp_list_make_empty(list.size(), memgraph.memory_ptr());
            for item in list.iter()? {
                let mgp_value = item.to_mgp_value(&memgraph)?;
                if ffi::mgp_list_append(mgp_list, mgp_value.ptr) == 0 {
                    ffi::mgp_list_destroy(mgp_list);
                    return Err(MgpError::UnableToMakeListValue);
                }
            }
            let mgp_value = ffi::mgp_value_make_list(mgp_list);
            if mgp_value.is_null() {
                ffi::mgp_list_destroy(mgp_list);
                return Err(MgpError::UnableToMakeListValue);
            }
            Ok(MgpValue::new(mgp_value, &memgraph))
        }
    }

    pub fn is_list(&self) -> bool {
        unsafe { ffi::mgp_value_is_list(self.ptr) != 0 }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] objects owns
    /// the new object.
    pub fn make_map(map: &Map, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            // The new object should be manually destroyed in case something within this function
            // fails.
            let mgp_map = ffi::mgp_map_make_empty(memgraph.memory_ptr());
            for item in map.iter()? {
                let mgp_value = match item.value.to_mgp_value(&memgraph) {
                    Ok(v) => v,
                    Err(_) => {
                        ffi::mgp_map_destroy(mgp_map);
                        return Err(MgpError::UnableToMakeMapValue);
                    }
                };
                if ffi::mgp_map_insert(mgp_map, item.key.as_ptr(), mgp_value.ptr) == 0 {
                    ffi::mgp_map_destroy(mgp_map);
                    return Err(MgpError::UnableToMakeMapValue);
                }
            }
            let mgp_value = ffi::mgp_value_make_map(mgp_map);
            if mgp_value.is_null() {
                ffi::mgp_map_destroy(mgp_map);
                return Err(MgpError::UnableToMakeMapValue);
            }
            Ok(MgpValue::new(mgp_value, &memgraph))
        }
    }

    pub fn is_map(&self) -> bool {
        unsafe { ffi::mgp_value_is_map(self.ptr) != 0 }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] objects owns
    /// the new object.
    pub fn make_vertex(vertex: &Vertex, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            // The new object should be manually destroyed in case something within this function
            // fails.
            let mgp_copy = ffi::mgp_vertex_copy(vertex.mgp_ptr(), memgraph.memory_ptr());
            if mgp_copy.is_null() {
                return Err(MgpError::UnableToMakeVertexValue);
            }
            let mgp_value = ffi::mgp_value_make_vertex(mgp_copy);
            if mgp_value.is_null() {
                ffi::mgp_vertex_destroy(mgp_copy);
                return Err(MgpError::UnableToMakeVertexValue);
            }
            Ok(MgpValue::new(mgp_value, &memgraph))
        }
    }

    pub fn is_vertex(&self) -> bool {
        unsafe { ffi::mgp_value_is_vertex(self.ptr) != 0 }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] objects owns
    /// the new object.
    pub fn make_edge(edge: &Edge, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            // The new object should be manually destroyed in case something within this function
            // fails.
            let mgp_copy = ffi::mgp_edge_copy(edge.mgp_ptr(), memgraph.memory_ptr());
            if mgp_copy.is_null() {
                return Err(MgpError::UnableToMakeEdgeValue);
            }
            let mgp_value = ffi::mgp_value_make_edge(mgp_copy);
            if mgp_value.is_null() {
                ffi::mgp_edge_destroy(mgp_copy);
                return Err(MgpError::UnableToMakeEdgeValue);
            }
            Ok(MgpValue::new(mgp_value, &memgraph))
        }
    }

    pub fn is_edge(&self) -> bool {
        unsafe { ffi::mgp_value_is_edge(self.ptr) != 0 }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] objects owns
    /// the new object.
    pub fn make_path(path: &Path, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        unsafe {
            // The new object should be manually destroyed in case something within this function
            // fails.
            let mgp_copy = ffi::mgp_path_copy(path.mgp_ptr(), memgraph.memory_ptr());
            if mgp_copy.is_null() {
                return Err(MgpError::UnableToMakePathValue);
            }
            let mgp_value = ffi::mgp_value_make_path(mgp_copy);
            if mgp_value.is_null() {
                ffi::mgp_path_destroy(mgp_copy);
                return Err(MgpError::UnableToMakePathValue);
            }
            Ok(MgpValue::new(mgp_value, &memgraph))
        }
    }

    pub fn is_path(&self) -> bool {
        unsafe { ffi::mgp_value_is_path(self.ptr) != 0 }
    }
}

/// Object containing/owning concrete underlying mgp objects (e.g., mgp_vertex).
///
/// User code should mostly deal with these objects.
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(CString),
    Vertex(Vertex),
    Edge(Edge),
    Path(Path),
    List(List),
    Map(Map),
}

impl Value {
    pub fn to_mgp_value(&self, memgraph: &Memgraph) -> MgpResult<MgpValue> {
        match self {
            Value::Null => MgpValue::make_null(&memgraph),
            Value::Bool(x) => MgpValue::make_bool(*x, &memgraph),
            Value::Int(x) => MgpValue::make_int(*x, &memgraph),
            Value::Float(x) => MgpValue::make_double(*x, &memgraph),
            Value::String(x) => MgpValue::make_string(&*x.as_c_str(), &memgraph),
            Value::List(x) => MgpValue::make_list(&x, &memgraph),
            Value::Map(x) => MgpValue::make_map(&x, &memgraph),
            Value::Vertex(x) => MgpValue::make_vertex(&x, &memgraph),
            Value::Edge(x) => MgpValue::make_edge(&x, &memgraph),
            Value::Path(x) => MgpValue::make_path(&x, &memgraph),
        }
    }
}

impl From<MgpValue> for Value {
    fn from(item: MgpValue) -> Self {
        match item.to_value() {
            Ok(v) => v,
            Err(_) => panic!("Unable to create Value from MgpValue."),
        }
    }
}

/// Creates copy of [mgp_value] object as a [Value] object.
///
/// NOTE: If would be more optimal not to copy [mgp_list], [mgp_map] and [mgp_path], but that's not
/// possible at this point because of the way how C API iterators are implemented. E.g., after each
/// `mgp_properties_iterator_next()` call, the previous `mgp_value` pointer shouldn't be used.
/// There is no known way of defining the right lifetime of the returned `MgpValue` object. A
/// solution would be to change the C API to preserve underlying values of the returned pointers
/// during the lifetime of the Rust iterator.
///
/// # Safety
///
/// Calls C API unsafe functions. The provided [mgp_value] object has to be a valid non-null
/// pointer.
pub(crate) unsafe fn mgp_raw_value_to_value(
    value: *const mgp_value,
    memgraph: &Memgraph,
) -> MgpResult<Value> {
    #[allow(non_upper_case_globals)]
    match ffi::mgp_value_get_type(value) {
        mgp_value_type_MGP_VALUE_TYPE_NULL => Ok(Value::Null),
        mgp_value_type_MGP_VALUE_TYPE_BOOL => Ok(Value::Bool(ffi::mgp_value_get_bool(value) == 0)),
        mgp_value_type_MGP_VALUE_TYPE_INT => Ok(Value::Int(ffi::mgp_value_get_int(value))),
        mgp_value_type_MGP_VALUE_TYPE_STRING => {
            let mgp_string = ffi::mgp_value_get_string(value);
            match create_cstring(mgp_string) {
                Ok(value) => Ok(Value::String(value)),
                Err(_) => Err(MgpError::UnableToMakeValueString),
            }
        }
        mgp_value_type_MGP_VALUE_TYPE_DOUBLE => Ok(Value::Float(ffi::mgp_value_get_double(value))),
        mgp_value_type_MGP_VALUE_TYPE_VERTEX => {
            let mgp_vertex = ffi::mgp_value_get_vertex(value);
            Ok(Value::Vertex(Vertex::mgp_copy(mgp_vertex, &memgraph)?))
        }
        mgp_value_type_MGP_VALUE_TYPE_EDGE => {
            let mgp_edge = ffi::mgp_value_get_edge(value);
            Ok(Value::Edge(Edge::mgp_copy(mgp_edge, &memgraph)?))
        }
        mgp_value_type_MGP_VALUE_TYPE_PATH => {
            let mgp_path = ffi::mgp_value_get_path(value);
            Ok(Value::Path(Path::mgp_copy(mgp_path, &memgraph)?))
        }
        mgp_value_type_MGP_VALUE_TYPE_LIST => {
            let mgp_list = ffi::mgp_value_get_list(value);
            Ok(Value::List(List::mgp_copy(mgp_list, &memgraph)?))
        }
        mgp_value_type_MGP_VALUE_TYPE_MAP => {
            let mgp_map = ffi::mgp_value_get_map(value);
            Ok(Value::Map(Map::mgp_copy(mgp_map, &memgraph)?))
        }
        _ => {
            panic!("Unable to create value object because of uncovered mgp_value type.");
        }
    }
}

#[cfg(test)]
mod tests;
