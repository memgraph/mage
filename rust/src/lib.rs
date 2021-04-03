#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_int;
#[macro_use]
extern crate c_str_macro;

//// START "library" part.

#[derive(Debug, Clone)]
struct MgpError;
impl std::fmt::Display for MgpError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "An error inside Rust procedure.")
    }
}

#[derive(Debug, Clone)]
struct MgpAllocationError;
impl std::fmt::Display for MgpAllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Unable to allocate memory inside Rust procedure.")
    }
}

#[derive(Debug, Clone)]
struct MgpPreparingResultError;
impl std::fmt::Display for MgpPreparingResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Unable to prepare result within Rust procedure.")
    }
}

#[derive(Debug, Clone)]
struct MgpAddProcedureParameterTypeError;
impl std::fmt::Display for MgpAddProcedureParameterTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Unable to add a type of procedure paramater in Rust Module."
        )
    }
}

struct MgpValue {
    value: *mut mgp_value,
}
impl Drop for MgpValue {
    fn drop(&mut self) {
        unsafe {
            mgp_value_destroy(self.value);
        }
    }
}

fn make_int_value(
    value: i64,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> Result<MgpValue, MgpAllocationError> {
    let unable_alloc_value_msg = c_str!("Unable to allocate an integer.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: mgp_value_make_int(value, memory),
        };
        if mg_value.value.is_null() {
            mgp_result_set_error_msg(result, unable_alloc_value_msg.as_ptr());
            return Err(MgpAllocationError);
        }
        return Ok(mg_value);
    }
}

fn make_bool_value(
    value: bool,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> Result<MgpValue, MgpAllocationError> {
    let unable_alloc_value_msg = c_str!("Unable to allocate boolean value.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: mgp_value_make_bool(if value == false { 0 } else { 1 }, memory),
        };
        if mg_value.value.is_null() {
            mgp_result_set_error_msg(result, unable_alloc_value_msg.as_ptr());
            return Err(MgpAllocationError);
        }
        return Ok(mg_value);
    }
}

struct MgpVerticesIterator {
    ptr: *mut mgp_vertices_iterator,
    is_first: bool,
}
impl Default for MgpVerticesIterator {
    fn default() -> Self {
        return Self {
            ptr: std::ptr::null_mut(),
            is_first: true,
        };
    }
}
impl Drop for MgpVerticesIterator {
    fn drop(&mut self) {
        unsafe {
            mgp_vertices_iterator_destroy(self.ptr);
        }
    }
}
impl Iterator for MgpVerticesIterator {
    type Item = MgpVertex;
    fn next(&mut self) -> Option<MgpVertex> {
        if self.is_first {
            self.is_first = false;
            unsafe {
                let data = mgp_vertices_iterator_get(self.ptr);
                if data.is_null() {
                    return None;
                } else {
                    return Some(MgpVertex { ptr: data });
                }
            }
        } else {
            unsafe {
                let data = mgp_vertices_iterator_next(self.ptr);
                if data.is_null() {
                    return None;
                } else {
                    return Some(MgpVertex { ptr: data });
                }
            }
        }
    }
}

struct MgpVertex {
    ptr: *const mgp_vertex,
}
impl MgpVertex {
    fn labels_count(&self) -> u64 {
        unsafe {
            return mgp_vertex_labels_count(self.ptr);
        }
    }

    fn has_label(&self, name: &str) -> bool {
        // TODO(gitbuda): Deal with additional allocation + panic.
        let c_str = CString::new(name).unwrap();
        unsafe {
            let c_mgp_label = mgp_label {
                name: c_str.as_ptr(),
            };
            return mgp_vertex_has_label(self.ptr, c_mgp_label) != 0;
        }
    }
}
// TODO(gitbuda): Implement all methods to access vertex data.

fn make_graph_vertices_iterator(
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> Result<MgpVerticesIterator, MgpAllocationError> {
    let unable_alloc_iter_msg = c_str!("Unable to allocate a vertices iterator.");
    unsafe {
        let iterator: MgpVerticesIterator = MgpVerticesIterator {
            ptr: mgp_graph_iter_vertices(graph, memory),
            ..Default::default()
        };
        if iterator.ptr.is_null() {
            mgp_result_set_error_msg(result, unable_alloc_iter_msg.as_ptr());
            return Err(MgpAllocationError);
        }
        return Ok(iterator);
    }
}

struct MgpResultRecord {
    record: *mut mgp_result_record,
}

fn make_result_record(result: *mut mgp_result) -> Result<MgpResultRecord, MgpAllocationError> {
    let record_fail_msg = c_str!("Unable to allocate record");
    unsafe {
        let record = mgp_result_new_record(result);
        if record.is_null() {
            mgp_result_set_error_msg(result, record_fail_msg.as_ptr());
            return Err(MgpAllocationError);
        }
        Ok(MgpResultRecord { record: record })
    }
}

fn insert_result_record(
    mgp_record: &MgpResultRecord,
    mgp_name: &CStr,
    mgp_value: &MgpValue,
    result: *mut mgp_result,
) -> Result<(), MgpPreparingResultError> {
    let name_not_inserted_msg = c_str!("Unable to insert record to the result.");
    unsafe {
        let inserted =
            mgp_result_record_insert(mgp_record.record, mgp_name.as_ptr(), mgp_value.value);
        if inserted == 0 {
            mgp_result_set_error_msg(result, name_not_inserted_msg.as_ptr());
            return Err(MgpPreparingResultError);
        }
        return Ok(());
    }
}

fn add_read_procedure(
    proc_ptr: extern "C" fn(*const mgp_list, *const mgp_graph, *mut mgp_result, *mut mgp_memory),
    name: &CStr,
    module: *mut mgp_module,
) -> *mut mgp_proc {
    unsafe {
        return mgp_module_add_read_procedure(module, name.as_ptr(), Some(proc_ptr));
    }
}

fn add_int_result_type(
    procedure: *mut mgp_proc,
    name: &CStr,
) -> Result<(), MgpAddProcedureParameterTypeError> {
    unsafe {
        if mgp_proc_add_result(procedure, name.as_ptr(), mgp_type_int()) == 0 {
            return Err(MgpAddProcedureParameterTypeError);
        }
        return Ok(());
    }
}

fn add_bool_result_type(
    procedure: *mut mgp_proc,
    name: &CStr,
) -> Result<(), MgpAddProcedureParameterTypeError> {
    unsafe {
        if mgp_proc_add_result(procedure, name.as_ptr(), mgp_type_bool()) == 0 {
            return Err(MgpAddProcedureParameterTypeError);
        }
        return Ok(());
    }
}

//// END "library" part.

extern "C" fn test_procedure(
    _args: *const mgp_list,
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) {
    match make_graph_vertices_iterator(graph, result, memory) {
        Ok(mgp_graph_iterator) => {
            for mgp_vertex in mgp_graph_iterator {
                match make_result_record(result) {
                    Ok(mgp_record) => {
                        let has_label = mgp_vertex.has_label("L3");
                        match make_bool_value(has_label, result, memory) {
                            Ok(mgp_value) => {
                                match insert_result_record(
                                    &mgp_record,
                                    c_str!("has_label"),
                                    &mgp_value,
                                    result,
                                ) {
                                    Ok(_) => {}
                                    Err(_) => {
                                        return;
                                    }
                                }
                            }
                            Err(_) => {
                                return;
                            }
                        };
                    }
                    Err(_) => {
                        return;
                    }
                }
            }
        }
        Err(_) => {
            return;
        }
    }
}

#[no_mangle]
pub extern "C" fn mgp_init_module(module: *mut mgp_module, _memory: *mut mgp_memory) -> c_int {
    let procedure = add_read_procedure(test_procedure, c_str!("test_procedure"), module);
    match add_bool_result_type(procedure, c_str!("has_label")) {
        Ok(_) => {}
        Err(_) => {
            return 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn mgp_shutdown_module() -> c_int {
    0
}
