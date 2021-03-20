#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CString;
use std::os::raw::c_int;

//// START "library" part.
// TODO(gitbuda): The expect is panicking (after CString::new), being in panic is not good in
// general, but in this particular case it will affect Memgraph (take it down). Figure out how to
// not panic and still operate correctly.

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
    let unable_alloc_value_msg = CString::new("Unable to allocate an integer.")
        .expect("CString::new failed prior to allocating an integer value.");
    unsafe {
        let mg_value: MgpValue = MgpValue {
            value: mgp_value_make_int(value, memory),
        };
        if mg_value.value.is_null() {
            mgp_result_set_error_msg(result, unable_alloc_value_msg.into_raw());
            return Err(MgpAllocationError);
        }
        return Ok(mg_value);
    }
}

struct MgpVerticesIterator {
    ptr: *mut mgp_vertices_iterator,
}

impl Drop for MgpVerticesIterator {
    fn drop(&mut self) {
        unsafe {
            mgp_vertices_iterator_destroy(self.ptr);
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
}
// TODO(gitbuda): Implement all methods to access vertex data.

fn make_graph_vertices_iterator(
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> Result<MgpVerticesIterator, MgpAllocationError> {
    let unable_alloc_iter_msg = CString::new("Unable to allocate a vertices iterator.")
        .expect("CString::new failed prior to allocating a vertices iterator.");
    unsafe {
        let iterator: MgpVerticesIterator = MgpVerticesIterator {
            ptr: mgp_graph_iter_vertices(graph, memory),
        };
        if iterator.ptr.is_null() {
            mgp_result_set_error_msg(result, unable_alloc_iter_msg.into_raw());
            return Err(MgpAllocationError);
        }
        return Ok(iterator);
    }
}

fn get_iterator_vertex(iterator: &MgpVerticesIterator) -> Result<MgpVertex, MgpError> {
    unsafe {
        return Ok(MgpVertex {
            ptr: mgp_vertices_iterator_get(iterator.ptr),
        });
    }
}

fn get_iterator_vertex_next(iterator: &MgpVerticesIterator) -> Result<MgpVertex, MgpError> {
    unsafe {
        return Ok(MgpVertex {
            ptr: mgp_vertices_iterator_next(iterator.ptr),
        });
    }
}

struct MgpResultRecord {
    record: *mut mgp_result_record,
}

fn make_result_record(result: *mut mgp_result) -> Result<MgpResultRecord, MgpAllocationError> {
    let record_fail_msg = CString::new("Unable to allocate record")
        .expect("CString::new failed on record allocation");
    unsafe {
        let record = mgp_result_new_record(result);
        if record.is_null() {
            mgp_result_set_error_msg(result, record_fail_msg.into_raw());
            return Err(MgpAllocationError);
        }
        Ok(MgpResultRecord { record: record })
    }
}

fn insert_result_record(
    mgp_record: &MgpResultRecord,
    mgp_name: String,
    mgp_value: &MgpValue,
    result: *mut mgp_result,
) -> Result<(), MgpPreparingResultError> {
    let name =
        CString::new(mgp_name.into_bytes()).expect("CString::new failed on allocating a name");
    let name_not_inserted_msg = CString::new("Unable to insert record to the result.")
        .expect("CString::new failed prior to inserting record.");

    unsafe {
        let inserted =
            mgp_result_record_insert(mgp_record.record, name.into_raw(), mgp_value.value);
        if inserted == 0 {
            mgp_result_set_error_msg(result, name_not_inserted_msg.into_raw());
            return Err(MgpPreparingResultError);
        }
        return Ok(());
    }
}

fn add_read_procedure(
    proc_ptr: extern "C" fn(*const mgp_list, *const mgp_graph, *mut mgp_result, *mut mgp_memory),
    name: String,
    module: *mut mgp_module,
) -> *mut mgp_proc {
    unsafe {
        return mgp_module_add_read_procedure(
            module,
            CString::new(name.into_bytes())
                .expect("A valid procedure name.")
                .into_raw(),
            Some(proc_ptr),
        );
    }
}

fn add_int_result_type(
    procedure: *mut mgp_proc,
    name: String,
) -> Result<(), MgpAddProcedureParameterTypeError> {
    unsafe {
        if mgp_proc_add_result(
            procedure,
            CString::new(name.into_bytes())
                .expect("A valid argument name.")
                .into_raw(),
            mgp_type_int(),
        ) == 0
        {
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
    use std::convert::TryFrom;
    match make_graph_vertices_iterator(graph, result, memory) {
        Ok(mgp_graph_iterator) => {
            for _index in 1..3 {
                match get_iterator_vertex_next(&mgp_graph_iterator) {
                    Ok(mgp_vertex) => match make_result_record(result) {
                        Ok(mgp_record) => match i64::try_from(mgp_vertex.labels_count()) {
                            Ok(labels_count) => {
                                match make_int_value(labels_count, result, memory) {
                                    Ok(mgp_value) => {
                                        match insert_result_record(
                                            &mgp_record,
                                            "labels_count".to_string(),
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
                        },
                        Err(_) => {
                            return;
                        }
                    },
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
    let procedure = add_read_procedure(test_procedure, "test_procedure".to_string(), module);
    match add_int_result_type(procedure, "labels_count".to_string()) {
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
