#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CString;
use std::os::raw::c_int;

struct MgValue {
    value: *mut mgp_value,
}
impl Drop for MgValue {
    fn drop(&mut self) {
        unsafe {
            mgp_value_destroy(self.value);
        }
    }
}

extern "C" fn test_procedure(
    _args: *const mgp_list,
    _graph: *const mgp_graph,
    _result: *mut mgp_result,
    _memory: *mut mgp_memory,
) {
    let unable_alloc_a_msg =
        CString::new("Unable to allocate a.").expect("CString::new failed on allocating a value");
    let a_name = CString::new("a").expect("CString::new failed on allocating a name");
    let a_not_insterted_msg = CString::new("Unable to insert a to the record")
        .expect("CString::new failed on inserting a");
    let record_fail_msg = CString::new("Unable to allocate record")
        .expect("CString::new failed on record allocation");

    unsafe {
        let record = mgp_result_new_record(_result);
        if record.is_null() {
            mgp_result_set_error_msg(_result, record_fail_msg.into_raw());
            return;
        }
        let a: MgValue = MgValue {
            value: mgp_value_make_int(0, _memory),
        };
        if a.value.is_null() {
            mgp_result_set_error_msg(_result, unable_alloc_a_msg.into_raw());
            return;
        }
        let a_inserted = mgp_result_record_insert(record, a_name.into_raw(), a.value);
        if a_inserted == 0 {
            mgp_result_set_error_msg(_result, a_not_insterted_msg.into_raw());
        }
    }
}

#[no_mangle]
pub extern "C" fn mgp_init_module(module: *mut mgp_module, _memory: *mut mgp_memory) -> c_int {
    unsafe {
        let procedure = mgp_module_add_read_procedure(
            module,
            CString::new("test_procedure")
                .expect("A valid function name.")
                .into_raw(),
            Some(test_procedure),
        );
        if !mgp_proc_add_result(
            procedure,
            CString::new("a")
                .expect("A valid argument name.")
                .into_raw(),
            mgp_type_int(),
        ) == 0
        {
            return 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn mgp_shutdown_module() -> c_int {
    0
}
