#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CString;
use std::os::raw::c_int;

extern "C" fn test_procedure(
    _args: *const mgp_list,
    _graph: *const mgp_graph,
    _result: *mut mgp_result,
    _memory: *mut mgp_memory,
) {
    //  TODO(gitbuda): Write an example.
}

#[no_mangle]
pub extern "C" fn mgp_init_module(module: *mut mgp_module, _memory: *mut mgp_memory) -> c_int {
    unsafe {
        mgp_module_add_read_procedure(
            module,
            CString::new("test_procedure")
                .expect("A valid function name.")
                .into_raw(),
            Some(test_procedure),
        );
    }
    0
}

#[no_mangle]
pub extern "C" fn mgp_shutdown_module() -> c_int {
    0
}
