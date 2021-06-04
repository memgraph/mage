use std::ffi::CStr;

use crate::context::*;
#[double]
use crate::mgp::ffi;
use crate::mgp::*;
use crate::result::*;
use mockall_double::double;

// Required because we want to use catch_unwind to control panics.
#[macro_export]
macro_rules! define_procedure {
    ($c_name:ident, $rs_func:expr) => {
        #[no_mangle]
        extern "C" fn $c_name(
            args: *const mgp_list,
            graph: *const mgp_graph,
            result: *mut mgp_result,
            memory: *mut mgp_memory,
        ) {
            let prev_hook = panic::take_hook();
            panic::set_hook(Box::new(|_| { /* Do nothing. */ }));

            let procedure_result = panic::catch_unwind(|| {
                let context = Memgraph::new(args, graph, result, memory);
                match $rs_func(&context) {
                    Ok(_) => (),
                    Err(e) => {
                        println!("{}", e);
                        let msg = e.to_string();
                        println!("{}", msg);
                        let c_msg =
                            CString::new(msg).expect("Unable to create Memgraph error message!");
                        set_memgraph_error_msg(&c_msg, &context);
                    }
                }
            });

            panic::set_hook(prev_hook);
            match procedure_result {
                Ok(_) => {}
                // TODO(gitbuda): Take cause on panic and pass to mgp_result_set_error_msg.
                // Until figuring out how to take info from panic object, set error in-place.
                // As far as I know iterator can't return Result object and set error in-place.
                Err(e) => {
                    println!("Procedure panic!");
                    match e.downcast::<&str>() {
                        Ok(panic_msg) => {
                            println!("{}", panic_msg);
                        }
                        Err(_) => {
                            println!("Unknown type of panic!.");
                        }
                    }
                    // TODO(gitbuda): Fix backtrace somehow.
                    println!("{:?}", Backtrace::new());
                }
            }
        }
    };
}

#[macro_export]
macro_rules! init_module {
    ($init_func:expr) => {
        #[no_mangle]
        pub extern "C" fn mgp_init_module(
            module: *mut mgp_module,
            memory: *mut mgp_memory,
        ) -> c_int {
            // TODO(gitbuda): Add error handling (catch_unwind, etc.).
            $init_func(module, memory)
        }
    };
}

#[macro_export]
macro_rules! close_module {
    ($close_func:expr) => {
        #[no_mangle]
        pub extern "C" fn mgp_shutdown_module() -> c_int {
            // TODO(gitbuda): Add error handling (catch_unwind, etc.).
            $close_func()
        }
    };
}

pub fn set_memgraph_error_msg(msg: &CStr, context: &Memgraph) {
    unsafe {
        let status = ffi::mgp_result_set_error_msg(context.result(), msg.as_ptr());
        if status == 0 {
            panic!("Unable to pass error message to the Memgraph engine.");
        }
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

pub fn get_type_any() -> *const mgp_type {
    unsafe { ffi::mgp_type_any() }
}

pub fn get_type_bool() -> *const mgp_type {
    unsafe { ffi::mgp_type_bool() }
}

pub fn get_type_string() -> *const mgp_type {
    unsafe { ffi::mgp_type_string() }
}

pub fn get_type_int() -> *const mgp_type {
    unsafe { ffi::mgp_type_int() }
}

pub fn get_type_float() -> *const mgp_type {
    unsafe { ffi::mgp_type_float() }
}

pub fn get_type_number() -> *const mgp_type {
    unsafe { ffi::mgp_type_number() }
}

pub fn get_type_map() -> *const mgp_type {
    unsafe { ffi::mgp_type_map() }
}

pub fn get_type_vertex() -> *const mgp_type {
    unsafe { ffi::mgp_type_node() }
}

pub fn get_type_edge() -> *const mgp_type {
    unsafe { ffi::mgp_type_relationship() }
}

pub fn get_type_path() -> *const mgp_type {
    unsafe { ffi::mgp_type_path() }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_int_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_int()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_bool_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_bool()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_string_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_string()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_double_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_float()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_vertex_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_node()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_edge_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_relationship()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_path_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_path()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_map_result_type(procedure: *mut mgp_proc, name: &CStr) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_map()) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_list_result_type(
    procedure: *mut mgp_proc,
    name: &CStr,
    item_type: *const mgp_type,
) -> MgpResult<()> {
    unsafe {
        if ffi::mgp_proc_add_result(procedure, name.as_ptr(), ffi::mgp_type_list(item_type)) == 0 {
            return Err(MgpError::AddProcedureParameterTypeError);
        }
        Ok(())
    }
}

// TODO(gitbuda): Add nullable result type.

// TODO(gitbuda): Add transaction management (abort) stuff.

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mgp::mock_ffi::*;
}
