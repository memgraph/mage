use std::ffi::CStr;

use crate::context::*;
#[double]
use crate::mgp::ffi;
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
                let context = Memgraph::new(args, graph, result, memory, std::ptr::null_mut());
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
            let memgraph = Memgraph::new(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                memory,
                module,
            );
            match $init_func(&memgraph) {
                Ok(_) => 0,
                Err(_) => 1,
            }
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

pub enum FieldType {
    Any,
    Bool,
    Number,
    Int,
    Double,
    String,
    Map,
    Vertex,
    Edge,
    Path,
    Nullable,
    List,
}

pub struct ResultFieldType<'a> {
    pub name: &'a CStr,
    pub types: &'a [FieldType],
}

#[macro_export]
macro_rules! define_type {
    ($name:literal, $($types:expr),+) => {
        ResultFieldType {
            name: &c_str!($name),
            types: &[$($types),*],
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

// TODO(gitbuda): Add transaction management (abort) stuff.
// TODO(gitbuda): Deal with optional arguments.
// TODO(gitbuda): Add support for depricated arguments.

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mgp::mock_ffi::*;
}
