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

pub enum SimpleType {
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
}

pub enum ComplexType {
    Nullable,
    List,
}

pub struct ResultFieldType {
    pub simple_type: SimpleType,
    pub complex_type: Option<ComplexType>,
}

impl Default for ResultFieldType {
    fn default() -> Self {
        Self {
            simple_type: SimpleType::Any,
            complex_type: None,
        }
    }
}

pub struct ResultField<'a> {
    pub name: &'a CStr,
    pub field_type: ResultFieldType,
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn add_read_procedure(
    proc_ptr: extern "C" fn(*const mgp_list, *const mgp_graph, *mut mgp_result, *mut mgp_memory),
    name: &CStr,
    module: *mut mgp_module,
    result_fields: &[ResultField],
) -> MgpResult<()> {
    unsafe {
        let procedure = ffi::mgp_module_add_read_procedure(module, name.as_ptr(), Some(proc_ptr));
        if procedure.is_null() {
            return Err(MgpError::UnableToRegisterReadProcedure);
        }
        for result_field in result_fields {
            let mut mgp_type = match &result_field.field_type.simple_type {
                SimpleType::Any => ffi::mgp_type_any(),
                SimpleType::Bool => ffi::mgp_type_bool(),
                SimpleType::Number => ffi::mgp_type_number(),
                SimpleType::Int => ffi::mgp_type_int(),
                SimpleType::Double => ffi::mgp_type_float(),
                SimpleType::String => ffi::mgp_type_string(),
                SimpleType::Map => ffi::mgp_type_map(),
                SimpleType::Vertex => ffi::mgp_type_node(),
                SimpleType::Edge => ffi::mgp_type_relationship(),
                SimpleType::Path => ffi::mgp_type_path(),
            };
            if let Some(value) = &result_field.field_type.complex_type {
                mgp_type = match value {
                    ComplexType::List => ffi::mgp_type_list(mgp_type),
                    ComplexType::Nullable => ffi::mgp_type_nullable(mgp_type),
                };
            }
            if ffi::mgp_proc_add_result(procedure, result_field.name.as_ptr(), mgp_type) == 0 {
                return Err(MgpError::AddProcedureParameterTypeError);
            }
        }
        Ok(())
    }
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

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mgp::mock_ffi::*;
}
