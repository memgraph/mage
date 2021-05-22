use c_str_macro::c_str;
use rsmgp_sys::mgp::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::vertex::*;
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;
use backtrace::Backtrace;

// TODO(gitbuda): If double free occures, Memgraph crashes -> prevent/ensure somehow.

// Required because we want to be able to propagate Result by using ? operator.
fn test_procedure(
    _args: *const mgp_list,
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> Result<(), MgpError> {
    let mgp_graph_iterator = make_graph_vertices_iterator(graph, result, memory)?;
    for mgp_vertex in mgp_graph_iterator {
        let mgp_record = make_result_record(result)?;

        let properties_string = mgp_vertex
            .properties()
            .map(|prop| {
                let prop_name = prop.name.to_str().unwrap();
                if let Value::Int(value) = prop.value {
                    return format!("{}: {}", prop_name, value);
                } else {
                    return format!(",");
                }
            })
            .collect::<Vec<String>>()
            .join(", ");
        insert_result_record(
            &mgp_record,
            c_str!("properties_string"),
            &make_string_value(
                CString::new(properties_string.into_bytes())
                    .unwrap()
                    .as_c_str(),
                result,
                memory,
            )?,
            result,
        )?;

        let labels_count = mgp_vertex.labels_count();
        insert_result_record(
            &mgp_record,
            c_str!("labels_count"),
            &make_int_value(labels_count as i64, result, memory)?,
            result,
        )?;

        if labels_count > 0 {
            let first_label = make_string_value(mgp_vertex.label_at(0)?, result, memory)?;
            insert_result_record(&mgp_record, c_str!("first_label"), &first_label, result)?;
        } else {
            insert_result_record(
                &mgp_record,
                c_str!("first_label"),
                &make_string_value(c_str!(""), result, memory)?,
                result,
            )?;
        }

        let name_property = mgp_vertex.property(c_str!("name"))?.value;
        if let Value::Null = name_property {
            let unknown_name = make_string_value(c_str!("unknown"), result, memory)?;
            insert_result_record(&mgp_record, c_str!("name_property"), &unknown_name, result)?;
        } else {
            let known_name = make_string_value(c_str!("known"), result, memory)?;
            insert_result_record(&mgp_record, c_str!("name_property"), &known_name, result)?;
        }

        let has_label = mgp_vertex.has_label(c_str!("L3"));
        let mgp_value = make_bool_value(has_label, result, memory)?;
        insert_result_record(&mgp_record, c_str!("has_L3_label"), &mgp_value, result)?;
    }
    Ok(())
}

// Required because we want to use catch_unwind to control panics.
#[no_mangle]
extern "C" fn test_procedure_c(
    args: *const mgp_list,
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) {
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| { /* Do nothing. */ }));
    let procedure_result =
        panic::catch_unwind(|| match test_procedure(args, graph, result, memory) {
            Ok(_) => (),
            Err(e) => {
                println!("{}", e);
            }
        });
    panic::set_hook(prev_hook);
    let procedure_panic_msg = c_str!("Procedure panic!");
    if procedure_result.is_err() {
        println!("Procedure panic!");
        // TODO(gitbuda): Fix backtrace print which is almost useless.
        println!("{:?}", Backtrace::new());
        unsafe {
            mgp_result_set_error_msg(result, procedure_panic_msg.as_ptr());
        }
    }
}

#[no_mangle]
pub extern "C" fn mgp_init_module(module: *mut mgp_module, _memory: *mut mgp_memory) -> c_int {
    let procedure = add_read_procedure(test_procedure_c, c_str!("test_procedure"), module);
    match add_int_result_type(procedure, c_str!("labels_count")) {
        Ok(_) => {}
        Err(_) => {
            return 1;
        }
    }
    match add_bool_result_type(procedure, c_str!("has_L3_label")) {
        Ok(_) => {}
        Err(_) => {
            return 1;
        }
    }
    match add_string_result_type(procedure, c_str!("first_label")) {
        Ok(_) => {}
        Err(_) => {
            return 1;
        }
    }
    match add_string_result_type(procedure, c_str!("name_property")) {
        Ok(_) => {}
        Err(_) => {
            return 1;
        }
    }
    match add_string_result_type(procedure, c_str!("properties_string")) {
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
