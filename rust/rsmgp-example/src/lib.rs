use backtrace::Backtrace;
use c_str_macro::c_str;
use rsmgp_sys::context::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::vertex::*;
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;
use std::rc::Rc;

// TODO(gitbuda): If double free occures, Memgraph crashes -> prevent/ensure somehow.

// Required because we want to be able to propagate Result by using ? operator.
fn test_procedure(context: Memgraph) -> Result<(), MgpError> {
    let mgp_graph_iterator = make_graph_vertices_iterator(&context)?;
    for mgp_vertex in mgp_graph_iterator {
        let mgp_record = make_result_record(&context)?;

        let properties_string = mgp_vertex
            .properties()
            .map(|prop| {
                let prop_name = prop.name.to_str().unwrap();
                if let Value::Int(value) = prop.value {
                    return format!("{}: {}", prop_name, value);
                } else if let Value::String(value) = prop.value {
                    return format!("{}: {}", prop_name, value.to_str().unwrap());
                } else if let Value::Float(value) = prop.value {
                    return format!("{}: {}", prop_name, value);
                } else {
                    ",".to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(", ");
        // TODO(gitbuda): Combine insert and make value.
        insert_result_record(
            &mgp_record,
            c_str!("properties_string"),
            &make_string_value(
                CString::new(properties_string.into_bytes())
                    .unwrap()
                    .as_c_str(),
                &context,
            )?,
            &context,
        )?;

        let labels_count = mgp_vertex.labels_count();
        insert_result_record(
            &mgp_record,
            c_str!("labels_count"),
            &make_int_value(labels_count as i64, &context)?,
            &context,
        )?;

        if labels_count > 0 {
            let first_label = make_string_value(mgp_vertex.label_at(0)?, &context)?;
            insert_result_record(&mgp_record, c_str!("first_label"), &first_label, &context)?;
        } else {
            insert_result_record(
                &mgp_record,
                c_str!("first_label"),
                &make_string_value(c_str!(""), &context)?,
                &context,
            )?;
        }

        let name_property = mgp_vertex.property(c_str!("name"))?.value;
        if let Value::Null = name_property {
            let unknown_name = make_string_value(c_str!("unknown"), &context)?;
            insert_result_record(
                &mgp_record,
                c_str!("name_property"),
                &unknown_name,
                &context,
            )?;
        } else if let Value::String(value) = name_property {
            let mgp_value = make_string_value(value.as_c_str(), &context)?;
            insert_result_record(&mgp_record, c_str!("name_property"), &mgp_value, &context)?;
        } else {
            let unknown_type = make_string_value(c_str!("not null and not string"), &context)?;
            insert_result_record(
                &mgp_record,
                c_str!("name_property"),
                &unknown_type,
                &context,
            )?;
        }

        let has_label = mgp_vertex.has_label(c_str!("L3"));
        let mgp_value = make_bool_value(has_label, &context)?;
        insert_result_record(&mgp_record, c_str!("has_L3_label"), &mgp_value, &context)?;

        // TODO(gitbuda): Figure out how to test vertex e2e.
        // let vertex_value = make_vertex_value(&mgp_vertex, &context)?;
        // insert_result_record(&mgp_record, c_str!("vertex"), &vertex_value, &context)?;
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
    let procedure_result = panic::catch_unwind(|| {
        let context = Memgraph {
            context: Rc::new(MgpMemgraph {
                args,
                graph,
                result,
                memory,
            }),
        };
        match test_procedure(context) {
            Ok(_) => (),
            Err(e) => {
                println!("{}", e);
            }
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
    // match add_vertex_result_type(procedure, c_str!("vertex")) {
    //     Ok(_) => {}
    //     Err(_) => {
    //         return 1;
    //     }
    // }
    0
}

#[no_mangle]
pub extern "C" fn mgp_shutdown_module() -> c_int {
    0
}
