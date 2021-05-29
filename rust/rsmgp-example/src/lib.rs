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

// Required because we want to be able to propagate Result by using ? operator.
fn test_procedure(context: &Memgraph) -> Result<(), MgpError> {
    let mgp_graph_iterator = make_graph_vertices_iterator(&context)?;
    for mgp_vertex in mgp_graph_iterator {
        let mgp_record = make_result_record(&context)?;

        let properties_string = mgp_vertex
            .properties()?
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
        )?;

        let labels_count = mgp_vertex.labels_count();
        insert_result_record(
            &mgp_record,
            c_str!("labels_count"),
            &make_int_value(labels_count as i64, &context)?,
        )?;

        if labels_count > 0 {
            let first_label = make_string_value(&mgp_vertex.label_at(0)?, &context)?;
            insert_result_record(&mgp_record, c_str!("first_label"), &first_label)?;
        } else {
            insert_result_record(
                &mgp_record,
                c_str!("first_label"),
                &make_string_value(c_str!(""), &context)?,
            )?;
        }

        let name_property = mgp_vertex.property(c_str!("name"))?.value;
        if let Value::Null = name_property {
            let unknown_name = make_string_value(c_str!("unknown"), &context)?;
            insert_result_record(&mgp_record, c_str!("name_property"), &unknown_name)?;
        } else if let Value::String(value) = name_property {
            let mgp_value = make_string_value(value.as_c_str(), &context)?;
            insert_result_record(&mgp_record, c_str!("name_property"), &mgp_value)?;
        } else {
            let unknown_type = make_string_value(c_str!("not null and not string"), &context)?;
            insert_result_record(&mgp_record, c_str!("name_property"), &unknown_type)?;
        }

        let has_label = mgp_vertex.has_label(c_str!("L3"));
        let mgp_value = make_bool_value(has_label, &context)?;
        insert_result_record(&mgp_record, c_str!("has_L3_label"), &mgp_value)?;

        // TODO(gitbuda): Figure out how to test vertex e2e.
        // let vertex_value = make_vertex_value(&mgp_vertex, &context)?;
        // insert_result_record(&mgp_record, c_str!("vertex"), &vertex_value, &context)?;

        match mgp_vertex.out_edges()?.next() {
            Some(edge) => {
                let edge_type = edge.edge_type()?;
                let mgp_value = make_string_value(&edge_type, &context)?;
                insert_result_record(&mgp_record, c_str!("first_edge_type"), &mgp_value)?;
            }
            None => {
                let mgp_value = make_string_value(c_str!("unknown_edge_type"), &context)?;
                insert_result_record(&mgp_record, c_str!("first_edge_type"), &mgp_value)?;
            }
        }

        let list_property = mgp_vertex.property(c_str!("list"))?.value;
        if let Value::List(list) = list_property {
            let mgp_value = make_list_value(&list, &context)?;
            insert_result_record(&mgp_record, c_str!("list"), &mgp_value)?;
        }
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
        match test_procedure(&context) {
            Ok(_) => (),
            Err(e) => {
                println!("{}", e);
                let msg = e.to_string();
                println!("{}", msg);
                let c_msg = CString::new(msg).expect("Unable to create Memgraph error message!");
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
        Err(_) => {
            println!("Procedure panic!");
            // TODO(gitbuda): Fix backtrace somehow.
            println!("{:?}", Backtrace::new());
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
    match add_string_result_type(procedure, c_str!("first_edge_type")) {
        Ok(_) => {}
        Err(_) => {
            return 1;
        }
    }
    match add_list_result_type(procedure, c_str!("list"), get_type_any()) {
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
