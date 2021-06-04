use backtrace::Backtrace;
use c_str_macro::c_str;
use rsmgp_sys::context::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::property::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

// Required because we want to be able to propagate Result by using ? operator.
fn test_procedure(context: &Memgraph) -> Result<(), MgpError> {
    for mgp_vertex in context.vertices_iter()? {
        let result = context.result_record()?;

        let mut properties: Vec<Property> = mgp_vertex.properties()?.collect();
        properties.sort_by(|a, b| {
            let a_name = a.name.to_str().unwrap();
            let b_name = b.name.to_str().unwrap();
            a_name.cmp(&b_name)
        });
        let properties_string = properties
            .iter()
            .map(|prop| {
                let prop_name = prop.name.to_str().unwrap();
                if let Value::Int(value) = prop.value {
                    return format!("{}: {}", prop_name, value);
                } else if let Value::String(value) = &prop.value {
                    return format!("{}: {}", prop_name, value.to_str().unwrap());
                } else if let Value::Float(value) = prop.value {
                    return format!("{}: {}", prop_name, value);
                } else {
                    ",".to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(", ");
        result.insert_string(
            c_str!("properties_string"),
            CString::new(properties_string.into_bytes())
                .unwrap()
                .as_c_str(),
        )?;

        let labels_count = mgp_vertex.labels_count();
        result.insert_int(c_str!("labels_count"), labels_count as i64)?;
        if labels_count > 0 {
            result.insert_string(c_str!("first_label"), &mgp_vertex.label_at(0)?)?;
        } else {
            result.insert_string(c_str!("first_label"), c_str!(""))?;
        }

        let name_property = mgp_vertex.property(c_str!("name"))?.value;
        if let Value::Null = name_property {
            result.insert_string(c_str!("name_property"), c_str!("unknown"))?;
        } else if let Value::String(value) = name_property {
            result.insert_string(c_str!("name_property"), &value)?;
        } else {
            result.insert_string(c_str!("name_property"), c_str!("not null and not string"))?
        }

        result.insert_bool(c_str!("has_L3_label"), mgp_vertex.has_label(c_str!("L3")))?;

        match mgp_vertex.out_edges()?.next() {
            Some(edge) => {
                let edge_type = edge.edge_type()?;
                result.insert_string(c_str!("first_edge_type"), &edge_type)?;
            }
            None => {
                result.insert_string(c_str!("first_edge_type"), c_str!("unknown_edge_type"))?;
            }
        }

        let list_property = mgp_vertex.property(c_str!("list"))?.value;
        if let Value::List(list) = list_property {
            result.insert_list(c_str!("list"), &list)?;
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
        let context = Memgraph::new(args, graph, result, memory);
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
