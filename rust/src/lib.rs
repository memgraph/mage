mod rsmgp;

use c_str_macro::c_str;
use rsmgp::*;
use rsmgp::mgp::*;
use std::os::raw::c_int;
use std::panic;

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
        let has_label = mgp_vertex.has_label("L3");
        let mgp_value = make_bool_value(has_label, result, memory)?;
        insert_result_record(&mgp_record, c_str!("has_label"), &mgp_value, result)?;
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
        // TODO(gitbuda): Implement set_error_msg.
        unsafe {
            mgp_result_set_error_msg(result, procedure_panic_msg.as_ptr());
        }
    }
}

#[no_mangle]
pub extern "C" fn mgp_init_module(module: *mut mgp_module, _memory: *mut mgp_memory) -> c_int {
    // TODO(gitbuda): Add catch_unwind.
    let procedure = add_read_procedure(test_procedure_c, c_str!("test_procedure"), module);
    match add_bool_result_type(procedure, c_str!("has_label")) {
        Ok(_) => {}
        Err(_) => {
            return 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn mgp_shutdown_module() -> c_int {
    // TODO(gitbuda): Add catch_unwind.
    0
}
