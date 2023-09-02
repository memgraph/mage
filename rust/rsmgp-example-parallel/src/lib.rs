mod example;

use crate::example::example as example_algorithm;
use crate::example::MemgraphGraph;
use c_str_macro::c_str;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::{close_module, define_optional_type, define_procedure, define_type, init_module};
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

init_module!(|memgraph: &Memgraph| -> Result<()> {
    memgraph.add_read_procedure(
        example,
        c_str!("example"),
        &[define_type!("node_list", Type::List, Type::Int)],
        &[],
        &[define_type!("node_id", Type::Int)],
    )?;
    Ok(())
});

fn write_nodes_to_records(memgraph: &Memgraph, nodes: Vec<i64>) -> Result<()> {
    for node_id in nodes {
        let record = memgraph.result_record()?;
        record.insert_int(c_str!("node_id"), node_id)?;
    }
    Ok(())
}

define_procedure!(example, |memgraph: &Memgraph| -> Result<()> {
    let args = memgraph.args()?;
    let Value::List(node_list) = args.value_at(0)? else {
        panic!("Failed to read node_list")
    };

    let node_list: Vec<i64> = node_list
        .iter()?
        .map(|value| match value {
            Value::Int(i) => i as i64,
            _ => panic!("Failed converting node_list to vector"),
        })
        .collect();

    let graph = MemgraphGraph::from_graph(memgraph);

    let result = example_algorithm(graph, &node_list);
    write_nodes_to_records(memgraph, result)?;
    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });
