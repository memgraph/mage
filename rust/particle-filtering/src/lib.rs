mod algorithm;
mod graphs;

use crate::algorithm::particle_filtering as algorithm_particle_filtering;
use crate::graphs::MemgraphGraph;
use c_str_macro::c_str;
use rsmgp_sys::list::*;
use rsmgp_sys::map::*;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::property::*;
use rsmgp_sys::result::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::vertex::Vertex;
use rsmgp_sys::{close_module, define_optional_type, define_procedure, define_type, init_module};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::panic;

init_module!(|memgraph: &Memgraph| -> Result<()> {
    memgraph.add_read_procedure(
        particle_filtering,
        c_str!("particle_filtering"),
        &[define_type!("node_list", Type::List, Type::Int)],
        &[],
        &[
            define_type!("node_id", Type::Int),
            define_type!("score", Type::Double),
        ],
    )?;

    Ok(())
});

fn write_ppr_nodes_to_records(memgraph: &Memgraph, ppr_nodes: HashMap<i64, f32>) -> Result<()> {
    for (node_id, score) in ppr_nodes {
        let record = memgraph.result_record()?;
        record.insert_int(c_str!("node_id"), node_id)?;
        record.insert_double(c_str!("score"), score as f64)?;
    }
    Ok(())
}

define_procedure!(particle_filtering, |memgraph: &Memgraph| -> Result<()> {
    let min_threshold = 0.1;
    let num_particles = 1000.0;

    let args = memgraph.args()?;
    let node_list: Value = args.value_at(0)?;

    let node_list = if let Value::List(node_list) = node_list {
        node_list
    } else {
        panic!("Failed to read node_list");
    };

    let vector: Vec<i64> = node_list
        .iter()?
        .map(|value| match value {
            Value::Int(i) => i as i64,
            _ => panic!("Failed converting node_list to vector"),
        })
        .collect();

    let graph = MemgraphGraph::from_graph(memgraph);

    let result = algorithm_particle_filtering(
        graph,
        &vector,
        min_threshold,
        num_particles,
    );
    write_ppr_nodes_to_records(memgraph, result)?;
    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });
