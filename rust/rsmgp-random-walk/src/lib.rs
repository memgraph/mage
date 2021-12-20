use c_str_macro::c_str;
use rsmgp_sys::edge::*;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::path::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::{close_module, define_procedure, define_type, init_module};
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;
extern crate rand;
use rand::Rng;

define_procedure!(get, |memgraph: &Memgraph| -> Result<()> {
    let args = memgraph.args()?;
    let result = memgraph.result_record()?;
    let input_start = args.value_at(0)?;
    let input_length = args.value_at(1)?;
    let path = if let Value::Vertex(ref vertex) = input_start {
        Path::make_with_start(vertex, memgraph)
    } else {
        panic!("Failed to create path from the start vertex.");
    }?;
    let length = if let Value::Int(value) = input_length {
        value
    } else {
        panic!("Failed to read desired path length.");
    };

    let mut vertex = if let Value::Vertex(vertex) = input_start {
        vertex
    } else {
        panic!("Failed to read start vertex.");
    };
    let mut rng = rand::thread_rng();
    for _ in 0..length {
        let edges: Vec<Edge> = vertex.out_edges()?.collect();
        if edges.is_empty() {
            break;
        }
        let num = rng.gen_range(0..edges.len());
        let edge = &edges[num];
        vertex = edge.to_vertex()?;
        path.expand(edge)?;
    }

    result.insert_path(c_str!("path"), &path)?;
    Ok(())
});

init_module!(|memgraph: &Memgraph| -> Result<()> {
    memgraph.add_read_procedure(
        get,
        c_str!("get"),
        &[
            define_type!("start", Type::Vertex),
            define_type!("length", Type::Int),
        ],
        &[],
        &[define_type!("path", Type::Path)],
    )?;
    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });
