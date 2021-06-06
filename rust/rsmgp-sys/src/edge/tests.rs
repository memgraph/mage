use c_str_macro::c_str;
use serial_test::serial;
use std::ptr::null_mut;

use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_mgp_copy() {
    mock_mgp_once!(mgp_edge_copy_context, |_, _| null_mut());

    with_dummy!(|memgraph: &Memgraph| {
        unsafe {
            let value = Edge::mgp_copy(null_mut(), &memgraph);
            assert!(value.is_err());
        }
    });
}

#[test]
#[serial]
fn test_id() {
    mock_mgp_once!(mgp_edge_get_id_context, |_| mgp_edge_id { as_int: 0 });

    with_dummy!(Edge, |edge: &Edge| {
        assert_eq!(edge.id(), 0);
    });
}

#[test]
#[serial]
fn test_edge_type() {
    let edge_type = CString::new("type").unwrap();
    mock_mgp_once!(mgp_edge_get_type_context, move |_| mgp_edge_type {
        name: edge_type.as_ptr(),
    });

    with_dummy!(Edge, |edge: &Edge| {
        let value = edge.edge_type().unwrap();
        assert_eq!(value, CString::new("type").unwrap());
    });
}

#[test]
#[serial]
fn test_from_vertex() {
    mock_mgp_once!(mgp_edge_get_from_context, |_| null_mut());
    mock_mgp_once!(mgp_vertex_copy_context, |_, _| null_mut());

    with_dummy!(Edge, |edge: &Edge| {
        assert!(edge.from_vertex().is_err());
    });
}

#[test]
#[serial]
fn test_to_vertex() {
    mock_mgp_once!(mgp_edge_get_to_context, |_| null_mut());
    mock_mgp_once!(mgp_vertex_copy_context, |_, _| null_mut());

    with_dummy!(Edge, |edge: &Edge| {
        assert!(edge.to_vertex().is_err());
    });
}

#[test]
#[serial]
fn test_property() {
    mock_mgp_once!(mgp_edge_get_property_context, |_, _, _| null_mut());

    with_dummy!(Edge, |edge: &Edge| {
        assert!(edge.property(c_str!("prop")).is_err());
    });
}

#[test]
#[serial]
fn test_properties_iterator() {
    mock_mgp_once!(mgp_edge_iter_properties_context, |_, _| null_mut());

    with_dummy!(Edge, |edge: &Edge| {
        assert!(edge.properties().is_err());
    });
}

#[test]
#[serial]
fn test_edges_iterator() {
    mock_mgp_once!(mgp_edges_iterator_get_context, |_| null_mut());
    mock_mgp_once!(mgp_edges_iterator_next_context, |_| null_mut());

    with_dummy!(|memgraph: &Memgraph| {
        let mut iterator = EdgesIterator::new(null_mut(), &memgraph);
        assert!(iterator.next().is_none());
        assert!(iterator.next().is_none());
    });
}
