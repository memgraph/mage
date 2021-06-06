use c_str_macro::c_str;
use serial_test::serial;
use std::ffi::{CStr, CString};
use std::ptr::null_mut;

use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_id() {
    mock_mgp_once!(mgp_vertex_get_id_context, |_| {
        mgp_vertex_id { as_int: 72 }
    });

    with_dummy!(Vertex, |vertex: &Vertex| {
        assert_eq!(vertex.id(), 72);
    });
}

#[test]
#[serial]
fn test_labels_count() {
    mock_mgp_once!(mgp_vertex_labels_count_context, |_| 2);

    with_dummy!(Vertex, |vertex: &Vertex| {
        assert_eq!(vertex.labels_count(), 2);
    });
}

#[test]
#[serial]
fn test_has_label() {
    mock_mgp_once!(mgp_vertex_has_label_context, |vertex, label| unsafe {
        assert_eq!(vertex, null_mut());
        assert_eq!(CStr::from_ptr(label.name), c_str!("labela"));
        1
    });

    with_dummy!(Vertex, |vertex: &Vertex| {
        assert_eq!(vertex.has_label(c_str!("labela")), true);
    });
}

#[test]
#[serial]
fn test_label_at() {
    let test_label = CString::new("test");
    mock_mgp_once!(mgp_vertex_label_at_context, move |vertex, _| {
        assert_eq!(vertex, null_mut());
        mgp_label {
            name: test_label.as_ref().unwrap().as_ptr(),
        }
    });

    with_dummy!(Vertex, |vertex: &Vertex| {
        assert_eq!(vertex.label_at(5).unwrap(), CString::new("test").unwrap());
    });
}

#[test]
#[serial]
fn test_property() {
    mock_mgp_once!(
        mgp_vertex_get_property_context,
        move |vertex, prop_name, memory| {
            assert_eq!(vertex, null_mut());
            assert_eq!(prop_name, c_str!("test").as_ptr());
            assert_eq!(memory, null_mut());
            null_mut()
        }
    );

    with_dummy!(Vertex, |vertex: &Vertex| {
        assert_eq!(
            vertex.property(c_str!("test")).err().unwrap(),
            MgpError::UnableToGetVertexProperty
        );
    });
}

#[test]
#[serial]
fn test_properties() {
    mock_mgp_once!(mgp_vertex_iter_properties_context, |_, _| { null_mut() });

    with_dummy!(Vertex, |vertex: &Vertex| {
        let iter = vertex.properties();
        assert!(iter.is_err());
    });
}

#[test]
#[serial]
fn test_in_edges() {
    mock_mgp_once!(mgp_vertex_iter_in_edges_context, |_, _| { null_mut() });

    with_dummy!(Vertex, |vertex: &Vertex| {
        let iter = vertex.in_edges();
        assert!(iter.is_err());
    });
}

#[test]
#[serial]
fn test_out_edges() {
    mock_mgp_once!(mgp_vertex_iter_out_edges_context, |_, _| { null_mut() });

    with_dummy!(Vertex, |vertex: &Vertex| {
        let iter = vertex.out_edges();
        assert!(iter.is_err());
    });
}
