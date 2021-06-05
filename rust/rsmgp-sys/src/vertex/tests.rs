use c_str_macro::c_str;
use serial_test::serial;
use std::ffi::{CStr, CString};

use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;

#[test]
#[serial]
fn test_vertex_id() {
    let ctx_1 = mgp_vertex_get_id_context();
    ctx_1.expect().times(1).returning(|_| {
        return mgp_vertex_id { as_int: 72 };
    });

    let memgraph = Memgraph::new_default();
    let vertex = Vertex::new(std::ptr::null_mut(), &memgraph);
    assert_eq!(vertex.id(), 72);
}

#[test]
#[serial]
fn test_vertex_labels_count() {
    let ctx_1 = mgp_vertex_labels_count_context();
    ctx_1.expect().times(1).returning(|_| 2);

    let memgraph = Memgraph::new_default();
    let vertex = Vertex::new(std::ptr::null_mut(), &memgraph);
    assert_eq!(vertex.labels_count(), 2);
}

#[test]
#[serial]
fn test_vertex_has_label() {
    let ctx_1 = mgp_vertex_has_label_context();
    ctx_1.expect().times(1).returning(|vertex, label| unsafe {
        assert_eq!(vertex, std::ptr::null_mut());
        assert_eq!(CStr::from_ptr(label.name), c_str!("labela"));
        1
    });

    let memgraph = Memgraph::new_default();
    let vertex = Vertex::new(std::ptr::null_mut(), &memgraph);
    assert_eq!(vertex.has_label(c_str!("labela")), true);
}

#[test]
#[serial]
fn test_vertex_label_at() {
    let ctx_1 = mgp_vertex_label_at_context();
    let test_label = CString::new("test");
    ctx_1.expect().times(1).returning(move |vertex, _| {
        assert_eq!(vertex, std::ptr::null_mut());
        return mgp_label {
            name: test_label.as_ref().unwrap().as_ptr(),
        };
    });

    let memgraph = Memgraph::new_default();
    let vertex = Vertex::new(std::ptr::null_mut(), &memgraph);
    assert_eq!(vertex.label_at(5).unwrap(), CString::new("test").unwrap());
}

#[test]
#[serial]
fn test_vertex_property() {
    let ctx_1 = mgp_vertex_get_property_context();
    ctx_1
        .expect()
        .times(1)
        .returning(move |vertex, prop_name, memory| {
            assert_eq!(vertex, std::ptr::null_mut());
            assert_eq!(prop_name, c_str!("test").as_ptr());
            assert_eq!(memory, std::ptr::null_mut());
            std::ptr::null_mut()
        });

    let memgraph = Memgraph::new_default();
    let vertex = Vertex::new(std::ptr::null_mut(), &memgraph);
    assert_eq!(
        vertex.property(c_str!("test")).err().unwrap(),
        MgpError::UnableToGetVertexProperty
    );
}
