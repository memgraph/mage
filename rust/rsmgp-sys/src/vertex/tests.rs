use super::*;
use crate::context::Memgraph;
use crate::mgp::mock_ffi::*;
use serial_test::serial;
use std::ffi::{CStr, CString};

#[test]
#[serial]
fn test_vertex_id() {
    let ctx_1 = mgp_vertex_get_id_context();
    ctx_1.expect().times(1).returning(|_| {
        return mgp_vertex_id { as_int: 72 };
    });

    let ctx_mg = Memgraph {
        ..Default::default()
    };

    let vertex = Vertex {
        ptr: std::ptr::null_mut(),
        context: ctx_mg,
    };
    assert_eq!(vertex.id(), 72);
}

#[test]
#[serial]
fn test_vertex_labels_count() {
    let ctx_1 = mgp_vertex_labels_count_context();
    ctx_1.expect().times(1).returning(|_| 2);

    let ctx_mg = Memgraph {
        ..Default::default()
    };

    let vertex = Vertex {
        ptr: std::ptr::null_mut(),
        context: ctx_mg,
    };
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

    let ctx_mg = Memgraph {
        ..Default::default()
    };

    let vertex = Vertex {
        ptr: std::ptr::null_mut(),
        context: ctx_mg,
    };
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

    let ctx_mg = Memgraph {
        ..Default::default()
    };

    let vertex = Vertex {
        ptr: std::ptr::null_mut(),
        context: ctx_mg,
    };
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

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(
            CStr::from_ptr(msg),
            c_str!("Unable to allocate vertex property.")
        );
        0
    });

    let ctx_mg = Memgraph {
        ..Default::default()
    };

    let vertex = Vertex {
        ptr: std::ptr::null_mut(),
        context: ctx_mg,
    };
    assert_eq!(
        vertex.property(c_str!("test")).err().unwrap(),
        MgpError::UnableToReturnVertexPropertyValueAllocationError
    );
}

#[test]
#[serial]
fn test_make_graph_vertices_iterator() {
    let ctx_1 = mgp_graph_iter_vertices_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let ctx_mg = Memgraph {
        ..Default::default()
    };

    let value = make_graph_vertices_iterator(&ctx_mg);
    assert!(value.is_err());
}
