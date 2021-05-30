use c_str_macro::c_str;
use serial_test::serial;

use super::*;
use crate::mgp::mock_ffi::*;

#[test]
#[serial]
fn test_mgp_copy() {
    let ctx_1 = mgp_edge_copy_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph {
        ..Default::default()
    };
    unsafe {
        let value = Edge::mgp_copy(std::ptr::null_mut(), &memgraph);
        assert!(value.is_err());
    }
}

#[test]
#[serial]
fn test_id() {
    let ctx_1 = mgp_edge_get_id_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_| mgp_edge_id { as_int: 0 });

    let memgraph = Memgraph {
        ..Default::default()
    };
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);

    let value = edge.id();
    assert_eq!(value, 0);
}

#[test]
#[serial]
fn test_edge_type() {
    let edge_type = CString::new("type").unwrap();
    let ctx_1 = mgp_edge_get_type_context();
    ctx_1.expect().times(1).returning(move |_| mgp_edge_type {
        name: edge_type.as_ptr(),
    });

    let memgraph = Memgraph {
        ..Default::default()
    };
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);

    let value = edge.edge_type().unwrap();
    assert_eq!(value, CString::new("type").unwrap());
}

#[test]
#[serial]
fn test_from_vertex() {
    let ctx_1 = mgp_edge_get_from_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());
    let ctx_2 = mgp_vertex_copy_context();
    ctx_2
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph {
        ..Default::default()
    };
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);

    let value = edge.from_vertex();
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_to_vertex() {
    let ctx_1 = mgp_edge_get_to_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());
    let ctx_2 = mgp_vertex_copy_context();
    ctx_2
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph {
        ..Default::default()
    };
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);

    let value = edge.to_vertex();
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_property() {
    let ctx_1 = mgp_edge_get_property_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _, _| std::ptr::null_mut());

    let memgraph = Memgraph {
        ..Default::default()
    };
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);

    let value = edge.property(c_str!("prop"));
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_properties_iterator() {
    let ctx_1 = mgp_edge_iter_properties_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph {
        ..Default::default()
    };
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);

    let value = edge.properties();
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_edges_iterator() {
    let ctx_1 = mgp_edges_iterator_get_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());
    let ctx_2 = mgp_edges_iterator_next_context();
    ctx_2.expect().times(1).returning(|_| std::ptr::null_mut());

    let memgraph = Memgraph {
        ..Default::default()
    };
    let mut iterator = EdgesIterator::new(std::ptr::null_mut(), &memgraph);

    let value_1 = iterator.next();
    assert!(value_1.is_none());

    let value_2 = iterator.next();
    assert!(value_2.is_none());
}
