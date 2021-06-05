use serial_test::serial;

use super::*;
use crate::mgp::mock_ffi::*;

#[test]
#[serial]
fn test_mgp_copy() {
    let ctx_1 = mgp_path_copy_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    unsafe {
        let path = Path::mgp_copy(std::ptr::null_mut(), &memgraph);
        assert!(path.is_err());
    }
}

#[test]
#[serial]
fn test_mgp_ptr() {
    let memgraph = Memgraph::new_default();

    let path = Path::new(std::ptr::null_mut(), &memgraph);
    let ptr = path.mgp_ptr();
    assert!(ptr.is_null());
}

#[test]
#[serial]
fn test_size() {
    let ctx_1 = mgp_path_size_context();
    ctx_1.expect().times(1).returning(|_| 0);

    let memgraph = Memgraph::new_default();
    let path = Path::new(std::ptr::null_mut(), &memgraph);
    assert_eq!(path.size(), 0);
}

#[test]
#[serial]
fn test_make_with_start() {
    let ctx_1 = mgp_path_make_with_start_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let vertex = Vertex::new(std::ptr::null_mut(), &memgraph);
    assert!(Path::make_with_start(&vertex, &memgraph).is_err());
}

#[test]
#[serial]
fn test_expand() {
    let ctx_1 = mgp_path_expand_context();
    ctx_1.expect().times(1).returning(|_, _| 0);

    let memgraph = Memgraph::new_default();
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);
    let path = Path::new(std::ptr::null_mut(), &memgraph);
    assert!(path.expand(&edge).is_err());
}

#[test]
#[serial]
fn test_vertex_at() {
    let ctx_1 = mgp_path_vertex_at_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let path = Path::new(std::ptr::null_mut(), &memgraph);
    assert!(path.vertex_at(0).is_err());
}

#[test]
#[serial]
fn test_edge_at() {
    let ctx_1 = mgp_path_edge_at_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let path = Path::new(std::ptr::null_mut(), &memgraph);
    assert!(path.edge_at(0).is_err());
}
