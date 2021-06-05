use serial_test::serial;

use super::*;
use crate::mgp::mock_ffi::*;

#[test]
#[serial]
fn test_memgraph_vertices_iterator() {
    let ctx_1 = mgp_graph_iter_vertices_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let value = memgraph.vertices_iter();
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_memgraph_vertex_by_id() {
    let ctx_1 = mgp_graph_get_vertex_by_id_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let value = memgraph.vertex_by_id(0);
    assert!(value.is_err());
}
