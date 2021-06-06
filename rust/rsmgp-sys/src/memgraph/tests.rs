use c_str_macro::c_str;
use serial_test::serial;
use std::ptr::null_mut;

use super::*;
use crate::define_type;
use crate::mgp::mock_ffi::*;
use crate::testing::alloc::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_vertices_iterator() {
    mock_mgp_once!(mgp_graph_iter_vertices_context, |_, _| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let value = memgraph.vertices_iter();
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_vertex_by_id() {
    mock_mgp_once!(mgp_graph_get_vertex_by_id_context, |_, _, _| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let value = memgraph.vertex_by_id(0);
        assert!(value.is_err());
    });
}

#[no_mangle]
extern "C" fn dummy_c_func(
    _: *const mgp_list,
    _: *const mgp_graph,
    _: *mut mgp_result,
    _: *mut mgp_memory,
) {
}

#[test]
#[serial]
fn test_add_read_procedure() {
    mock_mgp_once!(mgp_module_add_read_procedure_context, |_, _, _| unsafe {
        alloc_mgp_proc()
    });
    let ctx_any = mgp_type_any_context();
    ctx_any
        .expect()
        .times(3)
        .returning(|| unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_bool_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_number_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_int_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_float_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_string_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_map_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_node_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_relationship_context, || unsafe {
        alloc_mgp_type()
    });
    mock_mgp_once!(mgp_type_path_context, || unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_nullable_context, |_| unsafe { alloc_mgp_type() });
    mock_mgp_once!(mgp_type_list_context, |_| unsafe { alloc_mgp_type() });
    let ctx_add_result = mgp_proc_add_result_context();
    ctx_add_result.expect().times(12).returning(|_, _, _| 1);

    with_dummy!(|memgraph: &Memgraph| {
        assert!(memgraph
            .add_read_procedure(
                dummy_c_func,
                c_str!("dummy_c_func"),
                &[
                    define_type!("any", FieldType::Any),
                    define_type!("bool", FieldType::Bool),
                    define_type!("number", FieldType::Number),
                    define_type!("int", FieldType::Int),
                    define_type!("double", FieldType::Double),
                    define_type!("string", FieldType::String),
                    define_type!("map", FieldType::Map),
                    define_type!("vertex", FieldType::Vertex),
                    define_type!("edge", FieldType::Edge),
                    define_type!("path", FieldType::Path),
                    define_type!("nullable", FieldType::Nullable, FieldType::Any),
                    define_type!("list", FieldType::List, FieldType::Any),
                ],
            )
            .is_ok());
    });
}
