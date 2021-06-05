use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;
use c_str_macro::c_str;
use serial_test::serial;

#[test]
#[serial]
fn test_make_null_mgp_value() {
    let ctx_1 = mgp_value_make_null_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let value = MgpValue::make_null(&memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_false_bool_mgp_value() {
    let ctx_1 = mgp_value_make_bool_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 0);
        std::ptr::null_mut()
    });

    let memgraph = Memgraph::new_default();
    let value = MgpValue::make_bool(false, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_true_bool_mgp_value() {
    let ctx_1 = mgp_value_make_bool_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 1);
        std::ptr::null_mut()
    });

    let memgraph = Memgraph::new_default();
    let value = MgpValue::make_bool(true, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_int_mgp_value() {
    let ctx_1 = mgp_value_make_int_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 100);
        std::ptr::null_mut()
    });

    let memgraph = Memgraph::new_default();
    let value = MgpValue::make_int(100, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_double_mgp_value() {
    let ctx_1 = mgp_value_make_double_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 0.0);
        std::ptr::null_mut()
    });

    let memgraph = Memgraph::new_default();
    let value = MgpValue::make_double(0.0, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_string_mgp_value() {
    use std::ffi::CStr;

    let ctx_1 = mgp_value_make_string_context();
    ctx_1.expect().times(1).returning(|value, _| unsafe {
        assert_eq!(CStr::from_ptr(value), c_str!("test"));
        std::ptr::null_mut()
    });

    let memgraph = Memgraph::new_default();
    let value = MgpValue::make_string(c_str!("test"), &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_list_mgp_value() {
    let ctx_1 = mgp_list_size_context();
    ctx_1.expect().times(2).returning(|_| 0);

    let ctx_2 = mgp_list_make_empty_context();
    ctx_2
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let ctx_3 = mgp_value_make_list_context();
    ctx_3.expect().times(1).returning(|_| std::ptr::null_mut());

    let ctx_4 = mgp_list_destroy_context();
    ctx_4.expect().times(1).returning(|_| {});

    let memgraph = Memgraph::new_default();
    let list = List::new(std::ptr::null_mut(), &memgraph);
    let value = MgpValue::make_list(&list, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_map_mgp_value() {
    let ctx_1 = mgp_map_make_empty_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());

    let ctx_2 = mgp_map_iter_items_context();
    ctx_2
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let map = Map::new(std::ptr::null_mut(), &memgraph);
    let value = MgpValue::make_map(&map, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_vertex_mgp_value() {
    let ctx_1 = mgp_vertex_copy_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let vertex = Vertex::new(std::ptr::null_mut(), &memgraph);
    let value = MgpValue::make_vertex(&vertex, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_edge_mgp_value() {
    let ctx_1 = mgp_edge_copy_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let edge = Edge::new(std::ptr::null_mut(), &memgraph);
    let value = MgpValue::make_edge(&edge, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_path_mgp_value() {
    let ctx_1 = mgp_path_copy_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let path = Path::new(std::ptr::null_mut(), &memgraph);
    let value = MgpValue::make_path(&path, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_mgp_value_for_the_right_type() {
    let ctx_is_null = mgp_value_is_null_context();
    ctx_is_null.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_bool = mgp_value_is_bool_context();
    ctx_is_bool.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_int = mgp_value_is_int_context();
    ctx_is_int.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_double = mgp_value_is_double_context();
    ctx_is_double.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_string = mgp_value_is_string_context();
    ctx_is_string.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_list = mgp_value_is_list_context();
    ctx_is_list.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_map = mgp_value_is_map_context();
    ctx_is_map.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_vertex = mgp_value_is_vertex_context();
    ctx_is_vertex.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_edge = mgp_value_is_edge_context();
    ctx_is_edge.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let ctx_is_path = mgp_value_is_path_context();
    ctx_is_path.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });

    let value = MgpValue {
        ptr: std::ptr::null_mut(),
    };

    assert!(value.is_null());
    assert!(value.is_bool());
    assert!(value.is_int());
    assert!(value.is_double());
    assert!(value.is_string());
    assert!(value.is_list());
    assert!(value.is_map());
    assert!(value.is_vertex());
    assert!(value.is_edge());
    assert!(value.is_path());
}

#[test]
#[serial]
fn test_mgp_value_for_the_wrong_type() {
    let ctx_is_null = mgp_value_is_null_context();
    ctx_is_null.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_bool = mgp_value_is_bool_context();
    ctx_is_bool.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_int = mgp_value_is_int_context();
    ctx_is_int.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_double = mgp_value_is_double_context();
    ctx_is_double.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_string = mgp_value_is_string_context();
    ctx_is_string.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_list = mgp_value_is_list_context();
    ctx_is_list.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_map = mgp_value_is_map_context();
    ctx_is_map.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_vertex = mgp_value_is_vertex_context();
    ctx_is_vertex.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_edge = mgp_value_is_edge_context();
    ctx_is_edge.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let ctx_is_path = mgp_value_is_path_context();
    ctx_is_path.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });

    let value = MgpValue {
        ptr: std::ptr::null_mut(),
    };

    assert!(!value.is_null());
    assert!(!value.is_bool());
    assert!(!value.is_int());
    assert!(!value.is_double());
    assert!(!value.is_string());
    assert!(!value.is_list());
    assert!(!value.is_map());
    assert!(!value.is_vertex());
    assert!(!value.is_edge());
    assert!(!value.is_path());
}

#[test]
#[serial]
fn test_to_mgp_value() {
    let ctx_1 = mgp_value_make_null_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let value = Value::Null;
    let mgp_value = value.to_mgp_value(&memgraph);

    assert!(mgp_value.is_err());
}
