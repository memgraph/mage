use c_str_macro::c_str;
use serial_test::serial;
use std::ptr::null_mut;

use super::*;
use crate::mgp::mock_ffi::*;
use crate::testing::alloc::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_make_empty() {
    mock_mgp_once!(mgp_map_make_empty_context, |_| null_mut());

    with_dummy!(|memgraph: &Memgraph| {
        let value = Map::make_empty(&memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_mgp_copy() {
    mock_mgp_once!(mgp_map_make_empty_context, |_| unsafe { alloc_mgp_map() });
    let ctx_iter_items = mgp_map_iter_items_context();
    ctx_iter_items
        .expect()
        .times(1)
        .returning(|_, _| unsafe { alloc_mgp_map_items_iterator() });
    mock_mgp_once!(mgp_map_items_iterator_get_context, |_| { null_mut() });
    mock_mgp_once!(mgp_map_destroy_context, |_| {});
    mock_mgp_once!(mgp_map_items_iterator_destroy_context, |_| {});

    with_dummy!(|memgraph: &Memgraph| {
        unsafe {
            let value = Map::mgp_copy(null_mut(), &memgraph);
            assert!(value.is_ok());
        }
    });
}

#[test]
#[serial]
fn test_insert() {
    mock_mgp_once!(mgp_value_make_null_context, |_| unsafe {
        alloc_mgp_value()
    });
    mock_mgp_once!(mgp_map_insert_context, |_, _, _| 0);
    mock_mgp_once!(mgp_value_destroy_context, |_| {});

    with_dummy!(Map, |map: &Map| {
        let value = Value::Null;
        assert!(map.insert(c_str!("key"), &value).is_err());
    });
}

#[test]
#[serial]
fn test_size() {
    mock_mgp_once!(mgp_map_size_context, |_| 0);

    with_dummy!(Map, |map: &Map| {
        let value = map.size();
        assert_eq!(value, 0);
    });
}

#[test]
#[serial]
fn test_at() {
    mock_mgp_once!(mgp_map_at_context, |_, _| null_mut());

    with_dummy!(Map, |map: &Map| {
        let value = map.at(c_str!("key"));
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_empty_map_iter() {
    mock_mgp_once!(mgp_map_iter_items_context, |_, _| null_mut());

    with_dummy!(Map, |map: &Map| {
        let iter = map.iter();
        assert!(iter.is_err());
    });
}
