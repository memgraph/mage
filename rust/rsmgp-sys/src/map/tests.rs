use c_str_macro::c_str;
use serial_test::serial;

use super::*;
use crate::mgp::mock_ffi::*;

#[test]
#[serial]
fn test_mgp_copy() {
    let ctx_1 = mgp_map_iter_items_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let ctx_2 = mgp_map_make_empty_context();
    ctx_2.expect().times(1).returning(|_| std::ptr::null_mut());

    let ctx_3 = mgp_map_destroy_context();
    ctx_3.expect().times(1).returning(|_| ());

    let memgraph = Memgraph::new_default();
    unsafe {
        let value = Map::mgp_copy(std::ptr::null_mut(), &memgraph);
        assert!(value.is_err());
    }
}

#[test]
#[serial]
fn test_make_empty() {
    let ctx_1 = mgp_map_make_empty_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let value = Map::make_empty(&memgraph);
    assert!(value.is_err());
}

// TODO(gitbuda): Figure out how + test properly map mgp_copy because it's quite complex.
// TODO(gitbuda): Figure out how + test properly map insert.

#[test]
#[serial]
fn test_size() {
    let ctx_1 = mgp_map_size_context();
    ctx_1.expect().times(1).returning(|_| 0);

    let memgraph = Memgraph::new_default();
    let map = Map::new(std::ptr::null_mut(), &memgraph);
    let value = map.size();
    assert_eq!(value, 0);
}

#[test]
#[serial]
fn test_at() {
    let ctx_1 = mgp_map_at_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let map = Map::new(std::ptr::null_mut(), &memgraph);
    let value = map.at(c_str!("key"));
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_empty_map_iter() {
    let ctx_1 = mgp_map_iter_items_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let map = Map::new(std::ptr::null_mut(), &memgraph);
    let iter = map.iter();
    assert!(iter.is_err());
}
