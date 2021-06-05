use serial_test::serial;

use super::*;
use crate::mgp::mock_ffi::*;
use crate::testing::alloc::*;

#[test]
#[serial]
fn test_mgp_copy() {
    let ctx_1 = mgp_list_size_context();
    ctx_1.expect().times(1).returning(|_| 1);

    let ctx_2 = mgp_list_make_empty_context();
    ctx_2
        .expect()
        .times(1)
        .returning(|_, _| unsafe { alloc_mgp_list() });

    let ctx_3 = mgp_list_at_context();
    ctx_3
        .expect()
        .times(1)
        .returning(|_, _| unsafe { alloc_mgp_value() });

    let ctx_4 = mgp_list_append_context();
    ctx_4.expect().times(1).returning(|_, _| 0);

    let ctx_5 = mgp_list_destroy_context();
    ctx_5.expect().times(1).returning(|_| ());

    let memgraph = Memgraph::new_default();
    unsafe {
        let value = List::mgp_copy(std::ptr::null_mut(), &memgraph);
        assert!(value.is_err());
    }
}

#[test]
#[serial]
fn test_make_empty() {
    let ctx_1 = mgp_list_make_empty_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let value = List::make_empty(0, &memgraph);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_append() {
    let ctx_1 = mgp_list_append_context();
    ctx_1.expect().times(1).returning(|_, _| 0);

    let ctx_2 = mgp_value_make_null_context();
    ctx_2
        .expect()
        .times(1)
        .returning(|_| unsafe { alloc_mgp_value() });

    let ctx_3 = mgp_value_destroy_context();
    ctx_3.expect().times(1).returning(|_| ());

    let memgraph = Memgraph::new_default();
    let value = Value::Null;
    let list = List::new(std::ptr::null_mut(), &memgraph);
    let result = list.append(&value);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_append_extend() {
    let ctx_1 = mgp_list_append_extend_context();
    ctx_1.expect().times(1).returning(|_, _| 0);

    let ctx_2 = mgp_value_make_null_context();
    ctx_2
        .expect()
        .times(1)
        .returning(|_| unsafe { alloc_mgp_value() });

    let ctx_3 = mgp_value_destroy_context();
    ctx_3.expect().times(1).returning(|_| ());

    let memgraph = Memgraph::new_default();
    let value = Value::Null;
    let list = List::new(std::ptr::null_mut(), &memgraph);
    let result = list.append_extend(&value);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_size() {
    let ctx_1 = mgp_list_size_context();
    ctx_1.expect().times(1).returning(|_| 0);

    let memgraph = Memgraph::new_default();
    let list = List::new(std::ptr::null_mut(), &memgraph);
    let value = list.size();
    assert_eq!(value, 0);
}

#[test]
#[serial]
fn test_capacity() {
    let ctx_1 = mgp_list_capacity_context();
    ctx_1.expect().times(1).returning(|_| 0);

    let memgraph = Memgraph::new_default();
    let list = List::new(std::ptr::null_mut(), &memgraph);
    let value = list.capacity();
    assert_eq!(value, 0);
}

#[test]
#[serial]
fn test_value_at() {
    let ctx_1 = mgp_list_at_context();
    ctx_1
        .expect()
        .times(1)
        .returning(|_, _| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let list = List::new(std::ptr::null_mut(), &memgraph);
    let value = list.value_at(0);
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_empty_list_iter() {
    let ctx_1 = mgp_list_size_context();
    ctx_1.expect().times(1).returning(|_| 0);

    let memgraph = Memgraph::new_default();
    let list = List::new(std::ptr::null_mut(), &memgraph);
    let iter = list.iter();
    assert!(iter.is_ok());

    let value = iter.unwrap().next();
    assert!(value.is_none());
}
