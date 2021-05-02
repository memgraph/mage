use super::*;
use crate::mgp::mock_ffi::*;
use serial_test::serial;

#[test]
#[serial]
fn test_make_null_value() {
    let ctx_1 = mgp_value_make_null_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(CStr::from_ptr(msg), c_str!("Unable to allocate null."));
        0
    });

    let value = make_null_value(std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_false_bool_value() {
    let ctx_1 = mgp_value_make_bool_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 0);
        std::ptr::null_mut()
    });

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(CStr::from_ptr(msg), c_str!("Unable to allocate bool."));
        0
    });

    let value = make_bool_value(false, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_true_bool_value() {
    let ctx_1 = mgp_value_make_bool_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 1);
        std::ptr::null_mut()
    });

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(CStr::from_ptr(msg), c_str!("Unable to allocate bool."));
        0
    });

    let value = make_bool_value(true, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_int_value() {
    let ctx_1 = mgp_value_make_int_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 100);
        std::ptr::null_mut()
    });

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(CStr::from_ptr(msg), c_str!("Unable to allocate integer."));
        0
    });

    let value = make_int_value(100, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_string_value() {
    use std::ffi::CStr;

    let ctx_1 = mgp_value_make_string_context();
    ctx_1.expect().times(1).returning(|value, _| unsafe {
        assert_eq!(CStr::from_ptr(value), c_str!("test"));
        std::ptr::null_mut()
    });

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(CStr::from_ptr(msg), c_str!("Unable to allocate string."));
        0
    });

    let value = make_string_value(c_str!("test"), std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}

#[test]
#[serial]
fn test_make_double_value() {
    let ctx_1 = mgp_value_make_double_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 0.0);
        std::ptr::null_mut()
    });

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(CStr::from_ptr(msg), c_str!("Unable to allocate double."));
        0
    });

    let value = make_double_value(0.0, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}
