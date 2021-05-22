use super::*;
use crate::mgp::mock_ffi::*;
use serial_test::serial;

#[test]
#[serial]
fn test_make_null_mgp_value() {
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
fn test_make_false_bool_mgp_value() {
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
fn test_make_true_bool_mgp_value() {
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
fn test_make_int_mgp_value() {
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
fn test_make_string_mgp_value() {
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
fn test_make_double_mgp_value() {
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
    let ctx_is_string = mgp_value_is_string_context();
    ctx_is_string.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });
    let ctx_is_double = mgp_value_is_double_context();
    ctx_is_double.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        1
    });
    let value = MgpValue {
        value: std::ptr::null_mut(),
    };
    assert!(value.is_null());
    assert!(value.is_bool());
    assert!(value.is_int());
    assert!(value.is_string());
    assert!(value.is_double());
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
    let ctx_is_string = mgp_value_is_string_context();
    ctx_is_string.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });
    let ctx_is_double = mgp_value_is_double_context();
    ctx_is_double.expect().times(1).returning(|value| {
        assert_eq!(value, std::ptr::null_mut());
        0
    });
    let value = MgpValue {
        value: std::ptr::null_mut(),
    };
    assert!(!value.is_null());
    assert!(!value.is_bool());
    assert!(!value.is_int());
    assert!(!value.is_string());
    assert!(!value.is_double());
}

#[test]
#[serial]
fn test_to_result_mgp_value() {
    let ctx_1 = mgp_value_make_null_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|_, msg| unsafe {
        assert_eq!(CStr::from_ptr(msg), c_str!("Unable to allocate null."));
        0
    });

    let value = Value::Null;
    let mgp_value = value.to_result_mgp_value(std::ptr::null_mut(), std::ptr::null_mut());

    assert!(mgp_value.is_err());
}
