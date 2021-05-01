use crate::mgp::mock_ffi::*;
use super::*;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

#[test]
fn test_make_true_bool_value() {
    let ctx_1 = mgp_value_make_bool_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 1);
        std::ptr::null_mut()
    });
    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|result, msg| unsafe {
        print_type_of(&result);
        println!("{}", std::ffi::CStr::from_ptr(msg).to_str().unwrap());
        0
    });

    let int_value = make_bool_value(true, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(int_value.is_err());
}

#[test]
fn test_make_int_value() {
    let ctx_1 = mgp_value_make_int_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 0);
        std::ptr::null_mut()
    });
    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|result, msg| unsafe {
        print_type_of(&result);
        println!("{}", std::ffi::CStr::from_ptr(msg).to_str().unwrap());
        0
    });

    let int_value = make_int_value(0, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(int_value.is_err());
}

#[test]
fn test_make_string_value() {
    use std::ffi::{CStr};

    let ctx_1 = mgp_value_make_string_context();
    ctx_1.expect().times(1).returning(|value, _| unsafe {
        let string = CStr::from_bytes_with_nul(b"test\0").expect("CStr::new failed");
        let received = CStr::from_ptr(value);
        assert_eq!(string, received);
        std::ptr::null_mut()
    });

    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|result, msg| unsafe {
        print_type_of(&result);
        println!("{}", std::ffi::CStr::from_ptr(msg).to_str().unwrap());
        0
    });

    let string = CStr::from_bytes_with_nul(b"test\0").expect("CStr::new failed");
    let value = make_string_value(string, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}

#[test]
fn test_make_double_value() {
    let ctx_1 = mgp_value_make_double_context();
    ctx_1.expect().times(1).returning(|value, _| {
        assert_eq!(value, 0.0);
        std::ptr::null_mut()
    });
    let ctx_2 = mgp_result_set_error_msg_context();
    ctx_2.expect().times(1).returning(|result, msg| unsafe {
        print_type_of(&result);
        println!("{}", std::ffi::CStr::from_ptr(msg).to_str().unwrap());
        0
    });

    let value = make_double_value(0.0, std::ptr::null_mut(), std::ptr::null_mut());
    assert!(value.is_err());
}
