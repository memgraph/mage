// Copyright (c) 2016-2021 Memgraph Ltd. [https://memgraph.com]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use c_str_macro::c_str;
use serial_test::serial;
use std::ffi::CStr;
use std::ptr::null_mut;

use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_make_null_mgp_value() {
    mock_mgp_once!(mgp_value_make_null_context, |_| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let value = MgpValue::make_null(&memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_false_bool_mgp_value() {
    mock_mgp_once!(mgp_value_make_bool_context, |value, _| {
        assert_eq!(value, 0);
        null_mut()
    });

    with_dummy!(|memgraph: &Memgraph| {
        let value = MgpValue::make_bool(false, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_true_bool_mgp_value() {
    mock_mgp_once!(mgp_value_make_bool_context, |value, _| {
        assert_eq!(value, 1);
        null_mut()
    });

    with_dummy!(|memgraph: &Memgraph| {
        let value = MgpValue::make_bool(true, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_int_mgp_value() {
    mock_mgp_once!(mgp_value_make_int_context, |value, _| {
        assert_eq!(value, 100);
        null_mut()
    });

    with_dummy!(|memgraph: &Memgraph| {
        let value = MgpValue::make_int(100, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_double_mgp_value() {
    mock_mgp_once!(mgp_value_make_double_context, |value, _| {
        assert_eq!(value, 0.0);
        null_mut()
    });

    with_dummy!(|memgraph: &Memgraph| {
        let value = MgpValue::make_double(0.0, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_string_mgp_value() {
    mock_mgp_once!(mgp_value_make_string_context, |value, _| unsafe {
        assert_eq!(CStr::from_ptr(value), c_str!("test"));
        null_mut()
    });

    with_dummy!(|memgraph: &Memgraph| {
        let value = MgpValue::make_string(c_str!("test"), &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_list_mgp_value() {
    let mgp_list_size_context = mgp_list_size_context();
    mgp_list_size_context.expect().times(2).returning(|_| 0);
    mock_mgp_once!(mgp_list_make_empty_context, |_, _| { null_mut() });
    mock_mgp_once!(mgp_value_make_list_context, |_| { null_mut() });
    mock_mgp_once!(mgp_list_destroy_context, |_| {});

    with_dummy!(|memgraph: &Memgraph| {
        let list = List::new(null_mut(), &memgraph);
        let value = MgpValue::make_list(&list, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_map_mgp_value() {
    mock_mgp_once!(mgp_map_make_empty_context, |_| { null_mut() });
    mock_mgp_once!(mgp_map_iter_items_context, |_, _| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let map = Map::new(null_mut(), &memgraph);
        let value = MgpValue::make_map(&map, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_vertex_mgp_value() {
    mock_mgp_once!(mgp_vertex_copy_context, |_, _| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let vertex = Vertex::new(null_mut(), &memgraph);
        let value = MgpValue::make_vertex(&vertex, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_edge_mgp_value() {
    mock_mgp_once!(mgp_edge_copy_context, |_, _| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let edge = Edge::new(null_mut(), &memgraph);
        let value = MgpValue::make_edge(&edge, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_make_path_mgp_value() {
    mock_mgp_once!(mgp_path_copy_context, |_, _| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let path = Path::new(null_mut(), &memgraph);
        let value = MgpValue::make_path(&path, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_mgp_value_for_the_right_type() {
    mock_mgp_once!(mgp_value_is_null_context, |_| 1);
    mock_mgp_once!(mgp_value_is_bool_context, |_| 1);
    mock_mgp_once!(mgp_value_is_int_context, |_| 1);
    mock_mgp_once!(mgp_value_is_double_context, |_| 1);
    mock_mgp_once!(mgp_value_is_string_context, |_| 1);
    mock_mgp_once!(mgp_value_is_list_context, |_| 1);
    mock_mgp_once!(mgp_value_is_map_context, |_| 1);
    mock_mgp_once!(mgp_value_is_vertex_context, |_| 1);
    mock_mgp_once!(mgp_value_is_edge_context, |_| 1);
    mock_mgp_once!(mgp_value_is_path_context, |_| 1);

    let value = MgpValue::new(null_mut());
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
    mock_mgp_once!(mgp_value_is_null_context, |_| 0);
    mock_mgp_once!(mgp_value_is_bool_context, |_| 0);
    mock_mgp_once!(mgp_value_is_int_context, |_| 0);
    mock_mgp_once!(mgp_value_is_double_context, |_| 0);
    mock_mgp_once!(mgp_value_is_string_context, |_| 0);
    mock_mgp_once!(mgp_value_is_list_context, |_| 0);
    mock_mgp_once!(mgp_value_is_map_context, |_| 0);
    mock_mgp_once!(mgp_value_is_vertex_context, |_| 0);
    mock_mgp_once!(mgp_value_is_edge_context, |_| 0);
    mock_mgp_once!(mgp_value_is_path_context, |_| 0);

    let value = MgpValue::new(null_mut());
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
    mock_mgp_once!(mgp_value_make_null_context, |_| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let value = Value::Null;
        let mgp_value = value.to_mgp_value(&memgraph);
        assert!(mgp_value.is_err());
    });
}
