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

use serial_test::serial;
use std::ptr::null_mut;

use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;
use crate::testing::alloc::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_new_record() {
    mock_mgp_once!(mgp_result_new_record_context, |_| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let result_record = MgpResultRecord::new(&memgraph);
        assert!(result_record.is_err());
    });
}

#[test]
#[serial]
fn test_mgp_copy() {
    mock_mgp_once!(mgp_list_size_context, |_| 1);
    mock_mgp_once!(mgp_list_make_empty_context, |_, _| unsafe {
        alloc_mgp_list()
    });
    mock_mgp_once!(mgp_list_at_context, |_, _| unsafe { alloc_mgp_value() });
    mock_mgp_once!(mgp_list_append_context, |_, _| 0);
    mock_mgp_once!(mgp_list_destroy_context, |_| {});

    with_dummy!(|memgraph: &Memgraph| {
        unsafe {
            let value = List::mgp_copy(std::ptr::null_mut(), &memgraph);
            assert!(value.is_err());
        }
    });
}

#[test]
#[serial]
fn test_make_empty() {
    mock_mgp_once!(mgp_list_make_empty_context, |_, _| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let value = List::make_empty(0, &memgraph);
        assert!(value.is_err());
    });
}

#[test]
#[serial]
fn test_append() {
    mock_mgp_once!(mgp_list_append_context, |_, _| 0);
    mock_mgp_once!(mgp_value_make_null_context, |_| unsafe {
        alloc_mgp_value()
    });
    mock_mgp_once!(mgp_value_destroy_context, |_| {});

    with_dummy!(List, |list: &List| {
        assert!(list.append(&Value::Null).is_err());
    });
}

#[test]
#[serial]
fn test_append_extend() {
    mock_mgp_once!(mgp_list_append_extend_context, |_, _| 0);
    mock_mgp_once!(mgp_value_make_null_context, |_| unsafe {
        alloc_mgp_value()
    });
    mock_mgp_once!(mgp_value_destroy_context, |_| {});

    with_dummy!(List, |list: &List| {
        assert!(list.append_extend(&Value::Null).is_err());
    });
}

#[test]
#[serial]
fn test_size() {
    mock_mgp_once!(mgp_list_size_context, |_| 0);

    with_dummy!(List, |list: &List| {
        assert_eq!(list.size(), 0);
    });
}

#[test]
#[serial]
fn test_capacity() {
    mock_mgp_once!(mgp_list_capacity_context, |_| 0);

    with_dummy!(List, |list: &List| {
        assert_eq!(list.capacity(), 0);
    });
}

#[test]
#[serial]
fn test_value_at() {
    mock_mgp_once!(mgp_list_at_context, |_, _| null_mut());

    with_dummy!(List, |list: &List| {
        assert!(list.value_at(0).is_err());
    });
}

#[test]
#[serial]
fn test_empty_list_iter() {
    mock_mgp_once!(mgp_list_size_context, |_| 0);

    with_dummy!(List, |list: &List| {
        let iter = list.iter();
        assert!(iter.is_ok());
        let value = iter.unwrap().next();
        assert!(value.is_none());
    });
}
