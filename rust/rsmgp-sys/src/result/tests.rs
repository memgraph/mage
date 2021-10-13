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
use std::ptr::null_mut;

use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;
use crate::testing::alloc::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_create_record() {
    mock_mgp_once!(mgp_result_new_record_context, |_, _| {
        mgp_error::MGP_ERROR_UNABLE_TO_ALLOCATE
    });

    with_dummy!(|memgraph: &Memgraph| {
        let result_record = ResultRecord::create(&memgraph);
        assert!(result_record.is_err());
    });
}

macro_rules! mock_mgp_value_make_with_mem {
    ($c_func_name:ident) => {
        mock_mgp_once!($c_func_name, |_, _, value_ptr_ptr| unsafe {
            (*value_ptr_ptr) = alloc_mgp_value();
            mgp_error::MGP_ERROR_NO_ERROR
        });
    };
}

macro_rules! mock_mgp_value_make_without_mem {
    ($c_func_name:ident) => {
        mock_mgp_once!($c_func_name, |_, value_ptr_ptr| unsafe {
            (*value_ptr_ptr) = alloc_mgp_value();
            mgp_error::MGP_ERROR_NO_ERROR
        });
    };
}

#[test]
#[serial]
fn test_insert_value() {
    // TODO(antaljanosbenjamin) Try to free the independently allocates types (list, map, etc)
    // TODO(antaljanosbenjamin) Add temporal types
    mock_mgp_once!(mgp_value_make_null_context, |_, value_ptr_ptr| unsafe {
        (*value_ptr_ptr) = alloc_mgp_value();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_value_make_with_mem!(mgp_value_make_bool_context);
    mock_mgp_value_make_with_mem!(mgp_value_make_int_context);
    mock_mgp_value_make_with_mem!(mgp_value_make_double_context);
    mock_mgp_value_make_with_mem!(mgp_value_make_string_context);

    mock_mgp_once!(mgp_list_size_context, |_, size_ptr| unsafe {
        (*size_ptr) = 0;
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_once!(mgp_list_make_empty_context, |_, _, list_ptr_ptr| unsafe {
        (*list_ptr_ptr) = alloc_mgp_list();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_value_make_without_mem!(mgp_value_make_list_context);

    mock_mgp_once!(mgp_map_make_empty_context, |_, map_ptr_ptr| unsafe {
        (*map_ptr_ptr) = alloc_mgp_map();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_once!(mgp_map_iter_items_context, |_, _, iter_ptr_ptr| unsafe {
        (*iter_ptr_ptr) = alloc_mgp_map_items_iterator();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_once!(
        mgp_map_items_iterator_get_context,
        |_, item_ptr_ptr| unsafe {
            (*item_ptr_ptr) = null_mut();
            mgp_error::MGP_ERROR_NO_ERROR
        }
    );
    mock_mgp_value_make_without_mem!(mgp_value_make_map_context);
    mock_mgp_once!(mgp_map_items_iterator_destroy_context, |_| {});

    mock_mgp_once!(mgp_vertex_copy_context, |_, _, vertex_ptr_ptr| unsafe {
        (*vertex_ptr_ptr) = alloc_mgp_vertex();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_value_make_without_mem!(mgp_value_make_vertex_context);

    mock_mgp_once!(mgp_edge_copy_context, |_, _, edge_ptr_ptr| unsafe {
        (*edge_ptr_ptr) = alloc_mgp_edge();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_value_make_without_mem!(mgp_value_make_edge_context);

    mock_mgp_once!(mgp_path_copy_context, |_, _, path_ptr_ptr| unsafe {
        (*path_ptr_ptr) = alloc_mgp_path();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_value_make_without_mem!(mgp_value_make_path_context);

    mock_mgp_once!(mgp_result_new_record_context, |_, record_ptr_ptr| unsafe {
        (*record_ptr_ptr) = alloc_mgp_result_record();
        mgp_error::MGP_ERROR_NO_ERROR
    });
    let ctx_insert = mgp_result_record_insert_context();
    ctx_insert
        .expect()
        .times(10)
        .returning(|_, _, _| mgp_error::MGP_ERROR_UNABLE_TO_ALLOCATE);
    let ctx_destroy = mgp_value_destroy_context();
    ctx_destroy.expect().times(10).returning(|_| {});

    with_dummy!(|memgraph: &Memgraph| {
        let result_record = ResultRecord::create(&memgraph).unwrap();
        assert!(result_record.insert_null(c_str!("field")).is_err());
        assert!(result_record.insert_bool(c_str!("field"), true).is_err());
        assert!(result_record.insert_int(c_str!("field"), 1).is_err());
        assert!(result_record.insert_double(c_str!("field"), 0.1).is_err());
        assert!(result_record
            .insert_string(c_str!("field"), c_str!("string"))
            .is_err());
        let list = List::new(null_mut(), &memgraph);
        assert!(result_record.insert_list(c_str!("field"), &list).is_err());
        let map = Map::new(null_mut(), &memgraph);
        assert!(result_record.insert_map(c_str!("field"), &map).is_err());
        let vertex = Vertex::new(null_mut(), &memgraph);
        assert!(result_record
            .insert_vertex(c_str!("field"), &vertex)
            .is_err());
        let edge = Edge::new(null_mut(), &memgraph);
        assert!(result_record.insert_edge(c_str!("field"), &edge).is_err());
        let path = Path::new(null_mut(), &memgraph);
        assert!(result_record.insert_path(c_str!("field"), &path).is_err());
    });
}
