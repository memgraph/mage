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

use super::*;
use crate::memgraph::Memgraph;
use crate::mgp::mock_ffi::*;
use crate::testing::alloc::*;
use crate::{mock_mgp_once, with_dummy};
use libc::{c_void, free};
use serial_test::serial;

#[test]
#[serial]
fn test_from_naive_date() {
    let test_date = |date: NaiveDate| {
        mock_mgp_once!(
            mgp_date_from_parameters_context,
            move |date_params, _, date_ptr_ptr| unsafe {
                assert_eq!((*date_params).year, date.year());
                assert_eq!((*date_params).month as u32, date.month());
                assert_eq!((*date_params).day as u32, date.day());
                (*date_ptr_ptr) = alloc_mgp_date();
                mgp_error::MGP_ERROR_NO_ERROR
            }
        );
        mock_mgp_once!(mgp_date_destroy_context, |ptr| unsafe {
            free(ptr as *mut c_void);
        });

        with_dummy!(|memgraph: &Memgraph| {
            let _mgp_date = Date::from_naive_date(&date, &memgraph);
        });
    };
    test_date(NaiveDate::from_ymd(0, 1, 1));
    test_date(NaiveDate::from_ymd(1834, 1, 1));
    test_date(NaiveDate::from_ymd(1996, 12, 7));
    test_date(NaiveDate::from_ymd(9999, 12, 31));
}

#[test]
#[serial]
fn test_date_accessors() {
    let year = 1934;
    let month = 2;
    let day = 31;
    mock_mgp_once!(mgp_date_get_year_context, move |_, year_ptr| unsafe {
        (*year_ptr) = year;
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_once!(mgp_date_get_month_context, move |_, month_ptr| unsafe {
        (*month_ptr) = month;
        mgp_error::MGP_ERROR_NO_ERROR
    });
    mock_mgp_once!(mgp_date_get_day_context, move |_, day_ptr| unsafe {
        (*day_ptr) = day;
        mgp_error::MGP_ERROR_NO_ERROR
    });

    with_dummy!(Date, |date: &Date| {
        assert_eq!(date.year(), year);
        assert_eq!(date.month(), month);
        assert_eq!(date.day(), day);
    });
}

#[test]
#[serial]
fn test_invalid_date() {
    let test_invalid_date = |date: NaiveDate| {
        with_dummy!(|memgraph: &Memgraph| {
            let result = Date::from_naive_date(&date, &memgraph);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                Error::UnableToCreateDateFromNaiveDate
            );
        });
    };
    test_invalid_date(NaiveDate::from_ymd(-1, 12, 31));
    test_invalid_date(NaiveDate::from_ymd(10000, 1, 1));
}

#[test]
#[serial]
fn test_unable_to_allocate() {
    mock_mgp_once!(mgp_date_from_parameters_context, move |_, _, _| {
        mgp_error::MGP_ERROR_UNABLE_TO_ALLOCATE
    });

    with_dummy!(|memgraph: &Memgraph| {
        let error = Date::from_naive_date(&NaiveDate::from_num_days_from_ce(0), &memgraph);
        assert!(error.is_err());
        assert_eq!(error.err().unwrap(), Error::UnableToCreateDateFromNaiveDate);
    });
}
