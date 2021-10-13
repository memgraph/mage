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

use crate::memgraph::*;
#[double]
use crate::mgp::ffi;
use crate::mgp::*;
use crate::result::*;
use chrono::Timelike;
use chrono::{Datelike, NaiveDate, NaiveTime};
use mockall_double::double;

const MINIMUM_YEAR: i32 = 0;
const MAXIMUM_YEAR: i32 = 9999;
const NANOS_PER_MILLIS: u32 = 1_000_000;
const NANOS_PER_MICROS: u32 = 1_000;

pub(crate) struct Date {
    ptr: *mut mgp_date,
}

impl Drop for Date {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_date_destroy(self.ptr);
            }
        }
    }
}

impl Date {
    pub(crate) fn new(ptr: *mut mgp_date) -> Date {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create path because the given pointer is null."
        );

        Date { ptr }
    }

    pub fn from_naive_date(from: &NaiveDate, memgraph: &Memgraph) -> Result<Date> {
        let year = from.year();
        if year < MINIMUM_YEAR || year > MAXIMUM_YEAR {
            return Err(Error::UnableToCreateDateFromNaiveDate);
        }
        let mut date_params = mgp_date_parameters {
            year: from.year(),
            month: from.month() as i32,
            day: from.day() as i32,
        };
        unsafe {
            let date = Date::new(invoke_mgp_func_with_res!(
                *mut mgp_date,
                Error::UnableToCreateDateFromNaiveDate,
                ffi::mgp_date_from_parameters,
                &mut date_params,
                memgraph.memory_ptr()
            )?);
            Ok(date)
        }
    }

    pub fn to_naive_date(&self) -> NaiveDate {
        NaiveDate::from_ymd(self.year(), self.month(), self.day())
    }

    /// Returns the underlying [mgp_path] pointer.
    pub fn mgp_ptr(&self) -> *mut mgp_date {
        self.ptr
    }
    pub fn set_mgp_ptr(&mut self, new_ptr: *mut mgp_date) {
        self.ptr = new_ptr;
    }

    pub fn year(&self) -> i32 {
        unsafe { invoke_mgp_func!(i32, ffi::mgp_date_get_year, self.ptr).unwrap() }
    }

    pub fn month(&self) -> u32 {
        unsafe { invoke_mgp_func!(i32, ffi::mgp_date_get_month, self.ptr).unwrap() as u32 }
    }

    pub fn day(&self) -> u32 {
        unsafe { invoke_mgp_func!(i32, ffi::mgp_date_get_day, self.ptr).unwrap() as u32 }
    }
}

pub(crate) struct LocalTime {
    ptr: *mut mgp_local_time,
}

impl Drop for LocalTime {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_local_time_destroy(self.ptr);
            }
        }
    }
}

impl LocalTime {
    pub(crate) fn new(ptr: *mut mgp_local_time) -> LocalTime {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create path because the given pointer is null."
        );

        LocalTime { ptr }
    }

    pub fn from_naive_time(from: &NaiveTime, memgraph: &Memgraph) -> Result<LocalTime> {
        let eliminate_leap_seconds = |nanos: u32| {
            const NANOS_PER_SECONDS: u32 = 1_000_000_000;
            nanos % NANOS_PER_SECONDS
        };

        let nanoseconds = eliminate_leap_seconds(from.nanosecond());
        let mut local_time_params = mgp_local_time_parameters {
            hour: from.hour() as i32,
            minute: from.minute() as i32,
            second: from.second() as i32,
            millisecond: (nanoseconds / NANOS_PER_MILLIS) as i32,
            microsecond: (nanoseconds % NANOS_PER_MILLIS / NANOS_PER_MICROS) as i32,
        };

        unsafe {
            let date = LocalTime::new(invoke_mgp_func_with_res!(
                *mut mgp_local_time,
                Error::UnableToCreateLocalTimeFromNaiveTime,
                ffi::mgp_local_time_from_parameters,
                &mut local_time_params,
                memgraph.memory_ptr()
            )?);
            Ok(date)
        }
    }

    pub fn to_naive_time(&self) -> NaiveTime {
        NaiveTime::from_hms_nano(
            self.hour(),
            self.minute(),
            self.second(),
            self.millisecond() * NANOS_PER_MILLIS + self.microsecond() * NANOS_PER_MICROS,
        )
    }

    /// Returns the underlying [mgp_path] pointer.
    pub fn mgp_ptr(&self) -> *mut mgp_local_time {
        self.ptr
    }
    pub fn set_mgp_ptr(&mut self, new_ptr: *mut mgp_local_time) {
        self.ptr = new_ptr;
    }

    pub fn hour(&self) -> u32 {
        unsafe { invoke_mgp_func!(i32, ffi::mgp_local_time_get_hour, self.ptr).unwrap() as u32 }
    }

    pub fn minute(&self) -> u32 {
        unsafe { invoke_mgp_func!(i32, ffi::mgp_local_time_get_minute, self.ptr).unwrap() as u32 }
    }

    pub fn second(&self) -> u32 {
        unsafe { invoke_mgp_func!(i32, ffi::mgp_local_time_get_second, self.ptr).unwrap() as u32 }
    }

    pub fn millisecond(&self) -> u32 {
        unsafe {
            invoke_mgp_func!(i32, ffi::mgp_local_time_get_millisecond, self.ptr).unwrap() as u32
        }
    }

    pub fn microsecond(&self) -> u32 {
        unsafe {
            invoke_mgp_func!(i32, ffi::mgp_local_time_get_microsecond, self.ptr).unwrap() as u32
        }
    }
}

#[cfg(test)]
mod tests;
