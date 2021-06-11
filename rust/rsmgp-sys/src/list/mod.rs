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
//! All related to the list datatype.

use crate::memgraph::*;
use crate::mgp::*;
use crate::result::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

// NOTE: Not possible to implement [std::iter::IntoIterator] because the [ListIterator] holds the
// [List] reference which needs the lifetime specifier.
pub struct List {
    ptr: *mut mgp_list,
    memgraph: Memgraph,
}

impl Drop for List {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_list_destroy(self.ptr);
            }
        }
    }
}

pub struct ListIterator<'a> {
    list: &'a List,
    position: u64,
}

impl<'a> Iterator for ListIterator<'a> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        if self.position >= self.list.size() {
            return None;
        }
        let value = match self.list.value_at(self.position) {
            Ok(v) => v,
            Err(_) => panic!("Unable to access the next list value."),
        };
        self.position += 1;
        Some(value)
    }
}

impl List {
    pub fn new(ptr: *mut mgp_list, memgraph: &Memgraph) -> List {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create a new List because pointer is null."
        );

        List {
            ptr,
            memgraph: memgraph.clone(),
        }
    }

    pub fn make_empty(capacity: u64, memgraph: &Memgraph) -> MgpResult<List> {
        unsafe {
            let mgp_ptr = ffi::mgp_list_make_empty(capacity, memgraph.memory_ptr());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToCreateEmptyList);
            }
            Ok(List::new(mgp_ptr, &memgraph))
        }
    }

    /// Creates a new List based on [mgp_list].
    pub(crate) unsafe fn mgp_copy(ptr: *const mgp_list, memgraph: &Memgraph) -> MgpResult<List> {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to make list copy because list pointer is null."
        );

        let size = ffi::mgp_list_size(ptr);
        let mgp_copy = ffi::mgp_list_make_empty(size, memgraph.memory_ptr());
        if mgp_copy.is_null() {
            return Err(MgpError::UnableToCopyList);
        }
        for index in 0..size {
            let mgp_value = ffi::mgp_list_at(ptr, index);
            if ffi::mgp_list_append(mgp_copy, mgp_value) == 0 {
                ffi::mgp_list_destroy(mgp_copy);
                return Err(MgpError::UnableToCopyList);
            }
        }
        Ok(List::new(mgp_copy, &memgraph))
    }

    pub fn copy(&self) -> MgpResult<List> {
        unsafe { List::mgp_copy(self.ptr, &self.memgraph) }
    }

    /// Appends value to the list, but if there is no place, returns an error.
    pub fn append(&self, value: &Value) -> MgpResult<()> {
        unsafe {
            let mgp_value = value.to_mgp_value(&self.memgraph)?;
            if ffi::mgp_list_append(self.ptr, mgp_value.mgp_ptr()) == 0 {
                return Err(MgpError::UnableToAppendListValue);
            }
            Ok(())
        }
    }

    /// In case of a capacity change, the previously contained elements will move in
    /// memory and any references to them will be invalid.
    pub fn append_extend(&self, value: &Value) -> MgpResult<()> {
        unsafe {
            let mgp_value = value.to_mgp_value(&self.memgraph)?;
            if ffi::mgp_list_append_extend(self.ptr, mgp_value.mgp_ptr()) == 0 {
                return Err(MgpError::UnableToAppendExtendListValue);
            }
            Ok(())
        }
    }

    pub fn size(&self) -> u64 {
        unsafe { ffi::mgp_list_size(self.ptr) }
    }

    pub fn capacity(&self) -> u64 {
        unsafe { ffi::mgp_list_capacity(self.ptr) }
    }

    /// Always copies the underlying value because in case of the capacity change any references
    /// would become invalid.
    pub fn value_at(&self, index: u64) -> MgpResult<Value> {
        unsafe {
            let c_value = ffi::mgp_list_at(self.ptr, index);
            if c_value.is_null() {
                return Err(MgpError::UnableToAccessListValueByIndex);
            }
            mgp_raw_value_to_value(c_value, &self.memgraph)
        }
    }

    pub fn iter(&self) -> MgpResult<ListIterator> {
        Ok(ListIterator {
            list: self,
            position: 0,
        })
    }
}

#[cfg(test)]
mod tests;
