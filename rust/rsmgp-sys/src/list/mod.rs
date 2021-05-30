use crate::context::*;
use crate::mgp::*;
use crate::result::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct List {
    pub ptr: *mut mgp_list,
    pub context: Memgraph,
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
        if self.position == self.list.size() {
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
    // TODO(gitbuda): Add ability to create empty list object.

    pub fn append(_: Value) -> MgpResult<()> {
        // TODO(gitbuda): Implement list.append().
        Err(MgpError::UnableToAppendListValue)
    }

    pub fn append_extend(_: Value) -> MgpResult<()> {
        // TODO(gitbuda): Implement list.append_extend().
        Err(MgpError::UnableToAppendExtendListValue)
    }

    pub fn size(&self) -> u64 {
        unsafe { ffi::mgp_list_size(self.ptr) }
    }

    pub fn capacity(&self) -> u64 {
        unsafe { ffi::mgp_list_capacity(self.ptr) }
    }

    pub fn value_at(&self, index: u64) -> MgpResult<Value> {
        unsafe {
            let c_value = ffi::mgp_list_at(self.ptr, index);
            if c_value.is_null() {
                return Err(MgpError::UnableToAccessListValueByIndex);
            }
            mgp_raw_value_to_value(c_value, &self.context)
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
