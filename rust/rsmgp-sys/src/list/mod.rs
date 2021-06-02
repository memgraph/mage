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
    ptr: *mut mgp_list,
    context: Memgraph,
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
        // TODO(gitbuda): Implement ListIterator::next using mgp primitive methods.
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
    pub fn new(ptr: *mut mgp_list, context: &Memgraph) -> List {
        List {
            ptr,
            context: context.clone(),
        }
    }

    pub fn make_empty(capacity: u64, context: &Memgraph) -> MgpResult<List> {
        unsafe {
            let mgp_ptr = ffi::mgp_list_make_empty(capacity, context.memory());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToCreateEmptyList);
            }
            Ok(List::new(mgp_ptr, &context))
        }
    }

    pub(crate) unsafe fn mgp_copy(ptr: *const mgp_list, context: &Memgraph) -> MgpResult<List> {
        // Test passes null ptr because nothing else is possible.
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to make list copy because list pointer is null."
        );
        let size = ffi::mgp_list_size(ptr);
        // TODO(gitbuda): List::make_empty could be used but we have to inject the error context.
        let mgp_copy = ffi::mgp_list_make_empty(size, context.memory());
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
        Ok(List::new(mgp_copy, &context))
    }

    pub fn mgp_ptr(&self) -> *const mgp_list {
        self.ptr
    }

    pub fn copy(&self) -> MgpResult<List> {
        unsafe { List::mgp_copy(self.ptr, &self.context) }
    }

    pub fn append(&self, value: &Value) -> MgpResult<()> {
        unsafe {
            let mgp_value = value.to_result_mgp_value(&self.context)?;
            if ffi::mgp_list_append(self.ptr, mgp_value.ptr) == 0 {
                return Err(MgpError::UnableToAppendListValue);
            }
            Ok(())
        }
    }

    /// In case of a capacity change, the previously contained elements will move in
    /// memory and any references to them will be invalid.
    pub fn append_extend(&self, value: &Value) -> MgpResult<()> {
        unsafe {
            let mgp_value = value.to_result_mgp_value(&self.context)?;
            if ffi::mgp_list_append_extend(self.ptr, mgp_value.ptr) == 0 {
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
