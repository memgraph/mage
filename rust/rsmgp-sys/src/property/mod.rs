use std::ffi::CStr;
use std::marker::PhantomData;

use crate::context::*;
use crate::mgp::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct Property<'a> {
    // TODO(gitbuda): Property name should also be an owned CString.
    pub name: &'a CStr,
    pub value: Value,
}

pub struct PropertiesIterator<'a> {
    pub ptr: *mut mgp_properties_iterator,
    pub is_first: bool,
    pub context: Memgraph,
    pub phantom: PhantomData<&'a mgp_properties_iterator>,
}

impl<'a> Default for PropertiesIterator<'a> {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            is_first: true,
            context: Memgraph {
                ..Default::default()
            },
            phantom: PhantomData,
        }
    }
}

impl<'a> Drop for PropertiesIterator<'a> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_properties_iterator_destroy(self.ptr);
            }
        }
    }
}

impl<'a> Iterator for PropertiesIterator<'a> {
    type Item = Property<'a>;

    fn next(&mut self) -> Option<Property<'a>> {
        if self.is_first {
            self.is_first = false;
            unsafe {
                let data = ffi::mgp_properties_iterator_get(self.ptr);
                if data.is_null() {
                    None
                } else {
                    // TODO(gitbuda): Check if this unwrap is OK.
                    let data_ref = data.as_ref().unwrap();
                    Some(Property {
                        // TODO(gitbuda): This is just a wrapper for the underlying ptr.
                        name: CStr::from_ptr(data_ref.name),
                        value: match mgp_raw_value_to_value(data_ref.value, &self.context) {
                            Ok(value) => value,
                            Err(_) => Value::Null,
                        },
                    })
                }
            }
        } else {
            unsafe {
                let data = ffi::mgp_properties_iterator_next(self.ptr);
                if data.is_null() {
                    None
                } else {
                    // TODO(gitbuda): Check if this unwrap is OK.
                    let data_ref = data.as_ref().unwrap();
                    Some(Property {
                        name: CStr::from_ptr(data_ref.name),
                        value: match mgp_raw_value_to_value(data_ref.value, &self.context) {
                            Ok(value) => value,
                            Err(_) => Value::Null,
                        },
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
