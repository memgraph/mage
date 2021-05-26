use std::ffi::CString;

use crate::context::*;
use crate::mgp::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

// Property is used in the following contexts:
//   * Return Property from PropertiesIterator.
//   * Return Property directly from vertex/edge.
//
// Property owns CString and Value bacause the underlying C string or value could be deleted during
// the lifetime of the property. In other words, Property stores copies of underlying name and
// value.
#[derive(Debug)]
pub struct Property {
    pub name: CString,
    pub value: Value,
}

pub struct PropertiesIterator {
    pub ptr: *mut mgp_properties_iterator,
    pub is_first: bool,
    pub context: Memgraph,
}

impl Default for PropertiesIterator {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            is_first: true,
            context: Memgraph {
                ..Default::default()
            },
        }
    }
}

impl Drop for PropertiesIterator {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_properties_iterator_destroy(self.ptr);
            }
        }
    }
}

impl Iterator for PropertiesIterator {
    type Item = Property;

    fn next(&mut self) -> Option<Property> {
        unsafe {
            let data: *const mgp_property;
            if self.is_first {
                self.is_first = false;
                data = ffi::mgp_properties_iterator_get(self.ptr);
            } else {
                data = ffi::mgp_properties_iterator_next(self.ptr);
            }

            if data.is_null() {
                None
            } else {
                // Unwrap/panic is I think the only option here because if something fails the
                // whole procedure should be stopped. Returning empty Option is not an option
                // because it's not correct. The same applies in case of the property value.
                let data_ref = data.as_ref().unwrap();
                Some(Property {
                    name: match create_cstring(data_ref.name) {
                        Ok(v) => v,
                        Err(_) => panic!("Unable to provide next property. Name creation problem."),
                    },
                    value: match mgp_raw_value_to_value(data_ref.value, &self.context) {
                        Ok(v) => v,
                        Err(_) => panic!("Unable to provide next property. Value create problem."),
                    },
                })
            }
        }
    }
}

#[cfg(test)]
mod tests;
