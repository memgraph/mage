use std::ffi::CString;

use crate::memgraph::*;
use crate::mgp::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

/// Property is used in the following memgraphs:
///   * return Property from PropertiesIterator
///   * return Property directly from vertex/edge.
///
/// Property owns CString and Value bacause the underlying C string or value could be deleted during
/// the lifetime of the property. In other words, Property stores copies of underlying name and
/// value.
#[derive(Debug)]
pub struct Property {
    pub name: CString,
    pub value: Value,
}

pub struct PropertiesIterator {
    ptr: *mut mgp_properties_iterator,
    is_first: bool,
    memgraph: Memgraph,
}

impl PropertiesIterator {
    pub fn new(ptr: *mut mgp_properties_iterator, memgraph: &Memgraph) -> PropertiesIterator {
        PropertiesIterator {
            ptr,
            is_first: true,
            memgraph: memgraph.clone(),
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
                    value: match mgp_raw_value_to_value(data_ref.value, &self.memgraph) {
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
