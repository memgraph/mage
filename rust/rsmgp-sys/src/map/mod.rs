use std::ffi::{CStr, CString};

use crate::context::*;
use crate::mgp::*;
use crate::result::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct Map {
    pub ptr: *mut mgp_map,
    pub context: Memgraph,
}

impl Drop for Map {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_map_destroy(self.ptr);
            }
        }
    }
}

pub struct MapItem {
    pub key: CString,
    pub value: Value,
}

pub struct MapIterator {
    pub ptr: *mut mgp_map_items_iterator,
    pub is_first: bool,
    pub context: Memgraph,
}

impl Default for MapIterator {
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

impl Drop for MapIterator {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_map_items_iterator_destroy(self.ptr);
            }
        }
    }
}

impl Iterator for MapIterator {
    type Item = MapItem;

    fn next(&mut self) -> Option<MapItem> {
        unsafe {
            let data: *const mgp_map_item;
            if self.is_first {
                self.is_first = false;
                data = ffi::mgp_map_items_iterator_get(self.ptr);
            } else {
                data = ffi::mgp_map_items_iterator_next(self.ptr);
            }

            if data.is_null() {
                None
            } else {
                let mgp_map_item_key = ffi::mgp_map_item_key(data);
                let mgp_map_item_value = ffi::mgp_map_item_value(data);
                let key = match create_cstring(mgp_map_item_key) {
                    Ok(v) => v,
                    Err(_) => panic!("Unable to create map item key."),
                };
                let value = match mgp_raw_value_to_value(mgp_map_item_value, &self.context) {
                    Ok(v) => v,
                    Err(_) => panic!("Unable to create map item value."),
                };
                Some(MapItem { key, value })
            }
        }
    }
}

impl Map {
    // TODO(gitbuda): Add ability to create empty Map object.
    pub fn insert(&self, key: &CStr, value: &Value) -> MgpResult<()> {
        unsafe {
            let mgp_value = value.to_result_mgp_value(&self.context)?;
            // TODO(gitbuda): Check the Map ptr for null.
            if ffi::mgp_map_insert(self.ptr, key.as_ptr(), mgp_value.ptr) == 0 {
                return Err(MgpError::UnableToInsertMapValue);
            }
            Ok(())
        }
    }

    pub fn size(&self) -> u64 {
        unsafe { ffi::mgp_map_size(self.ptr) }
    }

    pub fn at(&self, key: &CStr) -> MgpResult<Value> {
        unsafe {
            let c_value = ffi::mgp_map_at(self.ptr, key.as_ptr());
            if c_value.is_null() {
                return Err(MgpError::UnableToAccessMapValue);
            }
            mgp_raw_value_to_value(c_value, &self.context)
        }
    }

    pub fn iter(&self) -> MgpResult<MapIterator> {
        unsafe {
            let mgp_iterator = ffi::mgp_map_iter_items(self.ptr, self.context.memory());
            if mgp_iterator.is_null() {
                return Err(MgpError::UnableToCreateMapIterator);
            }
            Ok(MapIterator {
                ptr: mgp_iterator,
                context: self.context.clone(),
                ..Default::default()
            })
        }
    }
}

#[cfg(test)]
mod tests;
