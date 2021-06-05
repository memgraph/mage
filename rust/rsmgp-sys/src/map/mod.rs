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
    ptr: *mut mgp_map,
    context: Memgraph,
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

/// MapItem has public fields because they are user facing object + they are easier to access.
pub struct MapItem {
    pub key: CString,
    pub value: Value,
}

pub struct MapIterator {
    ptr: *mut mgp_map_items_iterator,
    is_first: bool,
    context: Memgraph,
}

impl MapIterator {
    pub fn new(ptr: *mut mgp_map_items_iterator, context: &Memgraph) -> MapIterator {
        MapIterator {
            ptr,
            is_first: true,
            context: context.clone(),
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
    pub fn new(ptr: *mut mgp_map, context: &Memgraph) -> Map {
        Map {
            ptr,
            context: context.clone(),
        }
    }

    pub(crate) unsafe fn mgp_copy(ptr: *const mgp_map, context: &Memgraph) -> MgpResult<Map> {
        // Test passes null ptr because nothing else is possible.
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create map copy because map pointer is null."
        );

        let mgp_map_copy = ffi::mgp_map_make_empty(context.memory());
        let mgp_map_iterator = ffi::mgp_map_iter_items(ptr, context.memory());
        if mgp_map_iterator.is_null() {
            ffi::mgp_map_destroy(mgp_map_copy);
            return Err(MgpError::UnableToCreateMap);
        }
        let map_iterator = MapIterator::new(mgp_map_iterator, &context);
        for item in map_iterator {
            let mgp_value = item.value.to_mgp_value(&context)?;
            if ffi::mgp_map_insert(mgp_map_copy, item.key.as_ptr(), mgp_value.mgp_ptr()) == 0 {
                ffi::mgp_map_destroy(mgp_map_copy);
                return Err(MgpError::UnableToCreateMap);
            }
        }
        Ok(Map::new(mgp_map_copy, &context))
    }

    pub fn make_empty(context: &Memgraph) -> MgpResult<Map> {
        unsafe {
            let mgp_ptr = ffi::mgp_map_make_empty(context.memory());
            if mgp_ptr.is_null() {
                return Err(MgpError::UnableToCreateEmptyMap);
            }
            Ok(Map::new(mgp_ptr, &context))
        }
    }

    pub fn insert(&self, key: &CStr, value: &Value) -> MgpResult<()> {
        unsafe {
            let mgp_value = value.to_mgp_value(&self.context)?;
            // TODO(gitbuda): Check the Map ptr for null.
            if ffi::mgp_map_insert(self.ptr, key.as_ptr(), mgp_value.mgp_ptr()) == 0 {
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
            Ok(MapIterator::new(mgp_iterator, &self.context))
        }
    }
}

#[cfg(test)]
mod tests;
