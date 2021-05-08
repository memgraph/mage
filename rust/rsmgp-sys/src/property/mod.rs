use std::ffi::CStr;
use std::marker::PhantomData;

use crate::mgp::*;
use crate::value::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct MgpProperty<'a> {
    pub name: &'a CStr,
    pub value: MgpValue,
}

#[derive(Debug)]
pub struct MgpConstProperty<'a> {
    pub name: &'a CStr,
    pub value: MgpConstValue,
}

pub struct MgpPropertiesIterator<'a> {
    pub ptr: *mut mgp_properties_iterator,
    pub is_first: bool,
    pub phantom: PhantomData<&'a mgp_properties_iterator>,
}

impl<'a> Default for MgpPropertiesIterator<'a> {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            is_first: true,
            phantom: PhantomData,
        }
    }
}

impl<'a> Drop for MgpPropertiesIterator<'a> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_properties_iterator_destroy(self.ptr);
            }
        }
    }
}

impl<'a> Iterator for MgpPropertiesIterator<'a> {
    type Item = MgpConstProperty<'a>;

    fn next(&mut self) -> Option<MgpConstProperty<'a>> {
        if self.is_first {
            self.is_first = false;
            unsafe {
                let data = ffi::mgp_properties_iterator_get(self.ptr);
                if data.is_null() {
                    None
                } else {
                    let data_ref = data.as_ref().unwrap();
                    Some(MgpConstProperty {
                        name: CStr::from_ptr(data_ref.name),
                        value: MgpConstValue {
                            value: data_ref.value,
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
                    let data_ref = data.as_ref().unwrap();
                    Some(MgpConstProperty {
                        name: CStr::from_ptr(data_ref.name),
                        value: MgpConstValue {
                            value: data_ref.value,
                        },
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
