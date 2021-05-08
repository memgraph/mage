use c_str_macro::c_str;
use std::ffi::CStr;

use crate::mgp::*;
use crate::result::*;
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug)]
pub struct MgpProperty<'a> {
    pub name: &'a CStr,
    // TODO(gitbuda): Replace with MgpValue.
    pub value: *mut mgp_value,
}

// TODO(gitbuda): Figure out why this crashes Memgraph.
// impl<'a> Drop for MgpProperty<'a> {
//     fn drop(&mut self) {
//         unsafe {
//             if !self.value.is_null() {
//                 ffi::mgp_value_destroy(self.value);
//             }
//         }
//     }
// }

pub struct MgpVerticesIterator {
    ptr: *mut mgp_vertices_iterator,
    is_first: bool,
}

impl Default for MgpVerticesIterator {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            is_first: true,
        }
    }
}

impl Drop for MgpVerticesIterator {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_vertices_iterator_destroy(self.ptr);
            }
        }
    }
}

impl Iterator for MgpVerticesIterator {
    type Item = MgpVertex;

    fn next(&mut self) -> Option<MgpVertex> {
        if self.is_first {
            self.is_first = false;
            unsafe {
                let data = ffi::mgp_vertices_iterator_get(self.ptr);
                if data.is_null() {
                    None
                } else {
                    Some(MgpVertex { ptr: data })
                }
            }
        } else {
            unsafe {
                let data = ffi::mgp_vertices_iterator_next(self.ptr);
                if data.is_null() {
                    None
                } else {
                    Some(MgpVertex { ptr: data })
                }
            }
        }
    }
}

pub struct MgpVertex {
    ptr: *const mgp_vertex,
}

impl MgpVertex {
    pub fn id(&self) -> i64 {
        unsafe { ffi::mgp_vertex_get_id(self.ptr).as_int }
    }

    pub fn labels_count(&self) -> u64 {
        unsafe {
            // TODO(gitbuda): Figure out why this is not clippy::not_unsafe_ptr_arg_deref.
            ffi::mgp_vertex_labels_count(self.ptr)
        }
    }

    // TODO(gitbuda): Figure out the correct lifetime.
    pub fn label_at(&self, index: u64) -> MgpResult<&CStr> {
        unsafe {
            let c_label = ffi::mgp_vertex_label_at(self.ptr, index);
            if c_label.name.is_null() {
                return Err(MgpError::MgpOutOfBoundLabelIndex);
            }
            Ok(CStr::from_ptr(c_label.name))
        }
    }

    pub fn has_label(&self, name: &CStr) -> bool {
        unsafe {
            let c_mgp_label = mgp_label {
                name: name.as_ptr(),
            };
            ffi::mgp_vertex_has_label(self.ptr, c_mgp_label) != 0
        }
    }

    // TODO(gitbuda): This lifetime is probably problematic.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn property<'a>(
        &self,
        name: &'a CStr,
        memory: *mut mgp_memory,
    ) -> MgpResult<MgpProperty<'a>> {
        unsafe {
            let mgp_value = ffi::mgp_vertex_get_property(self.ptr, name.as_ptr(), memory);
            if mgp_value.is_null() {
                return Err(MgpError::MgpAllocationError);
            }
            Ok(MgpProperty {
                name,
                value: mgp_value,
            })
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_graph_vertices_iterator(
    graph: *const mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
) -> MgpResult<MgpVerticesIterator> {
    let unable_alloc_iter_msg = c_str!("Unable to allocate vertices iterator.");
    unsafe {
        let iterator: MgpVerticesIterator = MgpVerticesIterator {
            ptr: ffi::mgp_graph_iter_vertices(graph, memory),
            ..Default::default()
        };
        if iterator.ptr.is_null() {
            ffi::mgp_result_set_error_msg(result, unable_alloc_iter_msg.as_ptr());
            return Err(MgpError::MgpAllocationError);
        }
        Ok(iterator)
    }
}

#[cfg(test)]
mod tests;
