#[cfg(test)]
pub mod alloc {
    use libc::malloc;
    use std::mem::size_of;

    use crate::mgp::*;

    pub(crate) unsafe fn alloc_mgp_value() -> *mut mgp_value {
        malloc(size_of::<mgp_value>()) as *mut mgp_value
    }

    pub(crate) unsafe fn alloc_mgp_list() -> *mut mgp_list {
        malloc(size_of::<mgp_list>()) as *mut mgp_list
    }
}
