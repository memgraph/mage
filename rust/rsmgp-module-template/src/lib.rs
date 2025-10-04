use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::{close_module, define_procedure, init_module};
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

init_module!(|_memgraph: &Memgraph| -> Result<()> { Ok(()) });

define_procedure!(test_procedure, |_memgraph: &Memgraph| -> Result<()> {
    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });
