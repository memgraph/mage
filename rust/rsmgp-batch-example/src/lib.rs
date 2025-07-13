use c_str_macro::c_str;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::{
    close_module, define_batch_procedure_cleanup, define_batch_procedure_init, define_procedure,
    define_type, init_module,
};
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;
use std::cell::RefCell;

thread_local! {
    static COUNTER: RefCell<i32> = RefCell::new(0);
}

init_module!(|memgraph: &Memgraph| -> Result<()> {
    memgraph.add_batch_read_procedure(
        test_procedure,
        c_str!("test_procedure"),
        init_procedure,
        cleanup_procedure,
        &[],
        &[],
        &[define_type!("value", Type::Int)],
    )?;

    Ok(())
});

define_procedure!(test_procedure, |memgraph: &Memgraph| -> Result<()> {
    println!("Entered test procedure!");
    let value: i32 = COUNTER.with(|counter| {
        let mut val = counter.borrow_mut();
        *val += 1;
        *val
    });

    if value != 10 {
        let result = memgraph.result_record()?;
        result.insert_mgp_value(c_str!("value"), &MgpValue::make_int(value as i64, memgraph)?)?;
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    println!("Exiting test procedure! Value: {}", value);
    Ok(())
});

define_batch_procedure_init!(init_procedure, |memgraph: &Memgraph| -> Result<()> {
    println!("Entered init procedure!");
    println!("Exiting init procedure!");
    Ok(())
});

define_batch_procedure_cleanup!(cleanup_procedure, |memgraph: &Memgraph| -> Result<()> {
    println!("Entered cleanup procedure!");
    println!("Exiting cleanup procedure!");
    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });
