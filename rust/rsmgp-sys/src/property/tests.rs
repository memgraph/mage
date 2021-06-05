use serial_test::serial;

use super::*;
use crate::mgp::mock_ffi::*;

#[test]
#[serial]
fn test_properties_iterator() {
    let ctx_1 = mgp_properties_iterator_get_context();
    ctx_1.expect().times(1).returning(|_| std::ptr::null_mut());
    let ctx_2 = mgp_properties_iterator_next_context();
    ctx_2.expect().times(1).returning(|_| std::ptr::null_mut());

    let memgraph = Memgraph::new_default();
    let mut iterator = PropertiesIterator::new(std::ptr::null_mut(), &memgraph);

    let value_1 = iterator.next();
    assert!(value_1.is_none());

    let value_2 = iterator.next();
    assert!(value_2.is_none());
}
