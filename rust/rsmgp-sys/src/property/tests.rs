use serial_test::serial;
use std::ptr::null_mut;

use super::*;
use crate::mgp::mock_ffi::*;
use crate::{mock_mgp_once, with_dummy};

#[test]
#[serial]
fn test_properties_iterator() {
    mock_mgp_once!(mgp_properties_iterator_get_context, |_| { null_mut() });
    mock_mgp_once!(mgp_properties_iterator_next_context, |_| { null_mut() });

    with_dummy!(|memgraph: &Memgraph| {
        let mut iterator = PropertiesIterator::new(null_mut(), &memgraph);

        let value_1 = iterator.next();
        assert!(value_1.is_none());

        let value_2 = iterator.next();
        assert!(value_2.is_none());
    });
}
