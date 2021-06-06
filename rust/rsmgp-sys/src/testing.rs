#[cfg(test)]
pub mod alloc {
    use libc::malloc;
    use std::mem::size_of;

    use crate::mgp::*;

    pub(crate) unsafe fn alloc_mgp_type() -> *mut mgp_type {
        malloc(size_of::<mgp_type>()) as *mut mgp_type
    }

    pub(crate) unsafe fn alloc_mgp_value() -> *mut mgp_value {
        malloc(size_of::<mgp_value>()) as *mut mgp_value
    }

    pub(crate) unsafe fn alloc_mgp_list() -> *mut mgp_list {
        malloc(size_of::<mgp_list>()) as *mut mgp_list
    }

    pub(crate) unsafe fn alloc_mgp_map() -> *mut mgp_map {
        malloc(size_of::<mgp_map>()) as *mut mgp_map
    }

    pub(crate) unsafe fn alloc_mgp_map_items_iterator() -> *mut mgp_map_items_iterator {
        malloc(size_of::<mgp_map_items_iterator>()) as *mut mgp_map_items_iterator
    }

    pub(crate) unsafe fn alloc_mgp_vertex() -> *mut mgp_vertex {
        malloc(size_of::<mgp_vertex>()) as *mut mgp_vertex
    }

    pub(crate) unsafe fn alloc_mgp_edge() -> *mut mgp_edge {
        malloc(size_of::<mgp_edge>()) as *mut mgp_edge
    }

    pub(crate) unsafe fn alloc_mgp_path() -> *mut mgp_path {
        malloc(size_of::<mgp_path>()) as *mut mgp_path
    }

    pub(crate) unsafe fn alloc_mgp_proc() -> *mut mgp_proc {
        malloc(size_of::<mgp_proc>()) as *mut mgp_proc
    }

    pub(crate) unsafe fn alloc_mgp_result_record() -> *mut mgp_result_record {
        malloc(size_of::<mgp_result_record>()) as *mut mgp_result_record
    }

    #[macro_export]
    macro_rules! mock_mgp_once {
        ($c_func_name:ident, $rs_return_func:expr) => {
            let $c_func_name = $c_func_name();
            $c_func_name.expect().times(1).returning($rs_return_func);
        };
    }

    #[macro_export]
    macro_rules! with_dummy {
        ($rs_test_func:expr) => {
            let memgraph = Memgraph::new_default();
            $rs_test_func(&memgraph);
        };

        (List, $rs_test_func:expr) => {
            let memgraph = Memgraph::new_default();
            let list = List::new(null_mut(), &memgraph);
            $rs_test_func(&list);
        };

        (Map, $rs_test_func:expr) => {
            let memgraph = Memgraph::new_default();
            let map = Map::new(null_mut(), &memgraph);
            $rs_test_func(&map);
        };

        (Vertex, $rs_test_func:expr) => {
            let memgraph = Memgraph::new_default();
            let vertex = Vertex::new(null_mut(), &memgraph);
            $rs_test_func(&vertex);
        };

        (Edge, $rs_test_func:expr) => {
            let memgraph = Memgraph::new_default();
            let edge = Edge::new(null_mut(), &memgraph);
            $rs_test_func(&edge);
        };

        (Path, $rs_test_func:expr) => {
            let memgraph = Memgraph::new_default();
            let path = Path::new(null_mut(), &memgraph);
            $rs_test_func(&path);
        };
    }
}
