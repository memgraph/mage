use backtrace::Backtrace;
use c_str_macro::c_str;
use rsmgp_sys::context::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::property::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::{close_module, define_procedure, define_type, init_module};
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

init_module!(
    |module: *mut mgp_module, _: *mut mgp_memory| -> MgpResult<()> {
        add_read_procedure(
            test_procedure,
            c_str!("test_procedure"),
            module,
            &vec![
                define_type!("labels_count", FieldType::Int),
                define_type!("has_L3_label", FieldType::Bool),
                define_type!("first_label", FieldType::String),
                define_type!("name_property", FieldType::String),
                define_type!("properties_string", FieldType::String),
                define_type!("first_edge_type", FieldType::String),
                define_type!("list", FieldType::List, FieldType::Int),
            ],
        )?;
        Ok(())
    }
);

define_procedure!(
    test_procedure,
    |context: &Memgraph| -> Result<(), MgpError> {
        for mgp_vertex in context.vertices_iter()? {
            let result = context.result_record()?;

            let mut properties: Vec<Property> = mgp_vertex.properties()?.collect();
            properties.sort_by(|a, b| {
                let a_name = a.name.to_str().unwrap();
                let b_name = b.name.to_str().unwrap();
                a_name.cmp(&b_name)
            });
            let properties_string = properties
                .iter()
                .map(|prop| {
                    let prop_name = prop.name.to_str().unwrap();
                    if let Value::Int(value) = prop.value {
                        return format!("{}: {}", prop_name, value);
                    } else if let Value::String(value) = &prop.value {
                        return format!("{}: {}", prop_name, value.to_str().unwrap());
                    } else if let Value::Float(value) = prop.value {
                        return format!("{}: {}", prop_name, value);
                    } else {
                        ",".to_string()
                    }
                })
                .collect::<Vec<String>>()
                .join(", ");
            result.insert_string(
                c_str!("properties_string"),
                CString::new(properties_string.into_bytes())
                    .unwrap()
                    .as_c_str(),
            )?;

            let labels_count = mgp_vertex.labels_count();
            result.insert_int(c_str!("labels_count"), labels_count as i64)?;
            if labels_count > 0 {
                result.insert_string(c_str!("first_label"), &mgp_vertex.label_at(0)?)?;
            } else {
                result.insert_string(c_str!("first_label"), c_str!(""))?;
            }

            let name_property = mgp_vertex.property(c_str!("name"))?.value;
            if let Value::Null = name_property {
                result.insert_string(c_str!("name_property"), c_str!("unknown"))?;
            } else if let Value::String(value) = name_property {
                result.insert_string(c_str!("name_property"), &value)?;
            } else {
                result.insert_string(c_str!("name_property"), c_str!("not null and not string"))?
            }

            result.insert_bool(c_str!("has_L3_label"), mgp_vertex.has_label(c_str!("L3")))?;

            match mgp_vertex.out_edges()?.next() {
                Some(edge) => {
                    let edge_type = edge.edge_type()?;
                    result.insert_string(c_str!("first_edge_type"), &edge_type)?;
                }
                None => {
                    result.insert_string(c_str!("first_edge_type"), c_str!("unknown_edge_type"))?;
                }
            }

            let list_property = mgp_vertex.property(c_str!("list"))?.value;
            if let Value::List(list) = list_property {
                result.insert_list(c_str!("list"), &list)?;
            }
        }
        Ok(())
    }
);

close_module!(|| { 0 });
