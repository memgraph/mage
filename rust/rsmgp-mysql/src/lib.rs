use c_str_macro::c_str;
use mysql::*;
use mysql::prelude::*;
use rsmgp_sys::map::*;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::result::Result;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::Value;
use rsmgp_sys::{close_module, define_procedure, define_type, init_module};
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

const AURORA_HOST: &str = "localhost";
const AURORA_PORT: u16 = 3306;
const AURORA_USER: &str = "testuser";
const AURORA_PASSWORD: &str = "testpass";
const AURORA_DATABASE: &str = "testdb";

fn get_aurora_url() -> String {
    format!(
        "mysql://{user}:{pass}@{host}:{port}/{db}",
        user = AURORA_USER,
        pass = AURORA_PASSWORD,
        host = AURORA_HOST,
        port = AURORA_PORT,
        db = AURORA_DATABASE
    )
}

init_module!(|memgraph: &Memgraph| -> Result<()> {
    memgraph.add_read_procedure(
        migrate,
        c_str!("migrate"),
        &[define_type!("table_or_sql", Type::String),
        define_type!("config", Type::Map)],
        &[],
        &[define_type!("row", Type::Map),],
    )?;

    Ok(())
});

define_procedure!(migrate, |memgraph: &Memgraph| -> Result<()> {
    let args = memgraph.args()?;
    let sql_or_table = args.value_at(0)?;
    let config = args.value_at(1)?;

    let url = get_aurora_url();
    let opts = Opts::from_url(&url).expect("Invalid MySQL URL");
    let pool = Pool::new(opts).expect("Failed to create pool");
    let mut conn = pool.get_conn().expect("Failed to get connection");
    let query_result: Vec<Row> = conn.query("SELECT * FROM `Table`").expect("Query failed");
    for row in query_result.iter() {
        let result = memgraph.result_record()?;
        let mut row_map = Map::make_empty(&memgraph)?;
        let columns = row.columns_ref();
        for (i, col) in columns.iter().enumerate() {
            let col_name = CString::new(col.name_str().as_bytes()).unwrap();
            let val = row.as_ref(i).unwrap_or(&mysql::Value::NULL);
            let mg_val = match val {
                mysql::Value::NULL => Value::Null,
                mysql::Value::Int(i) => Value::Int(*i),
                mysql::Value::UInt(u) => Value::Int(*u as i64),
                mysql::Value::Float(f) => Value::Float(*f as f64),
                mysql::Value::Double(d) => Value::Float(*d),
                mysql::Value::Bytes(b) => {
                    // Convert bytes to CString safely, fallback to hex if not valid UTF-8
                    match CString::new(b.clone()) {
                        Ok(s) => Value::String(s),
                        Err(_) => Value::String(CString::new(hex::encode(b)).unwrap()),
                    }
                },
                _ => Value::Null,
            };
            row_map.insert(col_name.as_c_str(), &mg_val)?;
        }
        result.insert_map(c_str!("row"), &row_map)?;
    }

    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });
