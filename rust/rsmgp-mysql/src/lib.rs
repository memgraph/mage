use c_str_macro::c_str;
use chrono::{NaiveDate, NaiveDateTime};
use mysql::prelude::*;
use mysql::*;
use rsmgp_sys::map::*;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::result::Error;
use rsmgp_sys::result::Result;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::MgpValue;
use rsmgp_sys::value::Value;
use rsmgp_sys::{close_module, define_optional_type, define_procedure, define_type, init_module};
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

const AURORA_HOST: &str = "localhost";
const AURORA_PORT: i64 = 3306;
const AURORA_USER: &str = "username";
const AURORA_PASSWORD: &str = "password";
const AURORA_DATABASE: &str = "database";

fn get_aurora_url(config: &Map) -> String {
    use std::ffi::CString;

    // Helper to extract a string from the map, or fallback
    fn get_str(config: &Map, key: &str, default: &str) -> String {
        let ckey = CString::new(key).unwrap();
        match config.at(&ckey) {
            Ok(Value::String(s)) => s
                .to_str()
                .ok()
                .map(|s| s.to_string())
                .unwrap_or_else(|| default.to_string()),
            _ => default.to_string(),
        }
    }
    // Helper to extract a u16 from the map, or fallback
    fn get_port(config: &Map, key: &str, default: i64) -> i64 {
        let ckey = CString::new(key).unwrap();
        match config.at(&ckey) {
            Ok(Value::Int(i)) => i,
            _ => default,
        }
    }

    let user = get_str(config, "user", AURORA_USER);
    let pass = get_str(config, "password", AURORA_PASSWORD);
    let host = get_str(config, "host", AURORA_HOST);
    let db = get_str(config, "database", AURORA_DATABASE);
    let port = get_port(config, "port", AURORA_PORT);

    format!(
        "mysql://{user}:{pass}@{host}:{port}/{db}",
        user = user,
        pass = pass,
        host = host,
        port = port,
        db = db
    )
}

fn query_is_table(table_or_sql: &str) -> bool {
    table_or_sql.split_whitespace().count() == 1
}

init_module!(|memgraph: &Memgraph| -> Result<()> {
    memgraph.add_read_procedure(
        migrate,
        c_str!("migrate"),
        &[define_type!("table_or_sql", Type::String)],
        &[
            define_optional_type!(
                "config",
                &MgpValue::make_map(&Map::make_empty(&memgraph)?, &memgraph)?,
                Type::Map
            ),
            define_optional_type!(
                "config_path",
                &MgpValue::make_string(c_str!(""), &memgraph)?,
                Type::String
            ),
        ],
        &[define_type!("row", Type::Map)],
    )?;

    Ok(())
});

define_procedure!(migrate, |memgraph: &Memgraph| -> Result<()> {
    let args = memgraph.args()?;
    let query = match args.value_at(0)? {
        Value::String(ref cstr) => {
            let s = cstr.to_str().expect("CString is not valid UTF-8");
            if query_is_table(s) {
                format!("SELECT * FROM `{}`;", s)
            } else {
                s.to_string()
            }
        }
        _ => panic!("Expected String value in place of sql_or_table parameter!"),
    };

    let config_arg = args.value_at(1)?;
    let config = match config_arg {
        Value::Map(ref map) => map,
        _ => panic!("Expected Map value in place of config parameter!"),
    };

    let url = get_aurora_url(config);
    let opts: Opts = match Opts::from_url(&url) {
        Ok(opts) => opts,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::InvalidMySQLURL);
        }
    };
    let pool: Pool = match Pool::new(opts) {
        Ok(pool) => pool,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToCreateMySQLPool);
        }
    };

    let mut conn: PooledConn = match pool.get_conn() {
        Ok(conn) => conn,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToGetMySQLConnection);
        }
    };
    let prepared_statement = match conn.prep(&query) {
        Ok(stmt) => stmt,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToPrepareMySQLStatement);
        }
    };
    let query_result: Vec<Row> = match conn.exec(prepared_statement, ()) {
        Ok(rows) => rows,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToExecuteMySQLQuery);
        }
    };

    for row in query_result.iter() {
        let result = memgraph.result_record()?;
        let row_map = Map::make_empty(&memgraph)?;
        for column in row.columns_ref() {
            let col_name = CString::new(column.name_str().as_bytes()).unwrap();
            let column_value = &row[column.name_str().as_ref()];
            let mg_val = match column_value {
                mysql::Value::NULL => Value::Null,
                mysql::Value::Int(i) => Value::Int(*i),
                mysql::Value::UInt(u) => {
                    if *u <= i64::MAX as u64 {
                        Value::Int(*u as i64)
                    } else {
                        Value::String(CString::new(u.to_string()).unwrap())
                    }
                }
                mysql::Value::Float(f) => Value::Float(*f as f64),
                mysql::Value::Double(d) => Value::Float(*d as f64),
                mysql::Value::Bytes(b) => {
                    // Try to interpret as UTF-8 string, fallback to hex if not valid
                    match String::from_utf8(b.clone()) {
                        Ok(s) => {
                            if let Ok(d) = s.parse::<f64>() {
                                Value::Float(d as f64)
                            } else {
                                Value::String(CString::new(s).unwrap())
                            }
                        }
                        Err(_) => Value::String(CString::new(hex::encode(b)).unwrap()),
                    }
                }
                _ => Value::Null,
            };
            row_map.insert(col_name.as_c_str(), &mg_val)?;
        }
        result.insert_map(c_str!("row"), &row_map)?;
    }

    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });
