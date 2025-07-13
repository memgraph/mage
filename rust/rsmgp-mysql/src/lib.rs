use c_str_macro::c_str;
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
use rsmgp_sys::{
    close_module, define_batch_procedure_cleanup, define_batch_procedure_init,
    define_optional_type, define_procedure, define_type, init_module,
};
use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

thread_local! {
    static MYSQL_BASE_QUERY: RefCell<String> = RefCell::new("".to_string());
    static MYSQL_CONN: RefCell<Option<PooledConn>> = RefCell::new(None);
    static MYSQL_OFFSET: RefCell<usize> = RefCell::new(0);
}

const AURORA_HOST: &str = "localhost";
const AURORA_PORT: i64 = 3306;
const AURORA_USER: &str = "username";
const AURORA_PASSWORD: &str = "password";
const AURORA_DATABASE: &str = "database";

init_module!(|memgraph: &Memgraph| -> Result<()> {
    memgraph.add_read_procedure(
        test_connection,
        c_str!("test_connection"),
        &[],
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
        &[define_type!("success", Type::Bool)],
    )?;

    memgraph.add_read_procedure(
        show_tables,
        c_str!("show_tables"),
        &[],
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
        &[define_type!("table_name", Type::String)],
    )?;

    memgraph.add_read_procedure(
        describe_table,
        c_str!("describe_table"),
        &[define_type!("table", Type::String)],
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

    memgraph.add_read_procedure(
        execute,
        c_str!("execute"),
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

    memgraph.add_batch_read_procedure(
        batch,
        c_str!("batch"),
        init_batch,
        cleanup_batch,
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

define_procedure!(test_connection, |memgraph: &Memgraph| -> Result<()> {
    let args = memgraph.args()?;
    let config_arg = args.value_at(0)?;
    let config = match config_arg {
        Value::Map(ref map) => map,
        _ => panic!("Expected Map value in place of config parameter!"),
    };

    let url = get_aurora_url(config);
    let mut conn: PooledConn = get_connection(&url)?;

    let _: Vec<Row> = match conn.exec("SELECT 1 AS result".to_string(), ()) {
        Ok(result) => result,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToExecuteMySQLQuery);
        }
    };

    let result = memgraph.result_record()?;
    result.insert_bool(c_str!("success"), true)?;

    Ok(())
});

define_procedure!(show_tables, |memgraph: &Memgraph| -> Result<()> {
    let args = memgraph.args()?;
    let config_arg = args.value_at(0)?;
    let config = match config_arg {
        Value::Map(ref map) => map,
        _ => panic!("Expected Map value in place of config parameter!"),
    };

    let url = get_aurora_url(config);
    let mut conn: PooledConn = get_connection(&url)?;

    let rows: Vec<Row> = match conn.exec("SHOW TABLES".to_string(), ()) {
        Ok(result) => result,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToExecuteMySQLQuery);
        }
    };

    for row in rows {
        // The table name is the first column in the row
        let table_name: String = row
            .get(0)
            .and_then(|val| match val {
                mysql::Value::Bytes(bytes) => String::from_utf8(bytes.clone()).ok(),
                _ => None,
            })
            .unwrap_or_else(|| "<unknown>".to_string());

        let result = memgraph.result_record()?;
        result.insert_string(c_str!("table_name"), &CString::new(table_name).unwrap())?;
    }

    Ok(())
});

define_procedure!(describe_table, |memgraph: &Memgraph| -> Result<()> {
    let args = memgraph.args()?;
    let query = match args.value_at(0)? {
        Value::String(ref cstr) => format!("DESCRIBE `{}`", cstr.to_str().unwrap()),
        _ => panic!("Expected String value in place of table parameter!"),
    };

    let config_arg = args.value_at(1)?;
    let config = match config_arg {
        Value::Map(ref map) => map,
        _ => panic!("Expected Map value in place of config parameter!"),
    };

    let url = get_aurora_url(config);
    let mut conn: PooledConn = get_connection(&url)?;

    let prepared_statement = match conn.prep(&query) {
        Ok(stmt) => stmt,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToPrepareMySQLStatement);
        }
    };
    let result = match conn.exec_iter(prepared_statement, ()) {
        Ok(result) => result,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToExecuteMySQLQuery);
        }
    };

    for row_result in result {
        let row = match row_result {
            Ok(row) => row,
            Err(e) => {
                println!("Error fetching row: {}", e);
                continue;
            }
        };

        insert_row_into_memgraph(&row, &memgraph)?;
    }

    Ok(())
});

define_procedure!(execute, |memgraph: &Memgraph| -> Result<()> {
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
    let mut conn: PooledConn = get_connection(&url)?;

    let prepared_statement = match conn.prep(&query) {
        Ok(stmt) => stmt,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToPrepareMySQLStatement);
        }
    };
    let result = match conn.exec_iter(prepared_statement, ()) {
        Ok(result) => result,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToExecuteMySQLQuery);
        }
    };

    for row_result in result {
        let row = match row_result {
            Ok(row) => row,
            Err(e) => {
                println!("Error fetching row: {}", e);
                continue;
            }
        };

        insert_row_into_memgraph(&row, &memgraph)?;
    }

    Ok(())
});

define_batch_procedure_init!(init_batch, |memgraph: &Memgraph| -> Result<()> {
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
    let conn: PooledConn = get_connection(&url)?;

    // Store the connection in thread-local
    MYSQL_CONN.with(|conn_cell| {
        *conn_cell.borrow_mut() = Some(conn);
    });
    // Store the base query in thread-local
    MYSQL_BASE_QUERY.with(|base_query| {
        *base_query.borrow_mut() = query;
    });
    // Reset the offset counter to 0
    MYSQL_OFFSET.with(|counter| {
        *counter.borrow_mut() = 0;
    });

    Ok(())
});

define_procedure!(batch, |memgraph: &Memgraph| -> Result<()> {
    let base_query = MYSQL_BASE_QUERY.with(|cell| cell.borrow().clone());
    let offset = MYSQL_OFFSET.with(|cell| *cell.borrow());
    let batch_size = 100_000;
    let query = format!("{} LIMIT {} OFFSET {}", base_query, batch_size, offset);

    MYSQL_CONN.with(|conn_cell| {
        let mut conn_opt = conn_cell.borrow_mut();
        let conn = match conn_opt.as_mut() {
            Some(c) => c,
            None => {
                println!("No MySQL connection available");
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
        let result = match conn.exec_iter(prepared_statement, ()) {
            Ok(result) => result,
            Err(e) => {
                println!("Error: {}", e);
                return Err(Error::UnableToExecuteMySQLQuery);
            }
        };

        for row_result in result {
            let row = match row_result {
                Ok(row) => row,
                Err(e) => {
                    println!("Error fetching row: {}", e);
                    continue;
                }
            };

            insert_row_into_memgraph(&row, &memgraph)?;
        }
        Ok(())
    })?;

    MYSQL_OFFSET.with(|cell| {
        *cell.borrow_mut() += batch_size;
    });

    Ok(())
});

define_batch_procedure_cleanup!(cleanup_batch, |_memgraph: &Memgraph| -> Result<()> {
    MYSQL_CONN.with(|conn_cell| {
        *conn_cell.borrow_mut() = None;
    });
    MYSQL_BASE_QUERY.with(|base_query| {
        *base_query.borrow_mut() = String::new();
    });
    MYSQL_OFFSET.with(|counter| {
        *counter.borrow_mut() = 0;
    });

    Ok(())
});

close_module!(|| -> Result<()> { Ok(()) });

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

fn get_connection(url: &str) -> Result<PooledConn, Error> {
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
    let conn: PooledConn = match pool.get_conn() {
        Ok(conn) => conn,
        Err(e) => {
            println!("Error: {}", e);
            return Err(Error::UnableToGetMySQLConnection);
        }
    };

    Ok(conn)
}

fn query_is_table(table_or_sql: &str) -> bool {
    table_or_sql.split_whitespace().count() == 1
}

fn insert_row_into_memgraph(row: &Row, memgraph: &Memgraph) -> Result<()> {
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
            mysql::Value::Bytes(b) => match String::from_utf8(b.clone()) {
                Ok(s) => {
                    if let Ok(d) = s.parse::<f64>() {
                        Value::Float(d as f64)
                    } else {
                        Value::String(CString::new(s).unwrap())
                    }
                }
                Err(_) => Value::String(CString::new(hex::encode(b)).unwrap()),
            },
            _ => Value::Null,
        };
        row_map.insert(col_name.as_c_str(), &mg_val)?;
    }
    result.insert_map(c_str!("row"), &row_map)?;

    Ok(())
}
