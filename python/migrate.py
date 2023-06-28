import json
import mgp
import mysql.connector as mysql_connector
import oracledb
import pyodbc
import threading

from typing import Any, Dict


class Constants:
    I_COLUMN_NAME = 0
    CURSOR = "cursor"
    COLUMN_NAMES = "column_names"
    CONNECTION = "connection"
    BATCH_SIZE = 1000


##### MYSQL

mysql_dict = {}


def init_migrate_mysql(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
):
    global mysql_dict

    if params:
        _check_params_type(params)
    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if threading.get_native_id not in mysql_dict:
        mysql_dict[threading.get_native_id] = {}

    if Constants.CURSOR not in mysql_dict[threading.get_native_id]:
        mysql_dict[threading.get_native_id][Constants.CURSOR] = None

    if mysql_dict[threading.get_native_id][Constants.CURSOR] is None:
        connection = mysql_connector.connect(**config)
        cursor = connection.cursor(buffered=True)
        cursor.execute(table_or_sql, params=params)

        mysql_dict[threading.get_native_id][Constants.CONNECTION] = connection
        mysql_dict[threading.get_native_id][Constants.CURSOR] = cursor
        mysql_dict[threading.get_native_id][Constants.COLUMN_NAMES] = [
            column[Constants.I_COLUMN_NAME] for column in cursor.description
        ]


def mysql(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """
    With migrate.mysql you can access MySQL and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters (as in mysql.connector.connect),
    :param config_path: Path to the JSON file containing configuration parameters (as in mysql.connector.connect)
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The result table as a stream of rows
    """
    global mysql_dict
    cursor = mysql_dict[threading.get_native_id][Constants.CURSOR]
    column_names = mysql_dict[threading.get_native_id][Constants.COLUMN_NAMES]

    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_mysql():
    global mysql_dict
    mysql_dict[threading.get_native_id][Constants.CURSOR] = None
    mysql_dict[threading.get_native_id][Constants.CONNECTION].close()
    mysql_dict[threading.get_native_id][Constants.CONNECTION].commit()
    mysql_dict[threading.get_native_id][Constants.CONNECTION] = None
    mysql_dict[threading.get_native_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(mysql, init_migrate_mysql, cleanup_migrate_mysql)

### SQL SERVER

sql_server_dict = {}


def init_migrate_sql_server(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
):
    global sql_server_dict

    if params:
        _check_params_type(params, (list, tuple))
    else:
        params = []

    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if threading.get_native_id not in sql_server_dict:
        sql_server_dict[threading.get_native_id] = {}

    if Constants.CURSOR not in sql_server_dict[threading.get_native_id]:
        sql_server_dict[threading.get_native_id][Constants.CURSOR] = None

    if sql_server_dict[threading.get_native_id][Constants.CURSOR] is None:
        connection = pyodbc.connect(**config)
        cursor = connection.cursor()
        cursor.execute(table_or_sql, params=params)

        sql_server_dict[threading.get_native_id][Constants.CONNECTION] = connection
        sql_server_dict[threading.get_native_id][Constants.CURSOR] = cursor
        sql_server_dict[threading.get_native_id][Constants.COLUMN_NAMES] = [
            column[Constants.I_COLUMN_NAME] for column in cursor.description
        ]


def sql_server(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """
    With migrate.sql_server you can access SQL Server and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters (as in pyodbc.connect),
    :param config_path: Path to the JSON file containing configuration parameters (as in pyodbc.connect)
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The result table as a stream of rows
    """
    global sql_server_dict

    cursor = sql_server_dict[threading.get_native_id][Constants.CURSOR]
    column_names = sql_server_dict[threading.get_native_id][Constants.COLUMN_NAMES]
    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_sql_server():
    global sql_server_dict
    sql_server_dict[threading.get_native_id][Constants.CURSOR] = None
    sql_server_dict[threading.get_native_id][Constants.CONNECTION].close()
    sql_server_dict[threading.get_native_id][Constants.CONNECTION].commit()
    sql_server_dict[threading.get_native_id][Constants.CONNECTION] = None
    sql_server_dict[threading.get_native_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(sql_server, init_migrate_sql_server, cleanup_migrate_sql_server)

### Oracle DB

oracle_db_dict = {}


def init_migrate_oracle_db(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
):
    global oracle_db_dict

    if params:
        _check_params_type(params)

    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if not config:
        config = {}

    # To prevent query execution from hanging
    if "disable_oob" not in config:
        config["disable_oob"] = True

    if threading.get_native_id not in oracle_db_dict:
        oracle_db_dict[threading.get_native_id] = {}

    if Constants.CURSOR not in oracle_db_dict[threading.get_native_id]:
        oracle_db_dict[threading.get_native_id][Constants.CURSOR] = None

    if oracle_db_dict[threading.get_native_id][Constants.CURSOR] is None:
        connection = oracledb.connect(**config)
        cursor = connection.cursor(buffered=True)

        if not params:
            cursor.execute(table_or_sql)
        elif isinstance(params, (list, tuple)):
            cursor.execute(table_or_sql, params)
        else:
            cursor.execute(table_or_sql, **params)

        oracle_db_dict[threading.get_native_id][Constants.CONNECTION] = connection
        oracle_db_dict[threading.get_native_id][Constants.CURSOR] = cursor
        oracle_db_dict[threading.get_native_id][Constants.COLUMN_NAMES] = [
            column[Constants.I_COLUMN_NAME] for column in cursor.description
        ]


def oracle_db(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """
    With migrate.oracle_db you can access Oracle DB and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters (as in oracledb.connect),
    :param config_path: Path to the JSON file containing configuration parameters (as in oracledb.connect)
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The result table as a stream of rows
    """

    global oracle_db_dict
    cursor = oracle_db_dict[threading.get_native_id][Constants.CURSOR]
    column_names = oracle_db_dict[threading.get_native_id][Constants.COLUMN_NAMES]
    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_oracle_db():
    global oracle_db_dict
    oracle_db_dict[threading.get_native_id][Constants.CURSOR] = None
    oracle_db_dict[threading.get_native_id][Constants.CONNECTION].close()
    oracle_db_dict[threading.get_native_id][Constants.CONNECTION].commit()
    oracle_db_dict[threading.get_native_id][Constants.CONNECTION] = None
    oracle_db_dict[threading.get_native_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(oracle_db, init_migrate_oracle_db, cleanup_migrate_oracle_db)


def _query_is_table(table_or_sql: str) -> bool:
    return len(table_or_sql.split()) == 1


def _load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, mode="r") as config:
            return json.load(config)
    except Exception:
        raise OSError("Could not open/read file.")


def _combine_config(config: mgp.Map, config_path: str) -> Dict[str, Any]:
    assert len(config_path), "Path must not be empty"
    config_items = _load_config(path=config_path)

    for key, value in config_items.items():
        config[key] = value
    return config


def _name_row_cells(row_cells, column_names) -> Dict[str, Any]:
    return dict(map(lambda column, value: (column, value), column_names, row_cells))


def _check_params_type(params: Any, types=(dict, list, tuple)) -> None:
    if not isinstance(params, types):
        raise TypeError(
            "Database query parameter values must be passed in a container of type List[Any] or Map if supported."
        )
