import mgp
import mysql.connector
import oracledb
import pyodbc
import threading

from typing import Dict, Any


class Constants:
    I_COLUMN_NAME = 0
    CURSOR = "cursor"
    COLUMN_NAMES = "column_names"
    CONNECTION = "connection"
    BATCH_SIZE = 1000


def _query_is_table(table_or_sql: str) -> bool:
    return len(table_or_sql.split()) == 1


def _name_row_cells(row_cells, column_names) -> Dict[str, Any]:
    return dict(map(lambda column, value: (column, value), column_names, row_cells))


##### MYSQL

mysql_dict = {}


def init_migrate_mysql(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
):
    global mysql_dict
    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"MySQL query parameter values must be passed in a container of type List[Any] or Map."
        )
    if threading.get_native_id not in mysql_dict:
        mysql_dict[threading.get_native_id] = {}

    if Constants.CURSOR not in mysql_dict[threading.get_native_id]:
        mysql_dict[threading.get_native_id][Constants.CURSOR] = None

    if mysql_dict[threading.get_native_id][Constants.CURSOR] is None:
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor(buffered=True)
        cursor.execute(table_or_sql, params=params)

        mysql_dict[threading.get_native_id][Constants.CONNECTION] = connection
        mysql_dict[threading.get_native_id][Constants.CURSOR] = cursor
        mysql_dict[threading.get_native_id][Constants.COLUMN_NAMES] = [
            column[Constants.I_COLUMN_NAME] for column in cursor.description
        ]


def migrate_mysql(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
) -> mgp.Record(row=mgp.Any):
    global mysql_dict
    cursor = mysql_dict[threading.get_native_id][Constants.CURSOR]
    column_names = mysql_dict[threading.get_native_id][Constants.COLUMN_NAMES]
    results = []
    rows = cursor.fetchmany(Constants.BATCH_SIZE)
    for row in rows:
        results.append(mgp.Record(row=_name_row_cells(row, column_names)))
    return results


def cleanup_migrate_mysql():
    global mysql_dict
    mysql_dict[threading.get_native_id][Constants.CURSOR] = None
    mysql_dict[threading.get_native_id][Constants.CONNECTION].close()
    mysql_dict[threading.get_native_id][Constants.CONNECTION].commit()
    mysql_dict[threading.get_native_id][Constants.CONNECTION] = None
    mysql_dict[threading.get_native_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(migrate_mysql, init_migrate_mysql, cleanup_migrate_mysql)

### SQL SERVER

sql_server_dict = {}


def init_migrate_sql_server(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
):
    global sql_server_dict

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"SQL Server query parameter values must be passed in a container of type List[Any] or Map."
        )
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


def migrate_sql_server(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
) -> mgp.Record(row=mgp.Any):
    global sql_server_dict
    cursor = sql_server_dict[threading.get_native_id][Constants.CURSOR]
    column_names = sql_server_dict[threading.get_native_id][Constants.COLUMN_NAMES]
    results = []
    rows = cursor.fetchmany(Constants.BATCH_SIZE)
    for row in rows:
        results.append(mgp.Record(row=_name_row_cells(row, column_names)))
    return results


def cleanup_migrate_sql_server():
    global sql_server_dict
    sql_server_dict[threading.get_native_id][Constants.CURSOR] = None
    sql_server_dict[threading.get_native_id][Constants.CONNECTION].close()
    sql_server_dict[threading.get_native_id][Constants.CONNECTION].commit()
    sql_server_dict[threading.get_native_id][Constants.CONNECTION] = None
    sql_server_dict[threading.get_native_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(migrate_sql_server, init_migrate_sql_server, migrate_sql_server)

### Oracle DB

oracle_db_dict = {}


def init_migrate_oracle_db(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
):
    global oracle_db_dict

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"SQL Server query parameter values must be passed in a container of type List[Any] or Map."
        )

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


def migrate_oracle_db(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
) -> mgp.Record(row=mgp.Any):
    global oracle_db_dict
    cursor = oracle_db_dict[threading.get_native_id][Constants.CURSOR]
    column_names = oracle_db_dict[threading.get_native_id][Constants.COLUMN_NAMES]
    results = []
    rows = cursor.fetchmany(Constants.BATCH_SIZE)
    for row in rows:
        results.append(mgp.Record(row=_name_row_cells(row, column_names)))
    return results


def cleanup_migrate_oracle_db():
    global oracle_db_dict
    oracle_db_dict[threading.get_native_id][Constants.CURSOR] = None
    oracle_db_dict[threading.get_native_id][Constants.CONNECTION].close()
    oracle_db_dict[threading.get_native_id][Constants.CONNECTION].commit()
    oracle_db_dict[threading.get_native_id][Constants.CONNECTION] = None
    oracle_db_dict[threading.get_native_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(
    migrate_oracle_db, init_migrate_oracle_db, cleanup_migrate_oracle_db
)
