import csv
from enum import Enum
import os
from typing import Any

import mgp
import mysql.connector as mysql_connector
import oracledb
import pyodbc

CHUNK_SIZE = 10_000
I_COLUMN_NAME = 0


class WriteSize(Enum):
    ONE = "one"
    MULTIPLE = "more"


class Database(Enum):
    MYSQL = "MySQL"
    ORACLE_DB = "Oracle Database"
    SQL_SERVER = "SQL Server"


@mgp.read_proc
def mysql_to_csv(
    config: mgp.Map,
    file_path: str,
    table_or_sql: str,
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(file_path=str):
    """
    With migrate.mysql you can access MySQL and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures.

    :param config: Connection configuration parameters (as in mysql.connector.connect)
    :param file_path: The path to the export destination CSV file
    :param table_or_sql: Table name or an SQL query
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The path to the export destination CSV file
    """

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"MySQL query parameter values must be passed in a container of type List[Any] or Map."
        )

    with mysql_connector.connect(**config) as connection:
        cursor = connection.cursor()

        cursor.execute(table_or_sql, params=params)

        _export_source_db_to_csv(db=Database.MYSQL, cursor=cursor, file_path=file_path)

    return mgp.Record(file_path=file_path)


@mgp.read_proc
def oracle_db_to_csv(
    config: mgp.Map,
    file_path: str,
    table_or_sql: str,
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(file_path=str):
    """
    With migrate.oracle_db you can access Oracle DB and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures.

    :param config: Connection configuration parameters (as in oracledb.connect)
    :param file_path: The path to the export destination CSV file
    :param table_or_sql: Table name or an SQL query (Oracle Database doesn’t allow trailing semicolons for SQL code)
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The path to the export destination CSV file
    """

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql}"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"Oracle Database query parameter values must be passed in a container of type List[Any] or Map."
        )

    if not config:
        config = {}

    # To prevent query execution from hanging
    if "disable_oob" not in config:
        config["disable_oob"] = True

    with oracledb.connect(**config) as connection:
        cursor = connection.cursor()

        cursor.prefetchrows = CHUNK_SIZE + 1
        cursor.arraysize = CHUNK_SIZE

        if not params:
            cursor.execute(table_or_sql)
        elif isinstance(params, (list, tuple)):
            cursor.execute(table_or_sql, params)
        else:
            cursor.execute(table_or_sql, **params)

        _export_source_db_to_csv(
            db=Database.ORACLE_DB, cursor=cursor, file_path=file_path
        )

    return mgp.Record(file_path=file_path)


@mgp.read_proc
def sql_server_to_csv(
    config: mgp.Map,
    file_path: str,
    table_or_sql: str,
    params: mgp.Nullable[mgp.List[mgp.Any]] = None,
) -> mgp.Record(file_path=str):
    """
    With migrate.sql_server you can access SQL Server and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures.

    :param config: Connection configuration parameters (as in pyodbc.connect)
    :param file_path: The path to the export destination CSV file
    :param table_or_sql: Table name or an SQL query
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The path to the export destination CSV file
    """

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if not params:
        params = []

    with pyodbc.connect(**config) as connection:
        cursor = connection.cursor()

        cursor.execute(table_or_sql, *params)

        _export_source_db_to_csv(
            db=Database.SQL_SERVER, cursor=cursor, file_path=file_path
        )

    return mgp.Record(file_path=file_path)


@mgp.read_proc
def delete_csv(file_path: str) -> mgp.Record(file_path=str):
    """
    This procedure deletes the CSV file with the given path. It can be used for
    deleting the files created by this module’s mysql_to_csv, oracle_db_to_csv,
    and sql_server_to_csv procedures.

    :param file_path: The path to the CSV file to be deleted
    :return: The path to the now-deleted CSV file
    """
    try:
        os.remove(path=file_path)
    except PermissionError:
        raise PermissionError(
            "You don't have permission to delete the file. Make sure to give the necessary permissions to user memgraph."
        )
    except IsADirectoryError:
        raise IsADirectoryError("The given path points to a directory.")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find any file at the given path.")
    except Exception:
        raise OSError("Could not delete the file.")

    return mgp.Record(file_path=file_path)


def _export_source_db_to_csv(db: Database, cursor: Any, file_path: str):
    _write_to_csv(
        data=[column[I_COLUMN_NAME] for column in cursor.description],
        size=WriteSize.ONE,
        file_path=file_path,
        mode="w",
    )

    while True:
        rows = (
            cursor.fetchmany(size=CHUNK_SIZE)
            if (db is Database.MYSQL or db is Database.ORACLE_DB)
            else cursor.fetchmany(CHUNK_SIZE)
        )

        if not rows:
            break

        _write_to_csv(
            data=rows,
            size=WriteSize.MULTIPLE,
            file_path=file_path,
            mode="a",
        )


def _write_to_csv(
    data: Any,
    size: WriteSize,
    file_path: str,
    mode: str,
    newline: str = "",
    encoding: str = "utf8",
):
    try:
        with open(file_path, mode=mode, newline=newline, encoding=encoding) as csv_file:
            writer = csv.writer(csv_file)

            if size is WriteSize.ONE:
                writer.writerow(data)
            elif size is WriteSize.MULTIPLE:
                writer.writerows(data)
    except PermissionError:
        raise PermissionError(
            "You don't have permission to write into the file. Make sure to give the necessary permissions to user memgraph."
        )
    except csv.Error as e:
        raise csv.Error(
            "Could not write to the file {}, stopped at line {}: {}".format(
                file_path, writer.line_num, e
            )
        )
    except Exception:
        raise OSError("Could not open or write to the file.")


def _query_is_table(table_or_sql: str) -> bool:
    return len(table_or_sql.split()) == 1
