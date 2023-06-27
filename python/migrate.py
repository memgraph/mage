import json
from typing import Any, Dict, Union

import mgp
import mysql.connector as mysql_connector
import oracledb
import pyodbc

I_COLUMN_NAME = 0


@mgp.read_proc
def mysql(
    config: Any,
    table_or_sql: str,
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """
    With migrate.mysql you can access MySQL and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures.

    :param config: Connection configuration parameters, or a JSON file with them (as in mysql.connector.connect)
    :param table_or_sql: Table name or an SQL query
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The result table as a stream of rows
    """

    if not isinstance(config, dict) and not isinstance(config, str):
        pass  # TODO

    if isinstance(config, str):
        config = _load_config(path=config)

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"MySQL query parameter values must be passed in a container of type List[Any] or Map."
        )

    with mysql_connector.connect(**config) as connection:
        cursor = connection.cursor()

        cursor.execute(table_or_sql, params=params)

        return [
            mgp.Record(row=_name_row_cells(raw_row, cursor))
            for raw_row in cursor.fetchall()
        ]


@mgp.read_proc
def oracle_db(
    config: Any, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
) -> mgp.Record(row=mgp.Map):
    """
    With migrate.oracle_db you can access Oracle DB and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures.

    :param config: Connection configuration parameters, or a JSON file with them (as in oracledb.connect)
    :param table_or_sql: Table name or an SQL query (Oracle Database doesn’t allow trailing semicolons for SQL code)
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The result table as a stream of rows
    """

    if not isinstance(config, dict) and not isinstance(config, str):
        pass  # TODO

    if isinstance(config, str):
        config = _load_config(path=config)

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

        if not params:
            cursor.execute(table_or_sql)
        elif isinstance(params, (list, tuple)):
            cursor.execute(table_or_sql, params)
        else:
            cursor.execute(table_or_sql, **params)

        return [
            mgp.Record(row=_name_row_cells(raw_row, cursor))
            for raw_row in cursor.fetchall()
        ]


@mgp.read_proc
def sql_server(
    config: Any,
    table_or_sql: str,
    params: mgp.Nullable[mgp.List[mgp.Any]] = None,
) -> mgp.Record(row=mgp.Map):
    """
    With migrate.sql_server you can access SQL Server and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures.

    :param config: Connection configuration parameters, or a JSON file with them (as in pyodbc.connect)
    :param table_or_sql: Table name or an SQL query
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The result table as a stream of rows
    """

    if not isinstance(config, dict) and not isinstance(config, str):
        pass  # TODO

    if isinstance(config, str):
        config = _load_config(path=config)

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if not params:
        params = []

    with pyodbc.connect(**config) as connection:
        cursor = connection.cursor()

        cursor.execute(table_or_sql, *params)

        return [
            mgp.Record(row=_name_row_cells(raw_row, cursor))
            for raw_row in cursor.fetchall()
        ]


def _query_is_table(table_or_sql: str) -> bool:
    return len(table_or_sql.split()) == 1


def _load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, mode="r") as config:
            return json.load(config)
    except Exception:
        raise OSError("Could not open/read file.")


def _name_row_cells(row_cells, cursor) -> Dict[str, Any]:
    column_names = [column[I_COLUMN_NAME] for column in cursor.description]

    return dict(map(lambda column, value: (column, value), column_names, row_cells))
