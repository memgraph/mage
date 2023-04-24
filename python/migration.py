import mgp
import mysql.connector
import oracledb
import pyodbc

from typing import Dict, Any

from mage.migration_module import Results

I_COLUMN_NAME = 0


def _query_is_table(table_or_sql: str) -> bool:
    return len(table_or_sql.split()) == 1


@mgp.read_proc
def migrate_mysql(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
) -> mgp.Record(record_size=int):


    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"MySQL query parameter values must be passed in a container of type List[Any] or Map."
        )
    with mysql.connector.connect(**config) as connection:
        cursor = connection.cursor()

        cursor.execute(table_or_sql, params=params)

        Results.column_names = [column[I_COLUMN_NAME] for column in cursor.description]
        Results.records = []
        for raw_row in cursor.fetchall():
            Results.records.append(raw_row)
    
    Results.offset = 0
    Results.start = 0
    Results.end = 0
    return mgp.Record(
        record_size=len(Results.records),
    )


@mgp.read_proc
def migrate_sql_server(
    config: mgp.Map,
    table_or_sql: str,
    params: mgp.Nullable[mgp.List[mgp.Any]] = None,
) -> mgp.Record(record_size=int):


    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if not params:
        params = []

    with pyodbc.connect(**config) as connection:
        cursor = connection.cursor()

        cursor.execute(table_or_sql, *params)
        
        Results.column_names = [column[I_COLUMN_NAME] for column in cursor.description]
        Results.records = []
        for raw_row in cursor.fetchall():
            Results.records.append(raw_row)
    Results.offset = 0
    Results.start = 0
    Results.end = 0
    return mgp.Record(
        record_size=len(Results.records),
    )

@mgp.read_proc
def migrate_oracle_db(
    config: mgp.Map,
    table_or_sql: str,
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(record_size=int):


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

        Results.column_names = [column[I_COLUMN_NAME] for column in cursor.description]
        Results.records = []
        for raw_row in cursor.fetchall():
            Results.records.append(raw_row)
    Results.offset = 0
    Results.start = 0
    Results.end = 0
    return mgp.Record(
        record_size=len(Results.records),
    )



def _name_row_cells(row_cells, column_names) -> Dict[str, Any]:
    return dict(map(lambda column, value: (column, value), column_names, row_cells))


@mgp.read_proc
def yield_records(
  ctx: mgp.ProcCtx,
  chunk_size: mgp.Number
) -> mgp.Record(row=mgp.Any):
  offset = Results.start
  start = offset
  end = offset + chunk_size if offset + chunk_size < len(Results.records) else len(Results.records)  
  offset += chunk_size

  Results.start = start
  Results.end = end
  
  return [mgp.Record(row=_name_row_cells(row, Results.column_names))
          for row in Results.records[start:end]]


@mgp.read_proc
def yield_records_list(
  ctx: mgp.ProcCtx,
  chunk_size: mgp.Number
) -> mgp.Record(row_list=mgp.List[mgp.Number]):
  offset = Results.start
  start = offset
  end = offset + chunk_size if offset + chunk_size < len(Results.records) else len(Results.records)  
  offset += chunk_size

  Results.start = start
  Results.end = end

  rows = [_name_row_cells(row, Results.column_names) for row in Results.records[start:end]]
  return mgp.Record(row_list=rows)
