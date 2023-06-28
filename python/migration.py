

import mgp
import mysql.connector


def _query_is_table(table_or_sql: str) -> bool:
    return len(table_or_sql.split()) == 1


def _name_row_cells(row_cells, column_names):
    return dict(map(lambda column, value: (column, value), column_names, row_cells))




@mgp.read_proc
def migrate(
    config: mgp.Map, table_or_sql: str, params: mgp.Nullable[mgp.Any] = None
) -> mgp.Record(row=mgp.Any):
    global mysql_dict
    
    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    if params and not isinstance(params, (dict, list, tuple)):
        raise TypeError(
            f"MySQL query parameter values must be passed in a container of type List[Any] or Map."
        )
    

    connection = mysql.connector.connect(
        **config
    )
    cursor = connection.cursor(buffered=True)
    cursor.execute(table_or_sql, params=params)
    column_names = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    results = []
    for row in rows:
        results.append(mgp.Record(row=_name_row_cells(row, column_names)))
    return results



