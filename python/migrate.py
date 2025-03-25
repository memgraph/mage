import base64
from decimal import Decimal
import boto3
import csv
import duckdb as duckDB
import io
from gqlalchemy import Memgraph, Neo4j
import json
import mgp
import mysql.connector as mysql_connector
import oracledb
import os
import pyodbc
import psycopg2
import pyarrow.flight as flight
import re
import threading
from typing import Any, Dict, List


class Constants:
    BATCH_SIZE = 1000
    COLUMN_NAMES = "column_names"
    CONNECTION = "connection"
    CURSOR = "cursor"
    HOST = "host"
    I_COLUMN_NAME = 0
    PASSWORD = "password"
    PORT = "port"
    USERNAME = "username"


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
        
    thread_id = threading.get_native_id()
    if thread_id not in mysql_dict:
        mysql_dict[thread_id] = {}

    if Constants.CURSOR not in mysql_dict[thread_id]:
        mysql_dict[thread_id][Constants.CURSOR] = None

    if mysql_dict[thread_id][Constants.CURSOR] is None:
        connection = mysql_connector.connect(**config)
        cursor = connection.cursor()
        cursor.execute(table_or_sql, params=params)

        mysql_dict[thread_id][Constants.CONNECTION] = connection
        mysql_dict[thread_id][Constants.CURSOR] = cursor
        mysql_dict[thread_id][Constants.COLUMN_NAMES] = [
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
    
    thread_id = threading.get_native_id()
    cursor = mysql_dict[thread_id][Constants.CURSOR]
    column_names = mysql_dict[thread_id][Constants.COLUMN_NAMES]

    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_mysql():
    global mysql_dict
    
    thread_id = threading.get_native_id()
    mysql_dict[thread_id][Constants.CURSOR] = None
    mysql_dict[thread_id][Constants.CONNECTION].commit()
    mysql_dict[thread_id][Constants.CONNECTION].close()
    mysql_dict[thread_id][Constants.CONNECTION] = None
    mysql_dict[thread_id][Constants.COLUMN_NAMES] = None


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
        
    thread_id = threading.get_native_id()
    if thread_id not in sql_server_dict:
        sql_server_dict[thread_id] = {}

    if Constants.CURSOR not in sql_server_dict[thread_id]:
        sql_server_dict[thread_id][Constants.CURSOR] = None

    if sql_server_dict[thread_id][Constants.CURSOR] is None:
        connection = pyodbc.connect(**config)
        cursor = connection.cursor()
        cursor.execute(table_or_sql, *params)

        sql_server_dict[thread_id][Constants.CONNECTION] = connection
        sql_server_dict[thread_id][Constants.CURSOR] = cursor
        sql_server_dict[thread_id][Constants.COLUMN_NAMES] = [
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
    
    thread_id = threading.get_native_id()
    cursor = sql_server_dict[thread_id][Constants.CURSOR]
    column_names = sql_server_dict[thread_id][Constants.COLUMN_NAMES]
    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_sql_server():
    global sql_server_dict

    thread_id = threading.get_native_id()
    sql_server_dict[thread_id][Constants.CURSOR] = None
    sql_server_dict[thread_id][Constants.CONNECTION].commit()
    sql_server_dict[thread_id][Constants.CONNECTION].close()
    sql_server_dict[thread_id][Constants.CONNECTION] = None
    sql_server_dict[thread_id][Constants.COLUMN_NAMES] = None


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
        table_or_sql = f"SELECT * FROM {table_or_sql}"

    if not config:
        config = {}

    # To prevent query execution from hanging
    config["disable_oob"] = True
    
    thread_id = threading.get_native_id()
    if thread_id not in oracle_db_dict:
        oracle_db_dict[thread_id] = {}

    if Constants.CURSOR not in oracle_db_dict[thread_id]:
        oracle_db_dict[thread_id][Constants.CURSOR] = None

    if oracle_db_dict[thread_id][Constants.CURSOR] is None:
        connection = oracledb.connect(**config)
        cursor = connection.cursor()

        if not params:
            cursor.execute(table_or_sql)
        elif isinstance(params, (list, tuple)):
            cursor.execute(table_or_sql, params)
        else:
            cursor.execute(table_or_sql, **params)

        oracle_db_dict[thread_id][Constants.CONNECTION] = connection
        oracle_db_dict[thread_id][Constants.CURSOR] = cursor
        oracle_db_dict[thread_id][Constants.COLUMN_NAMES] = [
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

    thread_id = threading.get_native_id()
    cursor = oracle_db_dict[thread_id][Constants.CURSOR]
    column_names = oracle_db_dict[thread_id][Constants.COLUMN_NAMES]
    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_oracle_db():
    global oracle_db_dict

    thread_id = threading.get_native_id()
    oracle_db_dict[thread_id][Constants.CURSOR] = None
    oracle_db_dict[thread_id][Constants.CONNECTION].commit()
    oracle_db_dict[thread_id][Constants.CONNECTION].close()
    oracle_db_dict[thread_id][Constants.CONNECTION] = None
    oracle_db_dict[thread_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(oracle_db, init_migrate_oracle_db, cleanup_migrate_oracle_db)


# PostgreSQL dictionary to store connections and cursors by thread
postgres_dict = {}


def init_migrate_postgresql(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
):
    global postgres_dict

    if params:
        _check_params_type(params, (list, tuple))
    else:
        params = []

    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    if _query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    thread_id = threading.get_native_id()
    if thread_id not in postgres_dict:
        postgres_dict[thread_id] = {}

    if Constants.CURSOR not in postgres_dict[thread_id]:
        postgres_dict[thread_id][Constants.CURSOR] = None

    if postgres_dict[thread_id][Constants.CURSOR] is None:
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        cursor.execute(table_or_sql, params)

        postgres_dict[thread_id][Constants.CONNECTION] = connection
        postgres_dict[thread_id][Constants.CURSOR] = cursor
        postgres_dict[thread_id][Constants.COLUMN_NAMES] = [
            column.name for column in cursor.description
        ]


def postgresql(
    table_or_sql: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """
    With migrate.postgresql you can access PostgreSQL and execute queries. The result table is converted into a stream,
    and returned rows can be used to create or create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters (as in psycopg2.connect),
    :param config_path: Path to the JSON file containing configuration parameters (as in psycopg2.connect)
    :param params: Optionally, queries may be parameterized. In that case, `params` provides parameter values
    :return: The result table as a stream of rows
    """
    global postgres_dict
    
    thread_id = threading.get_native_id()
    cursor = postgres_dict[thread_id][Constants.CURSOR]
    column_names = postgres_dict[thread_id][Constants.COLUMN_NAMES]

    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_postgresql():
    global postgres_dict
    
    thread_id = threading.get_native_id()
    postgres_dict[thread_id][Constants.CURSOR] = None
    postgres_dict[thread_id][Constants.CONNECTION].commit()
    postgres_dict[thread_id][Constants.CONNECTION].close()
    postgres_dict[thread_id][Constants.CONNECTION] = None
    postgres_dict[thread_id][Constants.COLUMN_NAMES] = None


mgp.add_batch_read_proc(postgresql, init_migrate_postgresql, cleanup_migrate_postgresql)


##### S3
s3_dict = {}


def init_migrate_s3(
    file_path: str,
    config: mgp.Map,
    config_path: str = "",
):
    """
    Initialize an S3 connection and prepare to stream a CSV file.

    :param file_path: S3 file path in the format 's3://bucket-name/path/to/file.csv'
    :param config: Configuration map containing AWS credentials (access_key, secret_key, region, etc.)
    :param config_path: Path to a JSON file containing configuration parameters
    """
    global s3_dict

    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    # Extract S3 bucket and key
    if not file_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Expected 's3://bucket-name/path'.")

    file_path_no_protocol = file_path[5:]
    bucket_name, *key_parts = file_path_no_protocol.split("/")
    s3_key = "/".join(key_parts)

    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=config.get(
            "aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID", None)
        ),
        aws_secret_access_key=config.get(
            "aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY", None)
        ),
        aws_session_token=config.get(
            "aws_session_token", os.getenv("AWS_SESSION_TOKEN", None)
        ),
        region_name=config.get("region_name", os.getenv("AWS_REGION", None)),
    )

    # Fetch and read file as a streaming object
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    # Convert binary stream to text stream
    text_stream = io.TextIOWrapper(response["Body"], encoding="utf-8")

    # Read CSV headers
    csv_reader = csv.reader(text_stream)
    column_names = next(csv_reader)  # First row contains column names

    thread_id = threading.get_native_id()
    if thread_id not in s3_dict:
        s3_dict[thread_id] = {}

    s3_dict[thread_id][Constants.CURSOR] = csv_reader
    s3_dict[thread_id][Constants.COLUMN_NAMES] = column_names


def s3(
    file_path: str,
    config: mgp.Map,
    config_path: str = "",
) -> mgp.Record(row=mgp.Map):
    """
    Fetch rows from an S3 CSV file in batches.

    :param file_path: S3 file path in the format 's3://bucket-name/path/to/file.csv'
    :param config: AWS S3 connection parameters (AWS credentials, region, etc.)
    :param config_path: Optional path to a JSON file containing AWS credentials
    :return: The result table as a stream of rows
    """
    global s3_dict
    
    thread_id = threading.get_native_id()
    csv_reader = s3_dict[thread_id][Constants.CURSOR]
    column_names = s3_dict[thread_id][Constants.COLUMN_NAMES]

    batch_rows = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = next(csv_reader)
            batch_rows.append(mgp.Record(row=_name_row_cells(row, column_names)))
        except StopIteration:
            break

    return batch_rows


def cleanup_migrate_s3():
    """
    Clean up S3 dictionary references per-thread.
    """
    global s3_dict
    
    thread_id = threading.get_native_id()
    s3_dict.pop(thread_id, None)


mgp.add_batch_read_proc(s3, init_migrate_s3, cleanup_migrate_s3)


neo4j_dict = {}


def init_migrate_neo4j(
    label_or_rel_or_query: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
):
    global neo4j_dict

    thread_id = threading.get_native_id()
    if thread_id not in neo4j_dict:
        neo4j_dict[thread_id] = {}

    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    neo4j_db = Neo4j(**config)
    query = _formulate_cypher_query(label_or_rel_or_query)
    cursor = neo4j_db.execute_and_fetch(query, params)

    neo4j_dict[thread_id][Constants.CONNECTION] = neo4j_db
    neo4j_dict[thread_id][Constants.CURSOR] = cursor


def neo4j(
    label_or_rel_or_query: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """
    Migrate data from Neo4j to Memgraph. Can migrate a specific node label, relationship type, or execute a custom Cypher query.

    :param label_or_rel_or_query: Node label, relationship type, or a Cypher query
    :param config: Connection configuration for Neo4j
    :param config_path: Path to a JSON file containing connection parameters
    :param params: Optional query parameters
    :return: Stream of rows from Neo4j
    """
    global neo4j_dict
    
    thread_id = threading.get_native_id()
    cursor = neo4j_dict[thread_id][Constants.CURSOR]

    return [
        mgp.Record(row=row)
        for row in (next(cursor, None) for _ in range(Constants.BATCH_SIZE))
        if row is not None
    ]


def cleanup_migrate_neo4j():
    global neo4j_dict
    
    thread_id = threading.get_native_id()
    if Constants.CONNECTION in neo4j_dict[thread_id]:
        neo4j_dict[thread_id][Constants.CONNECTION].close()
    neo4j_dict.pop(thread_id, None)


mgp.add_batch_read_proc(neo4j, init_migrate_neo4j, cleanup_migrate_neo4j)


# Dictionary to store Flight connections per thread
flight_dict = {}


def init_migrate_arrow_flight(
    query: str,
    config: mgp.Map,
    config_path: str = "",
):
    global flight_dict

    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    host = config.get(Constants.HOST, None)
    port = config.get(Constants.PORT, None)
    username = config.get(Constants.USERNAME, "")
    password = config.get(Constants.PASSWORD, "")

    # Encode credentials
    auth_string = f"{username}:{password}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_string).decode("utf-8")

    # Establish Flight connection
    client = flight.connect(f"grpc://{host}:{port}")

    # Authenticate
    options = flight.FlightCallOptions(
        headers=[(b"authorization", f"Basic {encoded_auth}".encode("utf-8"))]
    )

    flight_info = client.get_flight_info(
        flight.FlightDescriptor.for_command(query), options
    )

    # Store connection per thread
    thread_id = threading.get_native_id()
    if thread_id not in flight_dict:
        flight_dict[thread_id] = {}

    flight_dict[thread_id][Constants.CONNECTION] = client
    flight_dict[thread_id][Constants.CURSOR] = iter(
        _fetch_flight_data(client, flight_info, options)
    )


def _fetch_flight_data(client, flight_info, options):
    """
    Efficiently fetches data in batches from Arrow Flight using RecordBatchReader.
    This prevents high memory usage by avoiding full table loading.
    """
    for endpoint in flight_info.endpoints:
        reader = client.do_get(endpoint.ticket, options)  # Stream the data
        for chunk in reader:  # Iterate over RecordBatches
            batch = chunk.data  # Convert each batch to an Arrow Table
            yield from batch.to_pylist()  # Convert to row dictionaries on demand


def arrow_flight(
    query: str,
    config: mgp.Map,
    config_path: str = "",
) -> mgp.Record(row=mgp.Map):
    """
    Execute a SQL query on Arrow Flight and stream results into Memgraph.

    :param query: SQL query to execute
    :param config: Arrow Flight connection configuration
    :param config_path: Path to a JSON config file
    :return: Stream of rows from Arrow Flight
    """
    global flight_dict

    thread_id = threading.get_native_id()
    cursor = flight_dict[thread_id][Constants.CURSOR]
    batch = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = _convert_row_types(next(cursor))
            batch.append(mgp.Record(row=row))
        except StopIteration:
            break

    return batch


def cleanup_migrate_arrow_flight():
    """
    Close the Flight connection per-thread.
    """
    global flight_dict

    thread_id = threading.get_native_id()
    if thread_id in flight_dict:
        flight_dict.pop(thread_id, None)


mgp.add_batch_read_proc(
    arrow_flight, init_migrate_arrow_flight, cleanup_migrate_arrow_flight
)


# Dictionary to store DuckDB connections and cursors per thread
duckdb_dict = {}


def init_migrate_duckdb(query: str, setup_queries: mgp.Nullable[List[str]] = None):
    """
    Initialize an in-memory DuckDB connection and execute the query.

    :param query: SQL query to execute
    :param config: Unused but kept for consistency with other migration functions
    :param config_path: Unused but kept for consistency with other migration functions
    """
    global duckdb_dict

    thread_id = threading.get_native_id()
    if thread_id not in duckdb_dict:
        duckdb_dict[thread_id] = {}

    # Ensure a fresh in-memory DuckDB instance for each thread
    connection = duckDB.connect()
    cursor = connection.cursor()
    if setup_queries is not None:
        for setup_query in setup_queries:
            cursor.execute(setup_query)

    cursor.execute(query)

    duckdb_dict[thread_id][Constants.CONNECTION] = connection
    duckdb_dict[thread_id][Constants.CURSOR] = cursor
    duckdb_dict[thread_id][Constants.COLUMN_NAMES] = [
        desc[0] for desc in cursor.description
    ]


def duckdb(query: str, setup_queries: mgp.Nullable[List[str]] = None) -> mgp.Record(row=mgp.Map):
    """
    Fetch rows from DuckDB in batches.

    :param query: SQL query to execute
    :param config: Unused but kept for consistency with other migration functions
    :param config_path: Unused but kept for consistency with other migration functions
    :return: The result table as a stream of rows
    """
    global duckdb_dict

    thread_id = threading.get_native_id()
    cursor = duckdb_dict[thread_id][Constants.CURSOR]
    column_names = duckdb_dict[thread_id][Constants.COLUMN_NAMES]

    rows = cursor.fetchmany(Constants.BATCH_SIZE)
    return [mgp.Record(row=_name_row_cells(row, column_names)) for row in rows]


def cleanup_migrate_duckdb():
    """
    Clean up DuckDB dictionary references per-thread.
    """
    global duckdb_dict

    thread_id = threading.get_native_id()
    if thread_id in duckdb_dict:
        if Constants.CONNECTION in duckdb_dict[thread_id]:
            duckdb_dict[thread_id][Constants.CONNECTION].close()
        duckdb_dict.pop(thread_id, None)


mgp.add_batch_read_proc(duckdb, init_migrate_duckdb, cleanup_migrate_duckdb)


memgraph_dict = {}


def init_migrate_memgraph(
    label_or_rel_or_query: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
):
    global memgraph_dict
    
    thread_id = threading.get_native_id()
    if thread_id not in memgraph_dict:
        memgraph_dict[thread_id] = {}

    if len(config_path) > 0:
        config = _combine_config(config=config, config_path=config_path)

    memgraph_db = Memgraph(**config)
    query = _formulate_cypher_query(label_or_rel_or_query)
    cursor = memgraph_db.execute_and_fetch(query, params)

    neo4j_dict[thread_id][Constants.CONNECTION] = memgraph_db
    neo4j_dict[thread_id][Constants.CURSOR] = cursor


def memgraph(
    label_or_rel_or_query: str,
    config: mgp.Map,
    config_path: str = "",
    params: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """
    Migrate data from Memgraph to another Memgraph instance. Can migrate a specific node label, relationship type, or execute a custom Cypher query.

    :param label_or_rel_or_query: Node label, relationship type, or a Cypher query
    :param config: Connection configuration for Memgraph
    :param config_path: Path to a JSON file containing connection parameters
    :param params: Optional query parameters
    :return: Stream of rows from Memgraph
    """
    global memgraph_dict
    
    thread_id = threading.get_native_id()
    cursor = memgraph_dict[thread_id][Constants.CURSOR]

    return [
        mgp.Record(row=row)
        for row in (next(cursor, None) for _ in range(Constants.BATCH_SIZE))
        if row is not None
    ]


def cleanup_migrate_memgraph():
    global memgraph_dict
    
    thread_id = threading.get_native_id()
    if Constants.CONNECTION in memgraph_dict[thread_id]:
        memgraph_dict[threading.get_native_id][Constants.CONNECTION].close()
    memgraph_dict.pop(threading.get_native_id, None)


mgp.add_batch_read_proc(memgraph, init_migrate_memgraph, cleanup_migrate_memgraph)


def _formulate_cypher_query(label_or_rel_or_query: str) -> str:
    words = label_or_rel_or_query.split()
    if len(words) > 1:
        return (
            label_or_rel_or_query  # Treat it as a Cypher query if multiple words exist
        )
    node_match = re.match(r"^\(\s*:(\w+)\s*\)$", label_or_rel_or_query)
    rel_match = re.match(r"^\[\s*:(\w+)\s*\]$", label_or_rel_or_query)

    if node_match:
        label = node_match.group(1)
        return (
            f"MATCH (n:{label}) RETURN labels(n) as labels, properties(n) as properties"
        )
    elif rel_match:
        rel_type = rel_match.group(1)
        return f"""
    MATCH (n)-[r:{rel_type}]->(m)
    RETURN 
        labels(n) as from_labels,
        labels(m) as to_labels, 
        properties(n) as from_properties, 
        properties(r) as edge_properties, 
        properties(m) as to_properties
    """
    return label_or_rel_or_query  # Assume it's a valid query


def _query_is_table(table_or_sql: str) -> bool:
    return len(table_or_sql.split()) == 1


def _combine_config(config: mgp.Map, config_path: str) -> Dict[str, Any]:
    assert len(config_path), "Path must not be empty"

    file_config = None
    try:
        with open(config_path, "r") as file:
            file_config = json.load(file)
    except Exception:
        raise OSError("Could not open/read file.")

    config.update(file_config)
    return config


def _name_row_cells(row_cells, column_names) -> Dict[str, Any]:
    return {
        column: (value if not isinstance(value, Decimal) else float(value))
        for column, value in zip(column_names, row_cells)
    }


def _convert_row_types(row_cells) -> Dict[str, Any]:
    return {
        column: (value if not isinstance(value, Decimal) else float(value))
        for column, value in row_cells.items()
    }


def _check_params_type(params: Any, types=(dict, list, tuple)) -> None:
    if not isinstance(params, types):
        raise TypeError(
            "Database query parameter values must be passed in a container of type List[Any] (or Map, if migrating from MySQL or Oracle DB)"
        )
