from neo4j import GraphDatabase
from neo4j.time import DateTime as Neo4jDateTime
from neo4j.time import Date as Neo4jDate
import json
import mgp
import re
import threading
from typing import Any, Dict


class Constants:
    BATCH_SIZE = 1000
    COLUMN_NAMES = "column_names"
    CONNECTION = "connection"
    CURSOR = "cursor"
    HOST = "host"
    I_COLUMN_NAME = 0
    PASSWORD = "password"
    PORT = "port"
    RESULT = "result"
    USERNAME = "username"
    SESSION = "session"
    DRIVER = "driver"
    DATABASE = "database"


neo4j_dict = {}


def _convert_neo4j_value(value):
    """Convert Neo4j values to Python-compatible formats."""
    if value is None:
        return None
    
    # Handle Neo4j DateTime objects
    try:
        if isinstance(value, Neo4jDateTime) or isinstance(value, Neo4jDate):
            return value.to_native()
    except ImportError:
        pass
    
    # Handle lists and dicts recursively
    if isinstance(value, list):
        return [_convert_neo4j_value(item) for item in value]
    
    if isinstance(value, dict):
        return {key: _convert_neo4j_value(val) for key, val in value.items()}
    
    # For other types, return as is
    return value


def _convert_neo4j_record(record):
    """Convert a Neo4j record to a Python dict with proper type conversion."""
    return {key: _convert_neo4j_value(value) for key, value in record.items()}


def _build_neo4j_uri(config: mgp.Map) -> str:
    host = config.get(Constants.HOST, "localhost")
    port = config.get(Constants.PORT, 7687)
    return f"bolt://{host}:{port}"


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

    uri = _build_neo4j_uri(config)
    username = config.get(Constants.USERNAME, "neo4j")
    password = config.get(Constants.PASSWORD, "password")
    database = config.get(Constants.DATABASE, None)  # None means default database
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Create session with optional database parameter
    if database:
        session = driver.session(database=database)
    else:
        session = driver.session()
        
    query = _formulate_cypher_query(label_or_rel_or_query)
    # Neo4j expects params to be a dict or None
    cypher_params = params if params is not None else {}
    result = session.run(query, parameters=cypher_params)

    neo4j_dict[thread_id][Constants.DRIVER] = driver
    neo4j_dict[thread_id][Constants.SESSION] = session
    neo4j_dict[thread_id][Constants.RESULT] = result


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
    result = neo4j_dict[thread_id][Constants.RESULT]

    # Fetch up to BATCH_SIZE records
    batch = []
    for record in result:
        # Convert neo4j.Record to dict with proper type conversion
        batch.append(mgp.Record(row=_convert_neo4j_record(record)))
        
        # Check if we've reached the batch size limit
        if len(batch) >= Constants.BATCH_SIZE:
            break
    
    return batch


def cleanup_migrate_neo4j():
    global neo4j_dict

    thread_id = threading.get_native_id()
    session = neo4j_dict[thread_id].get(Constants.SESSION)
    driver = neo4j_dict[thread_id].get(Constants.DRIVER)
    if session:
        session.close()
    if driver:
        driver.close()
    neo4j_dict.pop(thread_id, None)


mgp.add_batch_read_proc(neo4j, init_migrate_neo4j, cleanup_migrate_neo4j)


def _formulate_cypher_query(label_or_rel_or_query: str) -> str:
    words = label_or_rel_or_query.split()
    if len(words) > 1:
        return (
            label_or_rel_or_query  # Treat it as a Cypher query if multiple words exist
        )

    # Try to see if the syntax matches similar to (:Label) to migrate only nodes
    node_match = re.match(r"^\(\s*:(\w+)\s*\)$", label_or_rel_or_query)

    # Try to see if the syntax matches similar to [:REL_TYPE] to migrate only relationships
    rel_match = re.match(r"^\[\s*:(\w+)\s*\]$", label_or_rel_or_query)

    if node_match:
        label = node_match.group(1)
        return (
            f"MATCH (n:{label}) RETURN labels(n) as labels, properties(n) as properties"
        )

    if rel_match:
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