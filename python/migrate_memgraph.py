import threading
import yaml
import os
from typing import Any, Dict, List
import re

from gqlalchemy import Memgraph
import mgp


class Constants:
    BATCH_SIZE = 1000
    CONFIG_FILE_PATH = (
        "/var/lib/memgraph/internal_modules/migrate_config/memgraph_config.yaml"
    )
    CURSOR = "cursor"
    DATABASE = "database"
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 7687
    DRIVER = "driver"
    HOST = "host"
    MESSAGE = "message"
    PASSWORD = "password"
    PORT = "port"
    RESULT = "result"
    SESSION = "session"
    STATUS = "status"
    USERNAME = "username"
    URI_SCHEME = "uri_scheme"


memgraph_dict = {}


def init_query(
    query: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
):
    """
    Prepare Memgraph cursor for batch streaming.
    - query: Cypher query to execute
    - query_config: { limit?, batch_size? }
    - driver_config: either a map with config or string name to load from YAML
    """
    global memgraph_dict
    thread_id = threading.get_native_id()
    if thread_id not in memgraph_dict:
        memgraph_dict[thread_id] = {}

    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get Memgraph connection
    memgraph_db = _get_memgraph_connection(resolved_config)

    # Convert configs to dict, use empty map if None
    qcfg = dict(query_config) if query_config is not None else {}

    # Apply limit if specified
    limit_val = qcfg.get("limit")
    if limit_val is not None:
        try:
            limit_val = int(limit_val)
            query = f"{query} LIMIT {limit_val}"
        except Exception:
            raise Exception(f"Invalid 'limit' value: {limit_val}")

    # Execute query and get cursor
    cursor = memgraph_db.execute_and_fetch(query)

    # Stash per-thread
    memgraph_dict[thread_id][Constants.DRIVER] = memgraph_db
    memgraph_dict[thread_id][Constants.CURSOR] = cursor
    memgraph_dict[thread_id][Constants.RESULT] = cursor


def query(
    query: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """Stream up to BATCH_SIZE records as rows (per call)."""
    global memgraph_dict
    thread_id = threading.get_native_id()
    cursor = memgraph_dict[thread_id][Constants.RESULT]

    batch = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            record = next(cursor, None)
            if record is None:
                break
            batch.append(mgp.Record(row=_memgraph_to_primitive(record)))
        except Exception as e:
            break

    return batch


def cleanup_query():
    global memgraph_dict
    thread_id = threading.get_native_id()

    connection = memgraph_dict[thread_id].get(Constants.DRIVER)
    if connection:
        try:
            connection.close()
        except Exception as e:
            raise Exception(f"Failed to close connection: {str(e)}")

    memgraph_dict.pop(thread_id, None)


mgp.add_batch_read_proc(query, init_query, cleanup_query)


@mgp.read_proc
def test_connection(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(message=str):
    """
    Test Memgraph connection using configuration.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        Record with connection test results
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get Memgraph connection
    memgraph_db = _get_memgraph_connection(resolved_config)

    # Test Memgraph connection
    result = _test_memgraph_connection(memgraph_db, resolved_config)

    return mgp.Record(message=result)


@mgp.read_proc
def list_labels(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(label=str, count=int):
    """
    List all node labels in the Memgraph database.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per label with name and count
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get Memgraph connection
    memgraph_db = _get_memgraph_connection(resolved_config)

    # Query to get labels
    labels_query = """
    CALL node_labels() YIELD node_labels
    UNWIND node_labels as label
    WITH label, 
         size([(n:label) | n]) as count
    RETURN label, count
    ORDER BY label
    """

    result = memgraph_db.execute_and_fetch(labels_query)
    labels = list(result)

    memgraph_db.close()

    # Return one record per label
    return [
        mgp.Record(label=label["label"], count=label["count"])
        for label in labels
    ]


@mgp.read_proc
def list_relationship_types(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(relationship_type=str, count=int):
    """
    List all relationship types in the Memgraph database.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per relationship type with name and count
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get Memgraph connection
    memgraph_db = _get_memgraph_connection(resolved_config)

    # Query to get relationship types
    rel_types_query = """
    CALL relationship_types() YIELD relationship_types
    UNWIND relationship_types as relationship_type
    WITH relationship_type,
         size([()-[r:relationship_type]->() | r]) as count
    RETURN relationship_type, count
    ORDER BY relationship_type
    """

    result = memgraph_db.execute_and_fetch(rel_types_query)
    rel_types = list(result)

    memgraph_db.close()

    # Return one record per relationship type
    return [
        mgp.Record(relationship_type=rel["relationship_type"], count=rel["count"])
        for rel in rel_types
    ]


@mgp.read_proc
def describe_label(
    label: str,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(property_name=str, property_type=str, count=int):
    """
    Describe properties of a specific node label in Memgraph database.

    Args:
        label: Name of the label to describe
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per property with details
    """
    resolved_config = _resolve_driver_config(driver_config)

    # Get Memgraph connection
    memgraph_db = _get_memgraph_connection(resolved_config)

    # Query to get property information
    describe_query = """
    MATCH (n:`{label}`)
    UNWIND keys(n) as property_name
    WITH property_name, 
         collect(n[property_name])[0] as sample_value,
         count(n[property_name]) as count
    RETURN property_name,
           CASE 
               WHEN sample_value IS NULL THEN 'null'
               WHEN sample_value IS true OR sample_value IS false THEN 'boolean'
               WHEN sample_value IS INTEGER THEN 'integer'
               WHEN sample_value IS FLOAT THEN 'float'
               WHEN sample_value IS STRING THEN 'string'
               WHEN sample_value IS LIST THEN 'list'
               WHEN sample_value IS MAP THEN 'map'
               ELSE 'unknown'
           END as property_type,
           count
    ORDER BY property_name
    """.format(label=label)

    result = memgraph_db.execute_and_fetch(describe_query)
    properties = list(result)

    memgraph_db.close()

    # Return one record per property
    return [
        mgp.Record(
            property_name=prop["property_name"],
            property_type=prop["property_type"],
            count=prop["count"]
        )
        for prop in properties
    ]


@mgp.read_proc
def get_configurations() -> mgp.Record(name=str, config=mgp.Map):
    """
    Get configurations from the YAML file.

    Returns:
        One record per configuration with name and config
    """
    # Load all configurations
    configs = _load_configurations()

    # Return one record per configuration
    return [
        mgp.Record(name=name, config=config_dict)
        for name, config_dict in configs.items()
    ]


@mgp.read_proc
def add_configuration(
    configuration_name: str,
    driver_config: mgp.Map,
) -> mgp.Record(success=bool, message=str):
    """
    Add or update Memgraph configuration in YAML file.

    Args:
        configuration_name: Name of the configuration to save
        driver_config: Memgraph connection configuration

    Returns:
        Record with operation results
    """
    try:
        # Convert mgp.Map to dict
        config_dict = dict(driver_config)

        # Load existing configurations
        configs = _load_configurations()

        # Check if configuration already exists
        was_overridden = configuration_name in configs

        # Add or update the configuration
        configs[configuration_name] = config_dict

        # Save configurations back to file
        _save_configurations(configs)

        # Determine message based on whether it was overridden
        if was_overridden:
            message = (
                f"Configuration '{configuration_name}' was overridden successfully"
            )
        else:
            message = f"Configuration '{configuration_name}' was added successfully"

        return mgp.Record(
            success=True,
            message=message,
        )

    except Exception as e:
        raise Exception(f"Failed to save configuration: {str(e)}")


def _resolve_driver_config(driver_config: mgp.Nullable[mgp.Any]) -> Dict[str, Any]:
    """Resolve driver configuration from either map, string name, or None (auto-load single config)."""
    if driver_config is None:
        # Auto-load single configuration from file
        configs = _load_configurations()
        if len(configs) == 0:
            raise Exception("No configurations found in YAML file. Please use add_configuration to add a configuration.")
        elif len(configs) == 1:
            # Use the single configuration
            return list(configs.values())[0]
        else:
            raise Exception(f"Multiple configurations found ({len(configs)}). Please specify which configuration to use by name or provide a map.")
    elif isinstance(driver_config, str):
        # Load from YAML file by name
        config = _load_configuration_by_name(driver_config)
        if config is None:
            raise Exception(
                f"Configuration '{driver_config}' not found in YAML file. Please use the add_configuration procedure to add a new configuration."
            )
        return config
    elif isinstance(driver_config, mgp.Map):
        # Use the map directly
        return dict(driver_config)
    else:
        raise Exception(
            f"Invalid driver_config type: {type(driver_config)}. Expected a map object with the exact configuration parameters, a string name of the configuration, or None to auto-load single config."
        )


def _load_configuration_by_name(configuration_name: str) -> Dict[str, Any]:
    """Load a specific configuration by name from YAML file."""
    try:
        configs = _load_configurations()
        return configs.get(configuration_name)
    except Exception:
        return None


def _load_configurations() -> Dict[str, Any]:
    """Load configurations from YAML file. Create file if it doesn't exist."""
    config_file = Constants.CONFIG_FILE_PATH

    if not os.path.exists(config_file):
        # Create directory if it doesn't exist
        config_dir = os.path.dirname(config_file)
        try:
            os.makedirs(config_dir, exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to create directory {config_dir}: {str(e)}")

        # Create empty YAML file
        try:
            with open(config_file, "w") as file:
                yaml.dump({}, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise Exception(f"Failed to create file {config_file}: {str(e)}")

    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file) or {}
    except Exception as e:
        raise Exception(f"Failed to load file {config_file}: {str(e)}")


def _save_configurations(configs: Dict[str, Any]) -> None:
    """Save configurations to YAML file."""
    config_file = Constants.CONFIG_FILE_PATH

    # Create directory if it doesn't exist
    config_dir = os.path.dirname(config_file)
    try:
        os.makedirs(config_dir, exist_ok=True)
    except Exception as e:
        raise Exception(f"Failed to create directory {config_dir}: {str(e)}")

    try:
        with open(config_file, "w") as file:
            yaml.dump(configs, file, default_flow_style=False, indent=2)
    except Exception as e:
        raise Exception(f"Failed to write to {config_file}: {str(e)}")


def _memgraph_to_primitive(record):
    """Convert Memgraph record to primitive types."""
    result = {}
    for key, value in record.items():
        result[key] = _convert_memgraph_value(value)
    return result


def _convert_memgraph_value(value):
    """Convert Memgraph values to Python-compatible formats."""
    if value is None:
        return None

    # Handle lists and dicts recursively
    if isinstance(value, list):
        return [_convert_memgraph_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _convert_memgraph_value(val) for key, val in value.items()}

    # For other types, return as is
    return value


def _get_memgraph_connection(driver_config: Dict[str, Any]) -> Memgraph:
    """
    Get Memgraph connection.

    Args:
        driver_config: Memgraph connection configuration

    Returns:
        Memgraph connection instance
    """
    # Build connection parameters
    host = driver_config.get(Constants.HOST, Constants.DEFAULT_HOST)
    port = int(driver_config.get(Constants.PORT, Constants.DEFAULT_PORT))
    username = driver_config.get(Constants.USERNAME)
    password = driver_config.get(Constants.PASSWORD)
    
    # Create Memgraph connection
    return Memgraph(host=host, port=port, username=username, password=password)


def _test_memgraph_connection(
    memgraph_db: Memgraph, driver_config: Dict[str, Any]
) -> str:
    """Test Memgraph connection."""
    try:
        # Test connection with a simple query
        result = memgraph_db.execute_and_fetch("RETURN 1 as test")
        test_value = list(result)[0]["test"]
        memgraph_db.close()
        
        if test_value == 1:
            return "Memgraph connection successful"
        else:
            return "Memgraph connection test failed"
    except Exception as e:
        return f"Memgraph connection failed: {str(e)}"
