import threading
import yaml
import os
from typing import Any, Dict, List
from decimal import Decimal
import datetime

import psycopg2
import mgp


class Constants:
    BATCH_SIZE = 1000
    CONFIG_FILE_PATH = (
        "/var/lib/memgraph/internal_modules/migrate_config/postgresql_config.yaml"
    )
    CURSOR = "cursor"
    DATABASE = "database"
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 5432
    DRIVER = "driver"
    HOST = "host"
    MESSAGE = "message"
    PASSWORD = "password"
    PORT = "port"
    RESULT = "result"
    SESSION = "session"
    STATUS = "status"
    USERNAME = "username"
    SCHEMA = "schema"


postgresql_dict = {}


def init_query(
    query: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
):
    """
    Prepare PostgreSQL cursor for batch streaming.
    - query: SQL query to execute
    - query_config: { limit?, batch_size? }
    - driver_config: either a map with config or string name to load from YAML
    """
    global postgresql_dict
    thread_id = threading.get_native_id()
    if thread_id not in postgresql_dict:
        postgresql_dict[thread_id] = {}

    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get PostgreSQL connection
    connection = _get_postgresql_connection(resolved_config)

    # Convert configs to dict, use empty map if None
    qcfg = dict(query_config) if query_config is not None else {}

    # Setup cursor
    cursor = connection.cursor()

    # Apply limit if specified
    limit_val = qcfg.get("limit")
    if limit_val is not None:
        try:
            limit_val = int(limit_val)
            query = f"SELECT * FROM ({query}) AS subquery LIMIT {limit_val}"
        except Exception:
            raise Exception(f"Invalid 'limit' value: {limit_val}")

    # Execute query
    cursor.execute(query)

    # Stash per-thread
    postgresql_dict[thread_id][Constants.DRIVER] = connection
    postgresql_dict[thread_id][Constants.CURSOR] = cursor
    postgresql_dict[thread_id][Constants.RESULT] = cursor


def query(
    query: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """Stream up to BATCH_SIZE rows as records (per call)."""
    global postgresql_dict
    thread_id = threading.get_native_id()
    cursor = postgresql_dict[thread_id][Constants.RESULT]

    batch = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = cursor.fetchone()
            if row is None:
                break
            batch.append(mgp.Record(row=_postgresql_to_primitive(row, cursor.description)))
        except Exception as e:
            break

    return batch


def cleanup_query():
    global postgresql_dict
    thread_id = threading.get_native_id()

    cursor = postgresql_dict[thread_id].get(Constants.CURSOR)
    if cursor:
        try:
            cursor.close()
        except Exception as e:
            raise Exception(f"Failed to close cursor: {str(e)}")

    connection = postgresql_dict[thread_id].get(Constants.DRIVER)
    if connection:
        try:
            connection.close()
        except Exception as e:
            raise Exception(f"Failed to close connection: {str(e)}")

    postgresql_dict.pop(thread_id, None)


mgp.add_batch_read_proc(query, init_query, cleanup_query)


@mgp.read_proc
def test_connection(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(message=str):
    """
    Test PostgreSQL connection using configuration.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        Record with connection test results
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get PostgreSQL connection
    connection = _get_postgresql_connection(resolved_config)

    # Test PostgreSQL connection
    result = _test_postgresql_connection(connection, resolved_config)

    return mgp.Record(message=result)


@mgp.read_proc
def list_tables(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(table_name=str, table_type=str, schema_name=str):
    """
    List all table names in the PostgreSQL database.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per table with name, type, and schema
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get PostgreSQL connection
    connection = _get_postgresql_connection(resolved_config)

    # Get schema name
    schema = resolved_config.get(Constants.SCHEMA, "public")

    # Query to get tables and views
    tables_query = """
    SELECT table_name, table_type, table_schema
    FROM information_schema.tables
    WHERE table_schema = %s
    ORDER BY table_name
    """

    cursor = connection.cursor()
    cursor.execute(tables_query, [schema])
    tables = cursor.fetchall()

    cursor.close()
    connection.close()

    # Return one record per table
    return [
        mgp.Record(table_name=table[0], table_type=table[1], schema_name=table[2])
        for table in tables
    ]


@mgp.read_proc
def describe_table(
    table_name: str,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(column_name=str, data_type=str, is_nullable=str, character_maximum_length=int):
    """
    Describe table structure in PostgreSQL database.

    Args:
        table_name: Name of the table to describe
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per column with details
    """
    resolved_config = _resolve_driver_config(driver_config)

    # Get PostgreSQL connection
    connection = _get_postgresql_connection(resolved_config)

    # Get schema name
    schema = resolved_config.get(Constants.SCHEMA, "public")

    # Query to get column information
    describe_query = """
    SELECT column_name, data_type, is_nullable, character_maximum_length
    FROM information_schema.columns
    WHERE table_name = %s AND table_schema = %s
    ORDER BY ordinal_position
    """

    cursor = connection.cursor()
    cursor.execute(describe_query, [table_name, schema])
    columns = cursor.fetchall()

    cursor.close()
    connection.close()

    # Return one record per column
    return [
        mgp.Record(
            column_name=col[0],
            data_type=col[1],
            is_nullable=col[2],
            character_maximum_length=col[3] if col[3] is not None else 0
        )
        for col in columns
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
    Add or update PostgreSQL configuration in YAML file.

    Args:
        configuration_name: Name of the configuration to save
        driver_config: PostgreSQL connection configuration

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


def _postgresql_to_primitive(row, description):
    """Convert PostgreSQL row to primitive types."""
    result = {}
    for i, column in enumerate(description):
        column_name = column.name
        value = row[i]
        
        if value is None:
            result[column_name] = None
        elif isinstance(value, Decimal):
            result[column_name] = float(value)
        elif isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
            try:
                result[column_name] = value.isoformat()
            except Exception:
                result[column_name] = str(value)
        elif isinstance(value, (bytes, bytearray)):
            try:
                result[column_name] = value.decode("utf-8")
            except UnicodeDecodeError:
                result[column_name] = value.hex()
        else:
            result[column_name] = value
    
    return result


def _get_postgresql_connection(driver_config: Dict[str, Any]) -> psycopg2.connection:
    """
    Get PostgreSQL connection.

    Args:
        driver_config: PostgreSQL connection configuration

    Returns:
        PostgreSQL connection instance
    """
    # Create PostgreSQL connection
    host = driver_config.get(Constants.HOST, Constants.DEFAULT_HOST)
    port = int(driver_config.get(Constants.PORT, Constants.DEFAULT_PORT))
    database = driver_config.get(Constants.DATABASE)
    username = driver_config.get(Constants.USERNAME)
    password = driver_config.get(Constants.PASSWORD)
    
    return psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password
    )


def _test_postgresql_connection(
    connection: psycopg2.connection, driver_config: Dict[str, Any]
) -> str:
    """Test PostgreSQL connection."""
    try:
        # Test connection with a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result and result[0] == 1:
            return "PostgreSQL connection successful"
        else:
            return "PostgreSQL connection test failed"
    except Exception as e:
        return f"PostgreSQL connection failed: {str(e)}"
