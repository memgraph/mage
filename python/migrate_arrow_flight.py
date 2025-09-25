import threading
import yaml
import os
import base64
from typing import Any, Dict, List
from decimal import Decimal
import datetime

import pyarrow.flight as flight
import mgp


class Constants:
    BATCH_SIZE = 1000
    CONFIG_FILE_PATH = (
        "/var/lib/memgraph/internal_modules/migrate_config/arrow_flight_config.yaml"
    )
    CURSOR = "cursor"
    DATABASE = "database"
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 5005
    DRIVER = "driver"
    HOST = "host"
    MESSAGE = "message"
    PASSWORD = "password"
    PORT = "port"
    RESULT = "result"
    SESSION = "session"
    STATUS = "status"
    USERNAME = "username"


arrow_flight_dict = {}


def init_query(
    query: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
):
    """
    Prepare Arrow Flight cursor for batch streaming.
    - query: SQL query to execute
    - query_config: { limit?, batch_size? }
    - driver_config: either a map with config or string name to load from YAML
    """
    global arrow_flight_dict
    thread_id = threading.get_native_id()
    if thread_id not in arrow_flight_dict:
        arrow_flight_dict[thread_id] = {}

    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get Arrow Flight client
    client = _get_arrow_flight_client(resolved_config)

    # Convert configs to dict, use empty map if None
    qcfg = dict(query_config) if query_config is not None else {}

    # Apply limit if specified
    limit_val = qcfg.get("limit")
    if limit_val is not None:
        try:
            limit_val = int(limit_val)
            query = f"SELECT * FROM ({query}) AS subquery LIMIT {limit_val}"
        except Exception:
            raise Exception(f"Invalid 'limit' value: {limit_val}")

    # Encode credentials
    username = resolved_config.get(Constants.USERNAME, "")
    password = resolved_config.get(Constants.PASSWORD, "")
    auth_string = f"{username}:{password}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_string).decode("utf-8")

    # Authenticate
    options = flight.FlightCallOptions(
        headers=[(b"authorization", f"Basic {encoded_auth}".encode("utf-8"))]
    )

    # Get flight info
    flight_info = client.get_flight_info(
        flight.FlightDescriptor.for_command(query), options
    )

    # Create cursor from flight data
    cursor = iter(_fetch_flight_data(client, flight_info, options))

    # Stash per-thread
    arrow_flight_dict[thread_id][Constants.DRIVER] = client
    arrow_flight_dict[thread_id][Constants.CURSOR] = cursor
    arrow_flight_dict[thread_id][Constants.RESULT] = cursor


def query(
    query: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """Stream up to BATCH_SIZE rows as records (per call)."""
    global arrow_flight_dict
    thread_id = threading.get_native_id()
    cursor = arrow_flight_dict[thread_id][Constants.RESULT]

    batch = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = next(cursor)
            batch.append(mgp.Record(row=_arrow_flight_to_primitive(row)))
        except StopIteration:
            break
        except Exception as e:
            break

    return batch


def cleanup_query():
    global arrow_flight_dict
    thread_id = threading.get_native_id()

    client = arrow_flight_dict[thread_id].get(Constants.DRIVER)
    if client:
        try:
            client.close()
        except Exception as e:
            raise Exception(f"Failed to close client: {str(e)}")

    arrow_flight_dict.pop(thread_id, None)


mgp.add_batch_read_proc(query, init_query, cleanup_query)


@mgp.read_proc
def test_connection(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(message=str):
    """
    Test Arrow Flight connection using configuration.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        Record with connection test results
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get Arrow Flight client
    client = _get_arrow_flight_client(resolved_config)

    # Test Arrow Flight connection
    result = _test_arrow_flight_connection(client, resolved_config)

    return mgp.Record(message=result)


@mgp.read_proc
def list_tables(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(table_name=str, table_type=str, schema_name=str):
    """
    List all table names in the Arrow Flight database.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per table with name, type, and schema
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get Arrow Flight client
    client = _get_arrow_flight_client(resolved_config)

    # Query to get tables
    tables_query = "SHOW TABLES"

    # Encode credentials
    username = resolved_config.get(Constants.USERNAME, "")
    password = resolved_config.get(Constants.PASSWORD, "")
    auth_string = f"{username}:{password}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_string).decode("utf-8")

    # Authenticate
    options = flight.FlightCallOptions(
        headers=[(b"authorization", f"Basic {encoded_auth}".encode("utf-8"))]
    )

    # Get flight info
    flight_info = client.get_flight_info(
        flight.FlightDescriptor.for_command(tables_query), options
    )

    # Fetch data
    tables = []
    for row in _fetch_flight_data(client, flight_info, options):
        tables.append(row)

    client.close()

    # Return one record per table
    return [
        mgp.Record(
            table_name=table.get("table_name", ""),
            table_type=table.get("table_type", "TABLE"),
            schema_name=table.get("schema_name", "default")
        )
        for table in tables
    ]


@mgp.read_proc
def describe_table(
    table_name: str,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(column_name=str, data_type=str, is_nullable=str, column_default=str):
    """
    Describe table structure in Arrow Flight database.

    Args:
        table_name: Name of the table to describe
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per column with details
    """
    resolved_config = _resolve_driver_config(driver_config)

    # Get Arrow Flight client
    client = _get_arrow_flight_client(resolved_config)

    # Query to get column information
    describe_query = f"DESCRIBE {table_name}"

    # Encode credentials
    username = resolved_config.get(Constants.USERNAME, "")
    password = resolved_config.get(Constants.PASSWORD, "")
    auth_string = f"{username}:{password}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_string).decode("utf-8")

    # Authenticate
    options = flight.FlightCallOptions(
        headers=[(b"authorization", f"Basic {encoded_auth}".encode("utf-8"))]
    )

    # Get flight info
    flight_info = client.get_flight_info(
        flight.FlightDescriptor.for_command(describe_query), options
    )

    # Fetch data
    columns = []
    for row in _fetch_flight_data(client, flight_info, options):
        columns.append(row)

    client.close()

    # Return one record per column
    return [
        mgp.Record(
            column_name=col.get("column_name", ""),
            data_type=col.get("data_type", ""),
            is_nullable=col.get("is_nullable", "YES"),
            column_default=col.get("column_default", "")
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
    Add or update Arrow Flight configuration in YAML file.

    Args:
        configuration_name: Name of the configuration to save
        driver_config: Arrow Flight connection configuration

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


def _arrow_flight_to_primitive(row):
    """Convert Arrow Flight row to primitive types."""
    result = {}
    for key, value in row.items():
        if value is None:
            result[key] = None
        elif isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
            try:
                result[key] = value.isoformat()
            except Exception:
                result[key] = str(value)
        elif isinstance(value, (bytes, bytearray)):
            try:
                result[key] = value.decode("utf-8")
            except UnicodeDecodeError:
                result[key] = value.hex()
        else:
            result[key] = value
    return result


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


def _get_arrow_flight_client(driver_config: Dict[str, Any]) -> flight.FlightClient:
    """
    Get Arrow Flight client.

    Args:
        driver_config: Arrow Flight connection configuration

    Returns:
        Arrow Flight client instance
    """
    # Build connection parameters
    host = driver_config.get(Constants.HOST, Constants.DEFAULT_HOST)
    port = int(driver_config.get(Constants.PORT, Constants.DEFAULT_PORT))
    
    # Establish Flight connection
    return flight.connect(f"grpc://{host}:{port}")


def _test_arrow_flight_connection(
    client: flight.FlightClient, driver_config: Dict[str, Any]
) -> str:
    """Test Arrow Flight connection."""
    try:
        # Test connection with a simple query
        username = driver_config.get(Constants.USERNAME, "")
        password = driver_config.get(Constants.PASSWORD, "")
        auth_string = f"{username}:{password}".encode("utf-8")
        encoded_auth = base64.b64encode(auth_string).decode("utf-8")

        options = flight.FlightCallOptions(
            headers=[(b"authorization", f"Basic {encoded_auth}".encode("utf-8"))]
        )

        flight_info = client.get_flight_info(
            flight.FlightDescriptor.for_command("SELECT 1"), options
        )
        
        # Try to fetch one record
        for endpoint in flight_info.endpoints:
            reader = client.do_get(endpoint.ticket, options)
            for chunk in reader:
                batch = chunk.data
                records = batch.to_pylist()
                if records and len(records) > 0:
                    client.close()
                    return "Arrow Flight connection successful"
                break
            break
        
        client.close()
        return "Arrow Flight connection test failed"
    except Exception as e:
        return f"Arrow Flight connection failed: {str(e)}"
