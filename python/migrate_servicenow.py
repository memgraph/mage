import threading
import yaml
import os
from typing import Any, Dict, List

import requests
import mgp


class Constants:
    BATCH_SIZE = 1000
    CONFIG_FILE_PATH = (
        "/var/lib/memgraph/internal_modules/migrate_config/servicenow_config.yaml"
    )
    CURSOR = "cursor"
    MESSAGE = "message"
    RESULT = "result"
    SESSION = "session"
    STATUS = "status"
    USERNAME = "username"
    PASSWORD = "password"
    INSTANCE_URL = "instance_url"
    TABLE_NAME = "table_name"


servicenow_dict = {}


def init_query(
    endpoint: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
):
    """
    Prepare ServiceNow cursor for batch streaming.
    - endpoint: ServiceNow API endpoint (full URL)
    - query_config: { limit?, batch_size? }
    - driver_config: either a map with config or string name to load from YAML
    """
    global servicenow_dict
    thread_id = threading.get_native_id()
    if thread_id not in servicenow_dict:
        servicenow_dict[thread_id] = {}

    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Convert configs to dict, use empty map if None
    qcfg = dict(query_config) if query_config is not None else {}

    # Apply limit if specified
    limit_val = qcfg.get("limit")
    if limit_val is not None:
        try:
            limit_val = int(limit_val)
            # Add limit to endpoint
            if "?" in endpoint:
                endpoint += f"&sysparm_limit={limit_val}"
            else:
                endpoint += f"?sysparm_limit={limit_val}"
        except Exception:
            raise Exception(f"Invalid 'limit' value: {limit_val}")

    # Get authentication
    auth = (resolved_config.get(Constants.USERNAME), resolved_config.get(Constants.PASSWORD))
    headers = {"Accept": "application/json"}

    # Make request
    response = requests.get(endpoint, auth=auth, headers=headers)
    response.raise_for_status()

    data = response.json().get(Constants.RESULT, [])
    if not data:
        raise ValueError("No data found in ServiceNow response")

    # Stash per-thread
    servicenow_dict[thread_id][Constants.CURSOR] = iter(data)
    servicenow_dict[thread_id][Constants.RESULT] = iter(data)


def query(
    endpoint: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """Stream up to BATCH_SIZE records as rows (per call)."""
    global servicenow_dict
    thread_id = threading.get_native_id()
    data_iter = servicenow_dict[thread_id][Constants.RESULT]

    batch = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = next(data_iter)
            batch.append(mgp.Record(row=_servicenow_to_primitive(row)))
        except StopIteration:
            break
        except Exception as e:
            break

    return batch


def cleanup_query():
    global servicenow_dict
    thread_id = threading.get_native_id()

    # ServiceNow doesn't need explicit cleanup for the cursor
    servicenow_dict.pop(thread_id, None)


mgp.add_batch_read_proc(query, init_query, cleanup_query)


@mgp.read_proc
def test_connection(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(message=str):
    """
    Test ServiceNow connection using configuration.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        Record with connection test results
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Test ServiceNow connection
    result = _test_servicenow_connection(resolved_config)

    return mgp.Record(message=result)


@mgp.read_proc
def list_tables(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(table_name=str, table_label=str, table_type=str):
    """
    List all table names in the ServiceNow instance.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per table with name, label, and type
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get instance URL
    instance_url = resolved_config.get(Constants.INSTANCE_URL)
    if not instance_url:
        raise Exception("Instance URL is required in driver configuration")

    # Get authentication
    auth = (resolved_config.get(Constants.USERNAME), resolved_config.get(Constants.PASSWORD))
    headers = {"Accept": "application/json"}

    # Query to get tables
    tables_endpoint = f"{instance_url}/api/now/table/sys_db_object"
    params = {
        "sysparm_fields": "name,label,sys_class_name",
        "sysparm_query": "sys_class_name=sys_db_object^ORsys_class_name=sys_db_view"
    }

    try:
        response = requests.get(tables_endpoint, auth=auth, headers=headers, params=params)
        response.raise_for_status()
        
        tables = response.json().get(Constants.RESULT, [])
        
        return [
            mgp.Record(
                table_name=table.get("name", ""),
                table_label=table.get("label", ""),
                table_type=table.get("sys_class_name", "")
            )
            for table in tables
        ]
    except Exception as e:
        raise Exception(f"Failed to list tables: {str(e)}")


@mgp.read_proc
def describe_table(
    table_name: str,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(column_name=str, data_type=str, is_nullable=str, column_label=str):
    """
    Describe table structure in ServiceNow database.

    Args:
        table_name: Name of the table to describe
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per column with details
    """
    resolved_config = _resolve_driver_config(driver_config)

    # Get instance URL
    instance_url = resolved_config.get(Constants.INSTANCE_URL)
    if not instance_url:
        raise Exception("Instance URL is required in driver configuration")

    # Get authentication
    auth = (resolved_config.get(Constants.USERNAME), resolved_config.get(Constants.PASSWORD))
    headers = {"Accept": "application/json"}

    # Query to get column information
    columns_endpoint = f"{instance_url}/api/now/table/sys_dictionary"
    params = {
        "sysparm_fields": "element,internal_type,mandatory,column_label",
        "sysparm_query": f"name={table_name}",
        "sysparm_limit": 1000
    }

    try:
        response = requests.get(columns_endpoint, auth=auth, headers=headers, params=params)
        response.raise_for_status()
        
        columns = response.json().get(Constants.RESULT, [])
        
        return [
            mgp.Record(
                column_name=col.get("element", ""),
                data_type=col.get("internal_type", ""),
                is_nullable="NO" if col.get("mandatory", False) else "YES",
                column_label=col.get("column_label", "")
            )
            for col in columns
        ]
    except Exception as e:
        raise Exception(f"Failed to describe table: {str(e)}")


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
    Add or update ServiceNow configuration in YAML file.

    Args:
        configuration_name: Name of the configuration to save
        driver_config: ServiceNow connection configuration

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


def _servicenow_to_primitive(record):
    """Convert ServiceNow record to primitive types."""
    result = {}
    for key, value in record.items():
        if value is None:
            result[key] = None
        elif isinstance(value, (list, tuple)):
            result[key] = [_servicenow_to_primitive(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            result[key] = _servicenow_to_primitive(value)
        else:
            result[key] = value
    return result


def _test_servicenow_connection(driver_config: Dict[str, Any]) -> str:
    """Test ServiceNow connection."""
    try:
        # Get instance URL
        instance_url = driver_config.get(Constants.INSTANCE_URL)
        if not instance_url:
            return "ServiceNow connection failed: Instance URL is required"

        # Get authentication
        auth = (driver_config.get(Constants.USERNAME), driver_config.get(Constants.PASSWORD))
        headers = {"Accept": "application/json"}

        # Test connection with a simple query
        test_endpoint = f"{instance_url}/api/now/table/sys_user"
        params = {"sysparm_limit": 1}
        
        response = requests.get(test_endpoint, auth=auth, headers=headers, params=params)
        response.raise_for_status()
        
        return "ServiceNow connection successful"
    except Exception as e:
        return f"ServiceNow connection failed: {str(e)}"
