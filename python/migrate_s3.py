import threading
import yaml
import os
import io
import csv
from typing import Any, Dict, List

import boto3
import mgp


class Constants:
    BATCH_SIZE = 1000
    CONFIG_FILE_PATH = (
        "/var/lib/memgraph/internal_modules/migrate_config/s3_config.yaml"
    )
    CURSOR = "cursor"
    MESSAGE = "message"
    RESULT = "result"
    SESSION = "session"
    STATUS = "status"
    AWS_ACCESS_KEY_ID = "aws_access_key_id"
    AWS_SECRET_ACCESS_KEY = "aws_secret_access_key"
    AWS_SESSION_TOKEN = "aws_session_token"
    REGION_NAME = "region_name"
    BUCKET_NAME = "bucket_name"
    FILE_PATH = "file_path"


s3_dict = {}


def init_query(
    file_path: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
):
    """
    Prepare S3 cursor for batch streaming.
    - file_path: S3 file path in the format 's3://bucket-name/path/to/file.csv'
    - query_config: { limit?, batch_size? }
    - driver_config: either a map with config or string name to load from YAML
    """
    global s3_dict
    thread_id = threading.get_native_id()
    if thread_id not in s3_dict:
        s3_dict[thread_id] = {}

    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get S3 client
    s3_client = _get_s3_client(resolved_config)

    # Extract S3 bucket and key
    if not file_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Expected 's3://bucket-name/path'.")

    file_path_no_protocol = file_path[5:]
    bucket_name, *key_parts = file_path_no_protocol.split("/")
    s3_key = "/".join(key_parts)

    # Fetch and read file as a streaming object
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    # Convert binary stream to text stream
    text_stream = io.TextIOWrapper(response["Body"], encoding="utf-8")

    # Read CSV headers
    csv_reader = csv.reader(text_stream)
    column_names = next(csv_reader)  # First row contains column names

    # Convert configs to dict, use empty map if None
    qcfg = dict(query_config) if query_config is not None else {}

    # Apply limit if specified
    limit_val = qcfg.get("limit")
    if limit_val is not None:
        try:
            limit_val = int(limit_val)
            # For CSV files, we'll limit during iteration
        except Exception:
            raise Exception(f"Invalid 'limit' value: {limit_val}")
    else:
        limit_val = None

    # Stash per-thread
    s3_dict[thread_id][Constants.CURSOR] = csv_reader
    s3_dict[thread_id][Constants.RESULT] = csv_reader
    s3_dict[thread_id]["column_names"] = column_names
    s3_dict[thread_id]["limit"] = limit_val
    s3_dict[thread_id]["row_count"] = 0


def query(
    file_path: str,
    query_config: mgp.Nullable[mgp.Map] = None,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(row=mgp.Map):
    """Stream up to BATCH_SIZE rows as records (per call)."""
    global s3_dict
    thread_id = threading.get_native_id()
    csv_reader = s3_dict[thread_id][Constants.RESULT]
    column_names = s3_dict[thread_id]["column_names"]
    limit_val = s3_dict[thread_id]["limit"]
    row_count = s3_dict[thread_id]["row_count"]

    batch = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            # Check limit
            if limit_val is not None and row_count >= limit_val:
                break
                
            row = next(csv_reader)
            batch.append(mgp.Record(row=_s3_to_primitive(row, column_names)))
            s3_dict[thread_id]["row_count"] += 1
        except StopIteration:
            break
        except Exception as e:
            break

    return batch


def cleanup_query():
    global s3_dict
    thread_id = threading.get_native_id()

    # S3 doesn't need explicit cleanup for the cursor
    s3_dict.pop(thread_id, None)


mgp.add_batch_read_proc(query, init_query, cleanup_query)


@mgp.read_proc
def test_connection(
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(message=str):
    """
    Test S3 connection using configuration.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        Record with connection test results
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get S3 client
    s3_client = _get_s3_client(resolved_config)

    # Test S3 connection
    result = _test_s3_connection(s3_client, resolved_config)

    return mgp.Record(message=result)


@mgp.read_proc
def list_files(
    bucket_name: str,
    prefix: str = "",
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(file_name=str, file_size=int, last_modified=str):
    """
    List files in an S3 bucket.

    Args:
        bucket_name: Name of the S3 bucket
        prefix: Optional prefix to filter files
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per file with details
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get S3 client
    s3_client = _get_s3_client(resolved_config)

    # List objects in bucket
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append({
                    'file_name': obj['Key'],
                    'file_size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })
        
        return [
            mgp.Record(
                file_name=file['file_name'],
                file_size=file['file_size'],
                last_modified=file['last_modified']
            )
            for file in files
        ]
    except Exception as e:
        raise Exception(f"Failed to list files: {str(e)}")


@mgp.read_proc
def describe_file(
    file_path: str,
    driver_config: mgp.Nullable[mgp.Any] = None,
) -> mgp.Record(column_name=str, sample_value=str, data_type=str):
    """
    Describe CSV file structure by reading first few rows.

    Args:
        file_path: S3 file path in the format 's3://bucket-name/path/to/file.csv'
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per column with details
    """
    resolved_config = _resolve_driver_config(driver_config)

    # Get S3 client
    s3_client = _get_s3_client(resolved_config)

    # Extract S3 bucket and key
    if not file_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Expected 's3://bucket-name/path'.")

    file_path_no_protocol = file_path[5:]
    bucket_name, *key_parts = file_path_no_protocol.split("/")
    s3_key = "/".join(key_parts)

    try:
        # Fetch and read file
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        text_stream = io.TextIOWrapper(response["Body"], encoding="utf-8")
        csv_reader = csv.reader(text_stream)
        
        # Read headers
        column_names = next(csv_reader)
        
        # Read first few rows to analyze data types
        sample_rows = []
        for i, row in enumerate(csv_reader):
            if i >= 10:  # Sample first 10 rows
                break
            sample_rows.append(row)
        
        # Analyze columns
        columns = []
        for i, column_name in enumerate(column_names):
            sample_values = [row[i] if i < len(row) else "" for row in sample_rows]
            sample_value = sample_values[0] if sample_values else ""
            
            # Determine data type based on sample values
            data_type = _infer_data_type(sample_values)
            
            columns.append({
                'column_name': column_name,
                'sample_value': sample_value,
                'data_type': data_type
            })
        
        return [
            mgp.Record(
                column_name=col['column_name'],
                sample_value=col['sample_value'],
                data_type=col['data_type']
            )
            for col in columns
        ]
    except Exception as e:
        raise Exception(f"Failed to describe file: {str(e)}")


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
    Add or update S3 configuration in YAML file.

    Args:
        configuration_name: Name of the configuration to save
        driver_config: S3 connection configuration

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


def _s3_to_primitive(row, column_names):
    """Convert S3 CSV row to primitive types."""
    result = {}
    for i, column_name in enumerate(column_names):
        value = row[i] if i < len(row) else ""
        result[column_name] = value
    return result


def _infer_data_type(sample_values):
    """Infer data type from sample values."""
    if not sample_values:
        return "string"
    
    # Check if all values are integers
    try:
        for val in sample_values:
            if val and not val.isdigit():
                break
        else:
            return "integer"
    except:
        pass
    
    # Check if all values are floats
    try:
        for val in sample_values:
            if val and not val.replace('.', '').replace('-', '').isdigit():
                break
        else:
            return "float"
    except:
        pass
    
    # Check if all values are booleans
    boolean_values = {'true', 'false', 'yes', 'no', '1', '0'}
    if all(val.lower() in boolean_values for val in sample_values if val):
        return "boolean"
    
    # Default to string
    return "string"


def _get_s3_client(driver_config: Dict[str, Any]) -> boto3.client:
    """
    Get S3 client.

    Args:
        driver_config: S3 connection configuration

    Returns:
        S3 client instance
    """
    # Create S3 client
    return boto3.client(
        "s3",
        aws_access_key_id=driver_config.get(
            Constants.AWS_ACCESS_KEY_ID, os.getenv("AWS_ACCESS_KEY_ID", None)
        ),
        aws_secret_access_key=driver_config.get(
            Constants.AWS_SECRET_ACCESS_KEY, os.getenv("AWS_SECRET_ACCESS_KEY", None)
        ),
        aws_session_token=driver_config.get(
            Constants.AWS_SESSION_TOKEN, os.getenv("AWS_SESSION_TOKEN", None)
        ),
        region_name=driver_config.get(Constants.REGION_NAME, os.getenv("AWS_REGION", None)),
    )


def _test_s3_connection(
    s3_client: boto3.client, driver_config: Dict[str, Any]
) -> str:
    """Test S3 connection."""
    try:
        # Test connection by listing buckets
        s3_client.list_buckets()
        return "S3 connection successful"
    except Exception as e:
        return f"S3 connection failed: {str(e)}"
