import threading
import yaml
import os
from typing import Any, Dict

from bson import ObjectId, Decimal128, Binary
from bson.timestamp import Timestamp as BsonTimestamp
import mgp
from pymongo import MongoClient


class Constants:
    AUTH_SOURCE = "authSource"
    BATCH_SIZE = 1000
    CONFIG_FILE_PATH = (
        "/var/lib/memgraph/internal_modules/migrate_config/mongodb_config.yaml"
    )
    CURSOR = "cursor"
    DATABASE = "database"
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 27017
    DRIVER = "driver"
    HOST = "host"
    MESSAGE = "message"
    PASSWORD = "password"
    PORT = "port"
    REPLICA_SET = "replicaSet"
    RESULT = "result"
    SESSION = "session"
    STATUS = "status"
    TLS = "tls"
    URI = "uri"
    USERNAME = "username"


mongodb_dict = {}


def init_find(
    collection_name: str,  # 1) collection
    find_query: mgp.Map,  # 2) filter as MAP (find only)
    query_config: mgp.Map,  # 3) query config (NO collection here)
    driver_config: mgp.Any,  # 4) driver config (map) or config name (string)
):
    """
    Prepare MongoDB cursor for batch streaming.
    - collection_name: target collection (or view) within the database
    - find_query: MAP used as the filter for find()
    - query_config: { projection?, sort?, limit?, batch_size? }
    - driver_config: either a map with config or string name to load from YAML
    """
    global mongodb_dict
    thread_id = threading.get_native_id()
    if thread_id not in mongodb_dict:
        mongodb_dict[thread_id] = {}

    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get MongoDB client
    client = _get_mongo_client(resolved_config)

    # Convert configs to dict
    qcfg = dict(query_config)
    dcfg = dict(resolved_config)

    # Filter is already a map
    q_ast = dict(find_query) if find_query is not None else {}

    # database lives in driver_config
    database = dcfg.get(Constants.DATABASE)
    if not database:
        raise Exception("driver_config must include 'database'.")

    # Setup database and cursor
    db = client[database]
    cursor = _start_mongo_cursor(db, collection_name, q_ast, qcfg)

    # Stash per-thread
    mongodb_dict[thread_id][Constants.DRIVER] = client
    mongodb_dict[thread_id][Constants.SESSION] = db
    mongodb_dict[thread_id][Constants.RESULT] = cursor


def find(
    collection_name: str,
    find_query: mgp.Map,
    query_config: mgp.Map,
    driver_config: mgp.Any,
) -> mgp.Record(row=mgp.Map):
    """Stream up to BATCH_SIZE documents as rows (per call)."""
    global mongodb_dict
    thread_id = threading.get_native_id()
    cursor = mongodb_dict[thread_id][Constants.RESULT]

    batch = []
    for doc in cursor:
        batch.append(mgp.Record(row=_mongo_to_primitive(doc)))
        if len(batch) >= Constants.BATCH_SIZE:
            break
    return batch


def cleanup_find():
    global mongodb_dict
    thread_id = threading.get_native_id()

    cursor = mongodb_dict[thread_id].get(Constants.RESULT)
    if cursor:
        try:
            cursor.close()
        except Exception as e:
            raise Exception(f"Failed to close cursor: {str(e)}")

    client = mongodb_dict[thread_id].get(Constants.DRIVER)
    if client:
        try:
            client.close()
        except Exception as e:
            raise Exception(f"Failed to close client: {str(e)}")

    mongodb_dict.pop(thread_id, None)


mgp.add_batch_read_proc(find, init_find, cleanup_find)


@mgp.read_proc
def test_connection(
    driver_config: mgp.Any,  # Driver config (map) or config name (string)
) -> mgp.Record(message=str):
    """
    Test MongoDB connection using configuration.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        Record with connection test results
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get MongoDB client
    client = _get_mongo_client(resolved_config)

    # Test MongoDB connection
    result = _test_mongodb_connection(client, resolved_config)

    return mgp.Record(message=result)


@mgp.read_proc
def list_collections(
    driver_config: mgp.Any,  # Driver config (map) or config name (string)
) -> mgp.Record(name=str):
    """
    List all collection names in the MongoDB database.

    Args:
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        One record per collection name
    """
    # Resolve driver configuration
    resolved_config = _resolve_driver_config(driver_config)

    # Get MongoDB client
    client = _get_mongo_client(resolved_config)

    # Get database name
    database = resolved_config.get(Constants.DATABASE)
    if not database:
        raise Exception("Driver config must include 'database'.")

    # Get database and list collection names
    db = client[database]
    collection_names = db.list_collection_names()

    client.close()

    # Return one record per collection name
    return [mgp.Record(name=name) for name in collection_names]


@mgp.read_proc
def find_one(
    collection_name: str,  # Name of the collection
    driver_config: mgp.Any,  # Driver config (map) or config name (string)
) -> mgp.Record(document=mgp.Map):
    """
    Find one document from the specified collection.

    Args:
        collection_name: Name of the collection to query
        driver_config: Either a map with config or string name to load from YAML file

    Returns:
        Record with the found document
    """
    resolved_config = _resolve_driver_config(driver_config)

    # Get MongoDB client
    client = _get_mongo_client(resolved_config)

    # Get database name
    database = resolved_config.get(Constants.DATABASE)
    if not database:
        raise Exception("Driver config must include 'database'.")

    # Get database and collection
    db = client[database]
    collection = db[collection_name]

    # Find one document
    document = collection.find_one()

    client.close()

    if document is None:
        return mgp.Record(document={})

    return mgp.Record(document=_mongo_to_primitive(document))


@mgp.read_proc
def get_configurations() -> mgp.Record(name=str, config=mgp.Map):
    """
    Get configurations from the YAML file.

    Args:
        configuration_name: Name of the configuration
        config: Configuration dictionary

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
    configuration_name: str,  # Name of the configuration
    driver_config: mgp.Map,  # Driver configuration to save
) -> mgp.Record(success=bool, message=str):
    """
    Add or update MongoDB configuration in YAML file.

    Args:
        configuration_name: Name of the configuration to save
        driver_config: MongoDB connection configuration

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


def _resolve_driver_config(driver_config: mgp.Any) -> Dict[str, Any]:
    """Resolve driver configuration from either map or string name."""
    if isinstance(driver_config, str):
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
            f"Invalid driver_config type: {type(driver_config)}. Expected a map object with the exact configuration parameters or a string name of the configuration."
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


def _mongo_to_primitive(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Decimal128):
        return str(value)
    if isinstance(value, (bytes, bytearray, Binary)):
        return value.hex() if hasattr(value, "hex") else bytes(value).hex()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception as e:
            raise Exception(f"Failed to convert value {value} to isoformat: {str(e)}")
    if isinstance(value, BsonTimestamp):
        return {"ts_time": value.time, "ts_inc": value.inc}
    if isinstance(value, dict):
        return {str(k): _mongo_to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mongo_to_primitive(v) for v in value]
    return str(value)


def _get_mongo_client(driver_config: mgp.Map) -> MongoClient:
    """
    Get MongoDB client with client creation.

    Args:
        driver_config: MongoDB connection configuration

    Returns:
        MongoClient instance
    """
    # Convert mgp.Map to dict
    dcfg = dict(driver_config)

    # Create MongoDB client
    uri = dcfg.get(Constants.URI)
    if uri:
        return MongoClient(uri)

    kwargs = {}
    if Constants.USERNAME in dcfg:
        kwargs[Constants.USERNAME] = dcfg.get(Constants.USERNAME)
    if Constants.PASSWORD in dcfg:
        kwargs[Constants.PASSWORD] = dcfg.get(Constants.PASSWORD)
    if Constants.AUTH_SOURCE in dcfg:
        kwargs[Constants.AUTH_SOURCE] = dcfg.get(Constants.AUTH_SOURCE)
    if Constants.TLS in dcfg:
        kwargs[Constants.TLS] = bool(dcfg.get(Constants.TLS))
    if Constants.REPLICA_SET in dcfg:
        kwargs[Constants.REPLICA_SET] = dcfg.get(Constants.REPLICA_SET)

    host = dcfg.get(Constants.HOST, Constants.DEFAULT_HOST)
    port = int(dcfg.get(Constants.PORT, Constants.DEFAULT_PORT))
    return MongoClient(host=host, port=port, **kwargs)


def _start_mongo_cursor(db, collection: str, q_ast: dict, qcfg: dict):
    """
    Map-only => always find(). Optional projection/sort/limit/batch_size from qcfg.
    """
    coll = db[collection]

    # Normalize options
    projection = dict(qcfg.get("projection")) if qcfg.get("projection") else None
    limit_val = qcfg.get("limit")
    try:
        limit_val = int(limit_val) if limit_val is not None else None
    except Exception:
        raise Exception(f"Invalid 'limit' value: {limit_val}")

    batch_size_val = qcfg.get("batch_size")
    try:
        batch_size_val = int(batch_size_val) if batch_size_val is not None else None
    except Exception:
        raise Exception(f"Invalid 'batch_size' value: {batch_size_val}")

    sort_spec = qcfg.get("sort")  # expected like: [["age", -1], ["name", 1]]
    if sort_spec is not None:
        # Coerce to list of (field, dir) tuples with int dir
        try:
            sort_pairs = [(str(f), int(d)) for f, d in list(sort_spec)]
        except Exception:
            raise Exception(f"Invalid 'sort' value: {sort_spec}")
    else:
        sort_pairs = None

    # Build cursor without None-valued args
    cursor = coll.find(filter=q_ast or {}, projection=projection)

    if sort_pairs:
        cursor = cursor.sort(sort_pairs)
    if limit_val is not None:
        cursor = cursor.limit(limit_val)
    if batch_size_val is not None:
        cursor = cursor.batch_size(batch_size_val)

    return cursor


def _test_mongodb_connection(
    client: MongoClient, driver_config: mgp.Map
) -> Dict[str, Any]:
    """Test MongoDB connection."""
    # First test with ping
    client.admin.command("ping")

    # Test connection with a dummy query
    dcfg = dict(driver_config)

    database = dcfg.get(Constants.DATABASE)
    if not database:
        raise Exception("Driver config must include 'database'.")

    db = client[database]
    # Execute a simple query to test the connection
    db.list_collection_names()
    client.close()

    return "MongoDB connection successful"
