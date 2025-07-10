from io import TextIOWrapper
import json
try:
    import mgp
except ImportError:
    from . import mock_mgp as mgp
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
from typing import Any
from datetime import datetime, date, time, timedelta


def _convert_value_to_json_compatible(value: Any) -> Any:
    """Helper function to convert Memgraph values to JSON-compatible Python types."""
    if isinstance(value, mgp.Vertex):
        return {
            "type": "node",
            "id": value.id,
            "labels": [label.name for label in value.labels],
            "properties": {k: _convert_value_to_json_compatible(v) 
                          for k, v in value.properties.items()}
        }
    elif isinstance(value, mgp.Edge):
        return {
            "type": "relationship",
            "id": value.id,
            "start": value.from_vertex.id,
            "end": value.to_vertex.id,
            "relationship_type": value.type.name,
            "properties": {k: _convert_value_to_json_compatible(v) 
                          for k, v in value.properties.items()}
        }
    elif isinstance(value, mgp.Path):
        return {
            "type": "path",
            "start": _convert_value_to_json_compatible(value.vertices[0]),
            "end": _convert_value_to_json_compatible(value.vertices[-1]),
            "nodes": [_convert_value_to_json_compatible(v) for v in value.vertices],
            "relationships": [_convert_value_to_json_compatible(e) for e in value.edges]
        }
    elif isinstance(value, (list, tuple)):
        return [_convert_value_to_json_compatible(item) for item in value]
    elif isinstance(value, dict):
        return {k: _convert_value_to_json_compatible(v) for k, v in value.items()}
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, time):
        return value.isoformat()
    elif isinstance(value, timedelta):
        total_seconds = value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        microseconds = value.microseconds
        return f"P0DT{hours}H{minutes}M{seconds}.{microseconds:06d}S"
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    else:
        return str(value)


def extract_objects(file: TextIOWrapper):
    """Helper function to extract objects from a JSON file."""
    objects = json.load(file)
    if type(objects) is dict:
        objects = [objects]
    return objects


def to_json(ctx: mgp.ProcCtx, value: Any) -> mgp.Record:
    """
    Converts any value to its JSON string representation.
    Similar to Neo4j's apoc.convert.toJson().

    Parameters
    ----------
    value : Any
        The value to convert to JSON. Can be a node, relationship, path,
        or any primitive type.

    Returns
    -------
    mgp.Record
        A record containing the JSON string representation of the value.
    """
    converted = _convert_value_to_json_compatible(value)
    return mgp.Record(json=json.dumps(converted, ensure_ascii=False))


def from_json_list(ctx: mgp.ProcCtx, json_str: str) -> mgp.Record:
    """
    Converts a JSON string representing a list to a Python list.
    Similar to Neo4j's apoc.convert.fromJsonList().

    Parameters
    ----------
    json_str : str
        The JSON string to convert. Must represent a valid JSON array.

    Returns
    -------
    mgp.Record
        A record containing the Python list converted from JSON.

    Raises
    ------
    json.JSONDecodeError
        If the input string is not valid JSON.
    ValueError
        If the input JSON does not represent a list.
    """
    value = json.loads(json_str)
    if not isinstance(value, list):
        raise ValueError("Input JSON must represent a list")
    return mgp.Record(value=value)


def load_from_path(ctx: mgp.ProcCtx, path: str) -> mgp.Record:
    """
    Procedure to load JSON from a local file.

    Parameters
    ----------
    path : str
        Path to the JSON that is now being loaded.
    """
    file = Path(path)
    if file.exists():
        opened_file = open(file)
        objects = extract_objects(opened_file)
    else:
        raise FileNotFoundError("There is no file " + path)

    opened_file.close()

    return mgp.Record(objects=objects)


def load_from_url(ctx: mgp.ProcCtx, url: str) -> mgp.Record:
    """
    Procedure to load JSON from a remote address.

    Parameters
    ----------
    url : str
        URL to the JSON that is being loaded.
    """
    request = Request(url)
    request.add_header("User-Agent", "MAGE module")
    try:
        content = urlopen(request)
    except URLError as url_error:
        raise url_error
    else:
        objects = extract_objects(content)

    return mgp.Record(objects=objects) 
