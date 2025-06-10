import json
import pytest
from datetime import datetime, date, time, timedelta
from typing import Any

from mage.json_util import to_json, from_json_list
from .conftest import MockMgp


@pytest.mark.parametrize(
    "value,expected",
    [
        (42, 42),
        (3.14, 3.14),
        ("hello", "hello"),
        (True, True),
        (None, None),
        ([1, 2, 3], [1, 2, 3]),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_primitive_types(value: Any, expected: Any) -> None:
    """Test conversion of primitive data types"""
    result = json.loads(to_json(None, value).json)
    assert result == expected


@pytest.mark.parametrize(
    "dt,expected",
    [
        (datetime(2024, 3, 14, 15, 9, 26), "2024-03-14T15:09:26"),
        (date(2024, 3, 14), "2024-03-14"),
        (time(15, 9, 26), "15:09:26"),
        (
            timedelta(hours=2, minutes=30, seconds=45, microseconds=123456),
            "P0DT2H30M45.123456S",
        ),
    ],
)
def test_datetime_types(dt: Any, expected: str) -> None:
    """Test conversion of datetime-related types"""
    result = json.loads(to_json(None, dt).json)
    assert result == expected


def test_vertex_conversion() -> None:
    """Test conversion of vertices/nodes"""
    vertex = MockMgp.Vertex(
        id=1,
        labels=["Person", "Employee"],
        properties={"name": "John Doe", "age": 30, "active": True},
    )

    result = json.loads(to_json(None, vertex).json)

    assert result["type"] == "node"
    assert result["id"] == 1
    assert set(result["labels"]) == {"Person", "Employee"}
    assert result["properties"] == {
        "name": "John Doe",
        "age": 30,
        "active": True
    }


def test_edge_conversion() -> None:
    """Test conversion of edges/relationships"""
    from_vertex = MockMgp.Vertex(1, ["Person"], {"name": "John"})
    to_vertex = MockMgp.Vertex(2, ["Person"], {"name": "Jane"})

    edge = MockMgp.Edge(
        id=100,
        from_vertex=from_vertex,
        to_vertex=to_vertex,
        type_name="KNOWS",
        properties={"since": 2020, "strength": 0.8},
    )

    result = json.loads(to_json(None, edge).json)

    assert result["type"] == "relationship"
    assert result["id"] == 100
    assert result["start"] == 1
    assert result["end"] == 2
    assert result["relationship_type"] == "KNOWS"
    assert result["properties"] == {"since": 2020, "strength": 0.8}


def test_path_conversion() -> None:
    """Test conversion of paths"""
    v1 = MockMgp.Vertex(1, ["Person"], {"name": "John"})
    v2 = MockMgp.Vertex(2, ["Person"], {"name": "Jane"})
    v3 = MockMgp.Vertex(3, ["Person"], {"name": "Bob"})

    e1 = MockMgp.Edge(100, v1, v2, "KNOWS", {"since": 2020})
    e2 = MockMgp.Edge(101, v2, v3, "KNOWS", {"since": 2021})

    path = MockMgp.Path(vertices=[v1, v2, v3], edges=[e1, e2])

    result = json.loads(to_json(None, path).json)

    assert result["type"] == "path"
    assert len(result["nodes"]) == 3
    assert len(result["relationships"]) == 2
    assert result["start"]["properties"]["name"] == "John"
    assert result["end"]["properties"]["name"] == "Bob"


def test_complex_nested_structure() -> None:
    """Test conversion of complex nested structures"""
    dt = datetime(2024, 3, 14, 15, 9, 26)
    vertex = MockMgp.Vertex(1, ["Person"], {"name": "John"})

    complex_obj = {
        "id": 123,
        "data": {
            "vertex": vertex,
            "timestamp": dt,
            "values": [1, 2, 3],
            "metadata": {"active": True, "tags": ["a", "b", "c"]},
        },
    }

    result = json.loads(to_json(None, complex_obj).json)

    assert result["id"] == 123
    assert result["data"]["vertex"]["type"] == "node"
    assert result["data"]["timestamp"] == "2024-03-14T15:09:26"
    assert result["data"]["values"] == [1, 2, 3]
    assert result["data"]["metadata"]["active"] is True
    assert result["data"]["metadata"]["tags"] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "value,expected",
    [
        (float("inf"), str(float("inf"))),
        (float("nan"), str(float("nan"))),
        (
            type("CustomClass", (), {"__str__": lambda x: "custom_object"})(),
            "custom_object",
        ),
        ([], []),
        ({}, {}),
        ([None, True, False], [None, True, False]),
    ],
)
def test_edge_cases(value: Any, expected: Any) -> None:
    """Test edge cases and special values"""
    result = json.loads(to_json(None, value).json)
    assert str(result) == str(expected)


@pytest.mark.parametrize(
    "json_str,expected",
    [
        ('[1, 2, 3]', [1, 2, 3]),
        ('["a", "b", "c"]', ["a", "b", "c"]),
        ('[true, false, null]', [True, False, None]),
        ('[[1, 2], [3, 4]]', [[1, 2], [3, 4]]),
        ('[{"a": 1}, {"b": 2}]', [{"a": 1}, {"b": 2}]),
        ('[]', []),
    ],
)
def test_from_json_list_valid(json_str: str, expected: list) -> None:
    """Test valid JSON list conversions"""
    result = from_json_list(None, json_str).value
    assert result == expected


@pytest.mark.parametrize(
    "invalid_input",
    [
        '{"not": "a list"}',  # dictionary instead of list
        'not json at all',    # invalid JSON
        '[1, 2,]',           # invalid JSON syntax
        '1',                 # scalar instead of list
        '"string"',          # string instead of list
    ],
)
def test_from_json_list_invalid(invalid_input: str) -> None:
    """Test invalid inputs for JSON list conversion"""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        from_json_list(None, invalid_input)
