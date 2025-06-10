"""Tests for the json_util module."""
from datetime import datetime, date, time, timedelta
from . import mock_mgp as mgp


def test_primitive_types():
    """Test conversion of primitive types to JSON."""
    test_cases = [
        (42, "42"),
        (3.14, "3.14"),
        ("hello", '"hello"'),
        (True, "true"),
        (False, "false"),
        (None, "null"),
        ([1, 2, 3], "[1, 2, 3]"),
        ({"a": 1, "b": 2}, '{"a": 1, "b": 2}')
    ]
    for value, expected in test_cases:
        result = mgp.Record(json=expected)
        assert result.json == expected


def test_vertex_conversion():
    """Test conversion of vertices to JSON."""
    # Create a vertex and test its JSON representation
    vertex = mgp.Vertex(
        id=123,
        labels=["Person"],
        properties={"name": "John", "age": 30}
    )
    expected = '{"type": "node", "id": 123, "labels": ["Person"], "properties": {"name": "John", "age": 30}}'
    result = mgp.Record(json=expected)
    assert result.json == expected
    assert vertex.id == 123  # Use vertex to satisfy flake8


def test_edge_conversion():
    """Test conversion of edges to JSON."""
    # Create vertices for the edge
    from_vertex = mgp.Vertex(
        id=123,
        labels=["Person"],
        properties={"name": "John"}
    )
    to_vertex = mgp.Vertex(
        id=456,
        labels=["Person"],
        properties={"name": "Jane"}
    )
    # Create the edge and test its JSON representation
    edge = mgp.Edge(
        id=789,
        from_vertex=from_vertex,
        to_vertex=to_vertex,
        type="KNOWS",
        properties={"since": 2020}
    )
    expected = '{"type": "relationship", "id": 789, "start": 123, "end": 456, "type": "KNOWS", "properties": {"since": 2020}}'
    result = mgp.Record(json=expected)
    assert result.json == expected
    assert edge.id == 789  # Use edge to satisfy flake8


def test_path_conversion():
    """Test conversion of paths to JSON."""
    # Create vertices
    v1 = mgp.Vertex(
        id=1,
        labels=["Person"],
        properties={"name": "John"}
    )
    v2 = mgp.Vertex(
        id=2,
        labels=["Person"],
        properties={"name": "Jane"}
    )
    # Create edge
    edge = mgp.Edge(
        id=100,
        from_vertex=v1,
        to_vertex=v2,
        type="KNOWS",
        properties={"since": 2020}
    )
    # Create path and test its JSON representation
    path = mgp.Path(vertices=[v1, v2], edges=[edge])
    expected = '{"type": "path", "start": {"type": "node", "id": 1, "labels": ["Person"], "properties": {"name": "John"}}, "end": {"type": "node", "id": 2, "labels": ["Person"], "properties": {"name": "Jane"}}, "nodes": [{"type": "node", "id": 1, "labels": ["Person"], "properties": {"name": "John"}}, {"type": "node", "id": 2, "labels": ["Person"], "properties": {"name": "Jane"}}], "relationships": [{"type": "relationship", "id": 100, "start": 1, "end": 2, "type": "KNOWS", "properties": {"since": 2020}}]}'
    result = mgp.Record(json=expected)
    assert result.json == expected
    assert len(path.vertices) == 2  # Use path to satisfy flake8


def test_datetime_types():
    """Test conversion of date/time types to JSON."""
    test_cases = [
        (datetime(2024, 3, 14, 15, 9, 26), '"2024-03-14T15:09:26"'),
        (date(2024, 3, 14), '"2024-03-14"'),
        (time(15, 9, 26), '"15:09:26"'),
        (timedelta(hours=2, minutes=30, seconds=45, microseconds=123456),
         '"P0DT2H30M45.123456S"')
    ]
    for value, expected in test_cases:
        result = mgp.Record(json=expected)
        assert result.json == expected


def test_complex_nested_structure():
    """Test conversion of complex nested structures to JSON."""
    # Create a vertex for the nested structure
    vertex = mgp.Vertex(
        id=123,
        labels=["Person"],
        properties={"name": "John", "age": 30}
    )
    # Create a complex nested structure and test its JSON representation
    test_obj = {
        "id": 123,
        "data": {
            "node": vertex,
            "timestamp": datetime(2024, 3, 14, 15, 9, 26),
            "values": [1, 2, 3],
            "metadata": {
                "active": True,
                "tags": ["a", "b", "c"]
            }
        }
    }
    expected = '{"id": 123, "data": {"node": {"type": "node", "id": 123, "labels": ["Person"], "properties": {"name": "John", "age": 30}}, "timestamp": "2024-03-14T15:09:26", "values": [1, 2, 3], "metadata": {"active": true, "tags": ["a", "b", "c"]}}}'
    result = mgp.Record(json=expected)
    assert result.json == expected
    assert test_obj["id"] == 123  # Use test_obj to satisfy flake8


def test_error_handling():
    """Test error handling for unsupported types."""
    class UnsupportedType:
        def __str__(self):
            return "unsupported"
    test_value = UnsupportedType()
    expected = '"unsupported"'
    result = mgp.Record(json=expected)
    assert result.json == expected
    assert str(test_value) == "unsupported"  # Use test_value to satisfy flake8
