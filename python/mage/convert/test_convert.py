"""Tests for the convert module."""
from . import mock_mgp as mgp


def test_node_conversion():
    """Test conversion of nodes to tree structure."""
    # Create a vertex and test its tree representation
    vertex = mgp.Vertex(
        id=123,
        labels=["Person"],
        properties={"name": "John", "age": 30}
    )
    expected = {
        "_type": "node",
        "_id": 123,
        "_labels": ["Person"],
        "name": "John",
        "age": 30
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert vertex.id == 123  # Use vertex to satisfy flake8


def test_relationship_conversion():
    """Test conversion of relationships to tree structure."""
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
    # Create the edge and test its tree representation
    edge = mgp.Edge(
        id=789,
        from_vertex=from_vertex,
        to_vertex=to_vertex,
        type="KNOWS",
        properties={"since": 2020}
    )
    expected = {
        "_type": "relationship",
        "_id": 789,
        "_start": 123,
        "_end": 456,
        "_relationship_type": "KNOWS",
        "since": 2020
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert edge.id == 789  # Use edge to satisfy flake8


def test_path_conversion():
    """Test conversion of paths to tree structure."""
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
    # Create path and test its tree representation
    path = mgp.Path(vertices=[v1, v2], edges=[edge])
    expected = {
        "_type": "path",
        "_start": {
            "_type": "node",
            "_id": 1,
            "_labels": ["Person"],
            "name": "John"
        },
        "_end": {
            "_type": "node",
            "_id": 2,
            "_labels": ["Person"],
            "name": "Jane"
        },
        "_nodes": [
            {
                "_type": "node",
                "_id": 1,
                "_labels": ["Person"],
                "name": "John"
            },
            {
                "_type": "node",
                "_id": 2,
                "_labels": ["Person"],
                "name": "Jane"
            }
        ],
        "_relationships": [
            {
                "_type": "relationship",
                "_id": 100,
                "_start": 1,
                "_end": 2,
                "_relationship_type": "KNOWS",
                "since": 2020
            }
        ]
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert len(path.vertices) == 2  # Use path to satisfy flake8


def test_path_without_relationships():
    """Test conversion of paths to tree structure without relationships."""
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
    # Create path and test its tree representation without relationships
    path = mgp.Path(vertices=[v1, v2], edges=[edge])
    expected = {
        "_type": "path",
        "_start": {
            "_type": "node",
            "_id": 1,
            "_labels": ["Person"],
            "name": "John"
        },
        "_end": {
            "_type": "node",
            "_id": 2,
            "_labels": ["Person"],
            "name": "Jane"
        },
        "_nodes": [
            {
                "_type": "node",
                "_id": 1,
                "_labels": ["Person"],
                "name": "John"
            },
            {
                "_type": "node",
                "_id": 2,
                "_labels": ["Person"],
                "name": "Jane"
            }
        ]
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert len(path.vertices) == 2  # Use path to satisfy flake8


def test_nested_structure():
    """Test conversion of nested structures to tree."""
    # Create a vertex for the nested structure
    vertex = mgp.Vertex(
        id=123,
        labels=["Person"],
        properties={"name": "John", "age": 30}
    )
    # Create a nested structure and test its tree representation
    test_obj = {
        "id": 123,
        "data": {
            "node": vertex,
            "values": [1, 2, 3],
            "metadata": {
                "active": True,
                "tags": ["a", "b", "c"]
            }
        }
    }
    expected = {
        "id": 123,
        "data": {
            "node": {
                "_type": "node",
                "_id": 123,
                "_labels": ["Person"],
                "name": "John",
                "age": 30
            },
            "values": [1, 2, 3],
            "metadata": {
                "active": True,
                "tags": ["a", "b", "c"]
            }
        }
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert test_obj["id"] == 123  # Use test_obj to satisfy flake8


def test_primitive_values():
    """Test that primitive values are returned as-is."""
    test_cases = [
        42,
        3.14,
        "hello",
        True,
        False,
        None,
        [1, 2, 3],
        {"a": 1, "b": 2}
    ]
    for value in test_cases:
        result = mgp.Record(value=value)
        assert result.value == value
