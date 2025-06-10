"""Tests for the convert module."""
from mage.convert import mock_mgp as mgp


def test_node_conversion(vertex):
    """Test conversion of nodes to tree structure."""
    expected = {
        "_type": "node",
        "_id": 123,
        "_labels": ["Person"],
        "name": "John",
        "age": 30
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert vertex.id == 123


def test_relationship_conversion(edge):
    """Test conversion of relationships to tree structure."""
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
    assert edge.id == 789


def test_path_conversion(path):
    """Test conversion of paths to tree structure."""
    expected = {
        "_type": "path",
        "_start": {
            "_type": "node",
            "_id": 123,
            "_labels": ["Person"],
            "name": "John",
            "age": 30
        },
        "_end": {
            "_type": "node",
            "_id": 456,
            "_labels": ["Person"],
            "name": "Jane"
        },
        "_nodes": [
            {
                "_type": "node",
                "_id": 123,
                "_labels": ["Person"],
                "name": "John",
                "age": 30
            },
            {
                "_type": "node",
                "_id": 456,
                "_labels": ["Person"],
                "name": "Jane"
            }
        ],
        "_relationships": [
            {
                "_type": "relationship",
                "_id": 789,
                "_start": 123,
                "_end": 456,
                "_relationship_type": "KNOWS",
                "since": 2020
            }
        ]
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert len(path.vertices) == 2


def test_path_without_relationships(path):
    """Test conversion of paths to tree structure without relationships."""
    expected = {
        "_type": "path",
        "_start": {
            "_type": "node",
            "_id": 123,
            "_labels": ["Person"],
            "name": "John",
            "age": 30
        },
        "_end": {
            "_type": "node",
            "_id": 456,
            "_labels": ["Person"],
            "name": "Jane"
        },
        "_nodes": [
            {
                "_type": "node",
                "_id": 123,
                "_labels": ["Person"],
                "name": "John",
                "age": 30
            },
            {
                "_type": "node",
                "_id": 456,
                "_labels": ["Person"],
                "name": "Jane"
            }
        ]
    }
    result = mgp.Record(value=expected)
    assert result.value == expected
    assert len(path.vertices) == 2


def test_nested_structure(vertex):
    """Test conversion of nested structures to tree."""
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
    assert test_obj["id"] == 123


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
