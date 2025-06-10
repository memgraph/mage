"""Fixtures for convert module tests."""
import pytest
from mage.convert import mock_mgp as mgp


@pytest.fixture
def vertex():
    """Create a test vertex."""
    return mgp.Vertex(
        id=123,
        labels=["Person"],
        properties={"name": "John", "age": 30}
    )


@pytest.fixture
def edge(vertex):
    """Create a test edge."""
    to_vertex = mgp.Vertex(
        id=456,
        labels=["Person"],
        properties={"name": "Jane"}
    )
    return mgp.Edge(
        id=789,
        from_vertex=vertex,
        to_vertex=to_vertex,
        type="KNOWS",
        properties={"since": 2020}
    )


@pytest.fixture
def path(vertex):
    """Create a test path."""
    to_vertex = mgp.Vertex(
        id=456,
        labels=["Person"],
        properties={"name": "Jane"}
    )
    edge = mgp.Edge(
        id=789,
        from_vertex=vertex,
        to_vertex=to_vertex,
        type="KNOWS",
        properties={"since": 2020}
    )
    return mgp.Path(vertices=[vertex, to_vertex], edges=[edge])
