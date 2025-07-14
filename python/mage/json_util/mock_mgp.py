"""Mock MGP module for testing."""
from typing import Any, Dict, TypeVar, Generic

T = TypeVar('T')


class List(Generic[T]):
    """Mock List class."""
    def __init__(self, *args):
        self.items = list(args)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class Vertex:
    """Mock Vertex class."""
    def __init__(self, id: int, labels: List[str],
                 properties: Dict[str, Any]):
        self.id = id
        self.labels = labels
        self.properties = properties


class Edge:
    """Mock Edge class."""
    def __init__(self, id: int, from_vertex: Vertex, to_vertex: Vertex,
                 type: str, properties: Dict[str, Any]):
        self.id = id
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.type = type
        self.properties = properties


class Path:
    """Mock Path class."""
    def __init__(self, vertices: List[Vertex], edges: List[Edge]):
        self.vertices = vertices
        self.edges = edges


class Record:
    """Mock Record class."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, **kwargs):
        return self.__class__(**kwargs)


class ProcCtx:
    """Mock ProcCtx class."""
    pass


def read_proc(func):
    """Mock read_proc decorator."""
    return func


def write_proc(func):
    """Mock write_proc decorator."""
    return func 


def function(func):
    """Mock function decorator."""
    return func
