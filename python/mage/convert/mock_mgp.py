"""Mock MGP module for testing."""
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class Label:
    """Mock Label class."""
    name: str


@dataclass
class Type:
    """Mock Type class."""
    name: str


@dataclass
class Vertex:
    """Mock Vertex class."""
    id: int
    labels: List[Label]
    properties: Dict[str, Any]

    def __init__(self, id: int, labels: List[str], properties: Dict[str, Any]):
        self.id = id
        self.labels = [Label(name=label) for label in labels]
        self.properties = properties


@dataclass
class Edge:
    """Mock Edge class."""
    id: int
    from_vertex: Vertex
    to_vertex: Vertex
    type: Type
    properties: Dict[str, Any]

    def __init__(self, id: int, from_vertex: Vertex, to_vertex: Vertex,
                 type: str, properties: Dict[str, Any]):
        self.id = id
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.type = Type(name=type)
        self.properties = properties


@dataclass
class Path:
    """Mock Path class."""
    vertices: List[Vertex]
    edges: List[Edge]


@dataclass
class Record:
    """Mock Record class."""
    value: Optional[Any] = None
    json: Optional[str] = None 