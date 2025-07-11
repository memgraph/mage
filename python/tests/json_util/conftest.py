import os
import sys
from typing import List, Dict, Any

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "../.."))
sys.path.insert(0, project_root)

# Add any fixtures here if needed in the future


# Mock classes to simulate Memgraph objects for testing
class MockLabel:
    def __init__(self, name: str):
        self.name = name


class MockVertex:
    def __init__(self, id: int, labels: list, properties: dict):
        self.id = id
        self.labels = [MockLabel(label) for label in labels]
        self.properties = properties


class MockEdge:
    def __init__(self, id: int, from_vertex: MockVertex,
                 to_vertex: MockVertex, type_name: str, properties: dict):
        self.id = id
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.type = MockLabel(type_name)
        self.properties = properties


class MockPath:
    def __init__(self, vertices: list, edges: list):
        self.vertices = vertices
        self.edges = edges


# Mock mgp module
class MockMgp:
    class Label:
        def __init__(self, name: str):
            self.name = name

    class Vertex:
        def __init__(self, id: int, labels: List[str],
                     properties: Dict[str, Any]):
            self.id = id
            self.labels = [MockMgp.Label(label) for label in labels]
            self.properties = properties

    class Edge:
        def __init__(
            self,
            id: int,
            from_vertex: "MockMgp.Vertex",
            to_vertex: "MockMgp.Vertex",
            type_name: str,
            properties: Dict[str, Any],
        ):
            self.id = id
            self.from_vertex = from_vertex
            self.to_vertex = to_vertex
            self.type = MockMgp.Label(type_name)
            self.properties = properties

    class Path:
        def __init__(
            self, vertices: List["MockMgp.Vertex"],
            edges: List["MockMgp.Edge"]
        ):
            self.vertices = vertices
            self.edges = edges

    class Record:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ProcCtx:
        pass

    class Node(MockVertex):
        pass

    class Relationship(MockEdge):
        pass

    @staticmethod
    def read_proc(f):
        return f

    @staticmethod
    def function(f):
        return f

    # Mock List type
    List = list


# Mock the mgp module
sys.modules['mgp'] = MockMgp
