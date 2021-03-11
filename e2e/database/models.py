from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set, Tuple, Optional


@dataclass(frozen=True, eq=True)
class MemgraphConstraint(ABC):
    label: str

    @abstractmethod
    def to_cypher(self) -> str:
        pass


@dataclass(frozen=True, eq=True)
class MemgraphConstraintUnique(MemgraphConstraint):
    property: Tuple

    def to_cypher(self) -> str:
        properties_str = ""
        if isinstance(self.property, (tuple, set, list)):
            properties_str = ", ".join([f"n.{prop}" for prop in self.property])
        else:
            properties_str = f"n.{self.property}"
        return f"(n:{self.label}) ASSERT {properties_str} IS UNIQUE"


@dataclass(frozen=True, eq=True)
class MemgraphConstraintExists(MemgraphConstraint):
    property: str

    def to_cypher(self) -> str:
        return f"(n:{self.label}) ASSERT EXISTS (n.{self.property})"


@dataclass(frozen=True, eq=True)
class MemgraphIndex:
    label: str
    property: Optional[str] = None

    def to_cypher(self):
        property_cypher = f"({self.property})" if self.property else ""
        return f":{self.label}{property_cypher}"


class GraphObject:
    def __init__(self, object_id: Any, properties: Dict[str, Any] = None):
        self._id = object_id
        self._properties = properties or dict()

    @property
    def id(self) -> Any:
        return self._id

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    def __str__(self) -> str:
        return "<GraphObject id={self.id} properties={self.properties}>"

    def __repr__(self) -> str:
        return str(self)


class Node(GraphObject):
    def __init__(
        self,
        node_id: Any,
        labels: Iterable[str] = None,
        properties: Dict[str, Any] = None,
    ):
        super().__init__(node_id, properties)
        self._labels = set(labels) if labels else set()

    @property
    def labels(self) -> Set[str]:
        return self._labels

    def __str__(self) -> str:
        return "".join(
            (
                "<Node",
                f" id={self.id}",
                f" labels={self.labels}",
                f" properties={self.properties}",
                ">",
            )
        )


class Relationship(GraphObject):
    def __init__(
        self,
        rel_id: Any,
        rel_type: str,
        start_node: Node,
        end_node: Node,
        properties: Dict[str, Any] = None,
    ):
        super().__init__(rel_id, properties)
        self._type = rel_type
        self._start_node = start_node
        self._end_node = end_node

    @property
    def type(self) -> str:
        return self._type

    @property
    def end_node(self) -> Node:
        return self._start_node

    @property
    def start_node(self) -> Node:
        return self._end_node

    @property
    def nodes(self) -> Tuple[Node, Node]:
        return (self.start_node, self.end_node)

    def __str__(self) -> str:
        return "".join(
            (
                "<Relationship",
                f" id={self.id}",
                f" nodes={self.nodes}",
                f" type={self.type}",
                f" properties={self.properties}",
                ">",
            )
        )
