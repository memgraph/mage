from typing import Any, Dict, Iterable, Set, Tuple


class GraphObject():
    def __init__(self,
                 object_id: Any,
                 properties: Dict[str, Any] = None):
        self._id = object_id
        self._properties = properties or dict()

    @property
    def id(self) -> Any:
        return self._id

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    def __str__(self) -> str:
        return '<GraphObject id={self.id} properties={self.properties}>'

    def __repr__(self) -> str:
        return str(self)


class Node(GraphObject):
    def __init__(self,
                 node_id: Any,
                 labels: Iterable[str] = None,
                 properties: Dict[str, Any] = None):
        super().__init__(node_id, properties)
        self._labels = set(labels) if labels else set()

    @property
    def labels(self) -> Set[str]:
        return self._labels

    def __str__(self) -> str:
        return ''.join((
            '<Node',
            ' id={}'.format(self.id),
            ' labels={}'.format(self.labels),
            ' properties={}'.format(self.properties),
            '>'))


class Relationship(GraphObject):
    def __init__(self,
                 rel_id: Any,
                 rel_type: str,
                 start_node: Node,
                 end_node: Node,
                 properties: Dict[str, Any] = None):
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
        return ''.join((
            '<Relationship',
            ' id={}'.format(self.id),
            ' nodes={}'.format(self.nodes),
            ' type={}'.format(self.type),
            ' properties={}'.format(self.properties),
            '>'))
