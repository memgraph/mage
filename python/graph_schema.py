import mgp
from collections import defaultdict
from typing import Dict, Tuple, Union, Iterator

NodeKeyType = Tuple[str, ...]
EdgeKeyType = Tuple[NodeKeyType, str, NodeKeyType]

NODE_TYPE = "node"
EDGE_TYPE = "relationship"


class Counter:
    def __init__(self, initial_value: int = 0):
        self.total_count = initial_value
        self.count_by_property_name = defaultdict(int)

    def count(self) -> None:
        self.total_count += 1

    def count_property(self, property_name: str) -> None:
        self.count_by_property_name[property_name] += 1

    def to_dict(self):
        return {
            "count": self.total_count,
            "properties_count": self.count_by_property_name,
        }


@mgp.read_proc
def get_schema(
    context: mgp.ProcCtx, include_properties: bool = False
) -> mgp.Record(nodes=mgp.List[mgp.Map], edges=mgp.List[mgp.Map]):
    node_count_by_labels: Dict[NodeKeyType, Counter] = {}
    edge_count_by_labels: Dict[EdgeKeyType, Counter] = {}

    for node in context.graph.vertices:
        labels = tuple(sorted(l.name for l in node.labels))
        _update_counts(
            node_count_by_labels,
            key=labels,
            obj=node,
            include_properties=include_properties,
        )

        for edge in node.out_edges:
            target_labels = tuple(sorted(l.name for l in edge.to_vertex.labels))
            key = (labels, edge.type.name, target_labels)
            _update_counts(
                edge_count_by_labels,
                key=key,
                obj=edge,
                include_properties=include_properties,
            )

    node_index_by_labels = {key: i for i, key in enumerate(node_count_by_labels.keys())}
    nodes = list(_iter_nodes_as_map(node_count_by_labels, node_index_by_labels))
    edges = list(_iter_edges_as_map(edge_count_by_labels, node_index_by_labels))

    return mgp.Record(nodes=nodes, edges=edges)


def _update_counts(
    obj_count_by_key: Dict[Union[NodeKeyType, EdgeKeyType], Counter],
    key: Union[NodeKeyType, EdgeKeyType],
    obj: Union[mgp.Vertex, mgp.Edge],
    include_properties: bool = False,
) -> None:
    if key not in obj_count_by_key:
        obj_count_by_key[key] = Counter()

    obj_counter = obj_count_by_key[key]
    obj_counter.count()

    if include_properties:
        for property_name in obj.properties.keys():
            obj_counter.count_property(property_name)


def _iter_nodes_as_map(
    node_count_by_labels: Dict[NodeKeyType, Counter],
    node_index_by_labels: Dict[NodeKeyType, int],
) -> Iterator[mgp.Map]:
    for labels, counter in node_count_by_labels.items():
        yield {
            "id": node_index_by_labels.get(labels),
            "labels": labels,
            "properties": counter.to_dict(),
            "type": NODE_TYPE
        }


def _iter_edges_as_map(
    edge_count_by_labels: Dict[EdgeKeyType, Counter],
    node_index_by_labels: Dict[NodeKeyType, int],
) -> Iterator[mgp.Map]:
    for i, ((source_label, edge_label, target_label), counter) in enumerate(
        edge_count_by_labels.items()
    ):
        source_node_id = node_index_by_labels.get(source_label)
        target_node_id = node_index_by_labels.get(target_label)

        if source_node_id is not None and target_node_id is not None:
            yield {
                "id": i,
                "start": source_node_id,
                "end": target_node_id,
                "label": edge_label,
                "properties": counter.to_dict(),
                "type": EDGE_TYPE
            }
