import mgp

from collections import deque
from typing import Dict, List, Tuple, Set, Optional


@mgp.read_proc
def collapse(
    context: mgp.ProcCtx,
    vertices: mgp.List[mgp.Vertex],
    collapse_edge_types: mgp.List[str],
    collapse_pseudo_node_labels: mgp.Nullable[List[str]] = None,
) -> mgp.Record(from_vertex=mgp.Vertex, path=mgp.Path, to_vertex=mgp.Vertex):
    """
    Returns collapsed graph.

    Returned collapsed graph is a list of records: (from_node, path, to_node).
    Algorithm collapses all related vertices into a one of the top vertices.
    List of top vertices is given by vertices parameter
    If edge has one of the types defined in collapse_edge_types parameter,
    edge to_vertex will be collapsed in the same top vertex as edge from_vertex.

    collapse_pseudo_node_labels is optional parameter. This parameter must be used
    if top vertices are connected with pseudo vertex
    (vertices with exactly one input edge and exactly one output edge).

    Path can contain one edge or two edges.
    Path contains two edges in a pseudo vertex case.

    Procedure will raise error if any of pseudo vertices defined by collapse_pseudo_node_labels
    parameter will have more or less than one input and one output edges.

    Procedure will raise an error if collapsed groups aren't disjunctive sets.

    Example call:
        MATCH (n:Plant) WITH COLLECT(n) AS nodes
        CALL collapse.collapse(nodes, ["OWN"], ["Transport"])
        YIELD from_vertex, path, to_vertex
        RETURN from_vertex, nodes(path), to_vertex;
    """
    union_find = dict()
    node_to_edge = set(collapse_pseudo_node_labels) if collapse_pseudo_node_labels else None
    vertices_id = {v.id for v in vertices}
    vertex_neighbours = []
    visited_global = set()
    for v in vertices:
        paths, group = _bfs(v, vertices_id, collapse_edge_types, visited_global, node_to_edge)
        visited_global.update([child_id for child_id, _ in group.items()])
        union_find.update(group)
        vertex_neighbours.append((v, paths))

    records = []
    for from_node, paths in vertex_neighbours:
        for path in paths:
            mgp_path = mgp.Path(path[0].from_vertex)
            for edge in path:
                mgp_path.expand(edge)
            to_node = union_find.get(edge.to_vertex.id)
            if from_node == to_node and len(mgp_path.edges) == 2:
                continue
            if to_node is None:
                continue
            records.append(mgp.Record(from_vertex=from_node, path=mgp_path, to_vertex=to_node))

    return records


@mgp.read_proc
def groups(
    context: mgp.ProcCtx,
    vertices: mgp.List[mgp.Vertex],
    collapse_edge_types: mgp.List[str],
    collapse_pseudo_node_labels: mgp.Nullable[List[str]] = None,
) -> mgp.Record(top_vertex=mgp.Vertex, collapsed_vertices=mgp.List[mgp.Vertex]):
    """
    Returns top vertices with corresponding collapsed vertices.

    Returned list of records are: (top_vertex, collapsed_vertices).
    Algorithm collapses all related vertices into a one of the top vertices.
    List of top vertices is given by vertices parameter.
    If edge has one of the types defined in collapse_edge_types parameter,
    edge to_vertex will be collapsed in the same top vertex as edge from_vertex.

    collapse_pseudo_node_labels is optional parameter. This parameter must be used
    if top vertices are connected with pseudo vertex
    (vertices with exactly one input edge and exactly one output edge).

    Path can contain one edge or two edges.
    Path contains two edges in a pseudo vertex case.

    Procedure will raise error if any of pseudo vertices defined by collapse_pseudo_node_labels
    parameter will have more or less than one input and one output edges.

    Procedure will raise an error if collapsed groups aren't disjunctive sets.

    Example call:
        MATCH (n:Plant) WITH COLLECT(n) AS nodes
        CALL collapse.groups(nodes, ["OWN"], ["Transport"])
        YIELD *
        RETURN top_vertex, collapsed_vertices;
    """
    node_to_edge = set(collapse_pseudo_node_labels) if collapse_pseudo_node_labels else None
    vertices_id = {v.id for v in vertices}
    records = []
    visited_global = set()
    for v in vertices:
        _, group = _bfs(v, vertices_id, collapse_edge_types,visited_global, node_to_edge)
        visited_global.update([child_id for child_id, _ in group.items()])
        children = [context.graph.get_vertex_by_id(child_id) for child_id, _ in group.items()]
        records.append(mgp.Record(top_vertex=v, collapsed_vertices=children))

    return records


def _bfs(
    vertex: mgp.Vertex,
    target_vertices_id: Set[int],
    collapse_edge_types: mgp.List[str],
    visited_global: Optional[Set],
    collapse_pseudo_node_labels: Optional[Set],
) -> Tuple[List[mgp.Edge], Dict[mgp.Vertex, int]]:

    out_edges = []
    union_find = dict()
    visited = set()
    queue = deque()

    queue.append(vertex)
    while queue:
        top = queue.popleft()
        visited.add(top.id)
        union_find[top.id] = vertex
        for out_edge in top.out_edges:
            out_vertex = out_edge.to_vertex

            out_vertex_labels = {o.name for o in out_vertex.labels}
            if collapse_pseudo_node_labels and collapse_pseudo_node_labels.intersection(
                out_vertex_labels
            ):
                _check_edges(
                    out_vertex, out_vertex.in_edges, out_vertex.out_edges, collapse_edge_types
                )
                out_edges.append([next(out_vertex.in_edges), next(out_vertex.out_edges)])
                continue
            if out_edge.type.name not in collapse_edge_types or out_vertex.id in target_vertices_id:
                out_edges.append([out_edge])
                continue
            if out_vertex.id in visited:
                continue

            if out_vertex.id in visited_global:
                raise Exception(
                    "Invalid input graph. Node id={} can't be"
                    " included in multiple groups.".format(vertex.id)
                )
            queue.append(out_vertex)

    return out_edges, union_find


def _check_edges(
    vertex: mgp.Vertex,
    in_edges: List[mgp.Edge],
    out_edges: List[mgp.Edge],
    collapse_edge_types: mgp.List[str],
):

    check_list = [in_edges, out_edges]
    for edges in check_list:
        filtered = [e for e in edges if e.type.name not in collapse_edge_types]
        if len(filtered) == 1:
            continue
        raise Exception(
            "Node id={} is considered as edge"
            " and doesn't have one input and one output edge.".format(vertex.id)
        )
