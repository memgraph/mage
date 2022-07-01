from collections import defaultdict
from typing import Dict, List, Tuple

import igraph

import mgp


@mgp.read_proc
def get_flow(
    ctx: mgp.ProcCtx,
    start_v: mgp.Vertex,
    end_v: mgp.Vertex,
    edge_property: str = "weight",
) -> mgp.Record(max_flow=mgp.Number):

    graph, id_mapping, _ = create_igraph_from_ctx(ctx, directed=True)
    max_flow = graph.maxflow(
        id_mapping[start_v.id], id_mapping[end_v.id], capacity=edge_property
    )

    return mgp.Record(max_flow=max_flow.value)


@mgp.read_proc
def pagerank(
    ctx: mgp.ProcCtx,
    damping: mgp.Number = 0.85,
    max_iter: int = 100,
    tol: mgp.Number = 1e-06,
    weight: mgp.Nullable[str] = None,
) -> mgp.Record(node=mgp.Vertex, rank=float):
    graph, _, inverse_id_mapping = create_igraph_from_ctx(ctx, directed=True)
    pg = graph.pagerank(
        weights=weight, directed=True, niter=max_iter, damping=damping, eps=tol
    )
    return [
        mgp.Record(
            node=ctx.graph.get_vertex_by_id(inverse_id_mapping[node_id]), rank=rank
        )
        for node_id, rank in enumerate(pg)
    ]


@mgp.read_proc
def all_simple_paths(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    cutoff: int = -1,
) -> mgp.Record(paths=mgp.List[mgp.List[mgp.Vertex]]):
    graph, id_mapping, inverted_id_mapping = create_igraph_from_ctx(ctx, directed=True)
    paths = [
        convert_vertex_ids_to_mgp_vertices(path, ctx, inverted_id_mapping)
        for path in graph.get_all_simple_paths(
            v=id_mapping[source.id], to=id_mapping[target.id], cutoff=cutoff
        )
    ]

    return mgp.Record(paths=paths)


@mgp.read_proc
def min_cut(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    edge_property: mgp.Nullable[str] = None,
) -> mgp.Record(partition_vertex_ids=List[List[mgp.Vertex]], value=float):
    graph, id_mapping, inverted_id_mapping = create_igraph_from_ctx(ctx, directed=True)
    mincut = graph.mincut(
        source=id_mapping[source.id],
        target=id_mapping[target.id],
        capacity=edge_property,
    )

    return mgp.Record(
        partition_vertex_ids=[
            convert_vertex_ids_to_mgp_vertices(partition, ctx, inverted_id_mapping)
            for partition in mincut.partition
        ],
        value=mincut.value,
    )


@mgp.read_proc
def topological_sort(
    ctx: mgp.ProcCtx,
) -> mgp.Record(nodes=mgp.Nullable[mgp.List[mgp.Vertex]]):
    graph, _, inverted_id_mapping = create_igraph_from_ctx(ctx, directed=True)
    sorted = graph.topological_sorting(mode="out")

    return mgp.Record(
        nodes=convert_vertex_ids_to_mgp_vertices(sorted, ctx, inverted_id_mapping)
    )


@mgp.read_proc
def community_leiden(
    ctx: mgp.ProcCtx,
    edge_property: mgp.Nullable[str] = None,
    resolution_parameter: float = 0.6,
    number_of_iterations: int = -1,
) -> mgp.Record(community_id=int, community_members=List[mgp.Vertex]):
    graph, _, inverted_id_mapping = create_igraph_from_ctx(ctx, directed=False)

    communities = graph.community_leiden(
        resolution_parameter=resolution_parameter,
        weights=edge_property,
        n_iterations=number_of_iterations,
    )

    return [
        mgp.Record(
            community_id=i,
            community_members=convert_vertex_ids_to_mgp_vertices(
                members, ctx, inverted_id_mapping
            ),
        )
        for i, members in enumerate(communities)
    ]


@mgp.read_proc
def spanning_tree(
    ctx: mgp.ProcCtx, edge_property: mgp.Nullable[str] = None, directed: bool = True
) -> mgp.Record(tree=List[List[mgp.Vertex]]):
    graph, id_mapping, inverted_id_mapping = create_igraph_from_ctx(
        ctx, directed=directed
    )
    edge_weights = None
    if edge_property:
        edge_weights = graph.es[edge_property]

    min_span_tree_graph = graph.spanning_tree(weights=edge_weights)
    min_span_tree = get_min_span_tree_vertex_pairs(
        ctx=ctx,
        id_mapping=id_mapping,
        inverted_id_mapping=inverted_id_mapping,
        min_span_tree_graph=min_span_tree_graph,
    )

    return mgp.Record(tree=min_span_tree)


@mgp.read_proc
def shortest_path_length(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    edge_property: mgp.Nullable[str] = None,
    directed: bool = True,
) -> mgp.Record(length=float):
    graph, id_mapping, _ = create_igraph_from_ctx(ctx, directed=directed)

    return mgp.Record(
        length=graph.shortest_paths(
            source=id_mapping[source.id],
            target=id_mapping[target.id],
            weights=edge_property,
        )[0][0]
    )


@mgp.read_proc
def all_shortest_path_lengths(
    ctx: mgp.ProcCtx,
    edge_property: mgp.Nullable[str] = None,
    directed: bool = False,
) -> mgp.Record(lengths=List[List[float]]):
    graph, _, _ = create_igraph_from_ctx(ctx, directed=directed)

    return mgp.Record(
        lengths=graph.shortest_paths(
            weights=edge_property,
        )
    )


@mgp.read_proc
def get_shortest_path(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    edge_property: mgp.Nullable[str] = None,
    directed: bool = True,
) -> mgp.Record(path=List[mgp.Vertex]):
    graph, id_mapping, invert_id_mapping = create_igraph_from_ctx(
        ctx, directed=directed
    )
    path = graph.get_shortest_paths(
        v=id_mapping[source.id], to=id_mapping[target.id], weights=edge_property
    )[0]

    return mgp.Record(
        path=convert_vertex_ids_to_mgp_vertices(path, ctx, invert_id_mapping)
    )


def convert_vertex_ids_to_mgp_vertices(
    vertex_ids: List[int], ctx: mgp.ProcCtx, inverted_id_mapping: Dict[int, int]
) -> List[mgp.Vertex]:

    vertices = []
    for id in vertex_ids:
        vertices.append(ctx.graph.get_vertex_by_id(inverted_id_mapping[id]))

    return vertices


def get_min_span_tree_vertex_pairs(
    ctx: mgp.ProcCtx,
    id_mapping: Dict[int, int],
    inverted_id_mapping: Dict[int, int],
    min_span_tree_graph: igraph.Graph,
) -> List[List[mgp.Vertex]]:
    """Function for getting vertex pairs that are connected in minimum spanning tree.

    Args:
        ctx (mgp.ProcCtx): Memgraph ProcCtx object
        id_mapping (Dict[int, int]): Vertex id mappings
        inverted_id_mapping (Dict[int,int]): Inverted vertex id mappings
        min_span_tree_graph (igraph.Graph): Igraph graph containing minimum spanning tree

    Returns:
        List[List[mgp.Vertex]]: List of vertex pairs that are connected in minimum spanning tree
    """
    unique_edges_dict = dict()
    not_isolated_vertex_directed = dict()
    min_span_tree = []
    for vertex in ctx.graph.vertices:
        isolated = True
        for neighbor in min_span_tree_graph.neighbors(id_mapping[vertex.id]):
            if (id_mapping[vertex.id], neighbor) not in unique_edges_dict and (
                neighbor,
                id_mapping[vertex.id],
            ) not in unique_edges_dict:
                min_span_tree.append(
                    [
                        ctx.graph.get_vertex_by_id(vertex.id),
                        ctx.graph.get_vertex_by_id(inverted_id_mapping[neighbor]),
                    ]
                )
                unique_edges_dict[(neighbor, id_mapping[vertex.id])] = True
                isolated = False
                not_isolated_vertex_directed[neighbor] = 1

        if isolated and id_mapping[vertex.id] not in not_isolated_vertex_directed:
            min_span_tree.append([ctx.graph.get_vertex_by_id(vertex.id)])

    return min_span_tree


def create_igraph_from_ctx(
    ctx: mgp.ProcCtx, directed: bool = False
) -> Tuple[igraph.Graph, Dict[int, int], Dict[int, int]]:
    """Function for creating igraph.Graph from mgp.ProcCtx.

    Args:
        ctx (mgp.ProcCtx): memgraph ProcCtx object
        directed (bool, optional): Is graph directed. Defaults to False.

    Returns:
        Tuple[igraph.Graph, Dict[int, int], Dict[int, int]]: Returns Igraph.Graph object, vertex id mappings and inverted_id_mapping vertex id mappings
    """

    vertex_attrs = defaultdict(list)
    edge_list = []
    edge_attrs = defaultdict(list)
    id_mapping = {vertex.id: i for i, vertex in enumerate(ctx.graph.vertices)}
    inverted_id_mapping = {i: vertex.id for i, vertex in enumerate(ctx.graph.vertices)}
    for vertex in ctx.graph.vertices:
        for name, value in vertex.properties.items():
            vertex_attrs[name].append(value)
        for edge in vertex.out_edges:
            for name, value in edge.properties.items():
                edge_attrs[name].append(value)
            edge_list.append(
                (id_mapping[edge.from_vertex.id], id_mapping[edge.to_vertex.id])
            )

    graph = igraph.Graph(
        directed=directed,
        n=len(ctx.graph.vertices),
        edges=edge_list,
        edge_attrs=edge_attrs,
        vertex_attrs=vertex_attrs,
    )

    return graph, id_mapping, inverted_id_mapping
