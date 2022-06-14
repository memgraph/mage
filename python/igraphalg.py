import igraph
import mgp
from collections import defaultdict
from typing import List


@mgp.read_proc
def get_flow(
    ctx: mgp.ProcCtx,
    start_v: mgp.Vertex,
    end_v: mgp.Vertex,
    edge_property: str = "weight",
) -> mgp.Record(max_flow=mgp.Number):

    graph = create_igraph_from_ctx(ctx, directed=True)
    max_flow = graph.maxflow(start_v.id, end_v.id, capacity=edge_property)

    return mgp.Record(max_flow=max_flow.value)


@mgp.read_proc
def pagerank(
    ctx: mgp.ProcCtx,
    damping: mgp.Number = 0.85,
    max_iter: int = 100,
    tol: mgp.Number = 1e-06,
    weight: mgp.Nullable[str] = "weight",
) -> mgp.Record(node=mgp.Vertex, rank=float):

    graph = create_igraph_from_ctx(ctx, directed=False)
    pg = graph.pagerank(weights=weight, niter=max_iter, damping=damping, eps=tol)

    return [mgp.Record(node=k, rank=v) for k, v in enumerate(pg)]


@mgp.read_proc
def all_simple_paths(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    cutoff: mgp.Nullable[int] = None,
) -> mgp.Record(paths=mgp.List[mgp.List[mgp.Vertex]]):
    graph = create_igraph_from_ctx(ctx, directed=True)

    return mgp.Record(
        paths=list(graph.get_all_simple_paths(v=source.id, to=target.id, cutoff=cutoff))
    )


@mgp.read_proc
def min_cuts(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    edge_property: str = "weights"
) -> mgp.Record(partition=List[mgp.Vertex], cut=List[mgp.Edge], value=float):
    graph = create_igraph_from_ctx(ctx, directed=True)
    mincut = graph.mincut(source=source.id, target=target.id, capacity=edge_property)

    return mgp.Record(partition=mincut.partition, cut=mincut.cut, value=mincut.value)


@mgp.read_proc
def community_leiden(ctx: mgp.ProcCtx, edge_property: str = "weigths",
                     resolution_parameter: float = 0.6,
                     number_of_iterations: int = -1,
                     ) -> mgp.Record(community_index=int, community_members=List[mgp.Vertex]):
    graph = create_igraph_from_ctx(ctx, directed=True)
    communities = graph.community_leiden(resolution_parameter=resolution_parameter, weights=edge_property, n_iterations=number_of_iterations)

    return [mgp.Record(community_index=i, community_members=members) for i, members in enumerate(communities)]


@mgp.read_proc
def spanning_tree(ctx: mgp.ProcCtx,
                  edge_property: str = "weigths"
                  ) -> mgp.Record(tree=List[mgp.Vertex]):
    graph = create_igraph_from_ctx(ctx, directed=True)

    return mgp.Record(tree=graph.spanning_tree(edge_property, return_tree=False))

def create_igraph_from_ctx(ctx: mgp.ProcCtx, directed: bool = False) -> igraph.Graph:
    vertex_attrs = defaultdict(list)
    edge_list = []
    edge_attrs = defaultdict(list)
    for vertex in ctx.graph.vertices:
        for name, value in vertex.properties.items():
            vertex_attrs[name].append(value)
        for edge in vertex.out_edges:
            for name, value in edge.properties.items():
                edge_attrs[name].append(value)
            edge_list.append((edge.from_vertex.id, edge.to_vertex.id))

    graph = igraph.Graph(
        directed=directed,
        n=len(vertex_attrs),
        edges=edge_list,
        edge_attrs=edge_attrs,
        vertex_attrs=vertex_attrs,
    )

    return graph


def create_igraph_from_matrix(
    weighted_adjacency: List[List[float]], mode="directed", attr="weight", multi=False
) -> igraph.Graph:
    """Create igraph graph from weighted 2D matrix

    Args:
        matrix (List[List[float]]): weighted matrix

    Returns:
        Igraph graph

    """

    graph = igraph.Graph.Weighted_Adjacency(
        weighted_adjacency, mode=mode, attr=attr, loops=multi
    )

    return graph
