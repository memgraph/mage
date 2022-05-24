import igraph
import mgp
from collections import defaultdict
from typing import List
import random

from mgp_networkx import (
    MemgraphMultiDiGraph,
    MemgraphDiGraph,  # noqa: E402
    MemgraphMultiGraph,
    MemgraphGraph,
)

random.seed(0)
igraph.set_random_number_generator(random)


@mgp.read_proc
def get_flow(
    ctx: mgp.ProcCtx,
    start_v: mgp.Vertex,
    end_v: mgp.Vertex,
    edge_property: str = "weight",
) -> mgp.Record(max_flow=mgp.Number):

    # graph = create_igraph(context, 'directed', True)
    graph = create_igraph_from_ctx(ctx, directed=True)
    max_flow = graph.maxflow(start_v.id, end_v.id, capacity=edge_property)

    return mgp.Record(max_flow=max_flow.value)


def create_igraph_from_ctx(ctx: mgp.ProcCtx, directed: bool = False):
    vertex_attrs = defaultdict(list)
    edge_list = []
    edge_attrs = defaultdict(list)
    for vertex in ctx._graph.vertices:
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
):
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


def create_igraph(ctx: mgp.ProcCtx, mode: str, multi: bool):
    if mode == "directed" and multi:
        return igraph.Graph.from_networkx(MemgraphMultiDiGraph(ctx=ctx))
    elif mode == "undirected" and multi:
        return igraph.Graph.from_networkx(MemgraphMultiGraph(ctx=ctx))
    elif mode == "directed" and not multi:
        return igraph.Graph.from_networkx(MemgraphDiGraph(ctx=ctx))
    elif mode == "undirected" and not multi:
        return igraph.Graph.from_networkx(MemgraphGraph(ctx=ctx))
