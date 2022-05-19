import igraph
import mgp

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


## ZBOG PRETVORBE JEDNOG GRAFA U DRUGI PA TRECI igraph:250ms, Bruno max_flow:70ms
@mgp.read_proc
def get_flow(
    context: mgp.ProcCtx,
    start_v: mgp.Vertex,
    end_v: mgp.Vertex,
    edge_property: str = "weight",
) -> mgp.Record(max_flow=mgp.Number):

    graph = create_igraph(context, 'directed', True)
    max_flow = graph.maxflow(start_v.id,end_v.id,capacity=edge_property)

    return mgp.Record(max_flow=max_flow.value)

def create_igraph_from_matrix(weighted_adjacency: List[List[float]], mode = 'directed',attr = "weight",multi = False):
    """Create igraph graph from weighted 2D matrix 

    Args:
        matrix (List[List[float]]): weighted matrix

    Returns:
        Igraph graph

    """

    graph = igraph.Graph.Weighted_Adjacency(weighted_adjacency, mode = mode,attr = attr,loops = multi)

    return graph

#DOSTA SPORIJE JER PRVO NAS PRETVARA U NETWORKX GRAPH PA ONDA U IGRAPH GRAPH
def create_igraph(ctx: mgp.ProcCtx,mode: str,multi: bool):
    if mode == 'directed' and multi:
        return igraph.Graph.from_networkx(MemgraphMultiDiGraph(ctx=ctx))
    elif mode == 'undirected' and multi:
        return igraph.Graph.from_networkx(MemgraphMultiGraph(ctx=ctx))
    elif mode == 'directed' and not multi:
        return igraph.Graph.from_networkx(MemgraphDiGraph(ctx=ctx))
    elif mode == 'undirected' and not multi:
        return igraph.Graph.from_networkx(MemgraphGraph(ctx=ctx))
