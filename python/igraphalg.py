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

def create_igraph_from_matrix(weighted_adjacency: List[List[float]], mode = 'directed',attr = 'weight',multi = False):
    """Create igraph graph from weighted 2D matrix 

    Args:
        matrix (List[List[float]]): weighted matrix

    Returns:
        Igraph graph

    """

    graph = igraph.Graph.Weighted_Adjacency(weighted_adjacency, mode = mode,attr = attr,loops = multi)

    return graph

def create_igraph(ctx: mgp.ProcCtx,mode: str,multi: bool):
    if mode == 'directed' and multi:
        return igraph.Graph.from_networkx(MemgraphMultiDiGraph(ctx=ctx))
    elif mode == 'undirected' and multi:
        return igraph.Graph.from_networkx(MemgraphMultiGraph(ctx=ctx))
    elif mode == 'directed' and not multi:
        return igraph.Graph.from_networkx(MemgraphDiGraph(ctx=ctx))
    elif mode == 'undirected' and not multi:
        return igraph.Graph.from_networkx(MemgraphGraph(ctx=ctx))

def community_leiden(ctx: mgp.ProcCtx, resolution_parameter:float,n_iterations:int, mode:str,multi:bool, edge_property:str = "weight", weighted_adjacency: List[List[float]] = None):
    if weighted_adjacency != None:
        graph  = create_igraph_from_matrix(weighted_adjacency,mode,edge_property,multi)
    else:
        graph = create_igraph(ctx,mode,multi)
    
    return graph.community_leiden(weights=graph.es[edge_property], resolution_parameter=resolution_parameter, n_iterations = n_iterations)

def max_flow(ctx: mgp.ProcCtx, mode:str, multi:bool, start_index: int, end_index: int, capacity: str):
    graph = create_igraph(ctx, mode, multi)

    return graph.maxflow(start_index,end_index,capacity)