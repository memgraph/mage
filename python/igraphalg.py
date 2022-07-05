from typing import List

import mgp

from mgp_igraph import MemgraphIgraph


@mgp.read_proc
def maxflow(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    destination: mgp.Vertex,
    capacity: str = "weight",
) -> mgp.Record(max_flow=mgp.Number):

    graph = MemgraphIgraph(ctx=ctx, directed=True)
    max_flow = graph.maxflow(source=source, destination=destination, capacity=capacity)

    return mgp.Record(max_flow=max_flow.value)


@mgp.read_proc
def pagerank(
    ctx: mgp.ProcCtx,
    damping: mgp.Number = 0.85,
    niter: int = 100,
    eps: mgp.Number = 1e-06,
    weights: mgp.Nullable[str] = None,
) -> mgp.Record(node=mgp.Vertex, rank=float):
    graph = MemgraphIgraph(ctx=ctx, directed=True)
    ranks = graph.pagerank(
        weights=weights, directed=True, niter=niter, damping=damping, eps=eps
    )

    return [
        mgp.Record(node=graph.get_vertex_by_id(node_id), rank=rank)
        for node_id, rank in enumerate(ranks)
    ]


@mgp.read_proc
def all_simple_paths(
    ctx: mgp.ProcCtx,
    v: mgp.Vertex,
    to: mgp.Vertex,
    cutoff: int = -1,
) -> mgp.Record(paths=mgp.List[mgp.List[mgp.Vertex]]):
    graph = MemgraphIgraph(ctx=ctx, directed=True)

    return mgp.Record(paths=graph.get_all_simple_paths(v=v, to=to, cutoff=cutoff))


@mgp.read_proc
def min_cut(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    capacity: mgp.Nullable[str] = None,
) -> mgp.Record(partition_vertices=List[List[mgp.Vertex]], value=float):
    graph = MemgraphIgraph(ctx=ctx, directed=True)

    cut = graph.mincut(source=source, target=target, capacity=capacity)

    return mgp.Record(
        partition_vertices=[
            graph.convert_vertex_ids_to_mgp_vertices(vertex_ids=partition)
            for partition in cut.partition
        ],
        value=cut.value,
    )


@mgp.read_proc
def topological_sort(
    ctx: mgp.ProcCtx, directed: bool = True, mode: str = "out"
) -> mgp.Record(nodes=mgp.Nullable[mgp.List[mgp.Vertex]]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)
    sorted_nodes = graph.topological_sort(mode=mode)

    return mgp.Record(
        nodes=sorted_nodes,
    )


@mgp.read_proc
def community_leiden(
    ctx: mgp.ProcCtx,
    resolution_parameter: float = 0.6,
    n_iterations: int = -1,
    beta: float = 0.01,
    weights: mgp.Nullable[str] = None,
    objective_function="modularity",
    directed: bool = False,
) -> mgp.Record(community_id=int, community_members=List[mgp.Vertex]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    communities = graph.community_leiden(
        resolution_parameter=resolution_parameter,
        weights=weights,
        n_iterations=n_iterations,
        objective_function=objective_function,
        beta=beta,
    )

    return [
        mgp.Record(
            community_id=i,
            community_members=members,
        )
        for i, members in enumerate(communities)
    ]


@mgp.read_proc
def spanning_tree(
    ctx: mgp.ProcCtx, weights: mgp.Nullable[str] = None, directed: bool = True
) -> mgp.Record(tree=List[List[mgp.Vertex]]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    return mgp.Record(tree=graph.spanning_tree(weights=weights))


@mgp.read_proc
def shortest_path_length(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    weights: mgp.Nullable[str] = None,
    directed: bool = True,
) -> mgp.Record(length=float):
    graph = MemgraphIgraph(ctx, directed=directed)

    return mgp.Record(
        length=graph.shortest_path_length(
            source=source,
            target=target,
            weights=weights,
        )
    )


@mgp.read_proc
def all_shortest_path_lengths(
    ctx: mgp.ProcCtx,
    weights: mgp.Nullable[str] = None,
    directed: bool = False,
) -> mgp.Record(nodes=List[mgp.Vertex], lengths=List[List[float]]):
    graph = MemgraphIgraph(ctx, directed=directed)

    return mgp.Record(
        nodes=[vertex for vertex in ctx.graph.vertices],
        lengths=graph.all_shortest_path_lengths(weights=weights),
    )


@mgp.read_proc
def get_shortest_path(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    weights: mgp.Nullable[str] = None,
    directed: bool = True,
) -> mgp.Record(path=List[mgp.Vertex]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    return mgp.Record(
        path=graph.get_shortest_path(source=source, target=target, weights=weights)
    )
