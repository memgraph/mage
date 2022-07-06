from typing import List

import mgp

from mgp_igraph import (
    InvalidTopologicalSortingModeException,
    InvalidPageRankImplementationOption,
    MemgraphIgraph,
    PageRankImplementationOptions,
    TopologicalSortingModes,
)


@mgp.read_proc
def maxflow(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    capacity: str = "weight",
) -> mgp.Record(max_flow=mgp.Number):

    graph = MemgraphIgraph(ctx=ctx, directed=True)
    max_flow = graph.maxflow(source=source, target=target, capacity=capacity)

    return mgp.Record(max_flow=max_flow.value)


@mgp.read_proc
def pagerank(
    ctx: mgp.ProcCtx,
    damping: mgp.Number = 0.85,
    niter: int = 100,
    eps: mgp.Number = 1e-06,
    weights: mgp.Nullable[str] = None,
    directed: bool = True,
    implementation: str = "prpack",
) -> mgp.Record(node=mgp.Vertex, rank=float):
    if implementation not in [
        PageRankImplementationOptions.PRPACK.value,
        PageRankImplementationOptions.ARPACK.value,
        PageRankImplementationOptions.POWER.value,
    ]:
        raise InvalidPageRankImplementationOption(
            'Implementation can be "prpack", "arpack" or "power".'
        )
    graph = MemgraphIgraph(ctx=ctx, directed=directed)
    ranks = graph.pagerank(
        weights=weights,
        directed=directed,
        niter=niter,
        damping=damping,
        eps=eps,
        implementation=implementation,
    )

    return [
        mgp.Record(node=graph.get_vertex_by_id(node_id), rank=rank)
        for node_id, rank in enumerate(ranks)
    ]


@mgp.read_proc
def get_all_simple_paths(
    ctx: mgp.ProcCtx,
    v: mgp.Vertex,
    to: mgp.Vertex,
    cutoff: int = -1,
) -> mgp.Record(paths=mgp.List[mgp.List[mgp.Vertex]]):
    graph = MemgraphIgraph(ctx=ctx, directed=True)

    return mgp.Record(paths=graph.get_all_simple_paths(v=v, to=to, cutoff=cutoff))


@mgp.read_proc
def mincut(
    ctx: mgp.ProcCtx,
    source: mgp.Vertex,
    target: mgp.Vertex,
    capacity: mgp.Nullable[str] = None,
) -> mgp.Record(partition_vertices=List[List[mgp.Vertex]], value=float):
    graph = MemgraphIgraph(ctx=ctx, directed=True)

    partition_vertices, value = graph.mincut(
        source=source, target=target, capacity=capacity
    )

    return mgp.Record(partition_vertices=partition_vertices, value=value)


@mgp.read_proc
def topological_sorting(
    ctx: mgp.ProcCtx, mode: str = "out"
) -> mgp.Record(nodes=mgp.List[mgp.Vertex]):

    if mode not in [
        TopologicalSortingModes.IN.value,
        TopologicalSortingModes.OUT.value,
    ]:
        raise InvalidTopologicalSortingModeException(
            'Mode can only be either "out" or "in"'
        )

    graph = MemgraphIgraph(ctx=ctx, directed=True)
    sorted_nodes = graph.topological_sort(mode=mode)

    return mgp.Record(
        nodes=sorted_nodes,
    )


@mgp.read_proc
def community_leiden(
    ctx: mgp.ProcCtx,
    objective_function: str = "CPM",
    weights: mgp.Nullable[str] = None,
    resolution_parameter: float = 0.6,
    beta: float = 0.01,
    initial_membership: mgp.Nullable[
        List[mgp.Nullable[List[mgp.Nullable[float]]]]
    ] = None,
    n_iterations: int = 2,
    node_weights: mgp.Nullable[List[mgp.Nullable[float]]] = None,
    directed: bool = False,
) -> mgp.Record(community_id=int, community_members=List[mgp.Vertex]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    communities = graph.community_leiden(
        resolution_parameter=resolution_parameter,
        weights=weights,
        n_iterations=n_iterations,
        objective_function=objective_function,
        beta=beta,
        initial_membership=initial_membership,
        node_weights=node_weights,
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
    ctx: mgp.ProcCtx,
    weights: mgp.Nullable[str] = None,
) -> mgp.Record(tree=List[List[mgp.Vertex]]):
    graph = MemgraphIgraph(ctx=ctx, directed=False)

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
