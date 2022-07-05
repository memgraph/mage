import igraph
import mgp
from collections import defaultdict
from typing import Dict, List, Tuple


class MemgraphIgraph(igraph.Graph):
    def __init__(self, ctx: mgp.ProcCtx, directed: bool):
        self.ctx_graph = ctx.graph
        self.id_mappings, self.reverse_id_mappings = self._create_igraph_from_ctx(
            ctx=ctx, directed=directed
        )

    def maxflow(
        self, source: mgp.Vertex, destination: mgp.Vertex, capacity: str
    ) -> igraph.Flow:
        return super().maxflow(
            self.id_mappings[source.id],
            self.id_mappings[destination.id],
            capacity=capacity,
        )

    def pagerank(
        self, weights: str, directed: bool, niter: int, damping: float, eps: float
    ) -> List[float]:
        return super().pagerank(
            weights=weights, directed=directed, niter=niter, damping=damping, eps=eps
        )

    def get_all_simple_paths(
        self, v: mgp.Vertex, to: mgp.Vertex, cutoff: int
    ) -> List[List[mgp.Vertex]]:
        paths = [
            self.convert_vertex_ids_to_mgp_vertices(path)
            for path in super().get_all_simple_paths(
                v=self.id_mappings[v.id], to=self.id_mappings[to.id], cutoff=cutoff
            )
        ]
        return paths

    def topological_sort(self, mode: str) -> List[mgp.Vertex]:
        sorted_vertex_ids = super().topological_sorting(mode=mode)

        return self.convert_vertex_ids_to_mgp_vertices(sorted_vertex_ids)

    def community_leiden(
        self,
        resolution_parameter,
        weights,
        n_iterations,
        beta=0.01,
        objective_function="CPM",
    ) -> List[List[mgp.Vertex]]:
        communities = super().community_leiden(
            resolution_parameter=resolution_parameter,
            weights=weights,
            n_iterations=n_iterations,
            objective_function=objective_function,
            beta=beta,
        )
        return [
            self.convert_vertex_ids_to_mgp_vertices(members) for members in communities
        ]

    def mincut(
        self, source: mgp.Vertex, target: mgp.Vertex, capacity: str
    ) -> igraph.Cut:

        return super().mincut(
            source=self.id_mappings[source.id],
            target=self.id_mappings[target.id],
            capacity=capacity,
        )

    def spanning_tree(
        self,
        weights: str,
    ) -> List[List[mgp.Vertex]]:
        if weights:
            weights = self.es[weights]

        min_spanning_tree_edges = super().spanning_tree(
            weights=weights, return_tree=False
        )

        return self._get_min_span_tree_vertex_pairs(
            min_spanning_tree_edges=self.es[min_spanning_tree_edges]
        )

    def shortest_path_length(
        self, source: mgp.Vertex, target: mgp.Vertex, weights: str
    ) -> float:
        return super().shortest_paths(
            source=self.id_mappings[source.id],
            target=self.id_mappings[target.id],
            weights=weights,
        )[0][0]

    def all_shortest_path_lengths(self, weights: str) -> List[List[float]]:
        return super().shortest_paths(
            weights=weights,
        )

    def get_shortest_path(
        self, source: mgp.Vertex, target: mgp.Vertex, weights: str
    ) -> List[mgp.Vertex]:
        path = super().get_shortest_paths(
            v=self.id_mappings[source.id],
            to=self.id_mappings[target.id],
            weights=weights,
        )[0]

        return self.convert_vertex_ids_to_mgp_vertices(path)

    def get_vertex_by_id(self, id: int) -> mgp.Vertex:
        return self.ctx_graph.get_vertex_by_id(self.reverse_id_mappings[id])

    def convert_vertex_ids_to_mgp_vertices(
        self, vertex_ids: List[int]
    ) -> List[mgp.Vertex]:

        vertices = []
        for id in vertex_ids:
            vertices.append(
                self.ctx_graph.get_vertex_by_id(self.reverse_id_mappings[id])
            )

        return vertices

    def _get_min_span_tree_vertex_pairs(
        self,
        min_spanning_tree_edges: igraph.EdgeSeq,
    ) -> List[List[mgp.Vertex]]:
        """Function for getting vertex pairs that are connected in minimum spanning tree.

        Args:
            min_span_tree_graph (igraph.EdgeSeq): Igraph graph containing minimum spanning tree

        Returns:
            List[List[mgp.Vertex]]: List of vertex pairs that are connected in minimum spanning tree
        """

        min_span_tree = []
        for edge in min_spanning_tree_edges:
            min_span_tree.append(
                [
                    self.ctx_graph.get_vertex_by_id(
                        self.reverse_id_mappings[edge.source]
                    ),
                    self.ctx_graph.get_vertex_by_id(
                        self.reverse_id_mappings[edge.target]
                    ),
                ]
            )

        return min_span_tree

    def _create_igraph_from_ctx(
        self, ctx: mgp.ProcCtx, directed: bool = False
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
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
        inverted_id_mapping = {
            i: vertex.id for i, vertex in enumerate(ctx.graph.vertices)
        }
        for vertex in ctx.graph.vertices:
            for name, value in vertex.properties.items():
                vertex_attrs[name].append(value)
            for edge in vertex.out_edges:
                for name, value in edge.properties.items():
                    edge_attrs[name].append(value)
                edge_list.append(
                    (id_mapping[edge.from_vertex.id], id_mapping[edge.to_vertex.id])
                )

        super().__init__(
            directed=directed,
            n=len(ctx.graph.vertices),
            edges=edge_list,
            edge_attrs=edge_attrs,
            vertex_attrs=vertex_attrs,
        )

        return id_mapping, inverted_id_mapping
