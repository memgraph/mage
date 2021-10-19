from typing import List

import numpy as np

from mage.node2vec.graph import Graph

from utils import math_functions


class SecondOrderRandomWalk:
    def __init__(self, p: float, q: float, num_walks: int, walk_length: int):
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length

    def sample_node_walks(self, graph: Graph) -> List[List[int]]:
        self.set_first_travel_transition_probs(graph)
        self.set_graph_transition_probs(graph)
        walks = []
        for node in graph.nodes:
            for i in range(self.num_walks):
                walks.append(self.sample_walk(graph, node))

        return walks

    def sample_walk(self, graph: Graph, start_node_id_int: int) -> List[int]:
        walk = [start_node_id_int]
        while len(walk) < self.walk_length:
            current_node_id = walk[-1]
            node_neighbors = graph.get_neighbors(current_node_id)
            if not node_neighbors:
                break

            if len(walk) == 1:
                walk.append(
                    np.random.choice(
                        node_neighbors,
                        p=graph.get_node_first_travel_transition_probs(
                            (current_node_id)
                        ),
                    )
                )
                continue

            previous_node_id = walk[-2]

            next = np.random.choice(
                node_neighbors,
                p=graph.get_edge_transition_probs(
                    edge=(previous_node_id, current_node_id)
                ),
            )

            walk.append(next)
        return walk

    def set_first_travel_transition_probs(self, graph: Graph) -> None:

        for source_node_id in graph.nodes:
            unnormalized_probs = [
                graph.get_edge_weight(source_node_id, neighbor_id)
                for neighbor_id in graph.get_neighbors(source_node_id)
            ]

            graph.set_node_first_travel_transition_probs(
                source_node_id, math_functions.normalize(unnormalized_probs)
            )

    def calculate_edge_transition_probs(
        self, graph: Graph, src_node_id: int, dest_node_id: int
    ) -> List[float]:
        unnorm_trans_probs = []

        for dest_neighbor_id in graph.get_neighbors(dest_node_id):

            edge_weight = graph.get_edge_weight(dest_node_id, dest_neighbor_id)

            if dest_neighbor_id == src_node_id:
                unnorm_trans_probs.append(edge_weight / self.p)
            elif graph.has_edge(dest_neighbor_id, src_node_id):
                unnorm_trans_probs.append(edge_weight)
            else:
                unnorm_trans_probs.append(edge_weight / self.q)

        return math_functions.normalize(unnorm_trans_probs)

    def set_graph_transition_probs(self, graph: Graph) -> None:

        for (node_from, node_to) in graph.get_edges():
            graph.set_edge_transition_probs(
                (node_from, node_to),
                self.calculate_edge_transition_probs(graph, node_from, node_to),
            )
            if graph.is_directed:
                continue

            graph.set_edge_transition_probs(
                (node_to, node_from),
                self.calculate_edge_transition_probs(graph, node_to, node_from),
            )
