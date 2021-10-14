from typing import List

import numpy as np

from mage.node2vec.graph import Graph


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

    def sample_walk(self, graph: Graph, start_node_id_int) -> List[int]:
        walk = [start_node_id_int]

        while len(walk) < self.walk_length:
            current_node_id = walk[-1]
            node_neighbors = graph.get_neighbors(current_node_id)
            if len(node_neighbors) == 0:
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

    def set_first_travel_transition_probs(self, graph: Graph):
        for source_node_id in graph.nodes:
            unnormalized_probs = [
                graph.get_edge_weight(source_node_id, neighbor_id)
                for neighbor_id in graph.get_neighbors(source_node_id)
            ]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            graph.set_node_first_travel_transition_probs(
                source_node_id, normalized_probs
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

            if src_node_id == 0 and dest_node_id == 31:
                print(dest_neighbor_id, unnorm_trans_probs[-1])

        norm_const = sum(unnorm_trans_probs)
        norm_trans_probs = np.array(unnorm_trans_probs) / norm_const

        return norm_trans_probs

    def set_graph_transition_probs(self, graph: Graph):

        for edge in graph.get_edges():
            graph.set_edge_transition_probs(
                (edge[0], edge[1]),
                self.calculate_edge_transition_probs(graph, edge[0], edge[1]),
            )
            if not graph.is_directed:
                graph.set_edge_transition_probs(
                    (edge[1], edge[0]),
                    self.calculate_edge_transition_probs(graph, edge[1], edge[0]),
                )

        for edge in graph.get_edges():
            print(edge, graph.get_edge_transition_probs(edge))
        print()

        return
