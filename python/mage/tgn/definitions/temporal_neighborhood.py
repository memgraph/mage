from typing import Tuple, Dict, List

import numpy as np


class TemporalNeighborhood:
    def __init__(self):
        super(TemporalNeighborhood, self).__init__()

        self.neighborhood: Dict[int, List[Tuple[int, int, float]]] = {}

    def update_neighborhood(self, sources, destinations, edge_idx, timestamps):
        # idea is that smallest new timestamp is always greater than last biggest one in dict so we don't need to
        # sort arrays:)
        # if it doesn't exist, create empty, else overwrite
        self.neighborhood = {
            **{node: [] for node in set(sources).union(set(destinations))},
            **self.neighborhood,
        }
        for neighbor in zip(sources, destinations, edge_idx, timestamps):
            self.neighborhood[neighbor[0]].append((neighbor[1:]))
            self.neighborhood[neighbor[1]].append(
                (neighbor[0], neighbor[2], neighbor[3])
            )

    def get_neighborhood(self, node: int, timestamp: int, num_neighbors: int):
        # todo check if we need copy

        if node not in self.neighborhood:
            return (
                np.zeros(num_neighbors, dtype=int),
                np.zeros(num_neighbors, dtype=int),
                np.zeros(num_neighbors),
            )
        neighbors_tuple = self.neighborhood[node].copy()

        unzipped_tuple = list(zip(*neighbors_tuple))
        neighbors = list(unzipped_tuple[0])[-num_neighbors:]
        edge_idxs = list(unzipped_tuple[1])[-num_neighbors:]
        timestamps = list(unzipped_tuple[2])[-num_neighbors:]

        neighbors = np.append(
            arr=neighbors, values=np.zeros(num_neighbors - len(neighbors))
        )
        edge_idxs = np.append(
            arr=edge_idxs, values=np.zeros(num_neighbors - len(edge_idxs))
        )
        timestamps = np.append(
            arr=timestamps, values=np.zeros(num_neighbors - len(timestamps))
        )
        return neighbors, edge_idxs, timestamps

    def find_neighborhood(
        self, nodes: List[int], num_neighbors: int
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        return {node: self.neighborhood[node].copy()[:num_neighbors] for node in nodes}
