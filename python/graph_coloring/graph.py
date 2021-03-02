from typing import Tuple, List, Any, Dict, Iterator
from itertools import islice


class Graph:
    """A structure representing an undirected weighted graph.

    :param nodes: a list containing the labels of all nodes in the graph
    :param adj: a dictionary that associates each node with a list of its neighbors
    :name: the name of the graph

    """

    def __init__(self, nodes: List[Any], adj: Dict[Any, List[Tuple[Any, float]]], name: str = ""):
        self._ind_to_label = nodes
        self._label_to_ind = dict((label, ind) for ind, label in enumerate(nodes))
        self._nodes_count = len(nodes)
        self._neigh_position = []
        self._neighbors = []
        self._weights = []
        self._name = name

        for i in range(self._nodes_count):
            self._neighbors.extend([self._label_to_ind[x[0]] for x in adj[self._ind_to_label[i]]])
            self._weights.extend([x[1] for x in adj[self._ind_to_label[i]]])
            self._neigh_position.append(len(self._neighbors))

    def __str__(self):
        """Returns the name of the graph.
        If the name is not given at initialization, returns an empty string."""
        return self._name

    def __len__(self):
        """Returns the number of nodes in the graph."""
        return self._nodes_count

    def __getitem__(self, n: int) -> Iterator[int]:
        """Returns an iterator of neighbors of node n."""
        start = self._neigh_position[n - 1] if n != 0 else 0
        end = self._neigh_position[n]
        return islice(self._neighbors, start, end)

    @property
    def nodes(self) -> Iterator[int]:
        """Returns an iterator of nodes in the graph."""
        nodes = (i for i in range(self._nodes_count))
        return nodes

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._nodes_count

    def number_of_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return len(self._neighbors) // 2

    def neighbors(self, n: int) -> Iterator[int]:
        """Returns an iterator of neighbors of node n."""
        return self.__getitem__(n)

    def weighted_neighbors(self, n: int) -> Iterator[Tuple[int, float]]:
        """Returns an iterator of neighbor and weight tuples of node n."""
        start = self._neigh_position[n - 1] if n != 0 else 0
        end = self._neigh_position[n]
        return self._neigh_weight_tuples(start, end)

    def weight(self, node_1: int, node_2: int) -> float:
        """Returns the weight between two nodes."""
        weighted_neighs = self.weighted_neighbors(node_1)
        for node, weight in weighted_neighs:
            if node == node_2:
                return weight
        return 0

    def degree(self, n) -> int:
        """Returns the degree of the given node."""
        start = self._neigh_position[n - 1] if n != 0 else 0
        end = self._neigh_position[n]
        return start - end

    def label(self, node: int) -> Any:
        """Returns the node label."""
        return self._ind_to_label[node]

    def _neigh_weight_tuples(self, start: int, end: int) -> Iterator[Tuple[int, Any]]:
        return zip((self._neighbors[i] for i in range(start, end)), (self._weights[i] for i in range(start, end)))
