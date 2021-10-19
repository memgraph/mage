from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class Graph(ABC):
    def __init__(self, is_directed: bool):
        self._nodes: List[int] = []
        self._is_directed = False
        self._preprocessed_transition_probs = {}
        self._first_travel_transition_probs = {}
        self._is_directed = is_directed
        self._graph = {}

    @property
    def graph(self):
        return self._graph

    @property
    def is_directed(self):
        return self._is_directed

    @property
    def preprocessed_transition_probs(self) -> Dict[Tuple[int, int], List[float]]:
        return self._preprocessed_transition_probs

    @property
    def first_travel_transition_probs(self) -> Dict[int, List[float]]:
        return self._first_travel_transition_probs

    @property
    def nodes(self) -> List[int]:
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @nodes.deleter
    def nodes(self):
        print("deleter of nodes called")
        del self._nodes

    @preprocessed_transition_probs.setter
    def preprocessed_transition_probs(self, value):
        self._preprocessed_transition_probs = value

    @preprocessed_transition_probs.deleter
    def preprocessed_transition_probs(self):
        print("deleter of preprocessed_transition_probs called")
        del self._preprocessed_transition_probs

    @first_travel_transition_probs.setter
    def first_travel_transition_probs(self, value):
        self._first_travel_transition_probs = value

    @first_travel_transition_probs.deleter
    def first_travel_transition_probs(self):
        print("deleter of _first_travel_transition_probs called")
        del self._first_travel_transition_probs

    @abstractmethod
    def has_edge(self, src_node_id: int, dest_node_id: int) -> bool:
        pass

    @abstractmethod
    def get_edge_weight(self, src_node_id: int, dest_node_id: int) -> float:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[int]:
        pass

    @abstractmethod
    def get_edges(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def set_edge_transition_probs(
        self, edge: Tuple[int, int], transition_probs: List[float]
    ) -> None:
        pass

    @abstractmethod
    def get_edge_transition_probs(self, edge: Tuple[int, int]) -> List[float]:
        pass

    @abstractmethod
    def set_node_first_travel_transition_probs(
        self, source_node_id: int, normalized_probs: List[float]
    ) -> None:
        pass

    @abstractmethod
    def get_node_first_travel_transition_probs(
        self, source_node_id: int
    ) -> List[float]:
        pass


class BasicGraph(Graph):
    def __init__(self, edges_weights: Dict[Tuple[int, int], float], is_directed: bool):
        super().__init__(is_directed)
        self._edges_weights = edges_weights

        self.init_graph()

    def set_edge_transition_probs(
        self, edge: Tuple[int, int], transition_probs: List[float]
    ) -> None:
        self._preprocessed_transition_probs[edge] = transition_probs

    def get_edge_transition_probs(self, edge: Tuple[int, int]) -> List[float]:
        return self._preprocessed_transition_probs[edge]

    def set_node_first_travel_transition_probs(
        self, source_node_id: int, normalized_probs: List[float]
    ) -> None:
        self._first_travel_transition_probs[source_node_id] = normalized_probs

    def get_node_first_travel_transition_probs(
        self, source_node_id: int
    ) -> List[float]:
        return self._first_travel_transition_probs[source_node_id]

    def has_edge(self, src_node_id: int, dest_node_id: int) -> bool:
        return (src_node_id, dest_node_id) in self._edges_weights or (
            not self.is_directed and (dest_node_id, src_node_id) in self._edges_weights
        )

    def get_edges(self) -> List[Tuple[int, int]]:
        edges = list(self._edges_weights.keys())
        if self._is_directed:
            return edges
        edges_different_dir = []
        for edge in edges:
            edges_different_dir.append((edge[1], edge[0]))
        edges.extend(edges_different_dir)
        return edges

    def get_edge_weight(self, src_node_id: int, dest_node_id: int) -> float:
        if (src_node_id, dest_node_id) not in self._edges_weights and self.is_directed:
            raise ValueError
        if (src_node_id, dest_node_id) in self._edges_weights:
            return self._edges_weights[(src_node_id, dest_node_id)]
        return self._edges_weights[(dest_node_id, src_node_id)]

    def get_neighbors(self, node_id: int) -> List[int]:
        return self._graph[node_id] if node_id in self._graph else []

    def init_graph(self) -> None:
        for node_from, node_to in self._edges_weights:
            if edge[0] not in self._graph:
                self._graph[edge[0]] = set()
            self._graph[edge[0]].add(edge[1])

            if not self.is_directed:
                if edge[1] not in self._graph:
                    self._graph[edge[1]] = set()
                self._graph[edge[1]].add(edge[0])

        self._nodes = self._graph.keys()

        for node in self._graph:
            self._graph[node] = sorted(list(self._graph[node]))
