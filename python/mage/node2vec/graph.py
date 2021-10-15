from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class Graph(ABC):
    def __init__(self):
        self._nodes: List[int] = []
        self._is_directed = False

    @property
    def is_directed(self) -> bool:
        return self._is_directed

    @property
    def nodes(self) -> List[int]:
        return self._nodes

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
        super().__init__()
        self._edges_weights = edges_weights
        self._is_directed = is_directed
        self._graph = {}
        self._preprocessed_transition_probs = {}
        self._first_travel_transition_probs = {}
        self.init_graph()

    @property
    def nodes(self) -> List[int]:
        return list(self._graph.keys())

    @property
    def graph(self):
        return self._graph

    @property
    def is_directed(self) -> bool:
        return self._is_directed

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
        for edge in self._edges_weights:
            if edge[0] not in self._graph:
                self._graph[edge[0]] = list()
            self._graph[edge[0]].append(edge[1])

            if not self.is_directed:
                if edge[1] not in self._graph:
                    self._graph[edge[1]] = list()
                self._graph[edge[1]].append(edge[0])

        self._nodes = self._graph.keys()

        for node in self._graph:
            self._graph[node] = sorted(self._graph[node])
