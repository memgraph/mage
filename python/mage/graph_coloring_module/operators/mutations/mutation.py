from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.individual import Individual


class Mutation(ABC):
    """A class that represents a mutation."""

    @abstractmethod
    def mutate(
        self, graph: Graph, individual: Individual, parameters: Dict[str, Any] = None
    ) -> Tuple[Individual, List[int]]:
        """A function that mutates the given individual and
        returns the new individual and nodes that was changed."""
        pass
