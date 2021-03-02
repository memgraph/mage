from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from telco.graph import Graph
from telco.components.individual import Individual


class Mutation(ABC):
    """A class that represents mutation."""

    @abstractmethod
    def mutate(
            self,
            graph: Graph,
            indv: Individual,
            parameters: Dict[str, Any] = None) -> Tuple[Individual, List[int]]:
        """Mutate the given individual and returns the new individual and nodes that was changed."""
        pass
