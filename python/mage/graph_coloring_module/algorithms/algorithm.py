from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.individual import Individual


class Algorithm(ABC):
    """An abstract class that represents algorithm."""

    @abstractmethod
    def run(
            self,
            graph: Graph,
            parameters: Dict[str, Any]) -> Optional[Individual]:
        """Runs the algorithm and returns the best individual."""
        pass
