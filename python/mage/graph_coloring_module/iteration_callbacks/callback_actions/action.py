from abc import ABC, abstractmethod
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.population import Population
from typing import Dict, Any


class Action(ABC):
    @abstractmethod
    def execute(
        self, graph: Graph, population: Population, parameters: Dict[str, Any] = None
    ) -> None:
        pass
