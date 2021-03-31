from abc import ABC, abstractmethod
from typing import Dict, Any
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.population import Population


class IterationCallback(ABC):
    @abstractmethod
    def update(self, graph: Graph, population: Population, parameters: Dict[str, Any]):
        pass

    @abstractmethod
    def end(self, graph: Graph, population: Population, parameters: Dict[str, Any]):
        pass
