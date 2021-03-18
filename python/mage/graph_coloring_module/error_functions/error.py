from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population


class Error(ABC):
    """A class that represents error function."""

    @abstractmethod
    def individual_err(
            self,
            graph: Graph,
            indv: Individual,
            parameters: Dict[str, Any] = None) -> Optional[float]:
        """Calculates the error of the individual.
        If an error occurs None is returned"""
        pass

    @abstractmethod
    def population_err(
            self,
            graph: Graph,
            population: Population,
            parameters: Dict[str, Any] = None) -> Optional[float]:
        """Calculates the population error.
        If an error occurs None is returned"""
        pass
