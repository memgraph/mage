from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from graph_coloring_module.graph import Graph
from graph_coloring_module.components.individual import Individual
from graph_coloring_module.components.population import Population


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
            pop: Population,
            parameters: Dict[str, Any] = None) -> Optional[float]:
        """Calculates the population error.
        If an error occurs None is returned"""
        pass

    @abstractmethod
    def delta(
            self,
            graph: Graph,
            old_indv: Individual,
            new_indv: Individual,
            correlation_diff: int,
            parameters: Dict[str, Any] = None) -> Optional[float]:
        """Calculates the difference of the population error
        that occurred after the replacement of the individual."""
        pass
