from typing import List, Optional
from graph_coloring_module.components.individual import Individual
from graph_coloring_module.graph import Graph
from graph_coloring_module.components.population import Population


class ChainPopulation(Population):
    """A class that represents a population that contains
    all possible individuals. In this population, the last
    individual is followed by the first individual, and the
    predecessor of the first individual is the last.."""

    def __init__(
            self,
            graph: Graph,
            individuals: List[Individual]):

        super().__init__(graph, individuals)
        self._set_correlations()

    def _get_prev_correlation_ind(self, ind: int) -> int:
        """Returns the index of the correlation with the previous individual in the chain of individuals."""
        return ind - 1 if ind - 1 >= 0 else self.size - 1

    def _get_next_correlation_ind(self, ind: int) -> int:
        """Returns the index of the correlation with the next individual in the chain of individuals."""
        return ind

    def get_prev_individual(self, ind: int) -> Optional[Individual]:
        """Returns the individual that precedes the individual on index ind."""
        if ind < 0 or ind >= self.size:
            return None
        prev_ind = ind - 1 if ind - 1 >= 0 else self.size - 1
        return self.individuals[prev_ind]

    def get_next_individual(self, ind: int) -> Optional[Individual]:
        """Returns the individual that follows the individual on index ind."""
        if ind < 0 or ind >= self.size:
            return None
        next_ind = ind + 1 if ind + 1 < self.size else 0
        return self.individuals[next_ind]

    def _set_correlations(self) -> None:
        for i in range(self.size):
            j = i + 1 if i + 1 < self.size else 0
            c = self._calculate_correlation(self.individuals[i], self.individuals[j])
            self._correlation.append(c)
            self._cumulative_correlation += c
