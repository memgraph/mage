from typing import List, Optional
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph


class ChainChunk(Population):
    """A class that represents a population that is just a part
    of the whole population. First and last individuals of this
    population exchange information with individuals located in
    other parts of the entire population. Pieces of the population
    are ordered. The first individual communicates with the last in
    the previous piece of the population, and the last communicates
    with the first in the next piece of the population."""

    def __init__(
            self,
            graph: Graph,
            individuals: List[Individual],
            prev_indv: Individual,
            next_indv: Individual):

        super().__init__(graph, individuals)
        self._prev_indv = prev_indv
        self._next_indv = next_indv

    def get_prev_individual(self, ind: int) -> Optional[Individual]:
        """Returns the individual that precedes the individual on index ind."""
        if ind < 0 or ind >= self.size:
            return None
        if ind == 0:
            return self._prev_indv
        return self.individuals[ind - 1]

    def get_next_individual(self, ind: int) -> Optional[Individual]:
        """Returns the individual that follows the individual on index ind."""
        if ind < 0 or ind >= self.size:
            return None
        if ind + 1 == self.size:
            return self._next_indv
        return self.individuals[ind + 1]

    def set_prev_individual(self, indv: Individual) -> None:
        """Sets the unit that precedes the current piece of chain."""
        #self._cumulative_correlation -= self._correlation[self.size]
        #self._correlation[self.size] = self._calculate_correlation(indv, self.individuals[0])
        #self._cumulative_correlation += self._correlation[self.size]
        self._prev_indv = indv

    def set_next_individual(self, indv: Individual) -> None:
        """Sets the individual that follows the current piece of chain."""
        #self._cumulative_correlation -= self._correlation[self.size - 1]
        #self._correlation[self.size - 1] = self._calculate_correlation(self.individuals[self.size - 1], indv)
        #self._cumulative_correlation += self._correlation[self.size - 1]
        self._next_indv = indv
