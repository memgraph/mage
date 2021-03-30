from abc import ABC, abstractmethod
from typing import List, Callable
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


class Population(ABC):
    """An abstract class that represents a population. A population
    contains individuals that are placed in a chain and exchange
    information with individuals that are located next to it."""

    def __init__(self, graph: Graph, individuals: List[Individual]):

        self._size = len(individuals)
        self._individuals = individuals
        self._best_individuals = self._individuals[:]
        self._graph = graph

        self._contains_solution = False
        self._solution = None
        self._sum_conflicts_weight = 0

        self._calculate_metrics()

    def __len__(self) -> int:
        """Returns size of the population."""
        return self._size

    def __getitem__(self, ind: int) -> Individual:
        """Returns an individual that is placed on a given index."""
        return self._individuals[ind]

    @abstractmethod
    def get_prev_individual(self, ind: int) -> Individual:
        """Returns the individual that precedes the individual on a given index in the individuals chain."""
        pass

    @abstractmethod
    def get_next_individual(self, ind: int) -> Individual:
        """Returns the individual following the individual on a given index in the individuals chain."""
        pass

    @property
    def individuals(self) -> List[Individual]:
        """Returns a list of individuals."""
        return self._individuals

    @property
    def best_individuals(self) -> List[Individual]:
        """Returns a list of individuals that had the smallest error through iterations."""
        return self._best_individuals

    @property
    def size(self) -> int:
        """Returns size of the population."""
        return self._size

    @property
    def mean_conflicts_weight(self) -> float:
        """Returns the average sum of weights of conflicting edges
        in individuals contained in population."""
        return self._sum_conflicts_weight / self.size

    @property
    def sum_conflicts_weight(self) -> float:
        """Returns the sum of sum of weights of conflicting edges
        in individuals contained in population"""
        return self._sum_conflicts_weight

    @property
    def contains_solution(self) -> bool:
        """Returns True if population contains an individual with no conflicting edges,
        otherwise returns False."""
        return self._contains_solution

    def set_individual(self, ind: int, indv: Individual, diff_nodes: List[int]) -> int:
        """Sets the individual on the index ind to the given individual indv.
        Returns None if the wrong index is given."""
        old_indv = self._individuals[ind]
        self._individuals[ind] = indv
        self._update_metrics(ind, old_indv)

    def solution(self) -> Individual:
        return self._solution

    def solution_error(self) -> float:
        if self._solution is not None:
            return self._solution._conflicts_weight
        else:
            return min([indv.conflicts_weight for indv in self.individuals])

    def best_individual_index(
        self, error_func: Callable[[Graph, Individual], float]
    ) -> int:
        """Returns the index of the individual with the least error."""
        errors = self.individuals_errors(error_func)
        return min(range(len(errors)), key=errors.__getitem__)

    def worst_individual_index(
        self, error_func: Callable[[Graph, Individual], float]
    ) -> int:
        """Returns the index of the individual with the largest error."""
        errors = self.individuals_errors(error_func)
        return max(range(len(errors)), key=errors.__getitem__)

    def best_individual(
        self, error_func: Callable[[Graph, Individual], float]
    ) -> Individual:
        """Returns the individual with the least error."""
        return self._individuals[self.best_individual_index(error_func)]

    def worst_individual(
        self, error_func: Callable[[Graph, Individual], float]
    ) -> Individual:
        """Returns the individual with the largest error."""
        return self._individuals[self.worst_individual_index(error_func)]

    def individuals_errors(
        self, error_func: Callable[[Graph, Individual], float]
    ) -> List[float]:
        """Returns a list of individuals errors."""
        return [error_func(self._graph, indv) for indv in self.individuals]

    def min_error(self, error_func: Callable[[Graph, Individual], float]) -> float:
        """Returns the smallest error in the population."""
        return min(self.individuals_errors(error_func))

    def max_error(self, error_func: Callable[[Graph, Individual], float]) -> float:
        """Returns the largest error in the population."""
        return max(self.individuals_errors(error_func))

    def _calculate_metrics(self) -> None:
        for individual in self.individuals:
            self._sum_conflicts_weight += individual.conflicts_weight
            if individual.conflicts_weight == 0:
                self._contains_solution = True
                self._solution = individual

    def _update_metrics(self, ind: int, old_indv: Individual) -> None:
        if self._contains_solution and self._solution == old_indv:
            self._contains_solution = False
            self._solution = None
            for individual in self.individuals:
                if individual.conflicts_weight == 0:
                    self._contains_solution = True
                    self._solution = individual

        new_indv = self.individuals[ind]
        self._sum_conflicts_weight -= old_indv.conflicts_weight
        self._sum_conflicts_weight += new_indv.conflicts_weight

        best_conflicts_weight = self._best_individuals[ind].conflicts_weight
        new_conflicts_weight = new_indv.conflicts_weight
        if new_conflicts_weight < best_conflicts_weight:
            self._best_individuals[ind] = new_indv
