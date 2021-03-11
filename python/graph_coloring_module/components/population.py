from abc import ABC, abstractmethod
from typing import List, Callable, Tuple
from graph_coloring_module.components.individual import Individual
from graph_coloring_module.graph import Graph


class Population(ABC):
    """An abstract class that represents a population. A population
    contains individuals that are placed in a chain and exchange
    information with individuals that are located next to it."""

    def __init__(
            self,
            graph: Graph,
            individuals: List[Individual]):

        self._size = len(individuals)
        self._individuals = individuals
        self._best_individuals = self._individuals[:]
        self._graph = graph

        self._contains_solution = False
        self._sum_conflicts_weight = 0
        self._cumulative_correlation = 0
        self._correlation = []

        self._calculate_metrics()

    def __len__(self) -> int:
        """Returns size of the population."""
        return self._size

    def __getitem__(self, ind: int) -> Individual:
        """Returns an individual that is placed on a given index."""
        return self._individuals[ind]

    @abstractmethod
    def _set_correlations(self) -> None:
        """Calculates the correlations between individuals and saves them in a list of correlations."""
        pass

    @abstractmethod
    def _get_prev_correlation_ind(self, ind: int) -> int:
        """Returns the index of the correlation with the previous individual in the chain of individuals."""
        pass

    @abstractmethod
    def _get_next_correlation_ind(Self, ind: int) -> int:
        """Returns the index of the correlation with the next individual in the chain of individuals."""
        pass

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
    def correlation(self) -> int:
        return self._correlation

    @property
    def cumulative_correlation(self) -> int:
        """Returns the cumulative correlation of the population.
        If the population represents a piece of chain then cumulative correlation
        does not include correlation with the individual from the previous piece of chain."""
        return self._cumulative_correlation

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
    def correlations(self, ind: int) -> Tuple[int, int]:
        """Returns correlations between a given individual and the previous or next individual."""
        prev_ind = self._get_prev_correlation_ind(ind)
        next_ind = self._get_next_correlation_ind(ind)
        return self._correlation[prev_ind], self._correlation[next_ind]

    @property
    def contains_solution(self) -> bool:
        """Returns True if population contains an individual with no conflicting edges,
        otherwise returns False."""
        return self._contains_solution

    def set_individual(self, ind: int, indv: Individual, diff_nodes: List[int]) -> int:
        """Sets the individual on the index ind to the given individual indv.
        Returns None if the wrong index is given. Otherwise, returns the difference
        of cumulative correlation that occurred after the replacement of the individual."""
        if not 0 <= ind < self.size:
            return None
        old_indv = self._individuals[ind]
        self._individuals[ind] = indv
        correlation_diff = self._update_correlation(ind, old_indv, diff_nodes)
        self._update_metrics(ind, old_indv)
        return correlation_diff

    def solution(self) -> Individual:
        errors = [indv.conflicts_weight for indv in self._best_individuals]
        sol_ind = min(range(len(errors)), key=errors.__getitem__)
        return self.best_individuals[sol_ind]

    def solution_error(self) -> float:
        errors = [indv.conflicts_weight for indv in self._best_individuals]
        return min(errors)

    def best_individual_index(self, error_func: Callable[[Graph, Individual], float]) -> int:
        """Returns the index of the individual with the least error."""
        errors = self.individuals_errors(error_func)
        return min(range(len(errors)), key=errors.__getitem__)

    def worst_individual_index(self, error_func: Callable[[Graph, Individual], float]) -> int:
        """Returns the index of the individual with the largest error."""
        errors = self.individuals_errors(error_func)
        return max(range(len(errors)), key=errors.__getitem__)

    def best_individual(self, error_func: Callable[[Graph, Individual], float]) -> Individual:
        """Returns the individual with the least error."""
        return self._individuals[self.best_individual_index(error_func)]

    def worst_individual(self, error_func: Callable[[Graph, Individual], float]) -> Individual:
        """Returns the individual with the largest error."""
        return self._individuals[self.worst_individual_index(error_func)]

    def individuals_errors(self, error_func: Callable[[Graph, Individual], float]) -> List[float]:
        """Returns a list of individuals errors."""
        return [error_func(self._graph, indv) for indv in self.individuals]

    def min_error(self, error_func: Callable[[Graph, Individual], float]) -> float:
        """Returns the smallest error in the population."""
        return min(self.individuals_errors(error_func))

    def max_error(self, error_func: Callable[[Graph, Individual], float]) -> float:
        """Returns the largest error in the population."""
        return max(self.individuals_errors(error_func))

    def _calculate_correlation(self, first: Individual, second: Individual) -> float:
        correlation = 0
        for node_1 in self._graph.nodes:
            for node_2 in range(node_1 + 1, len(self._graph)):
                S_first = -1 if first[node_1] == first[node_2] else 1
                S_second = -1 if second[node_1] == second[node_2] else 1
                correlation += S_first * S_second
        return correlation

    def _calculate_metrics(self) -> None:
        for ind in range(self.size):
            err = self.individuals[ind].conflicts_weight
            self._sum_conflicts_weight += err
            if err == 0:
                self._contains_solution = True

    def _update_metrics(self, ind: int, old_indv: Individual) -> None:
        indv = self.individuals[ind]
        self._sum_conflicts_weight -= old_indv.conflicts_weight
        self._sum_conflicts_weight += indv.conflicts_weight
        if indv.conflicts_weight <= 0:
            self._contains_solution = True
        best_conflicts_weight = self._best_individuals[ind].conflicts_weight
        new_conflicts_weight = indv.conflicts_weight
        if new_conflicts_weight < best_conflicts_weight:
            self._best_individuals[ind] = indv

    def _update_correlation(self, ind: int, old_indv: Individual, nodes: List[int]) -> int:
        next_corr_ind = self._get_next_correlation_ind(ind)
        prev_corr_ind = self._get_prev_correlation_ind(ind)

        new_indv = self.individuals[ind]
        prev_indv = self.get_prev_individual(ind)
        next_indv = self.get_next_individual(ind)

        corr_prev_delta = 0
        corr_next_delta = 0
        processed = [False for _ in range(old_indv.no_of_units)]

        for node in nodes:
            for neigh in self._graph[node]:
                if not processed[neigh]:
                    S_old = -1 if old_indv[node] == old_indv[neigh] else 1
                    S_new = -1 if new_indv[node] == new_indv[neigh] else 1
                    S_prev = -1 if prev_indv[node] == prev_indv[neigh] else 1
                    S_next = -1 if next_indv[node] == next_indv[neigh] else 1
                    corr_prev_delta += (S_new * S_prev) - (S_old * S_prev)
                    corr_next_delta += (S_new * S_next) - (S_old * S_next)
            processed[node] = True

        self._correlation[prev_corr_ind] += corr_prev_delta
        self._correlation[next_corr_ind] += corr_next_delta
        delta_corr = corr_prev_delta + corr_next_delta
        self._cumulative_correlation += delta_corr
        return delta_corr
