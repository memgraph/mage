from abc import ABC, abstractmethod
from typing import List, Tuple
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph

class CorrelationPopulation(Population):

    def __init__(self, graph: Graph, individuals: List[Individual]):
        super().__init__(graph, individuals)
        self._cumulative_correlation = 0
        self._correlation = []

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

    def set_individual(self, ind: int, indv: Individual, diff_nodes: List[int]) -> int:
        """Sets the individual on the index ind to the given individual indv.
        Returns None if the wrong index is given."""
        old_indv = self._individuals[ind]
        self._individuals[ind] = indv
        self._update_correlation(ind, old_indv, diff_nodes)
        self._update_metrics(ind, old_indv)

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
    def correlations(self, ind: int) -> Tuple[int, int]:
        """Returns correlations between a given individual and the previous or next individual."""
        prev_ind = self._get_prev_correlation_ind(ind)
        next_ind = self._get_next_correlation_ind(ind)
        return self._correlation[prev_ind], self._correlation[next_ind]
    
    def _calculate_correlation(self, first: Individual, second: Individual) -> float:
        correlation = 0
        for node_1 in self._graph.nodes:
            for node_2 in range(node_1 + 1, len(self._graph)):
                S_first = -1 if first[node_1] == first[node_2] else 1
                S_second = -1 if second[node_1] == second[node_2] else 1
                correlation += S_first * S_second
        return correlation

    def _update_correlation(
        self, ind: int, old_indv: Individual, nodes: List[int]
    ) -> int:
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
