import logging
from typing import Dict, Any, Optional
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.error_functions.error import Error
from mage.graph_coloring_module.framework.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate


logger = logging.getLogger('telco')

class CorrelationPopulationDescriptor():
    def __init__(self, population: Population):
        self._population = population
        self._correlation = []
        self._cumulative_correlation = 0
        self._contains_solution = False
        self._population_size = len(population.individuals)
        self._set_correlations()
    
    def _calculate_correlation_between_individuals(self, first: Individual, second: Individual) -> float:
        correlation = 0
        for node_1 in self._graph.nodes:
            for node_2 in range(node_1 + 1, len(self._graph)):
                S_first = -1 if first[node_1] == first[node_2] else 1
                S_second = -1 if second[node_1] == second[node_2] else 1
                correlation += S_first * S_second
        return correlation
    
    def _set_correlations(self) -> None:
        for i in range(self.size):
            next_indv = self.population.get_next_individual()
            c = self._calculate_correlation(self.individuals[i], next_indv)
            self._correlation.append(c)
            self._cumulative_correlation += c

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


class ConflictError(Error):
    """A class that represents the error function described in the paper
    Graph Coloring with a Distributed Hybrid Quantum Annealing Algorithm."""

    def __str__(self):
        return "ConflictError"

    def individual_err(
            self,
            graph: Graph,
            indv: Individual,
            parameters: Dict[str, Any] = None) -> Optional[float]:
        """Calculates the error of the individual as the number of conflicting edges."""
        return indv.conflicts_weight

    @validate("alpha", "beta")
    def population_err(
            self,
            graph: Graph,
            pop: Population,
            parameters: Dict[str, Any] = None) -> Optional[float]:
        """Calculates the population error as the sum of potential and kinetic energy.
        If an error occurs None is returned, otherwise the total error of the population is returned."""

        alpha = param_value(graph, parameters, "alpha")
        beta = param_value(graph, parameters, "beta")

        potential_energy = alpha * pop.sum_conflicts_weight
        correlation = pop.cumulative_correlation
        kinetic_energy = (-1 * beta) * correlation
        error = potential_energy - kinetic_energy
        return error

    @validate("alpha", "beta")
    def delta(
            self,
            graph: Graph,
            old_indv: Individual,
            new_indv: Individual,
            correlation_diff: float,
            parameters: Dict[str, Any] = None) -> Optional[float]:
        """Calculates the difference of the population error
        that occurred after the replacement of the individual."""

        alpha = param_value(graph, parameters, "alpha")
        beta = param_value(graph, parameters, "beta")

        potential_delta = alpha * (new_indv.conflicts_weight - old_indv.conflicts_weight)
        kinetic_delta = (-1 * beta) * correlation_diff
        return potential_delta - kinetic_delta
