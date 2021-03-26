import logging
from typing import Dict, Any, Optional
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.error_functions.error import Error
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.parameters import Parameter


logger = logging.getLogger("telco")


class ConflictError(Error):
    """A class that represents the error function described in the paper
    Graph Coloring with a Distributed Hybrid Quantum Annealing Algorithm."""

    def __str__(self):
        return "ConflictError"

    def individual_err(
        self, graph: Graph, indv: Individual, parameters: Dict[str, Any] = None
    ) -> Optional[float]:
        """Calculates the error of the individual as the number of conflicting edges."""
        return indv.conflicts_weight

    @validate(Parameter.CONFLICT_ERR_ALPHA, Parameter.CONFLICT_ERR_BETA)
    def population_err(
        self, graph: Graph, pop: Population, parameters: Dict[str, Any] = None
    ) -> Optional[float]:
        """Calculates the population error as the sum of potential and kinetic energy.
        If an error occurs None is returned, otherwise the total error of the population is returned."""

        alpha = param_value(graph, parameters, Parameter.CONFLICT_ERR_ALPHA)
        beta = param_value(graph, parameters, Parameter.CONFLICT_ERR_BETA)

        potential_energy = alpha * pop.sum_conflicts_weight
        correlation = pop.cumulative_correlation
        kinetic_energy = (-1 * beta) * correlation
        error = potential_energy - kinetic_energy
        return error

    @validate(Parameter.CONFLICT_ERR_ALPHA, Parameter.CONFLICT_ERR_BETA)
    def delta(
        self,
        graph: Graph,
        old_indv: Individual,
        new_indv: Individual,
        correlation_diff: float,
        parameters: Dict[str, Any] = None,
    ) -> Optional[float]:
        """Calculates the difference of the population error
        that occurred after the replacement of the individual."""

        alpha = param_value(graph, parameters, Parameter.CONFLICT_ERR_ALPHA)
        beta = param_value(graph, parameters, Parameter.CONFLICT_ERR_BETA)

        potential_delta = alpha * (
            new_indv.conflicts_weight - old_indv.conflicts_weight
        )
        kinetic_delta = (-1 * beta) * correlation_diff
        return potential_delta - kinetic_delta
