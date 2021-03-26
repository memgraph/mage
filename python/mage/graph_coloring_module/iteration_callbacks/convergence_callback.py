from typing import Dict, Any
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.iteration_callbacks.iteration_callback import (
    IterationCallback,
)
from mage.graph_coloring_module.parameters import Parameter


class ConvergenceCallback(IterationCallback):
    def __init__(self):
        self._iteration = 0
        self._best_sol_error = float("inf")
        super().__init__()

    @validate(Parameter.ERROR, Parameter.CONVERGENCE_CALLBACK_TOLERANCE)
    def update(self, graph: Graph, population: Population, parameters: Dict[str, Any]):
        error = param_value(graph, parameters, Parameter.ERROR)
        convergence_callback_tolerance = param_value(
            graph, parameters, Parameter.CONVERGENCE_CALLBACK_TOLERANCE
        )

        if self._best_sol_error == float("inf"):
            self._iteration = 1
            self._best_sol_error = population.min_error(error.individual_err)
            return

        self._iteration += 1
        if population.min_error(error.individual_err) < self._best_sol_error:
            self._best_sol_error = population.min_error(error.individual_err)
            self._iteration = 0

        if self._iteration == convergence_callback_tolerance:
            self._convergence_detected(graph, population, parameters)

    @validate(Parameter.ERROR, Parameter.CONVERGENCE_CALLBACK_ACTIONS)
    def _convergence_detected(
        self, graph: Graph, population: Population, parameters: Dict[str, Any]
    ):
        error = param_value(graph, parameters, Parameter.ERROR)
        convergence_callback_actions = param_value(
            graph, parameters, Parameter.CONVERGENCE_CALLBACK_ACTIONS
        )

        for action in convergence_callback_actions:
            action.execute(graph, population, parameters)

        self._iteration = 0
        self._best_sol_error = population.min_error(error.individual_err)
