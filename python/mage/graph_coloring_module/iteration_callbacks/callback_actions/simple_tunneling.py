import random
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.iteration_callbacks.callback_actions.action import (
    Action,
)
from mage.graph_coloring_module.parameters import Parameter
from typing import Dict, Any


class SimpleTunneling(Action):
    @validate(
        Parameter.SIMPLE_TUNNELING_MUTATION,
        Parameter.SIMPLE_TUNNELING_PROBABILITY,
        Parameter.SIMPLE_TUNNELING_MAX_ATTEMPTS,
        Parameter.SIMPLE_TUNNELING_ERROR_CORRECTION,
    )
    def execute(
        self,
        graph: Graph,
        population: Population,
        parameters: Dict[str, Any] = None,
    ) -> None:

        simple_tunneling_max_attempts = param_value(
            graph, parameters, Parameter.SIMPLE_TUNNELING_MAX_ATTEMPTS
        )
        simple_tunneling_mutation = param_value(
            graph, parameters, Parameter.SIMPLE_TUNNELING_MUTATION
        )
        simple_tunneling_probability = param_value(
            graph, parameters, Parameter.SIMPLE_TUNNELING_PROBABILITY
        )
        simple_tunneling_error_correction = param_value(
            graph, parameters, Parameter.SIMPLE_TUNNELING_ERROR_CORRECTION
        )

        for i, old_indv in enumerate(population.individuals):
            if random.random() < simple_tunneling_probability:
                old_indv_error = old_indv.conflicts_weight
                for _ in range(simple_tunneling_max_attempts):
                    new_indv, diff_nodes = simple_tunneling_mutation.mutate(
                        graph, old_indv, parameters
                    )
                    if (
                        new_indv.conflicts_weight
                        <= simple_tunneling_error_correction * old_indv_error
                    ):
                        population.set_individual(i, new_indv, diff_nodes)
