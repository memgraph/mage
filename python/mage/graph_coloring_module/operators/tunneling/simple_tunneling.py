from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.population import Population
from typing import Dict, Any


@validate("max_attempts_tunneling", "mutation_tunneling")
def tunneling(
    self,
    ind: int,
    graph: Graph,
    population: Population,
    parameters: Dict[str, Any] = None,
) -> None:

    max_attempts_tunneling = param_value(graph, parameters, "max_attempts_tunneling")
    mutation = param_value(graph, parameters, "mutation_tunneling")

    old_indv = population.individuals[ind]
    old_indv_error = old_indv.conflicts_weight

    for _ in range(max_attempts_tunneling):
        new_indv, diff_nodes = mutation.mutate(graph, old_indv, parameters)
        if new_indv.conflicts_weight <= 2 * old_indv_error:
            population.set_individual(ind, new_indv, diff_nodes)
