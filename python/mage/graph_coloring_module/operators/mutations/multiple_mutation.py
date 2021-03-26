import random
from mage.graph_coloring_module.operators.mutations.mutation import Mutation
from typing import Dict, Any, Tuple, List
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.parameters import Parameter


class MultipleMutation(Mutation):
    """A class that represents a mutation that changes an individual
    by randomly changing colors of the given number of nodes."""

    def __str__(self):
        return "MultipleMutation"

    @validate(Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES, Parameter.NO_OF_COLORS)
    def mutate(
        self, graph: Graph, indv: Individual, parameters: Dict[str, Any] = None
    ) -> Tuple[Individual, List[int]]:
        """Mutate the given individual and returns the new individual and nodes that were changed."""

        no_of_nodes_to_mutate = param_value(
            graph, parameters, Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES
        )
        no_of_colors = param_value(graph, parameters, Parameter.NO_OF_COLORS)

        nodes = [
            random.randint(0, len(graph) - 1) for _ in range(no_of_nodes_to_mutate)
        ]
        colors = [
            random.randint(0, no_of_colors - 1) for _ in range(no_of_nodes_to_mutate)
        ]
        mutated_indv = indv.replace_units(nodes, colors)
        return mutated_indv, nodes
