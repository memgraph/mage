import random
from mage.graph_coloring_module.operators.mutations.mutation import Mutation
from typing import Dict, Any, Tuple, List
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.utils.available_colors import available_colors
from mage.graph_coloring_module.parameters import Parameter


class RandomMutation(Mutation):
    """A mutation that changes the color of a random node
    in a graph with a random_mutation_probability probability,
    and with probability 1 - random_mutation_probability changes
    the color of a conflict node in a graph."""

    def __str__(self):
        return "RandomMutation"

    @validate(Parameter.RANDOM_MUTATION_PROBABILITY, Parameter.NO_OF_COLORS)
    def mutate(
        self, graph: Graph, indv: Individual, parameters: Dict[str, Any] = None
    ) -> Tuple[Individual, List[int]]:
        """Mutate the given individual and returns the new individual and nodes that were changed."""

        random_mutation_probability = param_value(
            graph, parameters, Parameter.RANDOM_MUTATION_PROBABILITY
        )
        no_of_colors = param_value(graph, parameters, Parameter.NO_OF_COLORS)

        conflict_nodes = indv.conflict_nodes
        non_conflict_nodes = []
        for node in graph.nodes:
            if node not in conflict_nodes:
                non_conflict_nodes.append(node)

        if len(conflict_nodes) == 0:
            return indv, []

        if random.random() < random_mutation_probability:
            node = random.sample(non_conflict_nodes, 1)[0]
            color = random.randint(0, no_of_colors - 1)
        else:
            node = random.sample(conflict_nodes, 1)[0]
            colors = available_colors(graph, no_of_colors, indv.chromosome, node)
            if len(colors) > 0:
                color = random.sample(colors, 1)[0]
            else:
                color = random.randint(0, no_of_colors - 1)

        mutated_indv = indv.replace_unit(node, color)
        return mutated_indv, [node]
