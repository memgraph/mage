import random
from typing import Dict, Any
from telco.graph import Graph
from telco.components.individual import Individual
from telco.algorithms.algorithm import Algorithm
from telco.framework.parameters_utils import param_value
from telco.utils.available_colors import available_colors
from telco.utils.validation import validate


class LDO(Algorithm):
    """A class that represents LDO greedy algorithm. This algorithm sorts nodes
    considering their degrees and then colors them sequentially. If it is not possible
    to uniquely determine the color, color is chosen randomly. If coloring the node with
    any possible color would cause conflicts, then the color is chosen randomly."""

    def __str__(self):
        return "LDO"

    @validate("no_of_colors")
    def run(
            self,
            graph: Graph,
            parameters: Dict[str, Any] = None) -> Individual:
        """Returns the Individual that is the result of the LDO algorithm."""

        no_of_colors = param_value(graph, parameters, "no_of_colors")

        chromosome = [-1 for _ in graph.nodes]
        sorted_nodes = sorted(list(graph.nodes), key = lambda n: graph.degree(n), reverse = True)

        for node in sorted_nodes:
            colors = available_colors(graph, no_of_colors, chromosome, node)
            if len(colors) > 0:
                color = random.sample(colors, 1)[0]
            else:
                color = random.randint(0, no_of_colors - 1)
            chromosome[node] = color

        return Individual(no_of_colors, graph, chromosome)
