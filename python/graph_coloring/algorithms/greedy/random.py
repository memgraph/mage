from typing import Dict, Any
from telco.graph import Graph
from telco.components.individual import Individual
from telco.algorithms.algorithm import Algorithm
from telco.framework.parameters_utils import param_value
from telco.utils.validation import validate


class Random(Algorithm):
    """A class that represents the algorithm that randomly colors nodes."""

    def __str__(self):
        return "Random"

    @validate("no_of_colors")
    def run(
            self,
            graph: Graph,
            parameters: Dict[str, Any] = None) -> Individual:
        no_of_colors = param_value(graph, parameters, "no_of_colors")
        return Individual(no_of_colors, graph)
