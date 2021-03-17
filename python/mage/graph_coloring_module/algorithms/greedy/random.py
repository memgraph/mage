from typing import Dict, Any
from mage.graph_coloring_module import Graph
from mage.graph_coloring_module import Individual
from mage.graph_coloring_module import Algorithm
from mage.graph_coloring_module import param_value
from mage.graph_coloring_module import validate


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
