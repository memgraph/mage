import random
from graph_coloring_module.operators.mutation import Mutation
from typing import Dict, Any, Optional, Tuple, List
from mage.graph_coloring_module import Graph
from mage.graph_coloring_module import Individual


class SimpleMutation(Mutation):
    """A class that represents a simple mutation that takes
    one node in a graph that is involved in conflicts and change
    it to a new randomly selected color."""

    def __str__(self):
        return "SimpleMutation"

    def mutate(
            self,
            graph: Graph,
            indv: Individual,
            parameters: Dict[str, Any] = None) -> Optional[Tuple[Individual, List[int]]]:
        """Mutate the given individual and returns the new individual and nodes that were changed."""
        conflict_nodes = indv.conflict_nodes
        if len(conflict_nodes) == 0:
            return indv, []
        node = random.sample(conflict_nodes, 1)[0]
        color = random.randint(0, indv.no_of_colors - 1)
        mutated_indv = indv.replace_unit(node, color)
        return mutated_indv, [node]
