import random
from graph_coloring_module.operators.mutation import Mutation
from typing import Dict, Any, Tuple, List
from graph_coloring_module.graph import Graph
from graph_coloring_module.components.individual import Individual


class MISMutation(Mutation):
    """A class that represents a mutation that finds one maximal independent set
    and changes colors of all nodes in the set to the same color."""

    def __str__(self):
        return "MISMutation"

    def mutate(
            self,
            graph: Graph,
            indv: Individual,
            parameters: Dict[str, Any] = None) -> Tuple[Individual, List[int]]:
        """Mutate the given individual and returns the new individual and nodes that were changed."""
        nodes = self._MIS(graph)
        color = indv[nodes[0]]
        colors = [color for _ in range(len(nodes))]
        mutated_indv = indv.replace_units(nodes, colors)
        return mutated_indv, nodes

    def _MIS(
            self,
            graph: Graph) -> List[int]:
        """Finds the maximal independent set by shuffling the nodes and adding the first node to the MIs,
        and then sequentially adding all those nodes that do not have neighbors in the MIS."""

        nodes = list(graph.nodes)
        random.shuffle(nodes)
        MIS_flags = [False for _ in range(len(graph))]
        MIS = []

        for node in nodes:
            include = True
            for neigh in graph[node]:
                if MIS_flags[neigh]:
                    include = False
                    break

            if include:
                MIS.append(node)
                MIS_flags[node] = True
        return MIS
