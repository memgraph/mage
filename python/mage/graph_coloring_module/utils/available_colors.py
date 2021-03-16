from typing import List
from graph_coloring_module.graph import Graph


def available_colors(
        graph: Graph,
        no_of_colors: int,
        chromosome: List[int],
        node: int) -> List[int]:
    """Returns a list of colors that are not used to color neighbors of a given node."""

    used = [False for _ in range(no_of_colors)]
    for neigh in graph[node]:
        c = chromosome[neigh]
        if c != -1:
            used[c] = True

    available_colors = []
    for c in range(no_of_colors):
        if not used[c]:
            available_colors.append(c)

    return available_colors
