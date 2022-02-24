import mgp


def BFS_find_weight_min_max(start_v: mgp.Vertex, edge_property: str) -> mgp.Number:
    """
    Breadth-first search for finding the largest and smallest edge weight,
    largest being used for capacity scaling, and smallest for lower bound

    :param start_v: starting vertex
    :param edge_propery: str denoting the edge property used as weight

    :return: Number, the largest value of edge_property in graph
    """

    next_queue = [start_v]
    visited = set()
    max_weight = 0
    min_weight = float("Inf")

    while next_queue:
        current_v = next_queue.pop(0)
        visited.add(current_v)

        for e in current_v.out_edges:
            if e.properties[edge_property] > max_weight:
                max_weight = e.properties[edge_property]
            elif e.properties[edge_property] < min_weight:
                min_weight = e.properties[edge_property]

            if e.to_vertex not in visited:
                next_queue.append(e.to_vertex)

    return max_weight, min_weight
