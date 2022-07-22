import mgp
import random

@mgp.read_proc
def get_path(
    start: mgp.Vertex,
    length: int
) -> mgp.Record(path=mgp.Path):
    """Generates a random path of length `length` or less starting
    from the `start` vertex.

    :param mgp.Vertex start: The starting node of the walk.
    :param int length: The number of edges to traverse.
    :return: Random path.
    :rtype: mgp.Record(mgp.Path)
    """
    path = mgp.Path(start)
    visited = {start}
    temp = start

    for _ in range(length):
        available_edges = list(temp.out_edges)
        random.shuffle(available_edges)
        for edge in available_edges:
            if edge.to_vertex not in visited:
                temp = edge.to_vertex
                path.expand(edge)
                visited.add(temp)
                break
    #print(len(path.vertices))
    return mgp.Record(path=path)