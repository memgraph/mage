import mgp
import random


DEFAULT_NUMBER_OF_NODES = 10


@mgp.read_proc
def get_path(
    context: mgp.ProcCtx,
    start: mgp.Vertex,
    number_of_nodes: int = DEFAULT_NUMBER_OF_NODES,
) -> mgp.Record(path=mgp.Path):
    """Generates a random path of length `number_of_nodes` or less starting
    from the `start` vertex.
    """
    path = mgp.Path(start)
    vertex = start
    for _ in range(number_of_nodes):
        try:
            edge = random.choice(list(vertex.out_edges))
            path.expand(edge)
            vertex = edge.to_vertex
        except IndexError:
            break

    return mgp.Record(path=path)
