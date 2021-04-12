import mgp
import random


DEFAULT_NUMBER_OF_NODES = 10


@mgp.read_proc
def get_path(
    context: mgp.ProcCtx,
    start: mgp.Vertex,
    number_of_nodes: int = DEFAULT_NUMBER_OF_NODES,
) -> mgp.Record(path=mgp.Path):
    """
    """
    path = mgp.Path(start)
    vertex = start
    for _ in range(10):
        try:
            edge = random.choice(list(vertex.out_edges))
            path.expand(edge)
            vertex = edge.to_vertex()
        except Exception:
            break

    return mgp.Record(path=path)
