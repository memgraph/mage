import mgp
import dgl
import link_prediction_util
import torch

def test_conversion(graph: dgl.graph, new_to_old, ctx: mgp.ProcCtx) -> bool:
    """
    Tests whether conversion to dgl graph went successfully. Checks how features are mapped.
    :param graph: Reference to the dgl graph.
    :param new_to_old: Mapping from new indexes to old indexes.
    :param ctx: Reference to the context execution.
    :return: True if everything went ok and False if something fails.
    """
    for vertex in graph.nodes():
        vertex_id=vertex.item()
        old_id=new_to_old[vertex_id]
        vertex=link_prediction_util.search_vertex(ctx=ctx, id=old_id)
        if vertex is None:
            return False
        old_features=vertex.properties.get("features")
        if torch.equal(graph.ndata["features"][vertex_id], torch.tensor(old_features, dtype=torch.float32)) is False:
            return False

    # Check number of nodes
    if graph.number_of_nodes() != len(ctx.graph.vertices):
        print("Wrong number of nodes!")
        return False
    # Check number of edges
    if graph.number_of_edges() != link_prediction_util.get_number_of_edges(ctx):
        print("Wrong number of edges")
        return False

    return True

