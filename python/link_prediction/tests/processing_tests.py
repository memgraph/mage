from typing import Dict
import mgp
import dgl
import link_prediction_util
import torch

def test_conversion(graph: dgl.graph, new_to_old: Dict[int, int], ctx: mgp.ProcCtx, 
                    node_id_property: str, node_features_property: str) -> bool:
    """
    Tests whether conversion from ctx.ProcCtx graph to dgl graph went successfully. Checks how features are mapped.

    Args:
        graph (dgl.graph): Reference to the dgl graph.
        new_to_old (Dict[int, int]): Mapping from new indexes to old indexes.
        ctx (mgp.ProcCtx): Reference to the context execution.
        node_id_property (str): Property name where the the node id is saved.
        node_features_property (str): Property namer where the node features are saved`

    Returns:
        bool: True if everything went OK and False if test failed.
    """
    for vertex in graph.nodes():
        vertex_id=vertex.item()
        old_id=new_to_old[vertex_id]
        vertex=link_prediction_util.search_vertex(ctx=ctx, id=old_id, node_id_property=node_id_property)
        if vertex is None:
            return False
        old_features=vertex.properties.get(node_features_property)
        if torch.equal(graph.ndata[node_features_property][vertex_id], torch.tensor(old_features, dtype=torch.float32)) is False:
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

