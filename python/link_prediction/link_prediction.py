import mgp # Python API
import torch
import dgl  # geometric deep learning
from typing import Tuple, Dict
from tests.processing_tests import test_conversion
from numpy import int32

@mgp.read_proc
def train_eval(ctx: mgp.ProcCtx) -> mgp.Record(auc_score=float):
    """
    Trains model on training set and evaluates it on test set.
    """
    graph, new_to_old = _get_dgl_graph_data(ctx)  # dgl representation of the graph and dict new to old index
    if test_conversion(graph=graph, new_to_old=new_to_old, ctx=ctx) is False:
        print("Remapping failed")


    # Train g is a graph which has removed test edges
    # Others are positive and negative train and test graphs

    # NOTE: THIS LINE BELOW RESULTS IN RAM OVERFLOW
    # train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = link_prediction_util.preprocess(g)
    # Create a model and edge predictor
    # Send to the GraphSAGE model features
    # print(train_g.ndata)
    # model = GraphSAGE(train_g.ndata['features'].shape[1], 16)
    # pred = DotPredictor()

    # link_prediction_util.train(model, pred, train_g, train_pos_g, train_neg_g)
    # trained_hidden_features = model(train_g, train_g.ndata["features"])  # ndata returns a node-data view for getting/setting node features. This will return node feature fear = (Tensor) or dict(str, Tensor) if dealing
    # auc_score = link_prediction_util.test(trained_hidden_features, pred, test_pos_g, test_neg_g)

    # Return just some generic result
    rec=mgp.Record(auc_score=0.01)
    return rec



def _get_dgl_graph_data(ctx: mgp.ProcCtx) -> Tuple[dgl.graph, Dict[int32, int32]]:
    """
    Returns dgl representation of the graph.
    :param ctx: Reference to the context execution.
    :return: Tuple of DGL graph representation and dictionary of mapping new to old index.
    """
    src_nodes, dest_nodes=[], []  # for saving the edges

    new_to_old=dict()  # map of new node index to old node index
    old_to_new=dict()  # map of old node index to new node index
    features=[]  # map of new node index to its feature
    ind=0

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            src_node, dest_node=edge.from_vertex, edge.to_vertex
            src_id_old=int(src_node.properties.get("id"))
            src_features=src_node.properties.get("features")
            dest_id_old=int(dest_node.properties.get("id"))
            dest_features=dest_node.properties.get("features")

            if src_id_old not in old_to_new.keys():
                new_to_old[ind]=src_id_old
                old_to_new[src_id_old]=ind
                src_nodes.append(ind)
                features.append(src_features)  # do very simple remapping
                ind += 1
            else:
                src_nodes.append(old_to_new[src_id_old])

            # Repeat the same for destination node
            if dest_id_old not in old_to_new.keys():
                new_to_old[ind]=dest_id_old
                old_to_new[dest_id_old]=ind
                dest_nodes.append(ind)
                features.append(dest_features)
                ind += 1
            else:
                dest_nodes.append(old_to_new[dest_id_old])

    features=torch.tensor(features, dtype=torch.float32)  # use float for storing tensor of features
    g=dgl.graph((src_nodes, dest_nodes))
    g.ndata["features"]=features
    return g, new_to_old


@ mgp.read_proc
def memory_test(ctx: mgp.ProcCtx) -> mgp.Record(res=int):
    print("Inside memory test function")
    global a
    print("A = ", a)
    a=1 - a
    return mgp.Record(res=1)


@ mgp.read_proc
def predict_link_score(ctx: mgp.ProcCtx, nodes: mgp.List[mgp.Vertex], src: mgp.Vertex, dest: mgp.Vertex
                       ) -> mgp.Record(prediction=mgp.Number):
    """
    How to design method, does it calculate something on the subgraph or where?
    :params src: src vertex in prediction
    :params dest: dest vertex in prediction

    :return prediction: 0 if there isn't edge and 1 if there is an edge
    """

    rec=mgp.Record(prediction=0.05)
