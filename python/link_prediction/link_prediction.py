import mgp
from numpy import append  # PythonAPI
import torch
import dgl  # geometric deep learning
from typing import List
from click import pass_context
from GraphSAGE import GraphSAGE
import link_prediction_util
from DotPredictor import DotPredictor


# Which methods are we going to support? Creation of training and test set upfront or splitting
# here in the method?
# Second step is to create a graph that can be loaded to DGL graph

@mgp.read_proc
def train_eval(context: mgp.ProcCtx,
        nodes: mgp.List[mgp.Vertex],
        num_epochs: int,
        split_size: float) -> mgp.Record(auc_score=float):
    """
    Trains model on training set and evaluates it on test set.
    """

    # print(len(nodes))
    g = _get_dgl_graph_data(nodes)
    # print(g.edges())

    # Train g is a graph which has removed test edges
    # Others are positive and negative train and test graphs

    # NOTE: THIS LINE BELOW RESULTS IN RAM OVERFLOW
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = link_prediction_util.preprocess(g)
    # Create a model and edge predictor
    # Send to the GraphSAGE model features
    # print(train_g.ndata)
    model = GraphSAGE(train_g.ndata['features'].shape[1], 16)
    pred = DotPredictor()

    link_prediction_util.train(model, pred, train_g, train_pos_g, train_neg_g)
    trained_hidden_features = model(train_g, train_g.ndata["features"])  # ndata returns a node-data view for getting/setting node features. This will return node feature fear = (Tensor) or dict(str, Tensor) if dealing
    auc_score = link_prediction_util.test(trained_hidden_features, pred, test_pos_g, test_neg_g)

    # Return just some generic result
    rec = mgp.Record(auc_score=auc_score)
    return rec


@mgp.read_proc
def predict_link_score(ctx: mgp.ProcCtx, nodes: mgp.List[mgp.Vertex], src: mgp.Vertex, dest: mgp.Vertex
) -> mgp.Record(prediction=mgp.Number):
    """
    How to design method, does it calculate something on the subgraph or where?
    :params src: src vertex in prediction
    :params dest: dest vertex in prediction

    :return prediction: 0 if there isn't edge and 1 if there is an edge
    """

    rec = mgp.Record(prediction=0.05)


def test(nodes: mgp.List[mgp.Vertex], new_to_old, old_to_new):
    for vertex in nodes:
        for edge in vertex.out_edges:
            src_node, dest_node = edge.from_vertex, edge.to_vertex
            src_id_old = int(src_node.properties.get("id"))
            src_features = torch.tensor(src_node.properties.get("features"), dtype=torch.long)
            dest_id_old = int(dest_node.properties.get("id"))
            dest_features = torch.tensor(dest_node.properties.get("features"), dtype=torch.long)

            # print(src_id_old, old_to_new[src_id_old], new_to_old[old_to_new[src_id_old]])
            # print(dest_id_old, old_to_new[dest_id_old], new_to_old[old_to_new[dest_id_old]])
    print(len(new_to_old.keys()))
    print(len(new_to_old.values()))
    print(len(old_to_new.keys()))
    print(len(new_to_old.values()))



def _get_dgl_graph_data(nodes: mgp.List[mgp.Vertex]) -> dgl.graph:
    src_nodes, dest_nodes = [], []
    # features = torch.zeros((len(nodes), features_size), dtype=torch.float32)

    new_to_old = dict()  # map of new index to old index
    old_to_new = dict()
    ind = 0

    features = []

    for vertex in nodes:
        for edge in vertex.out_edges:
            src_node, dest_node = edge.from_vertex, edge.to_vertex    
            src_id_old = int(src_node.properties.get("id"))
            src_features = src_node.properties.get("features")
            # print(src_features)
            dest_id_old = int(dest_node.properties.get("id"))
            dest_features = dest_node.properties.get("features")

            if src_id_old not in old_to_new.keys():
                new_to_old[ind] = src_id_old
                old_to_new[src_id_old] = ind
                src_nodes.append(ind)
                features.append(src_features)
                ind += 1

            else:
                src_nodes.append(old_to_new[src_id_old])
                # features[old_to_new[src_id_old], :] = src_features

            if dest_id_old not in old_to_new.keys():                       
                new_to_old[ind] = dest_id_old
                old_to_new[dest_id_old] = ind
                dest_nodes.append(ind)
                features.append(dest_features)
                ind += 1
            else:
                dest_nodes.append(old_to_new[dest_id_old])
                # features[old_to_new[dest_id_old], :] = dest_features

    features = torch.tensor(features, dtype=torch.float32)
    # print("Features size: ", features.shape)
    # print(len(src_nodes))
    # print(len(dest_nodes))
    # print(ind)
    g = dgl.graph((src_nodes, dest_nodes))
    g.ndata["features"] = features
    # print(torch.sum(g.ndata["features"][1]))
    # test(nodes, new_to_old, old_to_new)
    return g



