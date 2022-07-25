from email import message
import mgp  # Python API
import torch
import dgl  # geometric deep learning
from typing import Tuple, Dict
from tests.processing_tests import test_conversion
from numpy import int32
from dataclasses import dataclass


##############################
# classes and data structures
##############################


@dataclass
class LinkPredictionParameters:
    """
    layer_type: str -> layer type
    num_epochs: int -> number of epochs for model training
    optimizer: str -> can be one of the following: ADAM, SGD, AdaGrad...
    learning_rate: float -> learning rate for optimizer
    split_ratio: float -> split ratio between training and validation set. There is not test dataset because it is assumed that user first needs to create new edges in dataset to test a model on them.
    epochs_printing: int ->  how often do you want to print results from your model? Results will be from validation dataset. 
    node_features: str -> property name where the node features are saved.
    device_type: str ->  If model will be trained using CPU or cuda GPU. Possible values are cpu and cuda.
    """
    num_of_layers: int = 10
    layer_type: str = "GAT"
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    num_epochs: int = 200
    split_ratio: float = 0.775
    epochs_printing: int = 10
    device_type: str = "cuda" 
    node_features: str = "features1"



##############################
# global parameters
##############################

link_prediction_parameters: LinkPredictionParameters = LinkPredictionParameters()
##############################
# All read procedures
##############################

@mgp.read_proc
def set_model_parameters(ctx: mgp.ProcCtx, parameters: mgp.Map) -> mgp.Record(status=mgp.Number, message=str):
    """
    Saves parameters to the global parameters link_prediction_parameters. Specific parsing is needed because we want enable user to call it with a subset of parameters, no need to send them all. We will use some
    kind of reflection to most easily update parameters. NOTE: We don't do any checks on values sent in here, code will fail later if some unsupported parameter is sent.
    :param ctx: Reference to the context execution.
    :param parameters:
        num_of_layers: int -> number of layers
        layer_type: str -> layer type
        num_epochs: int -> number of epochs for model training
        optimizer: str -> can be one of the following: ADAM, SGD, AdaGrad...
        learning_rate: float -> learning rate for optimizer
        split_ratio: float -> split ratio between training and validation set. There is not test dataset because it is assumed that user first needs to create new edges in dataset to test a model on them.
        epochs_printing: int ->  how often do you want to print results from your model? Results will be from validation dataset. 
        node_features: str -> property name where the node features are saved.
        device_type: str ->  If model will be trained using CPU or cuda GPU. Possible values are cpu and cuda.
    :return: 1 if all sent parameters are successfully saved, 0 otherwise.
    """
    global link_prediction_parameters
    for key, value in parameters.items():
        if hasattr(link_prediction_parameters, key) is False:
            return mgp.Record(status=0, message="No attribute " + key + " in class LinkPredictionParameters")
        try:
            link_prediction_parameters.__setattr__(key, value)
        except Exception as exception:
            return mgp.Record(status=0, message=repr(exception))

    return mgp.Record(status=1, message="OK")


@mgp.read_proc
def train_eval(ctx: mgp.ProcCtx) -> mgp.Record(auc_score=float):
    """
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
    rec = mgp.Record(auc_score=0.01)
    return rec


@mgp.read_proc
def predict_link_score(ctx: mgp.ProcCtx, src_vertex: mgp.Vertex, dest_vertex: mgp.Vertex) -> mgp.Record(score=mgp.Number):
    """
    :param ctx: Reference to the execution context.
    :param src_vertex: Source vertex
    :param dest_vertex: Destination vertex.
    :return: Score between 0 and 1. 
    """
    result = mgp.Record(score=0.61)
    return result


##############################
# Convert to DGL graph, consider extracting such methods to another file.
##############################


def _get_dgl_graph_data(ctx: mgp.ProcCtx) -> Tuple[dgl.graph, Dict[int32, int32]]:
    """
    Creates dgl representation of the graph.
    :param ctx: Reference to the context execution.
    :return: Tuple of DGL graph representation and dictionary of mapping new to old index.
    """
    src_nodes, dest_nodes = [], []  # for saving the edges

    new_to_old = dict()  # map of new node index to old node index
    old_to_new = dict()  # map of old node index to new node index
    features = []  # map of new node index to its feature
    ind = 0

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            src_node, dest_node = edge.from_vertex, edge.to_vertex
            src_id_old = int(src_node.properties.get("id"))
            src_features = src_node.properties.get("features")
            dest_id_old = int(dest_node.properties.get("id"))
            dest_features = dest_node.properties.get("features")

            if src_id_old not in old_to_new.keys():
                new_to_old[ind] = src_id_old
                old_to_new[src_id_old] = ind
                src_nodes.append(ind)
                features.append(src_features)  # do very simple remapping
                ind += 1
            else:
                src_nodes.append(old_to_new[src_id_old])

            # Repeat the same for destination node
            if dest_id_old not in old_to_new.keys():
                new_to_old[ind] = dest_id_old
                old_to_new[dest_id_old] = ind
                dest_nodes.append(ind)
                features.append(dest_features)
                ind += 1
            else:
                dest_nodes.append(old_to_new[dest_id_old])

    features = torch.tensor(features, dtype=torch.float32)  # use float for storing tensor of features
    g = dgl.graph((src_nodes, dest_nodes))
    g.ndata["features"] = features
    return g, new_to_old
