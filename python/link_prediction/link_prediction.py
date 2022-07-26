from email import message
import mgp  # Python API
import torch
import dgl  # geometric deep learning
from typing import List, Tuple, Dict
from tests.processing_tests import test_conversion
from numpy import int32
from dataclasses import dataclass
import link_prediction_util


##############################
# classes and data structures
##############################

# You can think of link prediction as link classification where the label determines whether edge is or no.
# Seed
# https://github.com/williamleif/GraphSAGE/issues/93
# Memgraph lab 2.2
# Memgraph internship

@dataclass
class LinkPredictionParameters:
    """Parameters user in LinkPrediction module.
    :param hidden_features_size: List[int] -> Defines the size of each hidden layer in the architecture. 
    :param layer_type: str -> Layer type
    :param num_epochs: int -> Number of epochs for model training
    :param optimizer: str -> Can be one of the following: ADAM, SGD, AdaGrad...
    :param learning_rate: float -> Learning rate for optimizer
    :param split_ratio: float -> Split ratio between training and validation set. There is not test dataset because it is assumed that user first needs to create new edges in dataset to test a model on them.
    :param node_features_property: str → Property name where the node features are saved.
    :param node_id_property: str -> property name where the node id is saved.
    :param device_type: str ->  If model will be trained using CPU or cuda GPU. Possible values are cpu and cuda. To run it on Cuda, user must set this flag to true and system must support cuda execution. 
                                System's support is checked with torch.cuda.is_available()
    :param console_log_freq: int ->  how often do you want to print results from your model? Results will be from validation dataset. 
    :param checkpoint_freq: int → Select the number of epochs on which the model will be saved. The model is persisted on disc.
    :param aggregator: str → Aggregator used in models. Can be one of the following: LSTM, pooling, mean.
    """
    hidden_features_size = [16, 16]  # TODO specify what does this mean in more detail.    
    layer_type: str = "GAT"
    num_epochs: int = 100
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    split_ratio: float = 0.8
    node_features_property: str = "features"
    node_id_property: str = "id"
    device_type: str = "cpu" 
    console_log_freq: int = 5
    checkpoint_freq: int = 10
    aggregator: str = "mean"

@dataclass
class LinkPredictionOutputResult:
    """
    Metrics that will be returned on query .train() and get_output_results. Idea is later enable customizing metrics that can be displayed and returned by using __setattr__ and del.
    :param epoch: Epoch when metrics were showing such results.
    :param loss: Loss value in the specified epoch.
    :param accuracy: Accuracy in the specified epoch. (TP+TN) / ALL
    :param auc_score: Area below AUC curve. Probability that a random positive example is positioned to the right of a random negative example.
    """
    epoch: int = -1;
    loss: float = -1.0;
    accuracy: float = -1.0;
    auc_score: float = -1.0;


##############################
# global parameters
##############################

link_prediction_parameters: LinkPredictionParameters = LinkPredictionParameters()  # parameters currently saved.
output_results: List[LinkPredictionOutputResult] = list()  # List of all output records.
graph: dgl.graph = None # Reference to the graph.

##############################
# All read procedures
##############################

@mgp.read_proc
def set_model_parameters(ctx: mgp.ProcCtx, parameters: mgp.Map) -> mgp.Record(status=mgp.Number, message=str):
    """
    Saves parameters to the global parameters link_prediction_parameters. Specific parsing is needed because we want enable user to call it with a subset of parameters, no need to send them all. We will use some
    kind of reflection to most easily update parameters.
    :param ctx: Reference to the context execution.
    :param parameters:
        hidden_features_size: mgp.List[int] -> Defines the size of each hidden layer in the architecture. 
        layer_type: str -> Layer type
        num_epochs: int -> Number of epochs for model training
        optimizer: str -> Can be one of the following: ADAM, SGD, AdaGrad...
        learning_rate: float -> Learning rate for optimizer
        split_ratio: float -> Split ratio between training and validation set. There is not test dataset because it is assumed that user first needs to create new edges in dataset to test a model on them.
        node_features_property: str → Property name where the node features are saved.
        node_id_property str -> property name where the node id is saved.
        device_type: str ->  If model will be trained using CPU or cuda GPU. Possible values are cpu and cuda. To run it on Cuda, user must set this flag to true and system must support cuda execution. 
                                System's support is checked with torch.cuda.is_available()
        console_log_freq: int ->  how often do you want to print results from your model? Results will be from validation dataset. 
        checkpoint_freq: int → Select the number of epochs on which the model will be saved. The model is persisted on disc.
        aggregator: str → Aggregator used in models. Can be one of the following: LSTM, pooling, mean.

    :return: 1 if all sent parameters are successfully saved, 0 otherwise.
    """
    global link_prediction_parameters

    validation_status, validation_message = _validate_user_parameters(parameters=parameters)
    if validation_status is False:
        return mgp.Record(status=validation_status, message=validation_message)

    for key, value in parameters.items():
        if hasattr(link_prediction_parameters, key) is False:
            return mgp.Record(status=0, message="No attribute " + key + " in class LinkPredictionParameters")
        try:
            link_prediction_parameters.__setattr__(key, value)
        except Exception as exception:
            return mgp.Record(status=0, message=repr(exception))

    # Device type handling
    if link_prediction_parameters.device_type == "cuda" and torch.cuda.is_available() is True:
        link_prediction_parameters.device_type = "cuda"
    else:
        link_prediction_parameters.device_type = "cpu"
    
    return mgp.Record(status=1, message="OK")


@mgp.read_proc
def train(ctx: mgp.ProcCtx) -> mgp.Record(epoch=int, loss=float, accuracy=float, auc_score=float):
    """ Train method is used for training the module on the dataset provided with ctx. By taking decision to split the dataset here and not in the separate method, it is impossible to retrain the same model. 

    Args:
        ctx (mgp.ProcCtx, optional): Reference to the process execution.

    Returns:
        mgp.Record: It returns performance metrics obtained during the training.
    """
    global output_results

    graph, new_to_old = _get_dgl_graph_data(ctx)  # dgl representation of the graph and dict new to old index
    if test_conversion(graph=graph, new_to_old=new_to_old, ctx=ctx, node_id_property=link_prediction_parameters.node_id_property, node_features_property=link_prediction_parameters.node_features_property) is False:
        print("Remapping failed")

    # Train g is a graph which has removed test edges
    # Others are positive and negative train and test graphs

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
    rec = mgp.Record(epoch=1, loss=0.95, accuracy=0.99, auc_score=0.86)
    return rec


@mgp.read_proc
def predict_link_score(ctx: mgp.ProcCtx, src_vertex: mgp.Vertex, dest_vertex: mgp.Vertex) -> mgp.Record(score=mgp.Number):
    """Predicts edge score determined by source and destination vertex.

    Args:
        ctx (mgp.Record): The reference to the context execution.
        src_vertex (mgp.Vertex): Source vertex.
        dest_vertex (mgp.Vertex) Destination vertex.

    Returns:
        mgp.Record: A score between 0 and 1.
    """    
    result = mgp.Record(score=0.61)
    return result


@mgp.read_proc
def get_output_results(ctx: mgp.ProcCtx) -> mgp.Record(epoch=int, loss=float, accuracy=float, auc_score=float):
    """This method is used when user wants to get performance data obtained from the last training. It is in the form of LinkPredictionOutputResult data structure.

    Args:
        ctx (mgp.ProcCtx): Reference to the context execution

    Returns:
        mgp.Record[List[LinkPredictionOutputResult]]: A list of LinkPredictionOutputResults. 
    """
    # For now just simulate adding results here
    global output_results
    # Unfortunately copying
    results = []
    for out_res in output_results:
        results.append(mgp.Record(epoch=out_res.epoch, loss=out_res.loss, accuracy=out_res.accuracy, auc_score=out_res.auc_score))
    
    return results

##############################
# Convert to DGL graph, consider extracting such methods to another file.
##############################


def _get_dgl_graph_data(ctx: mgp.ProcCtx) -> Tuple[dgl.graph, Dict[int32, int32]]:
    """Creates dgl representation of the graph.

    Args:
        ctx (mgp.ProcCtx): The reference to the context execution.

    Returns:
        Tuple[dgl.graph, Dict[int32, int32]]: Tuple of DGL graph representation and dictionary of mapping new to old index.
    """    
    src_nodes, dest_nodes = [], []  # for saving the edges

    new_to_old = dict()  # map of new node index to old node index
    old_to_new = dict()  # map of old node index to new node index
    features = []  # map of new node index to its feature
    ind = 0

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            src_node, dest_node = edge.from_vertex, edge.to_vertex
            src_id_old = int(src_node.properties.get(link_prediction_parameters.node_id_property))
            src_features = src_node.properties.get(link_prediction_parameters.node_features_property)
            dest_id_old = int(dest_node.properties.get(link_prediction_parameters.node_id_property))
            dest_features = dest_node.properties.get(link_prediction_parameters.node_features_property)

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

    features = torch.tensor(features, dtype=torch.float32, device=link_prediction_parameters.device_type)  # use float for storing tensor of features
    g = dgl.graph((src_nodes, dest_nodes))
    g.ndata[link_prediction_parameters.node_features_property] = features
    return g, new_to_old


def _validate_user_parameters(parameters: mgp.Map) -> Tuple[bool, str]:
    """Validates parameters user sent through method set_model_parameters

    Args:
        parameters (mgp.Map): Parameters sent by user.

    Returns:
        bool: True if every parameter value is appropriate, False otherwise.
    """

    # Layer type check
    layer_type = parameters["layer_type"]
    if layer_type != "graph_attn" and layer_type != "graph_sage":
        return False, "Unknown layer type, this module supports only graph_attn and graph_sage. "

    # Num epochs
    num_epochs = int(parameters["num_epochs"])
    if num_epochs <= 0:
        return False, "Number of epochs must be greater than 0. "

    # Optimizer check
    optimizer = parameters["optimizer"]
    if optimizer != "ADAM" and optimizer != "SGD":
        return False, "Unknown optimizer, this module supports only ADAM and SGD. "

    # Learning rate check
    learning_rate = float(parameters["learning_rate"])
    if learning_rate <= 0.0:
        return False, "Learning rate must be greater than 0. "

    # Split ratio check
    split_ratio = float(parameters["split_ratio"])
    if split_ratio <= 0.0:
        return False, "Split ratio must be greater than 0. "

    # node_features_property check
    node_features_property = parameters["node_features_property"]
    if node_features_property == "":
        return False, "You must specify name of nodes' features property. "
    
    # node_id_property check
    node_id_property = parameters["node_id_property"]
    if node_id_property == "":
        return False, "You must specify name of nodes' id property. "
    
    # device_type check
    device_type = parameters["device_type"]
    if device_type != "cpu" and torch.device != "cuda":
        return False, "Only cpu and cuda are supported as devices. "

    # console_log_freq check
    console_log_freq = int(parameters["console_log_freq"])
    if console_log_freq <= 0:
        return False, "Console log frequency must be greater than 0. "
    
    # checkpoint freq check
    checkpoint_freq = int(parameters["checkpoint_freq"])
    if checkpoint_freq <= 0:
        return False, "Checkpoint frequency must be greter than 0. "

    # aggregator check
    aggregator = parameters["aggregator"]
    if aggregator != "mean" and aggregator != "LSTM" and aggregator != "pooling":
        return False, "Aggregator must be one of the following: mean, pooling or LSTM. "

    return True, "OK"

    


