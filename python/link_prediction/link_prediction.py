import mgp  # Python API
import torch
import dgl  # geometric deep learning
from typing import List, Tuple, Dict
from tests.processing_tests import test_conversion
from numpy import dtype, int32
from dataclasses import dataclass, field
import link_prediction_util


##############################
# classes and data structures
##############################

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
    :param aggregator: str → Aggregator used in models. Can be one of the following: lstm, pool, mean, gcn. It is only used in graph_sage, not graph_attn
    :param metrics: mgp.List[str] -> Metrics used to evaluate model in training on the test/validation set(we don't use validation set to optimize parameters so everything is test set).
                                Epoch will always be displayed, you can add loss, accuracy, precision, recall, specificity, F1, auc_score etc. 
    :param predictor_type: str -> Type of the predictor. Predictor is used for combining node scores to edge scores. 
    :param predictor_hidden_size: int -> Size of the hidden layer in MLP predictor. It will be used only for the MLP predictor. 
    :param attn_num_heads: List[int] -> GAT can support usage of more than one head in each layers except last one. Only used in GAT, not in GraphSage.

    """
    hidden_features_size: List = field(default_factory=lambda: [1433, 16, 16])  # Cannot add typing because of the way Python is implemented(no default things in dataclass, list is immutable something like this)
    layer_type: str = "graph_attn"
    num_epochs: int = 5
    optimizer: str = "ADAM"
    learning_rate: float = 0.01
    split_ratio: float = 0.8
    node_features_property: str = "features"
    node_id_property: str = "id"
    device_type: str = "cpu" 
    console_log_freq: int = 1
    checkpoint_freq: int = 10
    aggregator: str = "mean"
    metrics: List = field(default_factory=lambda: ["loss", "accuracy", "auc_score", "precision", "recall", "f1", "num_wrong_examples"])
    predictor_type: str = "dot"
    predictor_hidden_size: int = 16
    attn_num_heads: List[int] = field(default_factory=lambda: [3, 1])

##############################
# global parameters
##############################

link_prediction_parameters: LinkPredictionParameters = LinkPredictionParameters()  # parameters currently saved.
training_results: List[Dict[str, float]] = list()  # List of all output records. String is the metric's name and float represents value.
graph: dgl.graph = None # Reference to the graph.
h: torch.Tensor = None # hidden features
predictor: torch.nn.Module = None # Predictor for calculating edge scores

##############################
# All read procedures
##############################

@mgp.read_proc
def set_model_parameters(ctx: mgp.ProcCtx, parameters: mgp.Map) -> mgp.Record(status=mgp.Any, message=str):
    """Saves parameters to the global parameters link_prediction_parameters. Specific parsing is needed because we want enable user to call it with a subset of parameters, no need to send them all. 
    We will use some kind of reflection to most easily update parameters.

    Args:
        ctx (mgp.ProcCtx):  Reference to the context execution.
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
        aggregator: str → Aggregator used in models. Can be one of the following: lstm, pool, mean, gcn. 
        metrics: mgp.List[str] -> Metrics used to evaluate model in training. 
        predictor_type str: Type of the predictor. Predictor is used for combining node scores to edge scores. 
        predictor_hidden_size: int -> Size of the hidden layer in MLP predictor. It will be used only for the MLP predictor. 
        attn_num_heads: List[int] -> GAT can support usage of more than one head in each layer except last one. Only used in GAT, not in GraphSage.

    Returns:
        mgp.Record:
            status (bool): True if everything went OK, False otherwise.
    """    
    global link_prediction_parameters

    print("START")
    print(link_prediction_parameters)

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

    print("END")
    print(link_prediction_parameters)
    
    return mgp.Record(status=1, message="OK")


@mgp.read_proc
def train(ctx: mgp.ProcCtx) -> mgp.Record(status=str, metrics=mgp.Any):
    """ Train method is used for training the module on the dataset provided with ctx. By taking decision to split the dataset here and not in the separate method, it is impossible to retrain the same model. 

    Args:
        ctx (mgp.ProcCtx, optional): Reference to the process execution.

    Returns:
        mgp.Record: It returns performance metrics obtained during the training.
    """
    global training_results, predictor, h
    training_results.clear() # clear records from previous training
    predictor = None # Delete old predictor and create a new one in link_prediction_util.train method\
    h = None # delete old hidden features
    
    graph, new_to_old = _get_dgl_graph_data(ctx)  # dgl representation of the graph and dict new to old index
    """ if test_conversion(graph=graph, new_to_old=new_to_old, ctx=ctx, node_id_property=link_prediction_parameters.node_id_property, node_features_property=link_prediction_parameters.node_features_property) is False:
        print("Remapping failed")
        return mgp.Record(status="Preprocessing failed", metrics=[])
    """
 
    """ Train g is a graph which has removed test edges. Others are positive and negative train and test graphs
    """
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = link_prediction_util.preprocess(graph, link_prediction_parameters.split_ratio)
    if train_g is None or train_pos_g is None or train_neg_g is None or test_pos_g is None or test_neg_g is None:
        print("Preprocessing failed. ")
        return mgp.Record(status="Preprocessing failed", metrics=[])


    training_results, h, predictor = link_prediction_util.train(link_prediction_parameters.hidden_features_size, link_prediction_parameters.layer_type,
        link_prediction_parameters.num_epochs, link_prediction_parameters.optimizer, link_prediction_parameters.learning_rate,
        link_prediction_parameters.node_features_property, link_prediction_parameters.console_log_freq, link_prediction_parameters.checkpoint_freq,
        link_prediction_parameters.aggregator, link_prediction_parameters.metrics, link_prediction_parameters.predictor_type, link_prediction_parameters.predictor_hidden_size,
        link_prediction_parameters.attn_num_heads, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g)


    return mgp.Record(status="OK", metrics=training_results)


@mgp.read_proc
def predict(ctx: mgp.ProcCtx, src_vertex: mgp.Vertex, dest_vertex: mgp.Vertex) -> mgp.Record(score=mgp.Number):
    """Predicts edge score determined by source and destination vertex.

    Args:
        ctx (mgp.Record): The reference to the context execution.
        src_vertex (mgp.Vertex): Source vertex.
        dest_vertex (mgp.Vertex) Destination vertex.

    Returns:
        mgp.Record: A score between 0 and 1.
    """
    global predictor, h

    src_features = src_vertex.properties.get(link_prediction_parameters.node_features_property)
    dest_features = dest_vertex.properties.get(link_prediction_parameters.node_features_property)
    features = []
    features.append(src_features)
    features.append(dest_features)
    features = torch.zeros((2708, 1433), dtype=torch.float32) # TODO: Cache this values from training. 
    features[0, :] = torch.FloatTensor(src_features)
    features[1, :] = torch.FloatTensor(dest_features)
    # Create graph representation of this edge
    graph = dgl.graph(([], []), num_nodes=2708)    
    graph.add_edges(0, 1)
    graph.ndata[link_prediction_parameters.node_features_property] = features

    # Call utils module
    edge_probabilities = link_prediction_util.predict(h=h, predictor=predictor, graph=graph)

    result = mgp.Record(score=edge_probabilities[0].item())
    return result


@mgp.read_proc
def get_training_results(ctx: mgp.ProcCtx) -> mgp.Record(metrics=mgp.Any):
    """This method is used when user wants to get performance data obtained from the last training. It is in the form of list of records where each record is a Dict[metric_name, metric_value].

    Args:
        ctx (mgp.ProcCtx): Reference to the context execution

    Returns:
        mgp.Record[List[LinkPredictionOutputResult]]: A list of LinkPredictionOutputResults. 
    """
    return mgp.Record(metrics=training_results)

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
    features = torch.tensor(features, dtype=torch.float32)  # use float for storing tensor of features
    g = dgl.graph((src_nodes, dest_nodes))
    g.ndata[link_prediction_parameters.node_features_property] = features
    # g = dgl.add_self_loop(g) # TODO: How, why what? But needed for GAT, otherwise 0-in-degree nodes:u
    return g, new_to_old


def _validate_user_parameters(parameters: mgp.Map) -> Tuple[bool, str]:
    """Validates parameters user sent through method set_model_parameters

    Args:
        parameters (mgp.Map): Parameters sent by user.

    Returns:
        bool: True if every parameter value is appropriate, False otherwise.
    """
    # Hidden features size
    hidden_features_size = parameters["hidden_features_size"]
    for hid_size in hidden_features_size:
        if hid_size <= 0:
            return False, "Layer size must be greater than 0. "

    # Layer type check
    layer_type = parameters["layer_type"].lower()
    if layer_type != "graph_attn" and layer_type != "graph_sage":
        return False, "Unknown layer type, this module supports only graph_attn and graph_sage. "

    # Num epochs
    num_epochs = int(parameters["num_epochs"])
    if num_epochs <= 0:
        return False, "Number of epochs must be greater than 0. "

    # Optimizer check
    optimizer = parameters["optimizer"].upper()
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
    device_type = parameters["device_type"].lower()
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
    aggregator = parameters["aggregator"].lower()
    if aggregator != "mean" and aggregator != "lstm" and aggregator != "pool" and aggregator != "gcn":
        return False, "Aggregator must be one of the following: mean, pool, lstm or gcn. "

    # metrics check
    metrics = parameters["metrics"]
    for metric in metrics:
        _metric = metric.lower()
        if _metric != "loss" and _metric != "accuracy" and _metric != "f1" and \
            _metric != "auc_score" and _metric != "precision" and _metric != "recall" and \
            _metric != "specificity" and _metric != "num_wrong_examples":
            return False, "Metric name " + _metric + " is not supported!"
            
    # Predictor type
    predictor_type = parameters["predictor_type"].lower()
    if predictor_type != "dot" and predictor_type != "mlp":
        return False, "Predictor " + predictor_type + " is not supported. "

    # Predictor hidden size
    predictor_hidden_size = int(parameters["predictor_hidden_size"])
    if predictor_type == "mlp" and predictor_hidden_size <= 0:
        return False, "Size of the hidden layer must be greater than 0 when using MLPPredictor. "

    # Attention heads
    attn_num_heads = parameters["attn_num_heads"]
    if layer_type == "graph_attn":

        if len(attn_num_heads) != len(hidden_features_size) - 1:
            return False, "Specified network with {} layers but given attention heads data for {} layers. ".format(len(hidden_features_size) -1, len(attn_num_heads))

        if attn_num_heads[-1] != 1:
            return False, "Last GAT layer must contain only one attention head. "

        for num_heads in attn_num_heads:
            if num_heads <= 0:
                return False, "GAT allows only positive, larger than 0 values for number of attention heads. "

    return True, "OK"

