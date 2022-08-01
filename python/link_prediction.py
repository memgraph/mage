import mgp  # Python API
import torch
import dgl  # geometric deep learning
from typing import List, Tuple, Dict
from numpy import int32
from dataclasses import dataclass, field
from mage.link_prediction import preprocess, inner_train, inner_predict, create_model, create_optimizer, create_predictor, search_vertex, get_number_of_edges

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
    :param attn_num_heads: List[int] -> GAT can support usage of more than one head in each layers except last one. Only used in GAT, not in GraphSage.
    :param tr_acc_patience: int -> Training patience, for how many epoch will accuracy drop on test set be tolerated before stopping the training. 
    :param model_save_path: str -> Path where the link prediction model will be saved every checkpoint_freq epochs.
    """
    hidden_features_size: List = field(default_factory=lambda: [1433, 128, 64, 32, 16])  # Cannot add typing because of the way Python is implemented(no default things in dataclass, list is immutable something like this)
    layer_type: str = "graph_sage"
    num_epochs: int = 100
    optimizer: str = "SGD"
    learning_rate: float = 0.4
    split_ratio: float = 0.8
    node_features_property: str = "features"
    node_id_property: str = "id"
    device_type: str = "cpu"
    console_log_freq: int = 1
    checkpoint_freq: int = 10
    aggregator: str = "pool"
    metrics: List = field(default_factory=lambda: ["loss", "accuracy", "auc_score", "precision", "recall", "f1", "num_wrong_examples"])
    predictor_type: str = "mlp"
    attn_num_heads: List[int] = field(default_factory=lambda: [8, 8, 1])
    tr_acc_patience: int = 8
    model_save_path: str = "/home/andi/Memgraph/code/mage/python/mage/link_prediction/model.pt"  # TODO: When the development finishes

##############################
# global parameters
##############################


link_prediction_parameters: LinkPredictionParameters = LinkPredictionParameters()  # parameters currently saved.
training_results: List[Dict[str, float]] = list()  # List of all output training records. String is the metric's name and float represents value.
validation_results: List[Dict[str, float]] = list()  # List of all output validation results. String is the metric's name and float represents value in the Dictionary inside.
graph: dgl.graph = None  # Reference to the graph. This includes training and validation.
new_to_old: Dict[int, int] = None  # Mapping of DGL indexes to original dataset indexes
old_to_new: Dict[int, int] = None  # Mapping of original dataset indexes to DGL indexes
predictor: torch.nn.Module = None  # Predictor for calculating edge scores
model: torch.nn.Module = None

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
        attn_num_heads: List[int] -> GAT can support usage of more than one head in each layer except last one. Only used in GAT, not in GraphSage.
        tr_acc_patience: int -> Training patience, for how many epoch will accuracy drop on test set be tolerated before stopping the training. 
        model_save_path: str -> Path where the link prediction model will be saved every checkpoint_freq epochs. 


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
        if not hasattr(link_prediction_parameters, key):
            return mgp.Record(status=0, message="No attribute " + key + " in class LinkPredictionParameters")
        try:
            setattr(link_prediction_parameters, key, value)
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
def train(ctx: mgp.ProcCtx) -> mgp.Record(status=str, training_results=mgp.Any, validation_results=mgp.Any):
    """ Train method is used for training the module on the dataset provided with ctx. By taking decision to split the dataset here and not in the separate method, it is impossible to retrain the same model. 

    Args:
        ctx (mgp.ProcCtx, optional): Reference to the process execution.

    Returns:
        mgp.Record: It returns performance metrics obtained during the training on the training and validation dataset.
    """
    # Get global context
    global training_results, validation_results, predictor, model, graph, new_to_old, old_to_new

    # Reset parameters of the old training
    _reset_train_predict_parameters()

    # Check if the dataset is empty. E2E handling.
    if len(ctx.graph.vertices) == 0:
        return mgp.Record(status="Empty dataset. ", training_results=training_results, validation_results=validation_results)

    # Get some
    graph, new_to_old, old_to_new = _get_dgl_graph_data(ctx)  # dgl representation of the graph and dict new to old index

    # Check if there are no edges in the dataset, assume that it cannot learn effectively without edges. E2E handling.
    if graph.number_of_edges() == 0:
        return mgp.Record(status="No edges in the dataset. ", training_results=training_results, validation_results=validation_results)

    # TEST: Currently disabled
    """ if conversion_to_dgl_test(graph=graph, new_to_old=new_to_old, ctx=ctx, node_id_property=link_prediction_parameters.node_id_property, node_features_property=link_prediction_parameters.node_features_property) is False:
        print("Remapping failed")
        return mgp.Record(status="Preprocessing failed", metrics=[])
     """

    """ Train g is a graph which has removed test edges. Others are positive and negative train and test graphs
    """

    train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g = preprocess(graph, link_prediction_parameters.split_ratio)
    if train_g is None or train_pos_g is None or train_neg_g is None or val_pos_g is None or val_neg_g is None:
        print("Preprocessing failed. ")
        return mgp.Record(status="Preprocessing failed", metrics=[])
 
    # Create a model
    model = create_model(layer_type=link_prediction_parameters.layer_type, hidden_features_size=link_prediction_parameters.hidden_features_size,
                         aggregator=link_prediction_parameters.aggregator, attn_num_heads=link_prediction_parameters.attn_num_heads)
    # Create a predictor
    predictor = create_predictor(predictor_type=link_prediction_parameters.predictor_type, predictor_hidden_size=link_prediction_parameters.hidden_features_size[-1])

    # Create an optimizer
    optimizer = create_optimizer(optimizer_type=link_prediction_parameters.optimizer, learning_rate=link_prediction_parameters.learning_rate,
                                 model=model, predictor=predictor)

    # Collect training and validation results from utils model
    training_results, validation_results = inner_train(model, predictor, optimizer,
                                                       link_prediction_parameters.num_epochs, link_prediction_parameters.node_features_property,
                                                       link_prediction_parameters.console_log_freq, link_prediction_parameters.checkpoint_freq,
                                                       link_prediction_parameters.metrics, link_prediction_parameters.tr_acc_patience, link_prediction_parameters.model_save_path,
                                                       train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g)

    return mgp.Record(status="OK", training_results=training_results, validation_results=validation_results)


@mgp.read_proc
def predict(ctx: mgp.ProcCtx, src_vertex: mgp.Vertex, dest_vertex: mgp.Vertex) -> mgp.Record(score=mgp.Number):
    """Predict method. We assume here semi-inductive learning process where queried nodes are somehow connected to the original graph. It is assumed that nodes are already added to the original graph and our goal
    is to predict whether there is an edge between two nodes or not. Even if the edge exists, method can be used. 


    Args:
        ctx (mgp.ProcCtx): A reference to the context execution

    Returns:
        score: Probability that two nodes are connected
    """
    global graph, predictor, model

    # Create dgl graph representation
    src_old_id = int(src_vertex.properties.get(link_prediction_parameters.node_id_property))
    dest_old_id = int(dest_vertex.properties.get(link_prediction_parameters.node_id_property))

    # Get dgl ids
    src_id = old_to_new[src_old_id]
    dest_id = old_to_new[dest_old_id]

    edge_added, edge_id = False, -1

    print("Number of edges before: ", graph.number_of_edges())

    # Check if there is an edge between two nodes
    if graph.has_edges_between(src_id, dest_id) is True:
        print("Nodes {} and {} are already connected. ".format(src_old_id, dest_old_id))
        edge_id = graph.edge_ids(src_id, dest_id)
    else:
        edge_added = True
        print("Nodes {} and {} are not connected. ".format(src_old_id, dest_old_id))
        graph.add_edges(src_id, dest_id)
        edge_id = graph.edge_ids(src_id, dest_id)

    print("Edge id: ", edge_id)
    print("Number of edges after adding new edge: ", graph.number_of_edges())
    # Call utils module
    score = inner_predict(model=model, predictor=predictor, graph=graph, node_features_property=link_prediction_parameters.node_features_property, edge_id=edge_id)
    result = mgp.Record(score=score)

    # Remove edge if necessary
    if edge_added is True:
        graph.remove_edges(edge_id)

    print("Number of edges after: ", graph.number_of_edges())

    return result


@mgp.read_proc
def benchmark(ctx: mgp.ProcCtx, num_runs: int) -> mgp.Record(status=str, test_results=mgp.Any):
    """Benchmark method runs train method on different seeds for num_runs times to get as much as possible accurate results. 

    Args:
        ctx (mgp.ProcCtx): A reference to the context execution. 
        num_runs (int): Number of runs.

    Returns:
        mgp.Record:
            status (str): Status message
            test_results: Average test results obtained after training for num_runs times. 
    """
    # NOTE: THIS SHOULD BE HIDDEN FOR OUT USERS, BUT ENABLE THIS IN THE DEVELOPMENT SUPPORT

    # Get the global context
    global training_results, validation_results, predictor, model
    # Reset parameters obtained from the training
    _reset_train_predict_parameters()

    # Get DGL graph data representation
    graph, new_to_old, _ = _get_dgl_graph_data(ctx)  # dgl representation of the graph and dict new to old index
    """ if test_conversion(graph=graph, new_to_old=new_to_old, ctx=ctx, node_id_property=link_prediction_parameters.node_id_property, node_features_property=link_prediction_parameters.node_features_property) is False:
        print("Remapping failed")
        return mgp.Record(status="Preprocessing failed", metrics=[])
    """

    """ Train g is a graph which has removed test edges. Others are positive and negative train and test graphs
    """

    # Split the data
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = preprocess(graph, link_prediction_parameters.split_ratio)
    if train_g is None or train_pos_g is None or train_neg_g is None or test_pos_g is None or test_neg_g is None:
        print("Preprocessing failed. ")
        return mgp.Record(status="Preprocessing failed", metrics=[])

    test_results = benchmark.get_avg_seed_results(num_runs, link_prediction_parameters.hidden_features_size, link_prediction_parameters.layer_type,
                                                  link_prediction_parameters.num_epochs, link_prediction_parameters.optimizer, link_prediction_parameters.learning_rate,
                                                  link_prediction_parameters.node_features_property, link_prediction_parameters.console_log_freq, link_prediction_parameters.checkpoint_freq,
                                                  link_prediction_parameters.aggregator, link_prediction_parameters.metrics, link_prediction_parameters.predictor_type, link_prediction_parameters.hidden_features_size[-1],
                                                  link_prediction_parameters.attn_num_heads, link_prediction_parameters.tr_acc_patience, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g)

    return mgp.Record(status="OK", test_results=test_results)


@mgp.read_proc
def get_training_results(ctx: mgp.ProcCtx) -> mgp.Record(tr_metrics=mgp.Any, val_metrics=mgp.Any):
    """This method is used when user wants to get performance data obtained from the last training. It is in the form of list of records where each record is a Dict[metric_name, metric_value]. Training and validation
    results are returned. 

    Args:
        ctx (mgp.ProcCtx): Reference to the context execution

    Returns:
        mgp.Record[List[LinkPredictionOutputResult]]: A list of LinkPredictionOutputResults. 
    """
    return mgp.Record(tr_metrics=training_results, val_metrics=validation_results)

##############################
# Convert to DGL graph, consider extracting such methods to another file.
##############################


def _process_vertex_help_func(old_index: int, nodes_list: List[int], features: List, ind: int,
                              old_to_new: Dict[int, int], new_to_old: Dict[int, int], node_features: List) -> int:
    """Helper function to avoid code duplication in _get_dgl_graph_data.

    Args:
        old_index (int): Index from context execution. 
        nodes_list (List[int]): src_nodes or dest_nodes. See _get_dgl_graph_data for more details. 
        features (List): List of features. See _get_dgl_graph_data for more details.
        ind (int): ind from _get_dgl_graph_data
        old_to_new (Dict[int, int]): Mapping of old to new index.
        new_to_old (Dict[int, int]): Mapping of new to old index. 
        node_features (List): Feature list.

    Returns:
        int: Since list and dict are mutable, function only needs to return new ind. It can stay the same or be incremented by one.
    """
    if old_index not in old_to_new.keys():
        new_to_old[ind] = old_index
        old_to_new[old_index] = ind
        nodes_list.append(ind)
        features.append(node_features)
        return ind+1
    else:
        nodes_list.append(old_to_new[old_index])
        return ind


def _get_dgl_graph_data(ctx: mgp.ProcCtx) -> Tuple[dgl.graph, Dict[int32, int32], Dict[int32, int32]]:
    """Creates dgl representation of the graph.

    Args:
        ctx (mgp.ProcCtx): The reference to the context execution.

    Returns:
        Tuple[dgl.graph, Dict[int32, int32], Dict[int32, int32]]: Tuple of DGL graph representation, dictionary of mapping new to old index and dictionary of mapping old to new index. 
    """
    src_nodes, dest_nodes = [], []  # for saving the edges

    new_to_old = dict()  # map of new node index to old node index
    old_to_new = dict()  # map of old node index to new node index
    features = []  # map of new node index to its feature
    ind = 0

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            # Process source vertex first
            src_node, dest_node = edge.from_vertex, edge.to_vertex
            src_id_old = int(src_node.properties.get(link_prediction_parameters.node_id_property))
            src_features = src_node.properties.get(link_prediction_parameters.node_features_property)
            ind = _process_vertex_help_func(src_id_old, src_nodes, features, ind, old_to_new, new_to_old, src_features)
            
            # Process destination vertex next
            dest_id_old = int(dest_node.properties.get(link_prediction_parameters.node_id_property))
            dest_features = dest_node.properties.get(link_prediction_parameters.node_features_property)
            ind = _process_vertex_help_func(dest_id_old, dest_nodes, features, ind, old_to_new, new_to_old, dest_features)

           
            
    features = torch.tensor(features, dtype=torch.float32)  # use float for storing tensor of features
    # print("Src nodes: ", src_nodes)
    # print("Dest nodes: ", dest_nodes)
    # print("Features: ", features)
    g = dgl.graph((src_nodes, dest_nodes))
    g.ndata[link_prediction_parameters.node_features_property] = features
    # g = dgl.add_self_loop(g) # TODO: How, why what? But needed for GAT, otherwise 0-in-degree nodes:u
    return g, new_to_old, old_to_new


def _validate_user_parameters(parameters: mgp.Map) -> Tuple[bool, str]:
    """Validates parameters user sent through method set_model_parameters

    Args:
        parameters (mgp.Map): Parameters sent by user.

    Returns:
        bool: True if every parameter value is appropriate, False otherwise.
    """
    # Hidden features size
    if "hidden_features_size" in parameters.keys():
        hidden_features_size = parameters["hidden_features_size"]
        for hid_size in hidden_features_size:
            if hid_size <= 0:
                return False, "Layer size must be greater than 0. "

    # Layer type check
    if "layer_type" in parameters.keys():
        layer_type = parameters["layer_type"].lower()
        if layer_type != "graph_attn" and layer_type != "graph_sage":
            return False, "Unknown layer type, this module supports only graph_attn and graph_sage. "

    # Num epochs
    if "num_epochs" in parameters.keys():
        num_epochs = int(parameters["num_epochs"])
        if num_epochs <= 0:
            return False, "Number of epochs must be greater than 0. "

    # Optimizer check
    if "optimizer" in parameters.keys():
        optimizer = parameters["optimizer"].upper()
        if optimizer != "ADAM" and optimizer != "SGD":
            return False, "Unknown optimizer, this module supports only ADAM and SGD. "

    # Learning rate check
    if "learning_rate" in parameters.keys():
        learning_rate = float(parameters["learning_rate"])
        if learning_rate <= 0.0:
            return False, "Learning rate must be greater than 0. "

    # Split ratio check
    if "split_ratio" in parameters.keys():
        split_ratio = float(parameters["split_ratio"])
        if split_ratio <= 0.0:
            return False, "Split ratio must be greater than 0. "

    # node_features_property check
    if "node_features_property" in parameters.keys():
        node_features_property = parameters["node_features_property"]
        if node_features_property == "":
            return False, "You must specify name of nodes' features property. "

    # node_id_property check
    if "node_id_property" in parameters.keys():
        node_id_property = parameters["node_id_property"]
        if node_id_property == "":
            return False, "You must specify name of nodes' id property. "

    # device_type check
    if "device_type" in parameters.keys():
        device_type = parameters["device_type"].lower()
        if device_type != "cpu" and torch.device != "cuda":
            return False, "Only cpu and cuda are supported as devices. "

    # console_log_freq check
    if "console_log_freq" in parameters.keys():
        console_log_freq = int(parameters["console_log_freq"])
        if console_log_freq <= 0:
            return False, "Console log frequency must be greater than 0. "

    # checkpoint freq check
    if "checkpoint_freq" in parameters.keys():
        checkpoint_freq = int(parameters["checkpoint_freq"])
        if checkpoint_freq <= 0:
            return False, "Checkpoint frequency must be greter than 0. "

    # aggregator check
    if "aggregator" in parameters.keys():
        aggregator = parameters["aggregator"].lower()
        if aggregator != "mean" and aggregator != "lstm" and aggregator != "pool" and aggregator != "gcn":
            return False, "Aggregator must be one of the following: mean, pool, lstm or gcn. "

    # metrics check
    if "metrics" in parameters.keys():
        metrics = parameters["metrics"]
        for metric in metrics:
            _metric = metric.lower()
            if _metric != "loss" and _metric != "accuracy" and _metric != "f1" and \
                    _metric != "auc_score" and _metric != "precision" and _metric != "recall" and \
                    _metric != "specificity" and _metric != "num_wrong_examples":
                return False, "Metric name " + _metric + " is not supported!"

    # Predictor type
    if "predictor_type" in parameters.keys():
        predictor_type = parameters["predictor_type"].lower()
        if predictor_type != "dot" and predictor_type != "mlp":
            return False, "Predictor " + predictor_type + " is not supported. "

    # Attention heads
    if "attn_num_heads" in parameters.keys():
        attn_num_heads = parameters["attn_num_heads"]
        if layer_type == "graph_attn":

            if len(attn_num_heads) != len(hidden_features_size) - 1:
                return False, "Specified network with {} layers but given attention heads data for {} layers. ".format(len(hidden_features_size) - 1, len(attn_num_heads))

            if attn_num_heads[-1] != 1:
                return False, "Last GAT layer must contain only one attention head. "

            for num_heads in attn_num_heads:
                if num_heads <= 0:
                    return False, "GAT allows only positive, larger than 0 values for number of attention heads. "

    # Training accuracy patience
    if "tr_acc_patience" in parameters.keys():
        tr_acc_patience = int(parameters["tr_acc_patience"])
        if tr_acc_patience <= 0:
            return False, "Training acc patience flag must be larger than 0."

    # model_save_path
    if "model_save_path" in parameters.keys():
        model_save_path = parameters["model_save_path"]
        if model_save_path == "":
            return False, "Path must be != "" "

    return True, "OK"


def _reset_train_predict_parameters() -> None:
    """Reset global parameters that are returned by train method and used by predict method. 
    """
    global training_results, validation_results, predictor, model, graph, new_to_old, old_to_new
    training_results.clear()  # clear training records from previous training
    validation_results.clear()  # clear validation record from previous training
    predictor = None  # Delete old predictor and create a new one in link_prediction_util.train method\
    model = None  # Annulate old model
    graph = None  # Set graph to None
    old_to_new = None
    new_to_old = None


def conversion_to_dgl_test(graph: dgl.graph, new_to_old: Dict[int, int], ctx: mgp.ProcCtx,
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
        vertex_id = vertex.item()
        print(f"Testing vertex: {vertex_id}")
        old_id = new_to_old[vertex_id]
        vertex = search_vertex(ctx=ctx, id=old_id, node_id_property=node_id_property)
        if vertex is None:
            return False
        old_features = vertex.properties.get(node_features_property)
        if torch.equal(graph.ndata[node_features_property][vertex_id], torch.tensor(old_features, dtype=torch.float32)) is False:
            return False

    # Check number of nodes
    if graph.number_of_nodes() != len(ctx.graph.vertices):
        print("Wrong number of nodes!")
        return False
    # Check number of edges
    if graph.number_of_edges() != get_number_of_edges(ctx):
        print("Wrong number of edges")
        return False

    return True
