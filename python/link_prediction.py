import mgp  # Python API
import torch
import dgl  # geometric deep learning
from typing import Callable, List, Tuple, Dict
from numpy import int32
from dataclasses import dataclass, field
from collections import defaultdict
from heapq import heappop, heappush 
from mage.link_prediction import (
    preprocess,
    inner_train,
    inner_predict,
    create_model,
    create_optimizer,
    create_predictor,
    GRAPH_SAGE,
    GRAPH_ATTN,
    ADAM_OPT,
    SGD_OPT,
    CUDA_DEVICE,
    CPU_DEVICE,
    DOT_PREDICTOR,
    MLP_PREDICTOR,
    MEAN_AGG,
    LSTM_AGG,
    POOL_AGG,
    GCN_AGG,
    HIDDEN_FEATURES_SIZE,
    LAYER_TYPE,
    NUM_EPOCHS,
    OPTIMIZER,
    LEARNING_RATE,
    SPLIT_RATIO,
    NODE_FEATURES_PROPERTY,
    DEVICE_TYPE,
    CONSOLE_LOG_FREQ,
    CHECKPOINT_FREQ,
    AGGREGATOR,
    METRICS,
    LAYER_TYPE,
    PREDICTOR_TYPE,
    ATTN_NUM_HEADS,
    TR_ACC_PATIENCE,
    MODEL_SAVE_PATH,
    CONTEXT_SAVE_DIR,
    TARGET_RELATION,
    NUM_NEG_PER_POS_EDGE,
    BATCH_SIZE,
    SAMPLING_WORKERS,
    MODEL_NAME,
    PREDICTOR_NAME,
    AUC_SCORE,
    ACCURACY,
    PRECISION,
    RECALL,
    POS_EXAMPLES,
    NEG_EXAMPLES,
    POS_PRED_EXAMPLES,
    NEG_PRED_EXAMPLES,
    LOSS,
    F1,
)
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
    :param device_type: str ->  If model will be trained using CPU or cuda GPU. Possible values are cpu and cuda. To run it on Cuda, user must set this flag to true and system must support cuda execution.
        System's support is checked with torch.cuda.is_available()
    :param console_log_freq: int ->  how often do you want to print results from your model? Results will be from validation dataset.
    :param checkpoint_freq: int → Select the number of epochs on which the model will be saved. The model is persisted on disc.
    :param aggregator: str → Aggregator used in models. Can be one of the following: lstm, pool, mean, gcn. It is only used in graph_sage, not graph_attn
    :param metrics: mgp.List[str] -> Metrics used to evaluate model in training on the test/validation set(we don't use validation set to optimize parameters so everything is test set).
        Epoch will always be displayed, you can add loss, accuracy, precision, recall, specificity, F1, auc_score etc.
    :param predictor_type: str -> Type of the predictor. Predictor is used for combining node scores to edge scores.
    :param attn_num_heads: List[int] -> GAT can support usage of more than one head in each layers except last one. Only used in GAT, not in GraphSage.
    :param tr_acc_patience: int -> Training patience, for how many epoch will accuracy drop on validation set be tolerated before stopping the training.
    :param context_save_dir: str -> Path where the model and predictor will be saved every checkpoint_freq epochs.
    :param target_relation: str -> Unique edge type that is used for training.
    :param num_neg_per_pos_edge (int): Number of negative edges that will be sampled per one positive edge in the mini-batch.
    :param num_layers (int): Number of layers in the GNN architecture.
    :param batch_size (int): Batch size used in both training and validation procedure.
    :param sampling_workers (int): Number of workers that will cooperate in the sampling procedure in the training and validation.

    """

    hidden_features_size: List = field(
        default_factory=lambda: [20, 10]
    )  # Cannot add typing because of the way Python is implemented(no default things in dataclass, list is immutable something like this)
    layer_type: str = GRAPH_SAGE
    num_epochs: int = 10
    optimizer: str = ADAM_OPT
    learning_rate: float = 0.01
    split_ratio: float = 0.8
    node_features_property: str = "features"
    device_type: str = CPU_DEVICE
    console_log_freq: int = 1
    checkpoint_freq: int = 10
    aggregator: str = POOL_AGG
    metrics: List = field(
        default_factory=lambda: [
            LOSS,
            ACCURACY,
            AUC_SCORE,
            PRECISION,
            RECALL,
            F1,
            POS_EXAMPLES,
            NEG_EXAMPLES,
            POS_PRED_EXAMPLES,
            NEG_PRED_EXAMPLES
        ]
    )
    predictor_type: str =  MLP_PREDICTOR
    attn_num_heads: List[int] = field(default_factory=lambda: [4, 1])
    tr_acc_patience: int = 5
    context_save_dir: str = "./python/mage/link_prediction/context/"  # TODO: When the development finishes
    target_relation: str = None
    num_neg_per_pos_edge: int = 5
    batch_size: int = 512
    sampling_workers: int = 4

##############################
# global parameters
##############################

link_prediction_parameters: LinkPredictionParameters = LinkPredictionParameters()  # parameters currently saved.
training_results: List[Dict[str, float]] = list()  # List of all output training records. String is the metric's name and float represents value.
validation_results: List[Dict[str, float]] = (list())  # List of all output validation results. String is the metric's name and float represents value in the Dictionary inside.
graph: dgl.graph = None  # Reference to the graph. This includes training and validation.
reindex_dgl: Dict[int, int] = None  # Mapping of DGL indexes to original dataset indexes
reindex_orig: Dict[int, int] = None  # Mapping of original dataset indexes to DGL indexes
predictor: torch.nn.Module = None  # Predictor for calculating edge scores
model: torch.nn.Module = None
features_size_loaded: bool = False  # If size of the features was already inserted.
HETERO = None 
labels_concat = ":"  # string to separate labels if dealing with multiple labels per node
self_loop_edge_type = "MEMGRAPH_SELF_LOOP"


# Lambda function to concat list of labels
merge_labels: Callable[[List[mgp.Label]], str] = lambda labels: labels_concat.join([label.name for label in labels])


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
        device_type: str ->  If model will be trained using CPU or cuda GPU. Possible values are cpu and cuda. To run it on Cuda, user must set this flag to true and system must support cuda execution.
                                System's support is checked with torch.cuda.is_available()
        console_log_freq: int ->  how often do you want to print results from your model? Results will be from validation dataset.
        checkpoint_freq: int → Select the number of epochs on which the model will be saved. The model is persisted on disc.
        aggregator: str → Aggregator used in models. Can be one of the following: lstm, pool, mean, gcn.
        metrics: mgp.List[str] -> Metrics used to evaluate model in training.
        predictor_type str: Type of the predictor. Predictor is used for combining node scores to edge scores.
        attn_num_heads: List[int] -> GAT can support usage of more than one head in each layer except last one. Only used in GAT, not in GraphSage.
        tr_acc_patience: int -> Training patience, for how many epoch will accuracy drop on test set be tolerated before stopping the training.
        context_save_dir: str -> Path where the model and predictor will be saved every checkpoint_freq epochs.
        target_relation: str -> Unique edge type that is used for training.
        num_neg_per_pos_edge (int): Number of negative edges that will be sampled per one positive edge in the mini-batch.
        num_layers (int): Number of layers in the GNN architecture.
        batch_size (int): Batch size used in both training and validation procedure.
        sampling_workers (int): Number of workers that will cooperate in the sampling procedure in the training and validation.

    Returns:
        mgp.Record:
            status (bool): True if parameters were successfully updated, False otherwise.
            message(str): Additional explanation why method failed or OK otherwise.
    """
    global link_prediction_parameters

    _validate_user_parameters(parameters=parameters)
    
    for key, value in parameters.items():
        if not hasattr(link_prediction_parameters, key):
            return mgp.Record(
                status=1,
                message="No attribute " + key + " in class LinkPredictionParameters",
            )
        try:
            setattr(link_prediction_parameters, key, value)
        except Exception as exception:
            return mgp.Record(status=1, message=repr(exception))

    # Device type handling
    if link_prediction_parameters.device_type == CUDA_DEVICE and torch.cuda.is_available() is True:
        link_prediction_parameters.device_type = CUDA_DEVICE
    else:
        link_prediction_parameters.device_type = CPU_DEVICE

    # Lists handling=generator expression + unpacking
    if type(link_prediction_parameters.hidden_features_size) == tuple:
        link_prediction_parameters.hidden_features_size = [*(x for x in link_prediction_parameters.hidden_features_size)]

    if type(link_prediction_parameters.attn_num_heads) == tuple:
        link_prediction_parameters.attn_num_heads = [*(x for x in link_prediction_parameters.attn_num_heads)]

    return mgp.Record(status=1, message="OK")

@mgp.read_proc
def train(ctx: mgp.ProcCtx,) -> mgp.Record(training_results=mgp.Any, validation_results=mgp.Any):
    """Train method is used for training the module on the dataset provided with ctx. By taking decision to split the dataset here and not in the separate method, it is impossible to retrain the same model.

    Args:
        ctx (mgp.ProcCtx, optional): Reference to the process execution.

    Returns:
        mgp.Record: It returns performance metrics obtained during the training on the training and validation dataset.
    """
    # Get global context
    global training_results, validation_results, predictor, model, graph, reindex_dgl, reindex_orig, features_size_loaded, HETERO

    # Reset parameters of the old training
    _reset_train_predict_parameters()

    # Get some data
    # Dealing with heterogeneous graphs
    graph, reindex_dgl, reindex_orig = _get_dgl_graph_data(ctx)  # dgl representation of the graph and dict new to old index
  
    # Insert in the hidden_features_size structure if needed
    if not features_size_loaded:
        if HETERO:
            ftr_size = max(value.shape[1] for value in graph.ndata[link_prediction_parameters.node_features_property].values())
        else:
            ftr_size = graph.ndata[link_prediction_parameters.node_features_property].shape[1]


        # Load feature size in the hidden_features_size 
        _load_feature_size(ftr_size)

    # Split the data
    train_eid_dict, val_eid_dict = preprocess(graph=graph, split_ratio=link_prediction_parameters.split_ratio, 
        target_relation=link_prediction_parameters.target_relation)

    # Create a model
    model = create_model(
        layer_type=link_prediction_parameters.layer_type,
        hidden_features_size=link_prediction_parameters.hidden_features_size,
        aggregator=link_prediction_parameters.aggregator,
        attn_num_heads=link_prediction_parameters.attn_num_heads,
        edge_types=graph.etypes
    )

    # Create a predictor
    predictor = create_predictor(
        predictor_type=link_prediction_parameters.predictor_type,
        predictor_hidden_size=link_prediction_parameters.hidden_features_size[-1],
    )

    # Create an optimizer
    optimizer = create_optimizer(
        optimizer_type=link_prediction_parameters.optimizer,
        learning_rate=link_prediction_parameters.learning_rate,
        model=model,
        predictor=predictor,
    )

    # Call training method
    training_results, validation_results = inner_train(graph,
        train_eid_dict, 
        val_eid_dict, 
        link_prediction_parameters.target_relation,
        model,
        predictor,
        optimizer,
        link_prediction_parameters.num_epochs,
        link_prediction_parameters.node_features_property,
        link_prediction_parameters.console_log_freq,
        link_prediction_parameters.checkpoint_freq,
        link_prediction_parameters.metrics,
        link_prediction_parameters.tr_acc_patience,
        link_prediction_parameters.context_save_dir,
        link_prediction_parameters.num_neg_per_pos_edge,
        len(link_prediction_parameters.hidden_features_size) -1,
        link_prediction_parameters.batch_size,
        link_prediction_parameters.sampling_workers
    )
    
    # Return results
    return mgp.Record(training_results=training_results, validation_results=validation_results)

@mgp.read_proc
def predict(ctx: mgp.ProcCtx, src_vertex: mgp.Vertex, dest_vertex: mgp.Vertex) -> mgp.Record(score=mgp.Number):
    """Predict method. It is assumed that nodes are added to the original Memgraph graph. It supports both situations, when the edge doesn't exist and when
    the edge exists.

    Args:
        ctx (mgp.ProcCtx): A reference to the context execution
        src_vertex (mgp.Vertex): Source vertex.
        dest_vertex (mgp.Vertex): Destination vertex.

    Returns:
        score: Probability that two nodes are connected
    """
    # TODO: what if new node is created after training and before predict? You have to map it dgl.graph representation immediately. 
    global graph, predictor, model, reindex_orig, reindex_dgl, HETERO, features_size_loaded

    # If the model isn't available. Model is available if this method is called right after training or loaded from context.
    # Same goes for predictor.
    if model is None or predictor is None:
        raise Exception("No trained model available to the system. Train or load it first. ")

    # Load graph again so you find nodes that were possibly added between train and prediction
    graph, reindex_dgl, reindex_orig = _get_dgl_graph_data(ctx)  # dgl representation of the graph and dict new to old index

    # Create dgl graph representation
    src_old_id, src_type = src_vertex.id, merge_labels(src_vertex.labels)
    dest_old_id, dest_type = dest_vertex.id, merge_labels(dest_vertex.labels)

    # Get dgl ids
    src_id = reindex_orig[src_type][src_old_id]
    dest_id = reindex_orig[dest_type][dest_old_id]
    
    # Init edge properties
    edge_added, edge_id = False, -1

    # Check if there is an edge between two nodes
    if not graph.has_edges_between(src_id, dest_id, etype=link_prediction_parameters.target_relation):
        edge_added = True
        # print("Nodes {} and {} are not connected. ".format(src_old_id, dest_old_id))
        graph.add_edges(src_id, dest_id, etype=link_prediction_parameters.target_relation)
    
    edge_id = graph.edge_ids(src_id, dest_id, etype=link_prediction_parameters.target_relation)

     # Insert in the hidden_features_size structure if needed and it is needed only if the session was lost between training and predict method call.
    if not features_size_loaded:
        if HETERO:
            ftr_size = max(value.shape[1] for value in graph.ndata[link_prediction_parameters.node_features_property].values())
        else:
            ftr_size = graph.ndata[link_prediction_parameters.node_features_property].shape[1]

        # Load feature size in the hidden_features_size 
        _load_feature_size(ftr_size)

    # Call utils module
    score = inner_predict(
        model=model, 
        predictor=predictor, 
        graph=graph, 
        node_features_property=link_prediction_parameters.node_features_property,
        src_node=src_id, 
        dest_node=dest_id,
        src_type=src_type,
        dest_type=dest_type,)

    result = mgp.Record(score=score)

    # Remove edge if necessary
    if edge_added:
        graph.remove_edges(edge_id, etype=link_prediction_parameters.target_relation)

    # print("Number of edges after: ", graph.number_of_edges())

    return result

@mgp.read_proc
def recommend(ctx: mgp.ProcCtx, src_vertex: mgp.Vertex, dest_vertices: mgp.List[mgp.Vertex], k: int) -> mgp.Record(score=mgp.Number, recommendation=mgp.Vertex):
    """Recommend method. It is assumed that nodes are already added to the original graph and our goal is to predict whether there is an edge between two nodes or not. Even if the edge exists,
     method can be used. Recommends k nodes based on edge scores. 


    Args:
        ctx (mgp.ProcCtx): A reference to the context execution
        src_vertex (mgp.Vertex): Source vertex.
        dest_vertex (mgp.Vertex): Destination vertex.

    Returns:
        score: Probability that two nodes are connected
    """
    global graph, predictor, model, reindex_orig, reindex_dgl, HETERO

    print(f"Dest vertices: {len(dest_vertices)}")

    # If the model isn't available
    if model is None:
        raise Exception("No trained model available to the system. Train or load it first. ")

    # You called predict after session was lost
    graph, reindex_dgl, reindex_orig = _get_dgl_graph_data(ctx)

     # Insert in the hidden_features_size structure if needed and it is needed only if the session was lost between training and predict method call.
    if not features_size_loaded:
        if HETERO:
            ftr_size = max(value.shape[1] for value in graph.ndata[link_prediction_parameters.node_features_property].values())
        else:
            ftr_size = graph.ndata[link_prediction_parameters.node_features_property].shape[1]

        # Load feature size in the hidden_features_size 
        _load_feature_size(ftr_size)
    
    # Create dgl graph representation
    src_old_id, src_type = src_vertex.id, merge_labels(src_vertex.labels)

    # Get dgl ids
    src_id = reindex_orig[src_type][src_old_id]
    
    # Save edge scores and vertices
    results = []
    for i, dest_vertex in enumerate(dest_vertices):  # TODO: Can be implemented much faster by directly using matrix multiplication.
        # Get dest vertex
        dest_old_id, dest_type = dest_vertex.id, merge_labels(dest_vertex.labels)
        dest_id = reindex_orig[dest_type][dest_old_id]
        
        # Init edge properties
        edge_added, edge_id = False, -1

        # Check if there is an edge between two nodes
        if not graph.has_edges_between(src_id, dest_id, etype=link_prediction_parameters.target_relation):
            edge_added = True
            # print("Nodes {} and {} are not connected. ".format(src_old_id, dest_old_id))
            graph.add_edges(src_id, dest_id, etype=link_prediction_parameters.target_relation)
    
        edge_id = graph.edge_ids(src_id, dest_id, etype=link_prediction_parameters.target_relation)

        # Call utils module
        score = inner_predict(
            model=model, 
            predictor=predictor, 
            graph=graph, 
            node_features_property=link_prediction_parameters.node_features_property,
            src_node=src_id, 
            dest_node=dest_id,
            src_type=src_type,
            dest_type=dest_type,)
    
         # Remove edge if necessary
        if edge_added:
           graph.remove_edges(edge_id, etype=link_prediction_parameters.target_relation)
        
        heappush(results, (-score, i, dest_vertex))  # Build in O(n). Add i to break ties where all predict values are the same.
    
    # Extract recommendations
    recommendations = []
    for i in range(k):
        score, i, recommendation = heappop(results)
        recommendations.append(mgp.Record(score=-score, recommendation=recommendation))

    return recommendations

@mgp.read_proc
def test(ctx: mgp.ProcCtx, src_vertices: List[int], dest_vertices: List[int]) -> mgp.Record(scores=mgp.Any, accuracy=mgp.Number):
    """Offers completely the same functionality as predict.
    Instead of receiving references to the source and destination vertex, it received list of source and destination vertices.
    Closed for public.

    Args:
        ctx (mgp.ProcCtx): A reference to the context execution.
        src_vertices (List[int]): Source vertices for every edge. Based on Memgraph's id property.
        dest_vertices (List[int]): Dest vertices for every edge. Based on Memgraph's id property.

    Returns:
        Scores for every edge and final accuracy.
    """

    scores = []

    for i in range(len(src_vertices)):
        src_vertex = ctx.graph.get_vertex_by_id(src_vertices[i])
        dest_vertex = ctx.graph.get_vertex_by_id(dest_vertices[i])
        record = predict(ctx, src_vertex, dest_vertex)
        scores.append(record.fields["score"])

    acc = torch.sum(torch.tensor(scores, requires_grad=False) > 0.5).item() / len(scores)

    return mgp.Record(scores=scores, accuracy=acc)

@mgp.read_proc
def get_training_results(ctx: mgp.ProcCtx,) -> mgp.Record(training_results=mgp.Any, validation_results=mgp.Any):

    """This method is used when user wants to get performance data obtained from the last training. It is in the form of list of records where each record is a Dict[metric_name, metric_value]. Training and validation
    results are returned.

    Args:
        ctx (mgp.ProcCtx): Reference to the context execution

    Returns:
        mgp.Record[List[LinkPredictionOutputResult]]: A list of results. If the train method wasn't called yet, it returns empty lists.
    """
    global training_results, validation_results

    return mgp.Record(training_results=training_results, validation_results=validation_results)

@mgp.read_proc
def load_context(ctx: mgp.ProcCtx, path: str = link_prediction_parameters.context_save_dir) -> mgp.Record(status=mgp.Any):
    """Loads torch model from given path. If the path doesn't exist, underlying exception is thrown.
    If the path argument is not given, it loads from the default path. If the user has changed path and the context was deleted 
    then he/she needs to send that parameter here.

    Args:
        ctx (mgp.ProcCtx): A reference to the context execution.

    Returns:
        status(mgp.Any): True just to indicate that loading went well.
    """

    global model, predictor
    model = torch.load(path + MODEL_NAME)
    predictor = torch.load(path + PREDICTOR_NAME)
    return mgp.Record(status=True)

##############################
# Private helper methods.
##############################
def _proj_0(graph: dgl.graph) -> None:
    """Performs projection on all node features to the max_feature_size by padding it with 0.

    Args:
        graph (dgl.graph): A reference to the original graph.
    """
    global link_prediction_parameters
    ftr_size_max = 0
    for node_type in graph.ntypes:  # Not costly, iterates only over node types.
        node_type_features = graph.nodes[node_type].data[link_prediction_parameters.node_features_property]
        ftr_size_max = max(ftr_size_max, node_type_features.shape[1])

    for node_type in graph.ntypes:
        p1d = (0, ftr_size_max - graph.nodes[node_type].data[link_prediction_parameters.node_features_property].shape[1])  # Padding left if 0 and padding right is dim_goal - arr.shape[1]
        
        graph.nodes[node_type].data[link_prediction_parameters.node_features_property] = torch.nn.functional.pad(graph.nodes[node_type].data[link_prediction_parameters.node_features_property], 
                    p1d, mode="constant", value=0)


def _load_feature_size(features_size: int):
    """Inserts feature size to the hidden_features_size array.

    Args:
        features_size (int): Features size.
    """
    global link_prediction_parameters, features_size_loaded

    link_prediction_parameters.hidden_features_size.insert(0, features_size)
    features_size_loaded = True

def _process_help_function(mem_indexes: Dict[str, int], old_index: int, type_: str, features: List[int], reindex_orig: Dict[str, Dict[int, int]], reindex_dgl: Dict[str, Dict[int, int]], 
                                  index_dgl_to_features: Dict[str, Dict[int, List[int]]]) -> None:
    """Helper function for mapping original Memgraph graph to DGL representation.

    Args:
        mem_indexes (Dict[str, int]): Saves counters for each node type. 
        old_index (int): Memgraph's node index.
        type_ (str): Node type.
        features (List[int]): Node features.
        reindex_orig (Dict[str, Dict[int, int]]): Mapping from original indexes to DGL indexes for each node type.
        reindex_dgl (Dict[str, Dict[int, int]]): Mapping from DGL indexes to the original indexes for each node type.
        index_dgl_to_features (Dict[str, Dict[int, List[int]]]): DGL indexes to features for each node type.
    """
    if type_ not in reindex_orig.keys():  # Node type not seen before
        reindex_orig[type_] = dict()  # Mapping of old to new indexes for given type_
        reindex_dgl[type_] = dict()   # Mapping of new to old indexes for given type_
        index_dgl_to_features[type_] = dict()  # Mapping of new index to its feature for given type_

    # Check if old_index has been seen for this label
    if old_index not in reindex_orig[type_].keys():
        ind = mem_indexes[type_]  # get current counter
        reindex_dgl[type_][ind] = old_index  # save new_to_old relationship
        reindex_orig[type_][old_index] = ind  # save old_to_new relationship
        # Check if list is given as a string
        if type(features) == str:
            index_dgl_to_features[type_][ind] = eval(features)  # Save new to features relationship. TODO: Remove that when we done with Cypher converting from String to List
        else:
            index_dgl_to_features[type_][ind] = features

        mem_indexes[type_] += 1

def _get_dgl_graph_data(ctx: mgp.ProcCtx,) -> Tuple[dgl.graph, Dict[int32, int32], Dict[int32, int32]]:
    """Creates dgl representation of the graph. It works with heterogeneous and homogeneous.

    Args:
        ctx (mgp.ProcCtx): The reference to the context execution.

    Returns:
        Tuple[dgl.graph, Dict[str, Dict[int32, int32]], Dict[str, Dict[int32, int32]]: Tuple of DGL graph representation, dictionary of mapping new 
        to old index and dictionary of mapping old to new index for each node type.
    """

    global link_prediction_parameters, HETERO

    reindex_dgl = dict()  # map of label to new node index to old node index
    reindex_orig = dict()  # map of label to old node index to new node index
    mem_indexes = defaultdict(int)  # map of label to indexes. All indexes are by default indexed 0.

    type_triplets = []  # list of tuples where each tuple is in following form(src_type, edge_type, dst_type), e.g. ("Customer", "SUBSCRIBES_TO", "Plan")
    index_dgl_to_features = dict()  # dgl indexes to features

    src_nodes, dest_nodes = defaultdict(list), defaultdict(list) # label to node IDs -> Tuple of node-tensors format from DGL

    edge_types = set()

    # Iterate over all vertices
    for vertex in ctx.graph.vertices:
        # Process source vertex
        src_id, src_type, src_features = vertex.id, merge_labels(vertex.labels), vertex.properties.get(link_prediction_parameters.node_features_property)
        
        # Process hetero label stuff
        _process_help_function(mem_indexes, src_id, src_type, src_features, reindex_orig, reindex_dgl, index_dgl_to_features) 
 
        # Get all out edges
        for edge in vertex.out_edges:
            # Get edge information
            edge_type = edge.type.name

            # Process destination vertex next
            dest_node = edge.to_vertex
            dest_id, dest_type, dest_features = dest_node.id, merge_labels(dest_node.labels), dest_node.properties.get(link_prediction_parameters.node_features_property)
 
            # Define type triplet
            type_triplet = (src_type, edge_type, dest_type)
            
            type_triplet_in = type_triplet in type_triplets
            
            # Before processing node dest node and edge, check if this edge_type has occurred with different src_type or dest_type
            if edge_type in edge_types and not type_triplet_in and edge_type == link_prediction_parameters.target_relation:
                raise Exception(f"Edges of edge type {edge_type} are used for training and there are already edges with this edge type but with different combination of source and destination node. ")

            # Add to the type triplets
            if not type_triplet_in:
                type_triplets.append(type_triplet)

            # Add to the edge_types set
            edge_types.add(edge_type)
            
            # Handle mappings
            _process_help_function(mem_indexes, dest_id, dest_type, dest_features, reindex_orig, reindex_dgl, index_dgl_to_features) 

            # Define edge
            src_nodes[type_triplet].append(reindex_orig[src_type][src_id])
            dest_nodes[type_triplet].append(reindex_orig[dest_type][dest_id])
            
            
    # Check if there are no edges in the dataset, assume that it cannot learn effectively without edges. E2E handling.
    if len(src_nodes.keys()) == 0:
        raise Exception("No edges in the dataset. ")
 
    # data_dict has specific type that DGL requires to create a heterograph 
    data_dict = dict()  

    # Create a heterograph
    for type_triplet in type_triplets:
        data_dict[type_triplet] = torch.tensor(src_nodes[type_triplet]), torch.tensor(dest_nodes[type_triplet])

    g = dgl.heterograph(data_dict)  
    
    # Create features
    for node_type in g.ntypes:
        node_features = []
        for node in g.nodes(node_type):
            node_id = node.item()
            node_features.append(index_dgl_to_features[node_type][node_id])

        node_features = torch.tensor(node_features, dtype=torch.float32)
        g.nodes[node_type].data[link_prediction_parameters.node_features_property] = node_features
    
    # Infer target relation
    if len(type_triplets) == 1:  # Only one type of src_type, edge_type and dest_type -> it must be homogeneous graph
        link_prediction_parameters.target_relation = type_triplets[0][1]
        HETERO = False
    else:
        # Target relation needs to be specified by the user
        HETERO = True

    # Test conversion. Note: Do a conversion before you upscale features.
    _conversion_to_dgl_test(graph=g, reindex_dgl=reindex_dgl, ctx=ctx, node_features_property=link_prediction_parameters.node_features_property)

    # Upscale features so they are all of same size
    _proj_0(g) 
    
    return g, reindex_dgl, reindex_orig

def _validate_user_parameters(parameters: mgp.Map) -> None:
    """Validates parameters user sent through method set_model_parameters

    Args:
        parameters (mgp.Map): Parameters sent by user.

    Returns:
        Nothing or raises an exception if something is wrong.
    """
    # Hacky Python
    def raise_(ex):
        raise ex

    # Define lambda type checkers
    type_checker = lambda arg, mess, real_type: None if type(arg) == real_type else raise_(Exception(mess))


    # Hidden features size
    if HIDDEN_FEATURES_SIZE in parameters.keys():
        hidden_features_size = parameters[HIDDEN_FEATURES_SIZE]

        # Because list cannot be sent through mgp.
        type_checker(hidden_features_size, "hidden_features_size not an iterable object. ", tuple)

        for hid_size in hidden_features_size:
            if hid_size <= 0:
                 raise Exception("Layer size must be greater than 0. ")

    # Layer type check
    if LAYER_TYPE in parameters.keys():
        layer_type = parameters[LAYER_TYPE]
        
        # Check typing
        type_checker(layer_type, "layer_type must be string. ", str)

        if layer_type != GRAPH_ATTN and layer_type != GRAPH_SAGE:
             raise Exception("Unknown layer type, this module supports only graph_attn and graph_sage. ")

    # Num epochs
    if NUM_EPOCHS in parameters.keys():
        num_epochs = parameters[NUM_EPOCHS]

        # Check typing
        type_checker(num_epochs, "num_epochs must be int. ", int)

        if num_epochs <= 0:
             raise Exception("Number of epochs must be greater than 0. ")

    # Optimizer check
    if OPTIMIZER in parameters.keys():
        optimizer = parameters[OPTIMIZER]

        # Check typing
        type_checker(optimizer, "optimizer must be a string. ", str)

        if optimizer != ADAM_OPT and optimizer != SGD_OPT:
             raise Exception("Unknown optimizer, this module supports only ADAM and SGD. ")

    # Learning rate check
    if LEARNING_RATE in parameters.keys():
        learning_rate = parameters[LEARNING_RATE]

        # Check typing
        type_checker(learning_rate, "learning rate must be a float. ", float)

        if learning_rate <= 0.0:
             raise Exception("Learning rate must be greater than 0. ")

    # Split ratio check
    if SPLIT_RATIO in parameters.keys():
        split_ratio = parameters[SPLIT_RATIO]

        # Check typing
        type_checker(split_ratio, "split_ratio must be a float. ", float)

        if split_ratio <= 0.0:
             raise Exception("Split ratio must be greater than 0. ")

    # node_features_property check
    if NODE_FEATURES_PROPERTY in parameters.keys():
        node_features_property = parameters[NODE_FEATURES_PROPERTY]

        # Check typing
        type_checker(node_features_property, "node_features_property must be a string. ", str)

        if node_features_property == "":
            raise Exception("You must specify name of nodes' features property. ")

    # device_type check
    if DEVICE_TYPE in parameters.keys():
        device_type = parameters[DEVICE_TYPE]

        # Check typing
        type_checker(device_type, "device_type must be a string. ", str)

        if device_type != CPU_DEVICE and torch.device != CUDA_DEVICE:
            raise Exception("Only cpu and cuda are supported as devices. ")

    # console_log_freq check
    if CONSOLE_LOG_FREQ in parameters.keys():
        console_log_freq = parameters[CONSOLE_LOG_FREQ]

        # Check typing
        type_checker(console_log_freq, "console_log_freq must be an int. ", int)

        if console_log_freq <= 0:
            raise Exception("Console log frequency must be greater than 0. ")

    # checkpoint freq check
    if CHECKPOINT_FREQ in parameters.keys():
        checkpoint_freq = parameters[CHECKPOINT_FREQ]

        # Check typing
        type_checker(checkpoint_freq, "checkpoint_freq must be an int. ", int)

        if checkpoint_freq <= 0:
             raise Exception("Checkpoint frequency must be greter than 0. ")

    # aggregator check
    if AGGREGATOR in parameters.keys():
        aggregator = parameters[AGGREGATOR]

        # Check typing
        type_checker(aggregator, "aggregator must be a string. ", str)

        if aggregator != MEAN_AGG and aggregator != LSTM_AGG and aggregator != POOL_AGG and aggregator != GCN_AGG:
             raise Exception("Aggregator must be one of the following: mean, pool, lstm or gcn. ")

    # metrics check
    if METRICS in parameters.keys():
        metrics = parameters[METRICS]

        # Check typing
        type_checker(metrics, "metrics must be an iterable object. ", tuple)

        for metric in metrics:
            _metric = metric.lower()
            if (
                _metric != LOSS
                and _metric != ACCURACY
                and _metric != F1
                and _metric != AUC_SCORE
                and _metric != PRECISION
                and _metric != RECALL
                and _metric != POS_EXAMPLES
                and _metric != NEG_EXAMPLES
                and _metric != POS_PRED_EXAMPLES
                and _metric != NEG_PRED_EXAMPLES
            ):
                 raise Exception("Metric name " + _metric + " is not supported!")

    # Predictor type
    if PREDICTOR_TYPE in parameters.keys():
        predictor_type = parameters[PREDICTOR_TYPE]

        # Check typing
        type_checker(predictor_type, "predictor_type must be a string. ", str)

        if predictor_type != DOT_PREDICTOR and predictor_type != MLP_PREDICTOR:
             raise Exception("Predictor " + predictor_type + " is not supported. ")

    # Attention heads
    if ATTN_NUM_HEADS in parameters.keys():
        attn_num_heads = parameters[ATTN_NUM_HEADS]

        if layer_type == GRAPH_ATTN:

            # Check typing
            type_checker(attn_num_heads, "attn_num_heads must be an iterable object. ", tuple)

            if len(attn_num_heads) != len(hidden_features_size):
                 raise Exception("Specified network with {} layers but given attention heads data for {} layers. ".format(len(hidden_features_size) - 1, len(attn_num_heads)))

            if attn_num_heads[-1] != 1:
                 raise Exception("Last GAT layer must contain only one attention head. ")

            for num_heads in attn_num_heads:
                if num_heads <= 0:
                    raise Exception("GAT allows only positive, larger than 0 values for number of attention heads. ")

    # Training accuracy patience
    if TR_ACC_PATIENCE in parameters.keys():
        tr_acc_patience = parameters[TR_ACC_PATIENCE]

        # Check typing
        type_checker(tr_acc_patience, "tr_acc_patience must be an iterable object. ", int)

        if tr_acc_patience <= 0:
             raise Exception("Training acc patience flag must be larger than 0.")

    # model_save_path
    if MODEL_SAVE_PATH in parameters.keys():
        model_save_path = parameters[MODEL_SAVE_PATH]

        # Check typing
        type_checker(model_save_path, "model_save_path must be a string. ", str)

        if model_save_path == "":
             raise Exception("Path must be != " " ")
    
    # context save dir
    if CONTEXT_SAVE_DIR in parameters.keys():
        context_save_dir = parameters[CONTEXT_SAVE_DIR]

        # check typing
        type_checker(context_save_dir, "context_save_dir must be a string. ", str)

        if context_save_dir == "":
            raise Exception("Path must not be empty string. ")
    
    # target edge type
    if TARGET_RELATION in parameters.keys():
        target_relation = parameters[TARGET_RELATION]

        # check typing
        type_checker(target_relation, "target edge tyupe must be a string. ", str)
    else:
        raise Exception("Target relation or target edge type must be specified. ")
    
    # num_neg_per_positive_edge
    if NUM_NEG_PER_POS_EDGE in parameters.keys():
        num_neg_per_pos_edge = parameters[NUM_NEG_PER_POS_EDGE]

        # Check typing
        type_checker(num_neg_per_pos_edge, "number of negative edges per positive one must be an int. ", int)
    
    # batch size
    if BATCH_SIZE in parameters.keys():
        batch_size = parameters[BATCH_SIZE]

        # Check typing
        type_checker(batch_size,"batch_size must be an int", int)

    # sampling workers
    if SAMPLING_WORKERS in parameters.keys():
        sampling_workers = parameters[SAMPLING_WORKERS]

        # check typing
        type_checker(sampling_workers, "sampling_workers must be and int", int)

def _reset_train_predict_parameters() -> None:
    """Reset global parameters that are returned by train method and used by predict method.
    No need to reset features_size_loaded currently. 
    """
    global training_results, validation_results, predictor, model, graph, reindex_dgl, reindex_orig, HETERO
    training_results.clear()  # clear training records from previous training
    validation_results.clear()  # clear validation record from previous training
    predictor = None  # Delete old predictor and create a new one in link_prediction_util.train method\
    model = None  # Annulate old model
    graph = None  # Set graph to None
    reindex_orig = None
    reindex_dgl = None
    HETERO = None

def _conversion_to_dgl_test(
    graph: dgl.graph,
    reindex_dgl: Dict[str, Dict[int, int]],
    ctx: mgp.ProcCtx,
    node_features_property: str,
) -> None:
    """
    Tests whether conversion from ctx.ProcCtx graph to dgl graph went successfully. Checks how features are mapped. Throws exception if something fails.

    Args:
        graph (dgl.graph): Reference to the dgl heterogeneous graph.
        reindex_dgl (Dict[int, int]): Mapping from new indexes to old indexes.
        ctx (mgp.ProcCtx): Reference to the context execution.
        node_features_property (str): Property namer where the node features are saved`
    """

    # Check if the dataset is empty. E2E handling.
    if len(ctx.graph.vertices) == 0:
        raise Exception("The conversion to DGL failed. The dataset is empty. ")

    # Find all node types.
    for node_type in graph.ntypes:
        for vertex in graph.nodes(node_type):
            # Get int from torch.Tensor
            vertex_id = vertex.item()
            # Find vertex in Memgraph
            old_id = reindex_dgl[node_type][vertex_id]
            vertex = ctx.graph.get_vertex_by_id(old_id)
            if vertex is None:
                raise Exception(f"The conversion to DGL failed. Vertex with id {vertex.id} is not mapped to DGL graph. ")

            # Get features, check if they are given as string
            if type(vertex.properties.get(node_features_property)) == str:
                old_features = eval(vertex.properties.get(node_features_property))  # TODO: After dealing with Cypher modules
            else:
                old_features = vertex.properties.get(node_features_property)
            
            # Check if equal
            if not torch.equal(graph.nodes[node_type].data[node_features_property][vertex_id], torch.tensor(old_features, dtype=torch.float32),):
                raise Exception("The conversion to DGL failed. Stored graph does not contain the same features as the converted DGL graph. ")
