from torch_geometric.data import Data
from mage.node_classification.models.inductive_model import InductiveModel
from mage.node_classification.models.gatjk import GATJK
from mage.node_classification.utils.metrics import metrics
from mage.node_classification.utils.extract_from_database import extract_from_database
from mage.node_classification.models.train_model import model_train_step
import mgp
from tqdm import tqdm
import torch
import numpy as np
import os


##############################
# constants
##############################


class ModelParams:
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"
    HIDDEN_FEATURES_SIZE = "hidden_features_size"
    LAYER_TYPE = "layer_type"
    AGGREGATOR = "aggregator"


class LayerType:
    gat = "GAT"
    gatv2 = "GATv2"
    sage = "SAGE"
    gatjk = "GATJK"


class OptimizerParams:
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"


class DataParams:
    SPLIT_RATIO = "split_ratio"
    METRICS = "metrics"


class MemgraphParams:
    NODE_FEATURES_PROPERTY = "node_features_property"
    NODE_ID_PROPERTY = "node_id_property"
    NODE_CLASS_PROPERTY = "node_class_property"


class TrainParams:
    NUM_EPOCHS = "num_epochs"
    CONSOLE_LOG_FREQ = "console_log_freq"
    CHECKPOINT_FREQ = "checkpoint_freq"


class OtherParams:
    DEVICE_TYPE = "device_type"
    PATH_TO_MODEL = "path_to_model"
    REINDEXING = "reindexing"


class Modelling:
    # all None until set_params are executed
    data: Data = None
    # model: InductiveModel = None
    model: GATJK = None
    opt = None
    criterion = None


global_params: mgp.Map
logged_data: mgp.List = []


DEFINED_INPUT_TYPES = {
    ModelParams.HIDDEN_FEATURES_SIZE: list,
    ModelParams.LAYER_TYPE: str,
    TrainParams.NUM_EPOCHS: int,
    OptimizerParams.LEARNING_RATE: float,
    OptimizerParams.WEIGHT_DECAY: float,
    DataParams.SPLIT_RATIO: float,
    MemgraphParams.NODE_FEATURES_PROPERTY: str,
    MemgraphParams.NODE_ID_PROPERTY: str,
    OtherParams.DEVICE_TYPE: str,
    TrainParams.CONSOLE_LOG_FREQ: int,
    TrainParams.CHECKPOINT_FREQ: int,
    ModelParams.AGGREGATOR: str,
    DataParams.METRICS: list,
    OtherParams.PATH_TO_MODEL: str,
    OtherParams.REINDEXING: dict,
}

DEFAULT_VALUES = {
    ModelParams.HIDDEN_FEATURES_SIZE: [16],
    ModelParams.LAYER_TYPE: "SAGE",
    TrainParams.NUM_EPOCHS: 100,
    OptimizerParams.LEARNING_RATE: 0.1,
    OptimizerParams.WEIGHT_DECAY: 5e-4,
    DataParams.SPLIT_RATIO: 0.8,
    MemgraphParams.NODE_FEATURES_PROPERTY: "features",
    MemgraphParams.NODE_ID_PROPERTY: "id",
    MemgraphParams.NODE_CLASS_PROPERTY: "class",
    OtherParams.DEVICE_TYPE: "cpu",
    TrainParams.CONSOLE_LOG_FREQ: 5,
    TrainParams.CHECKPOINT_FREQ: 5,
    ModelParams.AGGREGATOR: "mean",
    DataParams.METRICS: [
        "loss",
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "num_wrong_examples",
    ],
    OtherParams.PATH_TO_MODEL: "../pytorch_models/model",
    OtherParams.REINDEXING: {},
}


##############################
# set model parameters
##############################


def declare_globals(params: mgp.Map):
    """This function declares dictionary of global parameters to given dictionary.

    Args:
        params (mgp.Map): given dictionary of parameters
    """
    global global_params
    global_params = params


def declare_model_and_data(ctx: mgp.ProcCtx):
    """This function initializes global variables data, model, opt and criterion.

    Args:
        ctx (mgp.ProcCtx): current context
    """
    global global_params
    if Modelling.data == None:
        nodes = list(iter(ctx.graph.vertices))

        for i in range(len(nodes)):
            # inner DB id property
            global_params[OtherParams.REINDEXING][i] = nodes[i].properties.get("id")

        edges = []
        for vertex in ctx.graph.vertices:
            # print(vertex.properties.get(global_params[MemgraphParams.NODE_ID_PROPERTY]))
            for edge in vertex.out_edges:
                edges.append(edge)

        Modelling.data = extract_from_database(
            nodes,
            edges,
            global_params[DataParams.SPLIT_RATIO],
            global_params,
            global_params[OtherParams.REINDEXING],
        )

    global_params[ModelParams.IN_CHANNELS] = np.shape(
        Modelling.data.x.detach().numpy()
    )[1]

    global_params[ModelParams.OUT_CHANNELS] = len(
        set(Modelling.data.y.detach().numpy())
    )

    if global_params[ModelParams.LAYER_TYPE] not in {
        LayerType.gat,
        LayerType.gatv2,
        LayerType.gatjk,
        LayerType.sage,
    }:
        raise Exception("Available models are GAT, GATv2, GATJK and SAGE")

    if global_params[ModelParams.LAYER_TYPE] == "GATJK":
        Modelling.model = GATJK(
            in_channels=global_params[ModelParams.IN_CHANNELS],
            hidden_features_size=global_params[ModelParams.HIDDEN_FEATURES_SIZE],
            out_channels=global_params[ModelParams.OUT_CHANNELS],
            aggr=global_params[ModelParams.AGGREGATOR],
        )
    else:
        Modelling.model = InductiveModel(
            layer_type=global_params[ModelParams.LAYER_TYPE],
            in_channels=global_params[ModelParams.IN_CHANNELS],
            hidden_features_size=global_params[ModelParams.HIDDEN_FEATURES_SIZE],
            out_channels=global_params[ModelParams.OUT_CHANNELS],
            aggr=global_params[ModelParams.AGGREGATOR],
        )

    Modelling.opt = torch.optim.Adam(
        Modelling.model.parameters(),
        lr=global_params[OptimizerParams.LEARNING_RATE],
        weight_decay=global_params[OptimizerParams.WEIGHT_DECAY],
    )

    Modelling.criterion = torch.nn.CrossEntropyLoss()


@mgp.read_proc
def set_model_parameters(
    ctx: mgp.ProcCtx, params: mgp.Map = {}
) -> mgp.Record(
    in_channels=int,
    out_channels=int,
    hidden_features_size=list,
    layer_type=str,
    aggregator=str,
    num_samples=int,
    learning_rate=float,
    weight_decay=float,
    split_ratio=float,
    metrics=mgp.Any,
    node_features_property=str,
    node_id_property=str,
    node_class_property=str,
    num_epochs=int,
    console_log_freq=int,
    checkpoint_freq=int,
    device_type=str,
    path_to_model=str,
):
    """The purpose of this function is to initialize all global variables. Parameter
    params is used for variables written in query module. It first checks
    if (new) variables in params are defined appropriately. If so, map of default
    global parameters is overriden with user defined dictionary params.
    After that it executes previously defined functions declare_globals and
    declare_model_and_data and sets each global variable to some value.

    Args:
        ctx: (mgp.ProcCtx): current context,
        params: (mgp.Map, optional): user defined parameters from query module. Defaults to {}

    Raises:
        Exception: exception is raised if some variable in dictionary params is not
                    defined as it should be

    Returns:
    mgp.Record(
        in_channels (int): number of input channels
        out_channels (int): number of out channels
        hidden_features_size (list): list of hidden features
        layer_type (str): type of layer
        aggregator (str): type of aggregator
        num_samples (int): number of samples
        learning_rate (float): learning rate
        weight_decay (float): weight decay
        split_ratio (float): ratio between training and validation data
        metrics (list): list of metrics to be calculated
        node_features_property (str): name of nodes features property
        node_id_property (str): name of nodes id property
        node_class_property (str): name of nodes class property
        num_epochs (int): number of epochs
        console_log_freq (int): frequency of logging metrics
        checkpoint_freq (int): frequency of saving models
        device_type (str): cpu or cuda
        path_to_model (str): path where model is load and saved
        data (Any): data
        model (Any): model
        opt (Any): optimizer
        criterion (Any): criterion
    )
    """
    global DEFINED_INPUT_TYPES, DEFAULT_VALUES

    # function checks if input values in dictionary are correctly typed
    def is_correctly_typed(defined_types, input_values):
        if isinstance(defined_types, dict) and isinstance(input_values, dict):
            # defined_types is a dict of types
            return all(
                k in input_values  # check if exists
                and is_correctly_typed(
                    defined_types[k], input_values[k]
                )  # check for correct type
                for k in defined_types
            )
        elif isinstance(defined_types, type):
            return isinstance(input_values, defined_types)
        else:
            return False

    if (
        ModelParams.HIDDEN_FEATURES_SIZE in params.keys()
        and type(params[ModelParams.HIDDEN_FEATURES_SIZE]) == tuple
    ):
        params[ModelParams.HIDDEN_FEATURES_SIZE] = list(
            params[ModelParams.HIDDEN_FEATURES_SIZE]
        )
    if (
        DataParams.METRICS in params.keys()
        and type(params[DataParams.METRICS]) == tuple
    ):
        params[DataParams.METRICS] = list(params["metrics"])

    params = {**DEFAULT_VALUES, **params}  # override any default parameters

    print(params)

    if not is_correctly_typed(DEFINED_INPUT_TYPES, params):
        raise Exception(
            f"Input dictionary is not correctly typed. Expected following types {DEFINED_INPUT_TYPES}."
        )

    declare_globals(params)
    declare_model_and_data(ctx)

    return mgp.Record(
        in_channels=global_params[ModelParams.IN_CHANNELS],
        out_channels=global_params[ModelParams.OUT_CHANNELS],
        hidden_features_size=global_params[ModelParams.HIDDEN_FEATURES_SIZE],
        layer_type=global_params[ModelParams.LAYER_TYPE],
        aggregator=global_params[ModelParams.AGGREGATOR],
        num_samples=np.shape(Modelling.data.x)[0],
        learning_rate=global_params[OptimizerParams.LEARNING_RATE],
        weight_decay=global_params[OptimizerParams.WEIGHT_DECAY],
        split_ratio=global_params[DataParams.SPLIT_RATIO],
        metrics=global_params[DataParams.METRICS],
        node_features_property=global_params[MemgraphParams.NODE_FEATURES_PROPERTY],
        node_id_property=global_params[MemgraphParams.NODE_ID_PROPERTY],
        node_class_property=global_params[MemgraphParams.NODE_CLASS_PROPERTY],
        num_epochs=global_params[TrainParams.NUM_EPOCHS],
        console_log_freq=global_params[TrainParams.CONSOLE_LOG_FREQ],
        checkpoint_freq=global_params[TrainParams.CHECKPOINT_FREQ],
        device_type=global_params[OtherParams.DEVICE_TYPE],
        path_to_model=global_params[OtherParams.PATH_TO_MODEL],
    )


##############################
# train
##############################


@mgp.read_proc
def train(
    no_epochs: int = 100,
) -> mgp.Record(epoch=int, loss=float, train_log=mgp.Any, val_log=mgp.Any):
    """This function performs training of model. Before it, function set_model_parameters
    must be executed. Otherwise, global variables data and model will be equal
    to None and AssertionError will be raised.

    Args:
        no_epochs (int, optional): number of epochs. Defaults to 100 )->mgp.Record(.

    Returns:
        _type_: _description_
    """

    global Modelling
    if Modelling.data == None:
        raise Exception("Dataset is not loaded. Load dataset first!")

    global_params[TrainParams.NUM_EPOCHS] = no_epochs

    for epoch in tqdm(range(1, no_epochs + 1)):
        loss = model_train_step(
            Modelling.model, Modelling.opt, Modelling.data, Modelling.criterion
        )

        global logged_data

        if epoch % global_params[TrainParams.CONSOLE_LOG_FREQ] == 0:

            dict_train = metrics(
                Modelling.data.train_mask,
                Modelling.model,
                Modelling.data,
                global_params[DataParams.METRICS],
            )
            dict_val = metrics(
                Modelling.data.val_mask,
                Modelling.model,
                Modelling.data,
                global_params[DataParams.METRICS],
            )
            logged_data.append(
                {"epoch": epoch, "loss": loss, "train": dict_train, "val": dict_val}
            )

            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {logged_data[-1]["train"]["accuracy"]:.4f}, Accuracy: {logged_data[-1]["val"]["accuracy"]:.4f}'
            )
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    return [
        mgp.Record(
            epoch=logged_data[k]["epoch"],
            loss=logged_data[k]["loss"],
            train_log=logged_data[k]["train"],
            val_log=logged_data[k]["val"],
        )
        for k in range(len(logged_data))
    ]


##############################
# get training data
##############################


@mgp.read_proc
def get_training_data() -> mgp.Record(
    epoch=int, loss=float, train_log=mgp.Any, val_log=mgp.Any
):
    """This function is used so user can see what is logged data from training.


    Returns:
        mgp.Record(
            epoch (int): epoch number of record of logged data row
            loss (float): loss in logged data row
            train_log (mgp.Any): training parameters of record of logged data row
            val_log (mgp.Any): validation parameters of record of logged data row
            ): record to return


    """

    return [
        mgp.Record(
            epoch=logged_data[k]["epoch"],
            loss=logged_data[k]["loss"],
            train_log=logged_data[k]["train"],
            val_log=logged_data[k]["val"],
        )
        for k in range(len(logged_data))
    ]


##############################
# model loading and saving, predict
##############################


@mgp.read_proc
def save_model() -> mgp.Record(path=str):
    """This function saves model to previously defined path_to_model.

    Returns:
        mgp.Record(path (str): path to model): return record
    """

    if Modelling.model == None:
        raise AssertionError("model is not loaded")
    torch.save(Modelling.model.state_dict(), global_params[OtherParams.PATH_TO_MODEL])
    return mgp.Record(path=global_params[OtherParams.PATH_TO_MODEL])


@mgp.read_proc
def load_model() -> mgp.Record(path=str):
    """This function loads model to previously defined path_to_model.

    Returns:
        mgp.Record(path (str): path to model): return record
    """
    global model

    if not os.path.exists(os.path.abspath(global_params[OtherParams.PATH_TO_MODEL])):
        raise Exception(
            f"File {global_params[OtherParams.PATH_TO_MODEL]} not found on system. Please provide the valid path."
        )

    Modelling.model.load_state_dict(
        torch.load(global_params[OtherParams.PATH_TO_MODEL])
    )
    return mgp.Record(path=global_params[OtherParams.PATH_TO_MODEL])


@mgp.read_proc
def predict(vertex: mgp.Vertex) -> mgp.Record(predicted_value=int):
    """This function predicts metrics on one node. It is suggested that user previously
    loads unseen test data to predict on it.
    Subgraph (where predict is performed) is consisted of node and self loop to it.

    Example of usage:
        MATCH (n {id: 1}) CALL node_classification.predict(n) YIELD * RETURN predicted_value;

        # note: if node with property id = 1 doesn't exist, query module won't be called

    Args:
        vertex (mgp.Vertex): node to predict on

    Returns:
        mgp.Record(predicted_class (int): predicted class): record to return
    """
    global global_params
    id = vertex.properties.get(global_params[MemgraphParams.NODE_ID_PROPERTY])
    features = vertex.properties.get(
        global_params[MemgraphParams.NODE_FEATURES_PROPERTY]
    )

    # it is assumed vertex does not always have truth label
    # truth = vertex.properties.get(global_params[MemgraphParams.NODE_CLASS_PROPERTY])

    inv_reindexing = {v: k for k, v in global_params[OtherParams.REINDEXING].items()}

    Modelling.model.eval()
    out = Modelling.model(Modelling.data.x, Modelling.data.edge_index)
    pred = out.argmax(dim=1)

    if not torch.equal(
        torch.tensor(np.array(features), dtype=torch.float32),
        Modelling.data.x[inv_reindexing[id]],
    ):
        raise AssertionError("features from node are different from database features")

    predicted_class = (int)(pred.detach().numpy()[inv_reindexing[id]])

    return mgp.Record(predicted_class=predicted_class)


@mgp.read_proc
def reset() -> mgp.Record(status=str):
    if "global_params" in globals().keys():
        globals.pop("global_params")

    if "logged_data" in globals().keys():
        globals.pop("logged_data")

    return mgp.Record(status="Global parameters and logged data have been reseted.")
