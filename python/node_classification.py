from gqlalchemy import Memgraph
import mgp
from tqdm import tqdm
from mage.node_classification.metrics import *
from torch_geometric.data import Data
#from mage.node_classification.model.inductive_model import InductiveModel
from mage.node_classification.utils.convert_data import convert_data
from mage.node_classification.train_model import model_train_step
import torch
import numpy as np
import sys
import mgp
import os


################################################################
import torch
import torch_geometric
import torch.nn.functional as F
import sys
import mgp

class LayerType:
    gat = "GAT"
    gatv2 = "GATv2"
    sage = "SAGE"


class InductiveModel(torch.nn.Module):
    def __init__(
        self,
        layer_type: str,
        in_channels: int,
        hidden_features_size: mgp.List[int],
        out_channels: int,
        aggr: str,
        heads: int = 3,
        dropout: float = 0.6
    ):
        """Initialization of model.

        Args:
            layer_type (str): type of layer
            in_channels (int): dimension of input channels
            hidden_features_size (mgp.List[int]): list of dimensions of hidden features
            out_channels (int): dimension of output channels
            aggr (str): aggregator type
        """

        super(InductiveModel, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        if layer_type not in {LayerType.gat, LayerType.gatv2, LayerType.sage}:
            raise Exception("Available models are GAT, GATv2 and SAGE")
            
        conv = getattr(torch_geometric.nn, layer_type + "Conv")
        if len(hidden_features_size) > 0:
            # dodat heads
            self.convs.append(conv(in_channels, hidden_features_size[0], aggr=aggr, heads=heads, dropout=dropout, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_features_size[0]*heads))
            for i in range(0, len(hidden_features_size) - 1):
                self.convs.append(
                    conv(
                        hidden_features_size[i]*heads, hidden_features_size[i + 1], aggr=aggr, heads=heads, dropout=dropout, concat=True
                    )
                )
                self.bns.append(torch.nn.BatchNorm1d(hidden_features_size[i + 1]*heads))
            self.convs.append(conv(hidden_features_size[-1]*heads, out_channels, aggr=aggr, heads=heads, dropout=dropout, concat=False))
        else:
            self.convs.append(conv(in_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
        """Forward propagation

        Args:
            x (torch.tensor): matrix of embeddings
            edge_index (torch.tensor): matrix of edges

        Returns:
            torch.tensor: embeddings after last layer of network is applied
        """

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)

            # apply relu and dropout on all layers except last one
            if i < len(self.convs) - 1:
                #x = self.bns[i](x)
                x = x.relu()
                #x = F.dropout(x, p=0.5, training=self.training)

        return x

################################################################

##############################
# constants
##############################


class ModelParams:
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"
    HIDDEN_FEATURES_SIZE = "hidden_features_size"
    LAYER_TYPE = "layer_type"
    AGGREGATOR = "aggregator"


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


class Modelling:
    # all None until set_params are executed
    data: Data = None
    model: InductiveModel = None
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

        reindexing = {}
        for i in range(len(nodes)):
            # inner DB id property
            reindexing[i] = nodes[i].properties.get("id")

        edges = []
        for vertex in ctx.graph.vertices:
            #print(vertex.properties.get(global_params[MemgraphParams.NODE_ID_PROPERTY]))
            for edge in vertex.out_edges:
                edges.append(edge)

        Modelling.data = convert_data(
            nodes, edges, global_params[DataParams.SPLIT_RATIO], global_params, reindexing
        )

    # print(data)
    # print(data.edge_index.max())

    global_params[ModelParams.IN_CHANNELS] = np.shape(
        Modelling.data.x.detach().numpy()
    )[1]

    global_params[ModelParams.OUT_CHANNELS] = len(
        set(Modelling.data.y.detach().numpy())
    )

    Modelling.model = InductiveModel(
        layer_type=global_params[ModelParams.LAYER_TYPE],
        in_channels=global_params[ModelParams.IN_CHANNELS],
        hidden_features_size=global_params[ModelParams.HIDDEN_FEATURES_SIZE],
        out_channels=global_params[ModelParams.OUT_CHANNELS],
        aggr=global_params[ModelParams.AGGREGATOR],
    )

    # obtain function by its name from library
    Modelling.opt = torch.optim.Adam(
        Modelling.model.parameters(),
        lr=global_params[OptimizerParams.LEARNING_RATE],
        weight_decay=global_params[OptimizerParams.WEIGHT_DECAY],
    )

    Modelling.criterion = torch.nn.CrossEntropyLoss()


@mgp.read_proc
def set_model_parameters(
    ctx: mgp.ProcCtx, params: mgp.Map = {}
) -> mgp.Record(in_channels=int, out_channels=int, num_samples=int, path=str):
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
            in_channels (int): number of input channels,
            out_channels (int): number of output channels
            num_sample(int): number of samples
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
        num_samples=np.shape(Modelling.data.x)[0],
        path=global_params[OtherParams.PATH_TO_MODEL],
    )


##############################
# train
##############################


@mgp.read_proc
def train(
    no_epochs: int = 100,
) -> mgp.Record(epoch=int, loss=float, train_log=mgp.Any, val_log=mgp.Any):
    """This function performs training of model. Before her, function set_model_parameters
    must be defined. Otherwise, global variables data and model will be equal
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
            #######################
            out = Modelling.model(Modelling.data.x, Modelling.data.edge_index)
            pred = out.argmax(dim=1)
            confmat = ConfusionMatrix(num_classes=len(set(Modelling.data.y.detach().numpy())))
            print(confmat(pred[Modelling.data.train_mask],Modelling.data.y[Modelling.data.train_mask])) 
            #####################################  
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

    ###########################################################
    # import weightwatcher as ww

    # watcher = ww.WeightWatcher(model=Modelling.model)
    # details = watcher.analyze()
    # summary = watcher.get_summary(details)
    # print(summary)
    ###########################################################
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
        raise Exception(f"File {global_params[OtherParams.PATH_TO_MODEL]} not found on system. Please provide the valid path.")

    Modelling.model.load_state_dict(
        torch.load(global_params[OtherParams.PATH_TO_MODEL])
    )
    return mgp.Record(path=global_params[OtherParams.PATH_TO_MODEL])


@mgp.read_proc
def predict() -> mgp.Record(dict=mgp.Map):
    """This function predicts metrics on all nodes. It is suggested that user previously
    loads unseen test data to predict on it.

    Returns:
        mgp.Record(mgp.Map: dictionary of all metrics): record to return
    """

    dict = metrics(
        Modelling.data.train_mask + Modelling.data.val_mask,
        Modelling.model,
        Modelling.data,
        global_params[DataParams.METRICS],
    )

    return mgp.Record(dict=dict)


@mgp.read_proc
def reset() -> mgp.Record(status=str):
    if "global_params" in globals().keys():
        globals.pop("global_params")

    if "logged_data" in globals().keys():
        globals.pop("logged_data")

    return mgp.Record(status="Global parameters and logged data have been reseted.")


