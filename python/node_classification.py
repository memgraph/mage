import mgp
from tqdm import tqdm
from mage.node_classification.metrics import *
from mage.node_classification.train_model import *
from torch_geometric.data import Data
from mage.node_classification.model.inductive_model import InductiveModel
from mage.node_classification.utils.convert_data import convert_data
import torch
import numpy as np
import sys
import typing
import mgp

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
    OPTIMIZER = "optimizer"
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
    TOTAL_NO_EPOCHS = "total_no_epochs"

class OtherParams:
    DEVICE_TYPE = "device_type"
    PATH_TO_MODELS = "/home/mateo/mage/pytorch_models/model"

# all None until set_params are executed
data: Data = None
model: InductiveModel = None
opt = None
criterion = None
global_params: typing.Dict
logged_data: mgp.List = []

DEFINED_INPUT_TYPES = {
    ModelParams.HIDDEN_FEATURES_SIZE: list,
    ModelParams.LAYER_TYPE: str,
    TrainParams.NUM_EPOCHS: int,
    OptimizerParams.OPTIMIZER: str,
    OptimizerParams.LEARNING_RATE: float,
    OptimizerParams.WEIGHT_DECAY: float,
    DataParams.SPLIT_RATIO: float,
    MemgraphParams.NODE_FEATURES_PROPERTY: str,
    MemgraphParams.NODE_ID_PROPERTY: str,
    OtherParams.DEVICE_TYPE: str,
    TrainParams.CONSOLE_LOG_FREQ: int,
    TrainParams.CHECKPOINT_FREQ: int,
    ModelParams.AGGREGATOR: str,
    DataParams.METRICS: list
}

DEFAULT_VALUES = {
    ModelParams.HIDDEN_FEATURES_SIZE: [16],
    ModelParams.LAYER_TYPE: "SAGE",
    TrainParams.NUM_EPOCHS: 100,
    OptimizerParams.OPTIMIZER: "Adam",
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
    DataParams.METRICS: ["loss", "accuracy", "f1_score", "precision", "recall", "num_wrong_examples"],
}


##############################
# set model parameters
##############################

def declare_globals(params: typing.Dict):
    global global_params
    global_params = params

def declare_model_and_data(ctx: mgp.ProcCtx):
    global global_params
    nodes = list(iter(ctx.graph.vertices))
    
    reindexing = {}
    for i in range(len(nodes)):
        reindexing[i] = nodes[i].properties.get(global_params["node_id_property"])

    edges = []
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edges.append(edge)

    global data
    data = convert_data(
        nodes, edges, global_params[DataParams.SPLIT_RATIO], global_params, reindexing)

    # print(data)
    # print(data.edge_index.max())

    global_params[ModelParams.IN_CHANNELS] = np.shape(data.x.detach().numpy())[1]
    print(global_params[ModelParams.IN_CHANNELS])
    global_params[ModelParams.OUT_CHANNELS] = len(set(data.y.detach().numpy()))
    print(len(set(data.y.detach().numpy())))
    
    global model
    model = InductiveModel(
        layer_type=global_params[ModelParams.LAYER_TYPE], 
        in_channels=global_params[ModelParams.IN_CHANNELS], 
        hidden_features_size=global_params[ModelParams.HIDDEN_FEATURES_SIZE], 
        out_channels=global_params[ModelParams.OUT_CHANNELS],
        aggr = global_params[ModelParams.AGGREGATOR]
    )

    global opt
    # obtain function by its name from library
    Optim = getattr(torch.optim, global_params[OptimizerParams.OPTIMIZER])
    opt = Optim(
        model.parameters(), 
        lr=global_params[OptimizerParams.LEARNING_RATE], 
        weight_decay=global_params[OptimizerParams.WEIGHT_DECAY])

    global criterion
    criterion = torch.nn.CrossEntropyLoss()



@mgp.read_proc
def set_model_parameters(
    ctx: mgp.ProcCtx,
    params: mgp.Map = {}
) -> mgp.Record():
    """
    In: context, dictionary of all parameters

    Out: empty mgp.Record()
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

    # mgconsole bug
    if "hidden_features_size" in params.keys():
        params["hidden_features_size"] = list(params["hidden_features_size"])
    if "metrics" in params.keys():
        params["metrics"] = list(params["metrics"])

    params = {**DEFAULT_VALUES, **params}  # override any default parameters
    
    print(params)
    
    if not is_correctly_typed(DEFINED_INPUT_TYPES, params):
        raise Exception(
            f"Input dictionary is not correctly typed. Expected following types {DEFINED_INPUT_TYPES}."
        )

    
    declare_globals(params)
    declare_model_and_data(ctx)

    return mgp.Record()

##############################
# train
##############################


@mgp.read_proc
def train(
    no_epochs: int = 100
) -> mgp.Record():
    """
    training
    """
    
    global data
    try: 
        assert data != None, "Dataset is not loaded. Load dataset first!" 
    except AssertionError as e: 
        print(e)
        sys.exit(1)


    for epoch in tqdm(range(1, no_epochs+1)):
        loss = train_model(model, opt, data, criterion)
        
        global logged_data
        
        if epoch % global_params[TrainParams.CONSOLE_LOG_FREQ] == 0:
            dict_train = metrics(data.train_mask, model, data, global_params[DataParams.METRICS])
            dict_val = metrics(data.val_mask, model, data, global_params[DataParams.METRICS])
            logged_data.append({"epoch": epoch,"loss": loss, "train": dict_train, "val": dict_val})
        
        
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {logged_data[-1]["train"]["accuracy"]:.4f}, Accuracy: {logged_data[-1]["val"]["accuracy"]:.4f}' )
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    return mgp.Record()

##############################
# get training data
##############################

@mgp.read_proc
def get_training_data() -> mgp.Record(
    epoch=int, loss=float, train_log=mgp.Any, val_log=mgp.Any):
    
    return [
        mgp.Record(
            epoch=logged_data[k]["epoch"],
            loss=logged_data[k]["loss"],
            train_log=logged_data[k]["train"],
            val_log=logged_data[k]["val"]) for k in range(len(logged_data))
        ]

@mgp.read_proc
def save_model() -> mgp.Record():
    global model
    torch.save(model.state_dict(), OtherParams.PATH_TO_MODELS)
    return mgp.Record()

@mgp.read_proc
def load_model() -> mgp.Record():
    global model
    model.load_state_dict(torch.load(OtherParams.PATH_TO_MODELS))
    return mgp.Record()

@mgp.read_proc
def predict() -> mgp.Record(accuracy=float, precision=float):
    
    dict = metrics(data.train_mask+data.val_mask, model, data, global_params[DataParams.METRICS])
    
    return mgp.Record(accuracy=dict["accuracy"], precision=dict["precision"])