from mage.node_classification.model.inductive_model import InductiveModel
from mage.node_classification.utils.convert_data import convert_data
import torch
import numpy as np
import sys
import typing
import mgp

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

# all None until set_params are executed
data = None
model = None
opt = None
criterion = None
global_params = None

def declare_globals(params: typing.Dict):
    global global_params
    global_params = params

def declare_model_and_data(ctx: mgp.ProcCtx):
    global global_params
    nodes = list(iter(ctx.graph.vertices))
    edges = []
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edges.append(edge)

    global data
    data = convert_data(nodes, edges, global_params[DataParams.SPLIT_RATIO])
    
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
