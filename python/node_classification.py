import mgp
from mage.node_classification.utils.convert_data import *
from mage.node_classification.metrics import *
from mage.node_classification.train_model import *
from mage.node_classification.globals.globals import *

import sys

@mgp.read_proc
def set_model_parameters(
    hidden_features_size: mgp.List[int] = [16],
    layer_type: str = "GAT",
    num_epochs: int = 100,
    optimizer: str = "Adam",
    learning_rate: float = 0.001,
    split_ratio: float = 0.8,
    node_features_property: str = "features",
    node_id_property: str = "id",
    device_type: str = "cpu",
    console_log_freq: int = 5,
    checkpoint_freq: int = 5,
    aggregator: str = "mean"
) -> mgp.Record(
    hidden_features_size = mgp.List[int],
    layer_type = str,
    num_epochs = int,
    optimizer = str,
    learning_rate = float,
    split_ratio = float,
    node_features_property = str,
    node_id_property = str,
    device_type = str,
    console_log_freq = int,
    checkpoint_freq = int,
    aggregator = str
):
    """
    set parameters of model and opt

    opt is Adam, criterion is cross entropy loss
    """
    
    hidden_features_size = hidden_features_size 
    layer_type = layer_type
    num_epochs = num_epochs
    optimizer = optimizer
    learning_rate = learning_rate
    split_ratio = split_ratio
    node_features_property = node_features_property
    node_id_property = node_id_property
    device_type = device_type
    console_log_freq = console_log_freq
    checkpoint_freq = checkpoint_freq
    aggregator = aggregator
    

    return mgp.Record(
        hidden_features_size,
        layer_type,
        num_epochs,
        optimizer,
        learning_rate,
        split_ratio,
        node_features_property,
        node_id_property,
        device_type,
        console_log_freq,
        checkpoint_freq,
        aggregator
    )

@mgp.read_proc
def load_data(
    ctx: mgp.ProcCtx,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> mgp.Record(no_nodes=int, no_edges=int):
    """
    loading data, must be executed before training
    """

    nodes = list(iter(ctx.graph.vertices))
    edges = []
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edges.append(edge)
    
    global data
    data = convert_data(nodes, edges, train_ratio, val_ratio, test_ratio)
    
    return mgp.Record(no_nodes=len(nodes), no_edges=len(edges))


@mgp.read_proc
def train(
    no_epochs: int = 100
) -> mgp.Record(total_epochs=int, accuracy=float, precision=float, recall=float, f1_score=float):
    """
    training
    """
    
    try: 
        assert data != None, "Dataset is not loaded. Load dataset first!" 
    except AssertionError as e: 
        print(e)
        sys.exit(1)


    for epoch in range(1, no_epochs+1):
        loss = train_model(model, opt, data, criterion)
        a, p, r, f = metrics(data.val_mask, model, data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {a:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f:.4f}' )
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    global total_no_epochs
    total_no_epochs += no_epochs
    return mgp.Record(total_epochs=total_no_epochs, accuracy=a, precision=p, recall=r, f1_score=f)

@mgp.read_proc
def test() -> mgp.Record(accuracy=float, precision=float, recall=float, f1_score=float):
    """
    testing
    """
    a, p, r, f = metrics(data.test_mask)
    
    return mgp.Record(accuracy=a, precision=p, recall=r, f1_score=f)
