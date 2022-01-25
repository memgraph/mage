import numpy as np
import torch

from tgn.constants import TGNLayerType, MessageFunctionType, MessageAggregatorType, MemoryUpdaterType
from tgn.definitions.tgn import TGN
from utils import data_loader


def train_tgn(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features, edge_features, train_data = data_loader.get_data(config["dataset_name"])
    sources, destinations, timestamps, edge_idxs, labels = train_data

    current_train_indices = np.arange(0, 100)

    sources = sources[current_train_indices]
    destinations = destinations[current_train_indices]
    timestamps = timestamps[current_train_indices]
    edge_idxs = edge_idxs[current_train_indices]
    labels = labels[current_train_indices]
    edge_features = edge_features[current_train_indices]

    # dimensions
    num_edge_features = edge_features.shape[1]
    time_dimension = 1
    memory_dimension = 100
    num_node_features=100
    message_dimension=100

    # be careful if you will copy this later, because here mapping of indices works from 0 to 100, later it should also??
    edge_features = {edge_idxs[index]: np.array(feature, dtype=np.float32) for index, feature in
                     enumerate(
                         edge_features)}

    node_features = {node: np.zeros(num_node_features) for node in set(sources).union(destinations)}

    graph_data = (sources, destinations, timestamps, edge_idxs, edge_features, node_features)

    tgn = TGN(
        num_of_layers=config['num_of_layers'],
        layer_type=config['layer_type'],
        memory_dimension=memory_dimension,
        time_dimension=time_dimension,
        num_edge_features=num_edge_features,
        num_node_features=num_node_features,
        message_dimension=message_dimension,
        # if edge is identity, node must be MLP,
        # if edge is MLP, node also will be MLP
        # thats why we only determine edge type
        edge_message_function_type=MessageFunctionType.Identity,
        message_aggregator_type=MessageAggregatorType.Last,
        memory_updater_type=MemoryUpdaterType.GRU


    ).to(device)

    tgn.train()

    nodes_unnormalized_scores = tgn(graph_data)

    print(nodes_unnormalized_scores)


def get_args():
    tgn_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "layer_type": TGNLayerType.GraphSumEmbedding,
    }
    config = {**tgn_config, **{"dataset_name": "wikipedia"}}
    print(config)
    return config


def main():
    config = get_args()
    train_tgn(config)


if __name__ == '__main__':
    main()
