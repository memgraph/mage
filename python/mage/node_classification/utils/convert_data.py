import torch
from torch_geometric.data import Data
import mgp
import numpy as np
from tqdm import tqdm
import random
import typing
import sys


def convert_data(
    nodes: mgp.List[mgp.Vertex],
    edges: mgp.List[mgp.Edge],
    train_ratio: float,
    params: mgp.Map,
    reindexing: mgp.Map,
) -> Data:
    """This function converts data from nodes and edges to default dataset of type
    torch_geometric.data.Data.

    Args:
        nodes (mgp.List[mgp.Vertex]): list of all nodes in graph
        edges (mgp.List[mgp.Edge]): list of all edges in graph
        train_ratio (float): ratio of train vs validation data
        params (mgp.Map): dictionary of global parameters
        reindexing (mgp.Map): reindexing dictionary to order ids so they can be neig

    Returns:
        Data: data from nodes and edges organized to dataset
    """

    if len(nodes) == 0:
        raise AssertionError("dataset is empty")
    if train_ratio > 1 or train_ratio < 0:
        raise AssertionError("training ratio must be positive numbers in [0,1]")

    x = np.zeros(
        (len(nodes), len(nodes[0].properties.get(params["node_features_property"])))
    )
    y = np.zeros((len(nodes)))
    edge_index = np.zeros((2, len(edges)))
    train_mask = np.zeros((len(nodes)))
    val_mask = np.zeros((len(nodes)))

    masks = np.zeros((len(nodes)))

    inv_reindexing = {v: k for k, v in reindexing.items()}

    for i in range(len(nodes)):
        if i < train_ratio * len(nodes):
            masks[i] = 1
        else:
            masks[i] = 2

    random.shuffle(
        masks
    )  # this way we have randomized 80%/10%/10% train/valuation/test data

    print("Structuring data:")
    for i in tqdm(range(len(nodes))):
        x[i] = np.add(
            x[i], np.array(nodes[i].properties.get(params["node_features_property"]))
        )
        # y[i] = random.randint(0, num_of_classes - 1)
        # # because of homophility, we cannot initialize classes randomly
        # label for class is missing for temporary CORA dataset in Memgraph lab

        y[i] = nodes[i].properties.get(params["node_class_property"])

        if masks[i] == 1:
            train_mask[i] = 1
        elif masks[i] == 2:
            val_mask[i] = 1

    for i in tqdm(range(len(edges))):
        edge_index[0][i] = inv_reindexing[
            edges[i].from_vertex.properties.get(params["node_id_property"])
        ]
        edge_index[1][i] = inv_reindexing[
            edges[i].to_vertex.properties.get(params["node_id_property"])
        ]

    print("Finished converting data.")

    # print("=============================")
    # print(np.shape(x))
    # print(np.shape(y))
    # print(np.shape(edge_index))
    # print(np.shape(train_mask))
    # print(np.shape(val_mask))
    # print("=============================")

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)

    data = Data(
        x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask
    )

    return data
