import torch
from torch_geometric.data import Data
import mgp
import numpy as np
from tqdm import tqdm
import random
import typing
import sys

def convert_data(
    nodes: typing.List[mgp.Vertex], 
    edges: typing.List[mgp.Edge], 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float)->Data:
    
    try:
        assert len(nodes) > 0, "dataset is empty"
        assert train_ratio > 0 and val_ratio > 0 and test_ratio > 0, "ratios are positive numbers in [0,1]"
        assert train_ratio + val_ratio + test_ratio == 1, "data ratios don't add up to 100%"
    except AssertionError as e: 
        print(e)
        sys.exit(1)


    x = np.zeros((len(nodes),len(nodes[0].properties.get("features"))))
    y = np.zeros((len(nodes)))
    edge_index = np.zeros((2,len(edges)))
    train_mask = np.zeros((len(nodes)))
    val_mask = np.zeros((len(nodes)))
    test_mask = np.zeros((len(nodes)))
    
    masks = np.zeros((len(nodes)))
    
    
    # reindexing = {} # for unordered ids
    # for i in range(len(nodes)): 
    #     reindexing[nodes[i].properties.get("id")] = i



    for i in range(len(nodes)):
        if i < train_ratio * len(nodes):
            masks[i] = 1
        elif i >= train_ratio * len(nodes) and i < (train_ratio + val_ratio) * len(nodes):
            masks[i] = 2
        else:
            masks[i] = 3

    random.shuffle(masks) # this way we have randomized 80%/10%/10% train/valuation/test data

    print("Structuring data:")
    for i in tqdm(range(len(nodes))):
        x[i] = np.add(x[i], np.array(nodes[i].properties.get("features")))
        # y[i] = random.randint(0, num_of_classes - 1) 
        # # because of homophility, we cannot initialize classes randomly
        # label for class is missing for temporary CORA dataset in Memgraph lab

        y[i] = nodes[i].properties.get("class")

        if masks[i] == 1:
            train_mask[i] = 1
        elif masks[i] == 2:
            val_mask[i] = 1
        else:
            test_mask[i] = 1

    for i in tqdm(range(len(edges))):
        edge_index[0][i] = edges[i].from_vertex.properties.get("id")
        edge_index[1][i] = edges[i].to_vertex.properties.get("id")
        
    global NO_CLASSES
    NO_CLASSES = len(set(y))
    global NO_FEATURES
    NO_FEATURES = np.shape(x)[1]

    print("Finished converting data.")

    # print("=============================")
    # print(np.shape(x))
    # print(np.shape(y))
    # print(np.shape(edge_index))
    # print(np.shape(train_mask))
    # print(np.shape(val_mask))
    # print(np.shape(test_mask))
    # print("=============================")


    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.int64) 
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data