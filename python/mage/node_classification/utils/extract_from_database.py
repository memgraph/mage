import torch
from torch_geometric.data import Data, HeteroData
import mgp
import numpy as np
from tqdm import tqdm
import random
from collections import Counter
import torch_geometric.transforms as T
import typing

def extract_from_database(
    ctx: mgp.ProcCtx,
    train_ratio: float,
    features_name: str,
    class_name: str,
) -> typing.Tuple[HeteroData, str, typing.Dict, typing.Dict]:
    
    data = HeteroData()
    

    #################
    # NODES
    #################
    nodes = list(iter(ctx.graph.vertices))
    node_types = []
    embedding_lengths = {}
    observed_attribute = None
    
    for i in range(len(nodes)):
        if type(nodes[i].properties.get(features_name)) == type(None):
            continue
        node_types.append(nodes[i].labels[0].name)
        if nodes[i].labels[0].name not in embedding_lengths:
            embedding_lengths[nodes[i].labels[0].name] = len(
                nodes[i].properties.get(features_name)
            )

        # find attribute which has classification labels
        if observed_attribute == None and type(
            nodes[i].properties.get(class_name)
        ) != type(None):
            observed_attribute = nodes[i].labels[0].name

    if(node_types == []):
        raise Exception("no features vectors found")
        
    node_types = Counter(node_types)

    append_counter, reindexing, inv_reindexing = {}, {}, {}

    for k, v in node_types.items():
        data[k].x = torch.tensor(
            np.zeros((v, embedding_lengths[k])), dtype=torch.float32
        )
        if k == observed_attribute:
            data[k].y = torch.tensor(np.zeros((v,), dtype=int), dtype=torch.long)
            data[k].train_mask = torch.tensor(
                np.zeros((v,), dtype=int), dtype=torch.bool
            )
            data[k].val_mask = torch.tensor(np.zeros((v,), dtype=int), dtype=torch.bool)
            masks = torch.tensor(np.zeros((v,), dtype=int), dtype=torch.bool)

        append_counter[k] = 0
        reindexing[k] = {}
        inv_reindexing[k] = {}

    for i in range(len(nodes)):
        if type(nodes[i].properties.get(features_name)) == type(None):
            continue

        t = nodes[i].labels[0].name

        data[t].x[append_counter[t]] = np.add(
            data[t].x[append_counter[t]],
            np.array(nodes[i].properties.get(features_name)),
        )
        reindexing[t][append_counter[t]] = nodes[i].id
        inv_reindexing[t][nodes[i].id] = append_counter[t]

        if t == observed_attribute:
            data[t].y[append_counter[t]] = (int)(nodes[i].properties.get(class_name))

        append_counter[t] += 1

    print(node_types)

    #################
    # EDGES
    #################
    edges = []
    edge_types = []
    append_counter = {}

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            f, t, o = (
                edge.from_vertex.labels[0].name,
                edge.type.name,
                edge.to_vertex.labels[0].name,
            )
            if type(edge.from_vertex.properties.get(features_name)) == type(
                None
            ) or type(edge.to_vertex.properties.get(features_name)) == type(None):
                continue
            # if not (f in node_types and o in node_types):
            #     continue
            edge_types.append((f, t, o))
            edges.append(edge)

    edge_types = Counter(edge_types)

    for k, v in edge_types.items():
        data[k].edge_index = torch.tensor(np.zeros((2, v)), dtype=torch.long)
        append_counter[k] = 0

    for i in range(len(edges)):
        f, t, o = (
            edges[i].from_vertex.labels[0].name,
            edges[i].type.name,
            edges[i].to_vertex.labels[0].name,
        )

        k = (f, t, o)

        # if not (f in node_types and o in node_types):
        #     continue
        data[k].edge_index[0][append_counter[k]] = (int)(
            inv_reindexing[f][edges[i].from_vertex.id]
        )
        data[k].edge_index[1][append_counter[k]] = (int)(
            inv_reindexing[o][edges[i].to_vertex.id]
        )

        append_counter[k] += 1

    #################
    # MASKS
    #################
    print(data) 
    print(observed_attribute)
    
    no_observed = np.shape(data[observed_attribute].x)[0]
    masks = np.zeros((no_observed))

    for i in range(no_observed):
        if i < train_ratio * no_observed:
            masks[i] = 1
        else:
            masks[i] = 2

    random.shuffle(masks)

    for i in range(no_observed):
        data[observed_attribute].train_mask[i] = (bool)(2 - (int)(masks[i]))
        data[observed_attribute].val_mask[i] = (bool)((int)(masks[i]) - 1)

    data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)
    
    return (data, observed_attribute, reindexing, inv_reindexing)


# def extract_from_database(
#     ctx: mgp.ProcCtx,
#     train_ratio: float,
#     params: mgp.Map,
#     reindexing: mgp.Map,
# ) -> Data:
#     """This function converts data from nodes and edges to default dataset of type
#     torch_geometric.data.Data.

#     Args:
#         nodes (mgp.List[mgp.Vertex]): list of all nodes in graph
#         edges (mgp.List[mgp.Edge]): list of all edges in graph
#         train_ratio (float): ratio of train vs validation data
#         params (mgp.Map): dictionary of global parameters
#         reindexing (mgp.Map): reindexing dictionary to order ids so they can be neig

#     Returns:
#         Data: data from nodes and edges organized to dataset
#     """
#     nodes = list(iter(ctx.graph.vertices))

#     for i in range(len(nodes)):
#         # inner DB id property
#         reindexing[i] = nodes[i].properties.get("id")

#     edges = []
#     for vertex in ctx.graph.vertices:
#         # print(vertex.properties.get(DEFAULT_VALUES[MemgraphParams.NODE_ID_PROPERTY]))
#         for edge in vertex.out_edges:
#             edges.append(edge)

#     if len(nodes) == 0:
#         raise AssertionError("dataset is empty")
#     if train_ratio > 1 or train_ratio < 0:
#         raise AssertionError("training ratio must be positive numbers in [0,1]")

#     x = np.zeros(
#         (len(nodes), len(nodes[0].properties.get(params["node_features_property"])))
#     )
#     y = np.zeros((len(nodes)))
#     edge_index = np.zeros((2, len(edges)))
#     train_mask = np.zeros((len(nodes)))
#     val_mask = np.zeros((len(nodes)))

#     masks = np.zeros((len(nodes)))

#     inv_reindexing = {v: k for k, v in reindexing.items()}

#     for i in range(len(nodes)):
#         if i < train_ratio * len(nodes):
#             masks[i] = 1
#         else:
#             masks[i] = 2

#     random.shuffle(
#         masks
#     )  # this way we have randomized 80%/10%/10% train/valuation/test data

#     print("Structuring data:")
#     for i in tqdm(range(len(nodes))):
#         x[i] = np.add(
#             x[i], np.array(nodes[i].properties.get(params["node_features_property"]))
#         )
#         # y[i] = random.randint(0, num_of_classes - 1)
#         # # because of homophility, we cannot initialize classes randomly
#         # label for class is missing for temporary CORA dataset in Memgraph lab

#         y[i] = nodes[i].properties.get(params["node_class_property"])

#         train_mask[i], val_mask[i] = 2 - masks[i], masks[i] - 1

#     for i in tqdm(range(len(edges))):
#         edge_index[0][i] = inv_reindexing[
#             edges[i].from_vertex.properties.get(params["node_id_property"])
#         ]
#         edge_index[1][i] = inv_reindexing[
#             edges[i].to_vertex.properties.get(params["node_id_property"])
#         ]

#     print("Finished converting data.")

#     # print("=============================")
#     # print(np.shape(x))
#     # print(np.shape(y))
#     # print(np.shape(edge_index))
#     # print(np.shape(train_mask))
#     # print(np.shape(val_mask))
#     # print("=============================")

#     x = torch.tensor(x, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.int64)
#     edge_index = torch.tensor(edge_index, dtype=torch.int64)
#     train_mask = torch.tensor(train_mask, dtype=torch.bool)
#     val_mask = torch.tensor(val_mask, dtype=torch.bool)

#     data = Data(
#         x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask
#     )

#     return data
