import torch
from torch_geometric.data import Data, HeteroData
import mgp
import numpy as np
from tqdm import tqdm
import random
from collections import Counter
import torch_geometric.transforms as T
import typing
from collections import defaultdict


def nodes_fetching(
    ctx: mgp.ProcCtx, features_name: str, class_name: str, data: HeteroData
) -> typing.Tuple[HeteroData, typing.Dict, typing.Dict, str]:
    """
    This procedure fetches the nodes from the database and returns them in HeteroData object.

    Args:
        ctx: The context of the procedure.
        features_name: The name of the features field.
        class_name: The name of the class field.
        data: HeteroData object where nodes will be stored.

    Returns:
        Tuple of (HeteroData, reindexing, inv_reindexing, observed_attribute).
    """

    nodes = list(iter(ctx.graph.vertices))  # obtain nodes from context
    node_types = []  # variable for storing node types
    embedding_lengths = {}  # variable for storing embedding lengths
    observed_attribute = None  # variable for storing observed attribute

    for node in nodes:

        if features_name not in node.properties:
            continue  # if features are not available, skip the node

        # add first node type to list of node types (if other exist, they are ignored)
        node_types.append(node.labels[0].name)

        # add embedding length to dictionary of embedding lengths
        if node.labels[0].name not in embedding_lengths:
            embedding_lengths[node.labels[0].name] = len(
                node.properties.get(features_name)
            )

        # if observed attribute is not set, set it to the first node type which has classification labels
        if observed_attribute == None and class_name in node.properties:
            observed_attribute = node.labels[0].name

    # if node_types is empty, raise error
    if not node_types:
        raise Exception(
            "There are no feature vectors found. Please check your database."
        )

    # apply Counter to obtain the number of each node type
    node_types = Counter(node_types)

    # auxiliary dictionaries for reindexing and inverse reindexing
    append_counter = defaultdict(lambda: 0)
    reindexing, inv_reindexing = defaultdict(lambda: {}), defaultdict(lambda: {})

    # since node_types is Counter, key is the node type and value is the number of nodes of that type
    for node_type, no_nodes_of_type in node_types.items():

        # for each node type, create a tensor of size no_nodes_of_type x embedding_lengths[node_type]
        data[node_type].x = torch.tensor(
            np.zeros((no_nodes_of_type, embedding_lengths[node_type])),
            dtype=torch.float32,
        )

        # if node type is observed attribute, create other necessary tensors
        if node_type == observed_attribute:
            data[node_type].y = torch.tensor(
                np.zeros((no_nodes_of_type,), dtype=int), dtype=torch.long
            )
            data[node_type].train_mask = torch.tensor(
                np.zeros((no_nodes_of_type,), dtype=int), dtype=torch.bool
            )
            data[node_type].val_mask = torch.tensor(
                np.zeros((no_nodes_of_type,), dtype=int), dtype=torch.bool
            )

    # now fill the tensors with the nodes from the database
    for node in nodes:
        if features_name not in node.properties:
            continue  # if features are not available, skip the node

        node_type = node.labels[0].name

        # add feature vector from database to tensor
        data[node_type].x[append_counter[node_type]] = np.add(
            data[node_type].x[append_counter[node_type]],
            np.array(node.properties.get(features_name)),
        )

        # store reindexing and inverse reindexing
        reindexing[node_type][append_counter[node_type]] = node.id
        inv_reindexing[node_type][node.id] = append_counter[node_type]

        # if node type is observed attribute, add classification label to tensor
        if node_type == observed_attribute:
            data[node_type].y[append_counter[node_type]] = int(
                node.properties.get(class_name)
            )

        # increase append_counter by 1
        append_counter[node_type] += 1

    return data, reindexing, inv_reindexing, observed_attribute


def edges_fetching(
    ctx: mgp.ProcCtx, features_name: str, inv_reindexing: defaultdict, data: HeteroData
) -> HeteroData:
    """This procedure fetches the edges from the database and returns them in HeteroData object.

    Args:
        ctx: The context of the procedure.
        features_name: The name of the database features attribute.
        inv_reindexing: The inverse reindexing dictionary.
        data: HeteroData object where edges will be stored.

    Returns:
        HeteroData object with edges.
    """

    edges = []  # variable for storing edges
    edge_types = []  # variable for storing edge types
    append_counter = defaultdict(lambda: 0)  # variable for storing append counter

    # obtain edges from context
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            # edge_type is (from_vertex type, edge name, to_vertex type)
            edge_type = tuple(
                (
                    edge.from_vertex.labels[0].name,
                    edge.type.name,
                    edge.to_vertex.labels[0].name,
                )
            )

            if (
                features_name not in edge.from_vertex.properties
                or features_name not in edge.to_vertex.properties
            ):
                continue  # if from_vertex or out_vertex do not have features, skip the edge

            edge_types.append(edge_type)  # append edge type to list of edge types
            edges.append(edge)  # append edge to list of edges

    edge_types = Counter(
        edge_types
    )  # apply Counter to obtain the number of each edge type

    # set edge_index variables to empty tensors of size 2 x no_edge_type_edges
    for edge_type, no_edge_type_edges in edge_types.items():
        data[edge_type].edge_index = torch.tensor(
            np.zeros((2, no_edge_type_edges)), dtype=torch.long
        )

    for edge in edges:
        (from_vertex_type, edge_name, to_vertex_type) = (
            edge.from_vertex.labels[0].name,
            edge.type.name,
            edge.to_vertex.labels[0].name,
        )
        edge_type = tuple((from_vertex_type, edge_name, to_vertex_type))

        # add edge coordinates to edge_index tensors
        data[edge_type].edge_index[0][append_counter[edge_type]] = int(
            inv_reindexing[from_vertex_type][edge.from_vertex.id]
        )
        data[edge_type].edge_index[1][append_counter[edge_type]] = int(
            inv_reindexing[to_vertex_type][edge.to_vertex.id]
        )

        append_counter[edge_type] += 1

    return data


def masks_generating(
    data: HeteroData, train_ratio: float, observed_attribute: str
) -> HeteroData:
    """This procedure generates the masks for the nodes and edges.

    Args:
        data: HeteroData object with nodes and edges.

    Returns:
        HeteroData object with masks.
    """
    no_observed = np.shape(data[observed_attribute].x)[0]
    masks = np.zeros((no_observed))

    masks = np.add(
        masks,
        np.array(
            list(map(lambda i: 1 if i < train_ratio * no_observed else 0, range(no_observed)))
        ),
    )

    random.shuffle(masks)

    data[observed_attribute].train_mask = torch.tensor(masks, dtype=torch.bool)
    data[observed_attribute].val_mask = torch.tensor(1 - masks, dtype=torch.bool)

    data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)

    return data


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
    data, reindexing, inv_reindexing, observed_attribute = nodes_fetching(
        ctx, features_name, class_name, data
    )

    #################
    # EDGES
    #################

    data = edges_fetching(ctx, features_name, inv_reindexing, data)

    #################
    # MASKS
    #################

    print(data)
    print(observed_attribute)
    data = masks_generating(data, train_ratio, observed_attribute)

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
#         # print(vertex.properties.get(default_values[MemgraphParams.NODE_ID_PROPERTY]))
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
