import enum
from typing import Dict, Tuple, Set, List, Any

import mgp

import numpy as np
import torch
import torch.nn as nn
from mage.tgn.constants import TGNLayerType, MessageFunctionType, MessageAggregatorType, MemoryUpdaterType
from mage.tgn.definitions.tgn import TGNEdgesSelfSupervised, TGN
from dataclasses import dataclass


###################
# params and classes
##################

class TGNParameters:
    NUM_OF_LAYERS = "num_of_layers"
    LAYER_TYPE = "layer_type"
    MEMORY_DIMENSION = "memory_dimension"
    TIME_DIMENSION = "time_dimension"
    NUM_EDGE_FEATURES = "num_edge_features"
    NUM_NODE_FEATURES = "num_node_features"
    MESSAGE_DIMENSION = "message_dimension"
    NUM_NEIGHBORS = "num_neighbors"
    EDGE_FUNCTION_TYPE = "edge_message_function_type"
    MESSAGE_AGGREGATOR_TYPE = "message_aggregator_type"
    MEMORY_UPDATER_TYPE = "memory_updater_type"
    LEARNING_TYPE = "learning_type"  # enum self_supervised or supervised


class LearningType(enum.Enum):
    Supervised = "supervised"
    SelfSupervised = "self_supervised"


@dataclass
class QueryModuleTGN:
    config: Dict[str, Any]
    tgn: TGN
    criterion: nn.BCELoss
    optimizer: torch.optim.Adam
    device: torch.device
    m_loss: List[float]
    fc1: nn.Linear
    fc2: nn.Linear
    act: nn.ReLU


@dataclass
class QueryModuleTGNBatch:
    current_batch_size: int
    sources: np.array
    destinations: np.array
    timestamps: np.array
    edge_idxs: np.array
    node_features: Dict[int, torch.Tensor]
    edge_features: Dict[int, torch.Tensor]
    batch_size: int


##############################


# global tgn training variables
##############################

# used in negative sampling for self_supervised
all_edges: Set[Tuple[int, int]] = set()

# to get all embeddings
all_embeddings: Dict[int, List[float]] = {}

query_module_tgn: QueryModuleTGN
query_module_tgn_batch: QueryModuleTGNBatch


#####################################

# init function

#####################################

def set_tgn(config: Dict[str, Any]):
    """
    This is adapted for self-supervised learning
    todo: check if it works with supervised
    """
    global query_module_tgn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tgn = TGNEdgesSelfSupervised(
        num_of_layers=config[TGNParameters.NUM_OF_LAYERS],
        layer_type=config[TGNParameters.LAYER_TYPE],
        memory_dimension=config[TGNParameters.MEMORY_DIMENSION],
        time_dimension=config[TGNParameters.TIME_DIMENSION],
        num_edge_features=config[TGNParameters.NUM_EDGE_FEATURES],
        num_node_features=config[TGNParameters.NUM_NODE_FEATURES],
        message_dimension=config[TGNParameters.MESSAGE_DIMENSION],
        num_neighbors=config[TGNParameters.NUM_NEIGHBORS],
        # if edge is identity, node must be MLP,
        # if edge is MLP, node also will be MLP
        # thats why we only determine edge type
        edge_message_function_type=config[TGNParameters.EDGE_FUNCTION_TYPE],
        message_aggregator_type=config[TGNParameters.MESSAGE_AGGREGATOR_TYPE],
        memory_updater_type=config[TGNParameters.MEMORY_UPDATER_TYPE]
    ).to(device)

    # set training mode
    tgn.train()

    # init this as global param
    embedding_dim_crit_loss = (config[TGNParameters.MEMORY_DIMENSION] + config[
        TGNParameters.NUM_NODE_FEATURES]) * 2  # concat embeddings of source and destination
    fc1 = nn.Linear(embedding_dim_crit_loss, embedding_dim_crit_loss // 2)
    fc2 = nn.Linear(embedding_dim_crit_loss // 2, 1)
    act = nn.ReLU(inplace=False)

    nn.init.xavier_normal_(fc1.weight)
    nn.init.xavier_normal_(fc2.weight)

    # global params
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=0.0001)
    m_loss = []

    query_module_tgn = QueryModuleTGN(config, tgn, criterion, optimizer, device, m_loss, fc1, fc2, act)


def sample_negative(negative_num: int) -> (np.array, np.array):
    """
    Currently sampling of negative nodes is done in completely random fashion, and it is possible to sample
    source-dest pair that are real edges
    todo fix this problem
    """
    global all_edges

    all_src = list(set([src for src, dest in all_edges]))
    all_dest = list(set([src for src, dest in all_edges]))

    return np.random.choice(all_src, negative_num, replace=True), \
           np.random.choice(all_dest, negative_num, replace=True)


def train_batch_self_supervised():
    global query_module_tgn, query_module_tgn_batch

    sources, destinations, timestamps, edge_features, edge_idxs, node_features, current_batch_size = \
        query_module_tgn_batch.sources, query_module_tgn_batch.destinations, query_module_tgn_batch.timestamps, \
        query_module_tgn_batch.edge_features, query_module_tgn_batch.edge_idxs, \
        query_module_tgn_batch.node_features, query_module_tgn_batch.current_batch_size

    assert len(sources) == len(destinations) == len(timestamps) == len(edge_features) == len(edge_idxs) == current_batch_size, f"Batch size training error"

    negative_src, negative_dest = sample_negative(len(sources))

    graph_data = (
        sources, destinations, negative_src, negative_dest, timestamps, edge_features, edge_idxs, node_features)
    query_module_tgn.optimizer.zero_grad()

    embeddings, embeddings_negative = query_module_tgn.tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    embeddings_source_neg = embeddings_negative[:current_batch_size]
    embeddings_dest_neg = embeddings_negative[current_batch_size:]

    x1, x2 = torch.cat([embeddings_source, embeddings_source_neg], dim=0), torch.cat([embeddings_dest,
                                                                                      embeddings_dest_neg])
    x = torch.cat([x1, x2], dim=1)
    h = query_module_tgn.act(query_module_tgn.fc1(x))
    score = query_module_tgn.fc2(h).squeeze(dim=0)

    pos_score = score[:current_batch_size]
    neg_score = score[current_batch_size:]

    pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()
    with torch.no_grad():
        pos_label = torch.ones(current_batch_size, dtype=torch.float, device=query_module_tgn.device)
        neg_label = torch.zeros(current_batch_size, dtype=torch.float, device=query_module_tgn.device)

    loss = query_module_tgn.criterion(pos_prob.squeeze(), pos_label) + query_module_tgn.criterion(neg_prob.squeeze(),
                                                                                                  neg_label)

    # todo solve a problem of retaining a graph
    loss.backward(retain_graph=True)
    query_module_tgn.optimizer.step()
    query_module_tgn.m_loss.append(loss.item())


def train_batch_supervised():
    # todo implement supervised learning
    pass


def train_epochs(epochs: int):
    for epoch_num in range(epochs):
        pass
        # split all edges in batches
        # train every batch by calling
        # train_batch(edges_batch)

        # print("epoch num", epoch_num, " loss", loss, "mean loss,", np.mean(m_loss))

    # print("nodes unnormalized scores", nodes_unnormalized_scores)


def process_edges(ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]):
    global query_module_tgn_batch, query_module_tgn

    # sources: np.array
    # destinations: np.array
    # timestamps: np.array
    # edge_features: Dict[int, torch.Tensor] (id is edge_idx)
    # edge_idxs: np.array incremental and not repeatable
    # node_features: Dict[int, Torch.Tensor]


    for edge in edges:

        source = ctx.graph.get_vertex_by_id(edge.from_vertex.id)
        src_id = int(edge.from_vertex.id)

        dest = ctx.graph.get_vertex_by_id(edge.to_vertex.id)
        dest_id = int(edge.to_vertex.id)

        src_features = source.properties.get("features", None)
        dest_features = dest.properties.get("features", None)

        timestamp = edge.id
        edge_idx = edge.id

        edge_feature = edge.properties.get("features", None)

        if src_features is None:
            src_features = np.random.randint(0, 100, query_module_tgn.config[TGNParameters.NUM_NODE_FEATURES]) / 100
        else:
            print(src_features)
            print(type(src_features))
            assert type(src_features) is tuple
            src_features = np.array(src_features)

        if dest_features is None:
            dest_features = np.random.randint(0, 100, query_module_tgn.config[TGNParameters.NUM_NODE_FEATURES]) / 100
        else:
            assert type(dest_features) is tuple
            dest_features = np.array(dest_features)

        if edge_feature is None:
            edge_feature = np.random.randint(0, 100, query_module_tgn.config[TGNParameters.NUM_EDGE_FEATURES]) / 100
        else:
            assert type(edge_feature) is tuple
            edge_feature = np.array(edge_feature)

        src_features = torch.tensor(src_features, requires_grad=True)
        dest_features = torch.tensor(dest_features, requires_grad=True)
        edge_feature = torch.tensor(edge_feature, requires_grad=True)

        query_module_tgn_batch.node_features[src_id] = src_features
        query_module_tgn_batch.node_features[dest_id] = dest_features

        # print(edge_idx)
        # print(src_features)
        # print(dest_features)
        # print(edge_feature)

        query_module_tgn_batch.edge_features[edge_idx] = edge_feature

        query_module_tgn_batch.sources = np.append(query_module_tgn_batch.sources, src_id)
        query_module_tgn_batch.destinations = np.append(query_module_tgn_batch.destinations, dest_id)
        query_module_tgn_batch.timestamps = np.append(query_module_tgn_batch.timestamps, timestamp)
        query_module_tgn_batch.edge_idxs = np.append(query_module_tgn_batch.edge_idxs, edge_idx)


    # print("sources", sources)
    # print("destinations", destinations)
    # print("timestamps", timestamps)
    # print("edge_features", edge_features)
    # print("edge_idxs", edge_idxs)
    # print("node_features", node_features)


def save_current_batch_data():
    global query_module_tgn_batch
    sources, destinations = query_module_tgn_batch.sources, query_module_tgn_batch.destinations
    for i, (src, dest) in enumerate(zip(sources, destinations)):
        all_edges.add((src, dest))


def reset_current_batch_data(batch_size:int):
    global query_module_tgn_batch
    query_module_tgn_batch = QueryModuleTGNBatch(0, np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)),
                                                 np.empty((0, 1)), {}, {}, batch_size)


#####################################################

# all available read_procs

#####################################################

@mgp.read_proc
def update(ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]) -> mgp.Record():
    global query_module_tgn_batch, query_module_tgn

    process_edges(ctx, edges)

    query_module_tgn_batch.current_batch_size += len(edges)

    if query_module_tgn_batch.current_batch_size < query_module_tgn_batch.batch_size:
        return mgp.Record()

    learning_type = query_module_tgn.config[TGNParameters.LEARNING_TYPE]
    if learning_type == "self_supervised":
        train_batch_self_supervised()
    else:
        train_batch_supervised()

    save_current_batch_data()
    reset_current_batch_data(batch_size=query_module_tgn_batch.batch_size)

    return mgp.Record()


@mgp.read_proc
def set_params(
        learning_type: str,
        batch_size: int,
        num_of_layers: int,
        layer_type: str,
        memory_dimension: int,
        time_dimension: int,
        num_edge_features: int,
        num_node_features: int,
        message_dimension: int,
        num_neighbors: int,
        edge_message_function_type: str,
        message_aggregator_type: str,
        memory_updater_type: str) -> mgp.Record():
    """
    Warning: Every time you call this function, old TGN object is cleared and process of learning params is
    restarted
    """
    global query_module_tgn_batch
    query_module_tgn_batch = QueryModuleTGNBatch(0, np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)),
                                                 np.empty((0, 1)), {}, {}, batch_size)
    config = {}

    # tgn params
    config[TGNParameters.NUM_OF_LAYERS] = num_of_layers
    config[TGNParameters.MEMORY_DIMENSION] = memory_dimension
    config[TGNParameters.TIME_DIMENSION] = time_dimension
    config[TGNParameters.NUM_EDGE_FEATURES] = num_edge_features
    config[TGNParameters.NUM_NODE_FEATURES] = num_node_features
    config[TGNParameters.MESSAGE_DIMENSION] = message_dimension
    config[TGNParameters.NUM_NEIGHBORS] = num_neighbors

    config[TGNParameters.LAYER_TYPE] = get_tgn_layer_enum(layer_type)
    config[TGNParameters.EDGE_FUNCTION_TYPE] = get_edge_message_function_type(edge_message_function_type)
    config[TGNParameters.MESSAGE_AGGREGATOR_TYPE] = get_message_aggregator_type(message_aggregator_type)
    config[TGNParameters.MEMORY_UPDATER_TYPE] = get_memory_updater_type(memory_updater_type)

    # learning params
    config[TGNParameters.LEARNING_TYPE] = learning_type

    # set tgn
    set_tgn(config)

    # todo add to return
    # print(config)
    return mgp.Record()


#####################################

# helper functions

#####################################
def get_tgn_layer_enum(layer_type: str) -> TGNLayerType:
    if TGNLayerType(layer_type) is TGNLayerType.GraphAttentionEmbedding:
        return TGNLayerType.GraphAttentionEmbedding
    elif TGNLayerType(layer_type) is TGNLayerType.GraphSumEmbedding:
        return TGNLayerType.GraphSumEmbedding
    else:
        raise Exception(f"Wrong layer type, expected {TGNLayerType.GraphAttentionEmbedding} "
                        f"or {TGNLayerType.GraphSumEmbedding} ")


def get_edge_message_function_type(message_function_type: str) -> MessageFunctionType:
    if MessageFunctionType(message_function_type) is MessageFunctionType.Identity:
        return MessageFunctionType.Identity
    elif MessageFunctionType(message_function_type) is MessageFunctionType.MLP:
        return MessageFunctionType.MLP
    else:
        raise Exception(f"Wrong message function type, expected {MessageFunctionType.Identity} "
                        f"or {MessageFunctionType.MLP} ")


def get_message_aggregator_type(message_aggregator_type: str) -> MessageAggregatorType:
    if MessageAggregatorType(message_aggregator_type) is MessageAggregatorType.Mean:
        return MessageAggregatorType.Mean
    elif MessageAggregatorType(message_aggregator_type) is MessageAggregatorType.Last:
        return MessageAggregatorType.Last
    else:
        raise Exception(f"Wrong message aggregator type, expected {MessageAggregatorType.Last} "
                        f"or {MessageAggregatorType.Mean} ")


def get_memory_updater_type(memory_updater_type: str) -> MemoryUpdaterType:
    if MemoryUpdaterType(memory_updater_type) is MemoryUpdaterType.GRU:
        return MessageFunctionType.Mean

    elif MemoryUpdaterType(memory_updater_type) is MemoryUpdaterType.RNN:
        return MemoryUpdaterType.RNN

    elif MemoryUpdaterType(memory_updater_type) is MemoryUpdaterType.LSTM:
        return MemoryUpdaterType.LSTM
    else:
        raise Exception(f"Wrong memory updater type, expected {MemoryUpdaterType.GRU} "
                        f", {MemoryUpdaterType.RNN} or or {MemoryUpdaterType.LSTM}")
