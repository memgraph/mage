import enum
from typing import Dict, Tuple, Set, List, Any

import mgp

import numpy as np
import torch
import torch.nn as nn
from mage.tgn.constants import (
    TGNLayerType,
    MessageFunctionType,
    MessageAggregatorType,
    MemoryUpdaterType,
)
from mage.tgn.definitions.tgn import TGN, TGNGraphSumEdgeSelfSupervised, TGNGraphSumSupervised, \
    TGNGraphAttentionSupervised
from dataclasses import dataclass

from mage.tgn.definitions.tgn import TGNGraphAttentionEdgeSelfSupervised
from mage.tgn.helper.simple_mlp import MLP


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
    NUM_ATTENTION_HEADS = "num_attention_heads"


class OptimizerParameters:
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"


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
    mlp: MLP


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
    labels: np.array


##############################
# global tgn training variables
##############################

# used in negative sampling for self_supervised
all_edges: Set[Tuple[int, int]] = set()

# to get all embeddings
all_embeddings: Dict[int, np.array] = {}

query_module_tgn: QueryModuleTGN
query_module_tgn_batch: QueryModuleTGNBatch

tgn_learning_type = None
tgn_training_losses = np.empty(0)


#####################################

# init function

#####################################

def set_tgn(learning_type: LearningType, tgn_config: Dict[str, any], optimizer_config: Dict[str, any]):
    global query_module_tgn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if learning_type == LearningType.SelfSupervised:
        tgn, mlp = get_tgn_self_supervised(tgn_config, device)
    else:
        tgn, mlp = get_tgn_supervised(tgn_config, device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=optimizer_config[OptimizerParameters.LEARNING_RATE],
                                 weight_decay=optimizer_config[OptimizerParameters.WEIGHT_DECAY])
    m_loss = []

    query_module_tgn = QueryModuleTGN(tgn_config, tgn, criterion, optimizer, device, m_loss, mlp)


def get_tgn_self_supervised(config: Dict[str, Any], device: torch.device):
    """
    Set parameters for self supervised learning. Here we try to predict edges.
    """

    if config[TGNParameters.LAYER_TYPE] == TGNLayerType.GraphSumEmbedding:
        tgn = TGNGraphSumEdgeSelfSupervised(**config).to(device)
    else:
        tgn = TGNGraphAttentionEdgeSelfSupervised(**config).to(device)

    tgn.train()

    src_dest_embedd_concat_dim = (config[TGNParameters.MEMORY_DIMENSION] + config[TGNParameters.NUM_NODE_FEATURES]) * 2

    # used as probability calculator for edge
    mlp = MLP([src_dest_embedd_concat_dim, src_dest_embedd_concat_dim // 2, 1])

    return tgn, mlp


def get_tgn_supervised(config: Dict[str, Any], device: torch.device):
    """
    """

    if config[TGNParameters.LAYER_TYPE] == TGNLayerType.GraphSumEmbedding:
        tgn = TGNGraphSumSupervised(**config).to(device)
    else:
        tgn = TGNGraphAttentionSupervised(**config).to(device)

    tgn.train()

    src_dest_embedd_concat_dim = config[TGNParameters.MEMORY_DIMENSION] + config[
        TGNParameters.NUM_NODE_FEATURES]

    # used as probability calculator for edge
    mlp = MLP([src_dest_embedd_concat_dim, src_dest_embedd_concat_dim // 2, 1])

    return tgn, mlp


#
# Training - self_supervised
#


def sample_negative(negative_num: int) -> (np.array, np.array):
    """
    Currently sampling of negative nodes is done in completely random fashion, and it is possible to sample
    source-dest pair that are real edges
    """
    global all_edges

    all_src = list(set([src for src, dest in all_edges]))
    all_dest = list(set([src for src, dest in all_edges]))

    return np.random.choice(all_src, negative_num, replace=True), np.random.choice(
        all_dest, negative_num, replace=True
    )


def train_batch_self_supervised():
    """
    Uses sources, destinations, timestamps, edge_features and node_features from transactions.
    It is possible that current_batch_size is not always consistent, but it is always greater than minimum required.
    """
    global query_module_tgn, query_module_tgn_batch, all_embeddings

    (
        sources,
        destinations,
        timestamps,
        edge_features,
        edge_idxs,
        node_features,
        current_batch_size,
    ) = (
        query_module_tgn_batch.sources,
        query_module_tgn_batch.destinations,
        query_module_tgn_batch.timestamps,
        query_module_tgn_batch.edge_features,
        query_module_tgn_batch.edge_idxs,
        query_module_tgn_batch.node_features,
        query_module_tgn_batch.current_batch_size,
    )

    assert (
            len(sources)
            == len(destinations)
            == len(timestamps)
            == len(edge_features)
            == len(edge_idxs)
            == current_batch_size
    ), f"Batch size training error"

    negative_src, negative_dest = sample_negative(len(sources))

    graph_data = (
        sources,
        destinations,
        negative_src,
        negative_dest,
        timestamps,
        edge_idxs,
        edge_features,
        node_features,
    )
    query_module_tgn.optimizer.zero_grad()

    embeddings, embeddings_negative = query_module_tgn.tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    embeddings_source_neg = embeddings_negative[:current_batch_size]
    embeddings_dest_neg = embeddings_negative[current_batch_size:]

    x1, x2 = torch.cat([embeddings_source, embeddings_source_neg], dim=0), torch.cat(
        [embeddings_dest, embeddings_dest_neg]
    )
    x = torch.cat([x1, x2], dim=1)
    score = query_module_tgn.mlp(x).squeeze(dim=0)

    pos_score = score[:current_batch_size]
    neg_score = score[current_batch_size:]

    pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()
    with torch.no_grad():
        pos_label = torch.ones(
            current_batch_size, dtype=torch.float, device=query_module_tgn.device
        )
        neg_label = torch.zeros(
            current_batch_size, dtype=torch.float, device=query_module_tgn.device
        )

    loss = query_module_tgn.criterion(
        pos_prob.squeeze(), pos_label
    ) + query_module_tgn.criterion(neg_prob.squeeze(), neg_label)

    loss.backward()
    query_module_tgn.optimizer.step()
    query_module_tgn.m_loss.append(loss.item())

    embeddings_source_npy = embeddings_source.cpu().detach().numpy()
    embeddings_dest_npy = embeddings_dest.cpu().detach().numpy()

    for i, node in enumerate(sources):
        all_embeddings[node] = embeddings_source_npy[i]

    for i, node in enumerate(destinations):
        all_embeddings[node] = embeddings_dest_npy[i]


#
# Training - supervised
#


def train_batch_supervised():
    global query_module_tgn, query_module_tgn_batch, all_embeddings

    (
        sources,
        destinations,
        timestamps,
        edge_features,
        edge_idxs,
        node_features,
        current_batch_size,
        labels,
    ) = (
        query_module_tgn_batch.sources,
        query_module_tgn_batch.destinations,
        query_module_tgn_batch.timestamps,
        query_module_tgn_batch.edge_features,
        query_module_tgn_batch.edge_idxs,
        query_module_tgn_batch.node_features,
        query_module_tgn_batch.current_batch_size,
        query_module_tgn_batch.labels,
    )

    assert (
            len(sources)
            == len(destinations)
            == len(timestamps)
            == len(edge_features)
            == len(edge_idxs)
            == current_batch_size
            == len(labels)
    ), f"Batch size training error"



    graph_data = (
        sources,
        destinations,
        timestamps,
        edge_idxs,
        edge_features,
        node_features,
    )
    query_module_tgn.optimizer.zero_grad()

    embeddings = query_module_tgn.tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    x= torch.cat([embeddings_source, embeddings_dest], dim=0)


    score = query_module_tgn.mlp(x).squeeze(dim=0)

    src_score = score[:current_batch_size]
    dest_score = score[current_batch_size:]

    src_prob, dest_prob = src_score.sigmoid(), dest_score.sigmoid()
    with torch.no_grad():
        src_label = torch.tensor(
            labels[:,0], dtype=torch.float, device=query_module_tgn.device
        )
        dest_label = torch.tensor(
            labels[:,1], dtype=torch.float, device=query_module_tgn.device
        )


    loss = query_module_tgn.criterion(
        src_prob.squeeze(), src_label.squeeze()
    ) + query_module_tgn.criterion( dest_prob.squeeze(), dest_label.squeeze())

    loss.backward()
    query_module_tgn.optimizer.step()
    query_module_tgn.m_loss.append(loss.item())

    embeddings_source_npy = embeddings_source.cpu().detach().numpy()
    embeddings_dest_npy = embeddings_dest.cpu().detach().numpy()

    for i, node in enumerate(sources):
        all_embeddings[node] = embeddings_source_npy[i]

    for i, node in enumerate(destinations):
        all_embeddings[node] = embeddings_dest_npy[i]


def train_epochs(epochs: int):
    pass


def process_edges(ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]):
    global query_module_tgn_batch, query_module_tgn, all_edges

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

        all_edges.add((src_id, dest_id))

        src_features = source.properties.get("features", None)
        dest_features = dest.properties.get("features", None)

        src_label = source.properties.get("label", 0)
        dest_label = source.properties.get("label", 0)

        timestamp = edge.id
        edge_idx = int(edge.id)

        edge_feature = edge.properties.get("features", None)

        if src_features is None:
            # If node feature doesn't exist on one node, probably won't on any. Maybe this is wrong approach.
            src_features = (
                    np.random.randint(
                        0, 100, query_module_tgn.config[TGNParameters.NUM_NODE_FEATURES]
                    )
                    / 100
            )
        else:
            assert type(src_features) is tuple
            src_features = np.array(src_features)

        if dest_features is None:
            dest_features = (
                    np.random.randint(
                        0, 100, query_module_tgn.config[TGNParameters.NUM_NODE_FEATURES]
                    )
                    / 100
            )
        else:
            assert type(dest_features) is tuple
            dest_features = np.array(dest_features)

        if edge_feature is None:
            edge_feature = (
                    np.random.randint(
                        0, 100, query_module_tgn.config[TGNParameters.NUM_EDGE_FEATURES]
                    )
                    / 100
            )
        else:
            assert type(edge_feature) is tuple
            edge_feature = np.array(edge_feature)

        src_features = torch.tensor(src_features, requires_grad=True, dtype=torch.float)
        dest_features = torch.tensor(
            dest_features, requires_grad=True, dtype=torch.float
        )
        edge_feature = torch.tensor(edge_feature, requires_grad=True, dtype=torch.float)

        query_module_tgn_batch.node_features[src_id] = src_features
        query_module_tgn_batch.node_features[dest_id] = dest_features

        query_module_tgn_batch.edge_features[edge_idx] = edge_feature

        query_module_tgn_batch.sources = np.append(
            query_module_tgn_batch.sources, src_id
        )
        query_module_tgn_batch.destinations = np.append(
            query_module_tgn_batch.destinations, dest_id
        )
        query_module_tgn_batch.timestamps = np.append(
            query_module_tgn_batch.timestamps, timestamp
        )
        query_module_tgn_batch.edge_idxs = np.append(
            query_module_tgn_batch.edge_idxs, edge_idx
        )
        query_module_tgn_batch.labels = np.append(
            query_module_tgn_batch.labels, np.array([src_label, dest_label])
        )


def reset_tgn_batch(batch_size: int):
    global query_module_tgn_batch
    query_module_tgn_batch = QueryModuleTGNBatch(
        0,
        np.empty((0, 1), dtype=int),
        np.empty((0, 1), dtype=int),
        np.empty((0, 1), dtype=int),
        np.empty((0, 1), dtype=int),
        {},
        {},
        batch_size,
        np.empty((1, 2), dtype=int),
    )


def reset_tgn():
    global all_embeddings, query_module_tgn_batch, query_module_tgn, all_edges

    all_embeddings = {}
    reset_tgn_batch(0)
    all_edges = set()


#####################################################

# all available read_procs

#####################################################


@mgp.read_proc
def revert_from_database(ctx: mgp.ProcCtx) -> mgp.Record():
    """
    todo implement
    Revert from database and potential file in var/log/ to which we can save params
    """
    pass


@mgp.read_proc
def save_tgn_params(ctx: mgp.ProcCtx) -> mgp.Record():
    """
    todo implement
    After every batch we could add saving params as checkpoints to var/log/memgraph
    This is how it is done usually in ML
    """
    pass


@mgp.read_proc
def reset(ctx: mgp.ProcCtx) -> mgp.Record():
    reset_tgn()
    return mgp.Record()


@mgp.read_proc
def get(ctx: mgp.ProcCtx) -> mgp.Record(node=mgp.Vertex, embedding=mgp.List[float]):
    global all_embeddings

    embeddings_dict = {}

    for node_id, embedding in all_embeddings.items():
        embeddings_dict[node_id] = [float(e) for e in embedding]

    return [
        mgp.Record(node=ctx.graph.get_vertex_by_id(node_id), embedding=embedding)
        for node_id, embedding in embeddings_dict.items()
    ]


@mgp.read_proc
def update(ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]) -> mgp.Record():
    global query_module_tgn_batch, query_module_tgn, tgn_learning_type

    process_edges(ctx, edges)

    query_module_tgn_batch.current_batch_size += len(edges)

    if query_module_tgn_batch.current_batch_size < query_module_tgn_batch.batch_size:
        return mgp.Record()

    if tgn_learning_type == LearningType.SelfSupervised:
        train_batch_self_supervised()
    elif tgn_learning_type == LearningType.Supervised:
        train_batch_supervised()
    else:
        raise Exception(
            f"Wrong learning type, expected {LearningType.SelfSupervised} or {LearningType.Supervised}"
        )
    reset_tgn_batch(batch_size=query_module_tgn_batch.batch_size)

    return mgp.Record()


@mgp.read_proc
def set_params(
        ctx: mgp.ProcCtx,
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
        memory_updater_type: str,
        num_attention_heads=1,
        learning_rate=2e-2,
        weight_decay=5e-3,
) -> mgp.Record():
    """
    Warning: Every time you call this function, old TGN object is cleared and process of learning params is
    restarted
    """
    global query_module_tgn_batch, tgn_learning_type

    reset_tgn_batch(batch_size)

    tgn_config = {
        TGNParameters.NUM_OF_LAYERS: num_of_layers,
        TGNParameters.MEMORY_DIMENSION: memory_dimension,
        TGNParameters.TIME_DIMENSION: time_dimension,
        TGNParameters.NUM_EDGE_FEATURES: num_edge_features,
        TGNParameters.NUM_NODE_FEATURES: num_node_features,
        TGNParameters.MESSAGE_DIMENSION: message_dimension,
        TGNParameters.NUM_NEIGHBORS: num_neighbors,
        TGNParameters.LAYER_TYPE: get_tgn_layer_enum(layer_type),
        TGNParameters.EDGE_FUNCTION_TYPE: get_edge_message_function_type(
            edge_message_function_type
        ),
        TGNParameters.MESSAGE_AGGREGATOR_TYPE: get_message_aggregator_type(
            message_aggregator_type
        ),
        TGNParameters.MEMORY_UPDATER_TYPE: get_memory_updater_type(memory_updater_type),
    }
    optimizer_config = {
        OptimizerParameters.LEARNING_RATE: learning_rate,
        OptimizerParameters.WEIGHT_DECAY: weight_decay
    }
    # tgn params

    if tgn_config[TGNParameters.LAYER_TYPE] == TGNLayerType.GraphAttentionEmbedding:
        tgn_config[TGNParameters.NUM_ATTENTION_HEADS] = num_attention_heads

    # set tgn
    tgn_learning_type = get_learning_type(learning_type)

    set_tgn(tgn_learning_type, tgn_config, optimizer_config)

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
        raise Exception(
            f"Wrong layer type, expected {TGNLayerType.GraphAttentionEmbedding} "
            f"or {TGNLayerType.GraphSumEmbedding} "
        )


def get_edge_message_function_type(message_function_type: str) -> MessageFunctionType:
    if MessageFunctionType(message_function_type) is MessageFunctionType.Identity:
        return MessageFunctionType.Identity
    elif MessageFunctionType(message_function_type) is MessageFunctionType.MLP:
        return MessageFunctionType.MLP
    else:
        raise Exception(
            f"Wrong message function type, expected {MessageFunctionType.Identity} "
            f"or {MessageFunctionType.MLP} "
        )


def get_message_aggregator_type(message_aggregator_type: str) -> MessageAggregatorType:
    if MessageAggregatorType(message_aggregator_type) is MessageAggregatorType.Mean:
        return MessageAggregatorType.Mean
    elif MessageAggregatorType(message_aggregator_type) is MessageAggregatorType.Last:
        return MessageAggregatorType.Last
    else:
        raise Exception(
            f"Wrong message aggregator type, expected {MessageAggregatorType.Last} "
            f"or {MessageAggregatorType.Mean} "
        )


def get_memory_updater_type(memory_updater_type: str) -> MemoryUpdaterType:
    if MemoryUpdaterType(memory_updater_type) is MemoryUpdaterType.GRU:
        return MemoryUpdaterType.GRU

    elif MemoryUpdaterType(memory_updater_type) is MemoryUpdaterType.RNN:
        return MemoryUpdaterType.RNN

    else:
        raise Exception(
            f"Wrong memory updater type, expected {MemoryUpdaterType.GRU} or"
            f", {MemoryUpdaterType.RNN}"
        )


def get_learning_type(learning_type: str) -> LearningType:
    if LearningType(learning_type) is LearningType.SelfSupervised:
        return LearningType.SelfSupervised

    elif LearningType(learning_type) is LearningType.Supervised:
        return LearningType.Supervised

    else:
        raise Exception(
            f"Wrong learning type, expected {LearningType.Supervised} or"
            f", {LearningType.SelfSupervised}"
        )
