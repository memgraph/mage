import dataclasses
import enum
import time
from math import ceil
from typing import Dict, Tuple, Set, List, Any

import mgp

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from mage.tgn.constants import (
    TGNLayerType,
    MessageFunctionType,
    MessageAggregatorType,
    MemoryUpdaterType,
)
from mage.tgn.definitions.tgn import TGN

from mage.tgn.helper.simple_mlp import MLP

from mage.tgn.definitions.instances import (
    TGNGraphSumEdgeSelfSupervised,
    TGNGraphSumSupervised,
    TGNGraphAttentionSupervised,
    TGNGraphAttentionEdgeSelfSupervised,
)


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
    DEVICE = "device"


class OptimizerParameters:
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"


class LearningType(enum.Enum):
    Supervised = "supervised"
    SelfSupervised = "self_supervised"


class TGNMode(enum.Enum):
    Train = "train"
    Eval = "eval"


@dataclasses.dataclass
class QueryModuleTGN:
    config: Dict[str, Any]
    tgn: TGN
    criterion: nn.BCELoss
    optimizer: torch.optim.Adam
    device: torch.device
    m_loss: List[float]
    mlp: MLP
    tgn_mode: TGNMode
    learning_type: LearningType


@dataclasses.dataclass
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

train_edges: List[mgp.Edge] = []
eval_edges: List[mgp.Edge] = []

# to get all embeddings
all_embeddings: Dict[int, np.array] = {}

query_module_tgn: QueryModuleTGN
query_module_tgn_batch: QueryModuleTGNBatch


#####################################

# init function

#####################################


def set_tgn(
    learning_type: LearningType,
    tgn_config: Dict[str, any],
    optimizer_config: Dict[str, any],
):
    global query_module_tgn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgn_config[TGNParameters.DEVICE] = device

    if learning_type == LearningType.SelfSupervised:
        tgn, mlp = get_tgn_self_supervised(tgn_config, device)
    else:
        tgn, mlp = get_tgn_supervised(tgn_config, device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        tgn.parameters(),
        lr=optimizer_config[OptimizerParameters.LEARNING_RATE],
        weight_decay=optimizer_config[OptimizerParameters.WEIGHT_DECAY],
    )
    m_loss = []

    query_module_tgn = QueryModuleTGN(
        tgn_config,
        tgn,
        criterion,
        optimizer,
        device,
        m_loss,
        mlp,
        TGNMode.Train,
        learning_type,
    )


def get_tgn_self_supervised(config: Dict[str, Any], device: torch.device):
    """
    Set parameters for self supervised learning. Here we try to predict edges.
    """

    if config[TGNParameters.LAYER_TYPE] == TGNLayerType.GraphSumEmbedding:
        tgn = TGNGraphSumEdgeSelfSupervised(**config).to(device)
    else:
        tgn = TGNGraphAttentionEdgeSelfSupervised(**config).to(device)

    # when we TGN outputs features for source and destinations, since we are working with edges and edge predictions
    # we will concatenate their features together and get prediction with MLP whether it is edge or it isn't
    mlp_in_features_dim = (
        config[TGNParameters.MEMORY_DIMENSION] + config[TGNParameters.NUM_NODE_FEATURES]
    ) * 2

    mlp = MLP([mlp_in_features_dim, mlp_in_features_dim // 2, 1]).to(device=device)

    return tgn, mlp


def get_tgn_supervised(config: Dict[str, Any], device: torch.device):
    """ """

    if config[TGNParameters.LAYER_TYPE] == TGNLayerType.GraphSumEmbedding:
        tgn = TGNGraphSumSupervised(**config).to(device)
    else:
        tgn = TGNGraphAttentionSupervised(**config).to(device)

    mlp_in_features_dim = (
        config[TGNParameters.MEMORY_DIMENSION] + config[TGNParameters.NUM_NODE_FEATURES]
    )

    # used as probability calculator for label
    mlp = MLP([mlp_in_features_dim, 64, 1]).to(device=device)

    return tgn, mlp


#
# Helper functions
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


def unpack_tgn_batch_data():
    global query_module_tgn_batch
    return dataclasses.astuple(query_module_tgn_batch)


def update_mode_reset_grads_check_dims():
    global query_module_tgn, query_module_tgn_batch

    if query_module_tgn.tgn_mode == TGNMode.Train:
        # set training mode
        query_module_tgn.tgn.train()

        query_module_tgn.optimizer.zero_grad()
        query_module_tgn.tgn.detach_tensor_grads()

        # todo add so that we only work with latest 128 neighbors
        # query_module_tgn.tgn.subsample_neighborhood()
    else:
        query_module_tgn.tgn.eval()

    (
        _,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        _,
        edge_features,
        _,
        labels,
    ) = unpack_tgn_batch_data()
    assert (
        len(sources)
        == len(destinations)
        == len(timestamps)
        == len(edge_features)
        == len(edge_idxs)
        == len(labels)
    ), f"Batch size training error"


def update_embeddings(
    embeddings_source: np.array,
    embeddings_dest: np.array,
    sources: np.array,
    destinations: np.array,
) -> None:
    global all_embeddings
    for i, node in enumerate(sources):
        all_embeddings[node] = embeddings_source[i]

    for i, node in enumerate(destinations):
        all_embeddings[node] = embeddings_dest[i]


#
# Batch processing - self_supervised
#
def process_batch_self_supervised() -> float:
    """
    Uses sources, destinations, timestamps, edge_features and node_features from transactions.
    It is possible that current_batch_size is not always consistent, but it is always greater than minimum required.
    """
    global query_module_tgn, query_module_tgn_batch

    # do all necessary checks and updates of gradients
    update_mode_reset_grads_check_dims()

    (
        _,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        node_features,
        edge_features,
        _,
        labels,
    ) = unpack_tgn_batch_data()

    current_batch_size = len(sources)
    negative_src, negative_dest = sample_negative(current_batch_size)

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

    embeddings, embeddings_negative = query_module_tgn.tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    embeddings_source_neg = embeddings_negative[:current_batch_size]
    embeddings_dest_neg = embeddings_negative[current_batch_size:]

    # first row concatenation
    x1, x2 = torch.cat([embeddings_source, embeddings_source_neg], dim=0), torch.cat(
        [embeddings_dest, embeddings_dest_neg], dim=0
    )
    # columns concatenation
    x = torch.cat([x1, x2], dim=1)
    # score calculation = shape (num_positive_edges + num_negative_edges, 1) ->
    # (num_positive_edges + num_negative_edges)
    # num_positive_edges == num_negative_edges == current_batch_size
    # todo update so that  num_negative_edges in range [10,25]
    score = query_module_tgn.mlp(x).squeeze(dim=0)

    pos_score = score[:current_batch_size]
    neg_score = score[current_batch_size:]

    pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()

    if query_module_tgn.tgn_mode == TGNMode.Train:

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

    print("POS PROB | NEG PROB", pos_prob.squeeze().cpu(), neg_prob.squeeze().cpu())
    pred_score = np.concatenate(
        [
            (pos_prob.squeeze()).cpu().detach().numpy(),
            (neg_prob.squeeze()).cpu().detach().numpy(),
        ]
    )
    true_label = np.concatenate(
        [np.ones(current_batch_size), np.zeros(current_batch_size)]
    )

    accuracy = average_precision_score(true_label, pred_score)

    # update embeddings to newest ones that we can return on user request
    update_embeddings(
        embeddings_source.cpu().detach().numpy(),
        embeddings_dest.cpu().detach().numpy(),
        sources,
        destinations,
    )

    return accuracy


#
# Training - supervised
#


def process_batch_supervised() -> float:
    global query_module_tgn, query_module_tgn_batch

    # do all necessary checks and updates of gradients
    update_mode_reset_grads_check_dims()

    (
        _,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        node_features,
        edge_features,
        current_batch_size,
        labels,
    ) = unpack_tgn_batch_data()

    graph_data = (
        sources,
        destinations,
        timestamps,
        edge_idxs,
        edge_features,
        node_features,
    )

    embeddings = query_module_tgn.tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    x = torch.cat([embeddings_source, embeddings_dest], dim=0)  # along rows

    score = query_module_tgn.mlp(x).squeeze(dim=0)

    src_score = score[:current_batch_size]
    dest_score = score[current_batch_size:]

    src_prob, dest_prob = src_score.sigmoid(), dest_score.sigmoid()

    pred_score = np.concatenate(
        [
            (src_prob.squeeze()).cpu().detach().numpy(),
            (dest_prob.squeeze()).cpu().detach().numpy(),
        ]
    )
    true_label = np.concatenate([np.array(labels[:, 0]), np.array(labels[:, 1])])

    accuracy = average_precision_score(true_label, pred_score)

    # update embeddings to newest ones that we can return on user request
    update_embeddings(
        embeddings_source.cpu().detach().numpy(),
        embeddings_dest.cpu().detach().numpy(),
        sources,
        destinations,
    )

    if query_module_tgn.tgn_mode == TGNMode.Eval:
        return accuracy

    # backprop only in case of training
    with torch.no_grad():
        src_label = torch.tensor(
            labels[:, 0], dtype=torch.float, device=query_module_tgn.device
        )
        dest_label = torch.tensor(
            labels[:, 1], dtype=torch.float, device=query_module_tgn.device
        )

    loss = query_module_tgn.criterion(
        src_prob.squeeze(), src_label.squeeze()
    ) + query_module_tgn.criterion(dest_prob.squeeze(), dest_label.squeeze())
    loss.backward()
    query_module_tgn.optimizer.step()
    query_module_tgn.m_loss.append(loss.item())

    return accuracy


def process_edges(edges: mgp.List[mgp.Edge]):
    global query_module_tgn_batch, query_module_tgn, all_edges

    for edge in edges:

        source = edge.from_vertex
        src_id = edge.from_vertex.id

        dest = edge.to_vertex
        dest_id = int(edge.to_vertex.id)

        all_edges.add((src_id, dest_id))

        src_features = source.properties.get("features", None)
        dest_features = dest.properties.get("features", None)

        src_label = source.properties.get("label", 0)
        dest_label = dest.properties.get("label", 0)

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

        src_features = torch.tensor(
            src_features,
            requires_grad=True,
            device=query_module_tgn.device,
            dtype=torch.float,
        )
        dest_features = torch.tensor(
            dest_features,
            requires_grad=True,
            dtype=torch.float,
            device=query_module_tgn.device,
        )
        edge_feature = torch.tensor(
            edge_feature,
            requires_grad=True,
            dtype=torch.float,
            device=query_module_tgn.device,
        )

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
            query_module_tgn_batch.labels, np.array([[src_label, dest_label]]), axis=0
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
        np.empty((0, 2), dtype=int),
    )


def reset_tgn():
    global all_embeddings, query_module_tgn_batch, query_module_tgn, all_edges

    all_embeddings = {}
    reset_tgn_batch(0)
    all_edges = set()


def update_batch(edges: mgp.List[mgp.Edge]) -> int:
    global query_module_tgn_batch, query_module_tgn

    edge_processing_start_time = time.time()
    process_edges(edges)
    edge_processing_time = time.time() - edge_processing_start_time

    query_module_tgn_batch.current_batch_size += len(edges)

    return int(edge_processing_time)


def train_eval_epochs(
    epochs: int, train_edges: List[mgp.Edge], eval_edges: List[mgp.Edge]
):
    global query_module_tgn, query_module_tgn_batch, all_edges

    batch_size = query_module_tgn_batch.batch_size
    num_train_edges = len(train_edges)
    num_train_batches = ceil(num_train_edges / batch_size)

    num_eval_edges = len(eval_edges)
    num_eval_batches = ceil(num_eval_edges / batch_size)
    assert batch_size > 0

    for epoch in range(epochs):
        query_module_tgn.tgn.init_memory()
        query_module_tgn.tgn.init_temporal_neighborhood()
        query_module_tgn.tgn.init_message_store()

        all_edges = set()

        query_module_tgn.m_loss = []

        reset_tgn_batch(batch_size)

        # train
        query_module_tgn.tgn_mode = TGNMode.Train
        query_module_tgn.tgn.train()
        for i in range(num_train_batches):
            # sample edges we need
            start_index_train_batch = i * batch_size
            end_index_train_batch = min((i + 1) * batch_size, num_train_edges - 1)
            current_batch = train_edges[start_index_train_batch:end_index_train_batch]

            # prepare batch
            process_edges(current_batch)

            batch_start_time = time.time()
            accuracy = (
                process_batch_self_supervised()
                if query_module_tgn.learning_type == LearningType.SelfSupervised
                else process_batch_supervised()
            )
            batch_process_time = time.time() - batch_start_time

            train_batch_loss = query_module_tgn.m_loss[-1]
            train_avg_loss = sum(query_module_tgn.m_loss) / len(query_module_tgn.m_loss)

            # reset for next batch
            reset_tgn_batch(batch_size=query_module_tgn_batch.batch_size)

            # create record, but print for now
            print(
                f"EPOCH {epoch} || BATCH {i}, | batch_process_time={batch_process_time} | train_batch_loss={train_batch_loss} | train_avg_loss={train_avg_loss} | accuracy={accuracy}"
            )

        # eval
        query_module_tgn.tgn_mode = TGNMode.Eval
        query_module_tgn.tgn.eval()
        for i in range(num_eval_batches):
            # sample edges we need
            start_index_eval_batch = i * batch_size
            end_index_eval_batch = min((i + 1) * batch_size, num_eval_edges - 1)
            current_batch = eval_edges[start_index_eval_batch:end_index_eval_batch]

            # prepare batch
            process_edges(current_batch)

            batch_start_time = time.time()
            accuracy = (
                process_batch_self_supervised()
                if query_module_tgn.learning_type == LearningType.SelfSupervised
                else process_batch_supervised()
            )
            batch_process_time = time.time() - batch_start_time

            # reset for next batch
            reset_tgn_batch(batch_size=query_module_tgn_batch.batch_size)

            # create record, but print for now
            print(
                f"EPOCH {epoch} || BATCH {i}, | batch_process_time={batch_process_time}  | accuracy={accuracy}"
            )


#####################################################

# all available read_procs

#####################################################


@mgp.read_proc
def process_epochs(
    ctx: mgp.ProcCtx, num_epochs: int, train_eval_split: float = 0.8
) -> mgp.Record():
    vertices = ctx.graph.vertices
    curr_all_edges = []
    for vertex in vertices:
        curr_all_edges.extend(list(vertex.out_edges))

    curr_all_edges = sorted(curr_all_edges, key=lambda x: x.id)
    num_edges = len(curr_all_edges)
    train_eval_index_split = int(num_edges * train_eval_split)

    train_eval_epochs(
        num_epochs,
        curr_all_edges[:train_eval_index_split],
        curr_all_edges[train_eval_index_split:],
    )
    return mgp.Record()


@mgp.read_proc
def set_mode(ctx: mgp.ProcCtx, mode: str) -> mgp.Record():
    global query_module_tgn
    if query_module_tgn.tgn is None:
        raise Exception("TGN module is not set")
    if mode == "train":
        query_module_tgn.tgn_mode = TGNMode.Train
    elif mode == "eval":
        query_module_tgn.tgn_mode = TGNMode.Eval
    else:
        raise Exception(f"Expected mode {TGNMode.Train} or {TGNMode.Eval}, got {mode}")

    return mgp.Record()


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
def update(
    ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]
) -> mgp.Record(
    batch_process_time=mgp.Number,
    edge_process_time=mgp.Number,
    train_batch_loss=mgp.Number,
    train_avg_loss=mgp.Number,
    accuracy=mgp.Number,
):
    global query_module_tgn_batch, query_module_tgn
    batch_process_time, train_batch_loss, train_avg_loss, accuracy = [0, 0, 0, 0]

    edge_process_time = update_batch(edges)

    current_batch_size = query_module_tgn_batch.current_batch_size

    if current_batch_size < query_module_tgn_batch.batch_size:
        return mgp.Record(
            batch_process_time=batch_process_time,
            edge_process_time=edge_process_time,
            train_batch_loss=train_batch_loss,
            train_avg_loss=train_avg_loss,
            accuracy=accuracy,
        )

    batch_start_time = time.time()
    accuracy = (
        process_batch_self_supervised()
        if query_module_tgn.learning_type == LearningType.SelfSupervised
        else process_batch_supervised()
    )
    batch_process_time = time.time() - batch_start_time

    if query_module_tgn.tgn_mode == TGNMode.Train:
        train_batch_loss = query_module_tgn.m_loss[-1]
        train_avg_loss = sum(query_module_tgn.m_loss) / len(query_module_tgn.m_loss)

    # reset for next batch
    reset_tgn_batch(batch_size=query_module_tgn_batch.batch_size)

    return mgp.Record(
        batch_process_time=batch_process_time,
        edge_process_time=edge_process_time,
        train_batch_loss=train_batch_loss,
        train_avg_loss=train_avg_loss,
        accuracy=accuracy,
    )


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
    learning_rate=1e-4,
    weight_decay=5e-5,
) -> mgp.Record():
    """
    Warning: Every time you call this function, old TGN object is cleared and process of learning params is
    restarted
    """
    global query_module_tgn_batch

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
        OptimizerParameters.WEIGHT_DECAY: weight_decay,
    }
    # tgn params

    if tgn_config[TGNParameters.LAYER_TYPE] == TGNLayerType.GraphAttentionEmbedding:
        tgn_config[TGNParameters.NUM_ATTENTION_HEADS] = num_attention_heads

    # set learning type
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
