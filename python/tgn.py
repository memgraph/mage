from typing import Dict, Tuple, Set, List

import mgp

import numpy as np
import torch

from mage.tgn.constants import TGNLayerType, MessageFunctionType, MessageAggregatorType, MemoryUpdaterType
from mage.tgn.definitions.tgn import TGNEdgesSelfSupervised


class Parameters:
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


##########################
# global params
##########################
config = {}
BATCH_SIZE = 0
tgn = None
criterion = None
optimizer = None
device = None
m_loss = None
fc1 = None
fc2 = None
act = None

all_edges: Set[Tuple[int, int]] = set()
current_batch_size = 0
current_batch_pool = (np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1)), {}, np.empty((1, 1)), {})
events_last_idx = 0


#########

def set_tgn():
    global tgn, config, criterion, optimizer, device, m_loss, fc1, fc2, act
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tgn = TGNEdgesSelfSupervised(
        num_of_layers=config[Parameters.NUM_OF_LAYERS],
        layer_type=config[Parameters.LAYER_TYPE],
        memory_dimension=config[Parameters.MEMORY_DIMENSION],
        time_dimension=config[Parameters.TIME_DIMENSION],
        num_edge_features=config[Parameters.NUM_EDGE_FEATURES],
        num_node_features=config[Parameters.NUM_NODE_FEATURES],
        message_dimension=config[Parameters.MESSAGE_DIMENSION],
        num_neighbors=config[Parameters.NUM_NEIGHBORS],
        # if edge is identity, node must be MLP,
        # if edge is MLP, node also will be MLP
        # thats why we only determine edge type
        edge_message_function_type=config[Parameters.EDGE_FUNCTION_TYPE],
        message_aggregator_type=config[Parameters.MESSAGE_AGGREGATOR_TYPE],
        memory_updater_type=config[Parameters.MEMORY_UPDATER_TYPE]
    ).to(device)

    # set training mode
    tgn.train()

    # init this as global param
    embedding_dim_crit_loss = (config[Parameters.MEMORY_DIMENSION] + config[
        Parameters.NUM_NODE_FEATURES]) * 2  # concat embeddings of src and dest
    fc1 = torch.nn.Linear(embedding_dim_crit_loss, embedding_dim_crit_loss // 2)
    fc2 = torch.nn.Linear(embedding_dim_crit_loss // 2, 1)
    act = torch.nn.ReLU(inplace=False)

    torch.nn.init.xavier_normal_(fc1.weight)
    torch.nn.init.xavier_normal_(fc2.weight)

    # global params
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=0.0001)
    m_loss = []


def sample_negative(negative_num: int) -> (np.array, np.array):
    all_src = list(set([src for src, dest in all_edges]))
    all_dest = list(set([src for src, dest in all_edges]))

    return np.random.choice(all_src, negative_num, replace=True), \
           np.random.choice(all_dest, negative_num, replace=True)


def get_train_data(train_indices: np.array, sources: np.array, destinations: np.array, timestamps: np.array,
                   edge_features: np.array, edge_idxs: np.array, node_features: Dict[int, np.array]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, torch.Tensor], Dict[
        int, torch.Tensor]]:
    # todo add assert

    sources_train = sources[train_indices]
    destinations_train = destinations[train_indices]
    timestamps_train = timestamps[train_indices]
    edge_idxs_train = edge_idxs[train_indices]
    edge_features_train = edge_features[train_indices]

    for i, (src, dest) in enumerate(zip(sources_train, destinations_train)):
        all_edges.add((src, dest))

    negative_src, negative_dest = sample_negative(len(train_indices))

    edge_features_train_dict = {edge_idxs[index]: torch.tensor(np.array(feature, dtype=np.float32), requires_grad=True)
                                for index, feature in zip(train_indices, edge_features_train)}

    node_features_dict = {node: torch.tensor(np.array(node_features[node], dtype=np.float32), requires_grad=True) for
                          node in
                          node_features}

    return sources_train, destinations_train, negative_src, negative_dest, timestamps_train, edge_idxs_train, \
           edge_features_train_dict, node_features_dict


def train_batch():
    global current_batch_pool, current_batch_size, optimizer, events_last_idx, device, m_loss, BATCH_SIZE, act, fc1, fc2, criterion

    sources, destinations, timestamps, edge_features, edge_idxs, node_features = current_batch_pool
    graph_data = (sources, destinations, timestamps, edge_features, edge_idxs, node_features)
    optimizer.zero_grad()

    embeddings, embeddings_negative = tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    embeddings_source_neg = embeddings_negative[:current_batch_size]
    embeddings_dest_neg = embeddings_negative[current_batch_size:]

    x1, x2 = torch.cat([embeddings_source, embeddings_source_neg], dim=0), torch.cat([embeddings_dest,
                                                                                      embeddings_dest_neg])
    x = torch.cat([x1, x2], dim=1)
    h = act(fc1(x))
    score = fc2(h).squeeze(dim=0)

    pos_score = score[:current_batch_size]
    neg_score = score[current_batch_size:]

    pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()
    with torch.no_grad():
        pos_label = torch.ones(current_batch_size, dtype=torch.float, device=device)
        neg_label = torch.zeros(current_batch_size, dtype=torch.float, device=device)

    loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

    # todo solve a problem of retaining a graph
    loss.backward(retain_graph=True)
    optimizer.step()
    m_loss.append(loss.item())


def train_epochs(epochs: int):
    for epoch_num in range(epochs):
        pass
        # split all edges in batches
        # train every batch by calling
        # train_batch(edges_batch)

        # print("epoch num", epoch_num, " loss", loss, "mean loss,", np.mean(m_loss))

    # print("nodes unnormalized scores", nodes_unnormalized_scores)


def process_edges(ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]):
    global current_batch_pool
    # sources: np.array
    # destinations: np.array
    # timestamps: np.array
    # edge_features: Dict[int, torch.Tensor] (id is edge_idx)
    # edge_idxs: np.array incremental and not repeatable
    # node_features: Dict[int, Torch.Tensor]
    sources, destinations, timestamps, edge_features, edge_idxs, node_features = current_batch_pool

    global config

    for edge in edges:

        source = ctx.graph.get_vertex_by_id(edge.from_vertex)
        src_id = source.id

        dest = ctx.graph.get_vertex_by_id(edge.to_vertex)
        dest_id = dest.id

        src_features = source.properties.get("features", None)
        dest_features = dest.properties.get("features", None)

        timestamp = edge.id
        edge_idx = edge.id

        edge_feature = edge.properties.get("features", None)

        if src_features is None:
            src_features = np.random.randint(0, 100, config[Parameters.NUM_NODE_FEATURES]) / 100
        else:
            assert type(src_features) is List
            src_features = np.array(src_features)

        if dest_features is None:
            dest_features = np.random.randint(0, 100, config[Parameters.NUM_NODE_FEATURES]) / 100
        else:
            assert type(dest_features) is List
            dest_features = np.array(dest_features)

        if edge_feature is None:
            edge_feature = np.random.randint(0, 100, config[Parameters.NUM_EDGE_FEATURES]) / 100
        else:
            assert type(edge_feature) is List
            edge_feature = np.array(edge_feature)

        src_features = torch.tensor(src_features, requires_grad=True)
        dest_features = torch.tensor(dest_features, requires_grad=True)
        edge_feature = torch.tensor(edge_feature, requires_grad=True)

        node_features[src_id] = src_features
        node_features[dest_id] = dest_features

        edge_feature[edge_idx] = edge_feature

        sources = np.append(sources, src_id)
        destinations = np.append(destinations, dest_id)
        timestamps = np.append(timestamps, timestamp)

    current_batch_pool = sources, destinations, timestamps, edge_features, edge_idxs, node_features

    print("sources", sources)
    print("destinations", destinations)
    print("timestamps", timestamps)
    print("edge_features", edge_features)
    print("edge_idxs", edge_idxs)
    print("node_features", node_features)


@mgp.read_proc
def update(ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]) -> mgp.Record():
    global BATCH_SIZE, current_batch_size, all_edges
    process_edges(ctx, edges)
    current_batch_size += len(edges)
    if current_batch_size >= BATCH_SIZE:
        train_batch()

    return mgp.Record()


@mgp.read_proc
def set_params(
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
    global config, BATCH_SIZE

    config[Parameters.NUM_OF_LAYERS] = num_of_layers
    config[Parameters.MEMORY_DIMENSION] = memory_dimension
    config[Parameters.TIME_DIMENSION] = time_dimension
    config[Parameters.NUM_EDGE_FEATURES] = num_edge_features
    config[Parameters.NUM_NODE_FEATURES] = num_node_features
    config[Parameters.MESSAGE_DIMENSION] = message_dimension
    config[Parameters.NUM_NEIGHBORS] = num_neighbors

    config[Parameters.LAYER_TYPE] = get_tgn_layer_enum(layer_type)
    config[Parameters.EDGE_FUNCTION_TYPE] = get_edge_message_function_type(edge_message_function_type)
    config[Parameters.MESSAGE_DIMENSION] = get_message_aggregator_type(message_aggregator_type)
    config[Parameters.MEMORY_UPDATER_TYPE] = get_memory_updater_type(memory_updater_type)

    BATCH_SIZE = batch_size

    print(config)

    set_tgn()

    return mgp.Record()


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
        return MessageFunctionType.Mean
    elif MessageAggregatorType(message_aggregator_type) is MessageAggregatorType.Last:
        return MessageFunctionType.Last
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
