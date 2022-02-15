from typing import Dict, Tuple, Set

import numpy as np
import torch

from tgn.constants import TGNLayerType, MessageFunctionType, MessageAggregatorType, MemoryUpdaterType
from tgn.definitions.tgn import TGN, TGNEdgesSelfSupervised
from utils import data_loader

NUM_NODE_FEATURES = 100
MESSAGE_DIM = 100
MEMORY_DIM = 100
TIME_DIM = 1
BATCH_SIZE = 5
NUM_NEIGHBORS = 5

all_edges: Set[Tuple[int, int]] = set()


def sample_negative(negative_num: int) -> (np.array, np.array):
    all_src = list(set([src for src, dest in all_edges]))
    all_dest = list(set([src for src, dest in all_edges]))

    return np.random.choice(all_src, negative_num, replace=True), \
           np.random.choice(all_dest, negative_num, replace=True)


def get_train_data(train_indices: np.array, sources: np.array, destinations: np.array, timestamps: np.array,
                   edge_features: np.array, edge_idxs: np.array) -> Tuple[
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

    # be careful if you will copy this later, because here mapping of indices works from 0 to 100, later it should
    # also??
    # dict -> (edge_indx, np.array)
    edge_features_train_dict = {edge_idxs[index]: torch.tensor(np.array(feature, dtype=np.float32), requires_grad=True)
                                for index, feature in zip(train_indices, edge_features_train)}

    node_features_dict = {node: torch.tensor(np.zeros(NUM_NODE_FEATURES), requires_grad=True) for node in
                          set(sources).union(destinations)}

    return sources_train, destinations_train, negative_src, negative_dest, timestamps_train, edge_idxs_train, \
           edge_features_train_dict, node_features_dict


def train_tgn(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features, edge_features, train_data = data_loader.get_data(config["dataset_name"])
    sources, destinations, timestamps, edge_idxs, labels = train_data

    # dimensions
    num_edge_features = edge_features.shape[1]

    tgn = TGNEdgesSelfSupervised(
        num_of_layers=config['num_of_layers'],
        layer_type=config['layer_type'],
        memory_dimension=MEMORY_DIM,
        time_dimension=TIME_DIM,
        num_edge_features=num_edge_features,
        num_node_features=NUM_NODE_FEATURES,
        message_dimension=MESSAGE_DIM,
        num_neighbors=NUM_NEIGHBORS,
        # if edge is identity, node must be MLP,
        # if edge is MLP, node also will be MLP
        # thats why we only determine edge type
        edge_message_function_type=MessageFunctionType.Identity,
        message_aggregator_type=MessageAggregatorType.Last,
        memory_updater_type=MemoryUpdaterType.GRU
    ).to(device)

    # set training mode
    tgn.train()
    epochs = config["epochs"]
    assert len(sources) == len(destinations) == len(timestamps) == len(edge_idxs) == len(labels), f'Error! Expected' \
                                                                                                  f'same lengths, but got' \
                                                                                                  f'it wrong'

    embedding_dim_crit_loss = (MEMORY_DIM + NUM_NODE_FEATURES) * 2  # concat embeddings of src and dest
    fc1 = torch.nn.Linear(embedding_dim_crit_loss, embedding_dim_crit_loss // 2)
    fc2 = torch.nn.Linear(embedding_dim_crit_loss // 2, 1)
    act = torch.nn.ReLU(inplace=False)

    torch.nn.init.xavier_normal_(fc1.weight)
    torch.nn.init.xavier_normal_(fc2.weight)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=0.0001)
    m_loss = []
    torch.autograd.set_detect_anomaly(True)
    for epoch_num in range(epochs):
        # todo when using epochs we create same event few times (epochs_mum), is this a good practice?
        loss = 0
        batch_num = int(len(sources) / BATCH_SIZE)

        for i in range(batch_num):
            print("Ba")

            optimizer.zero_grad()

            if i == batch_num - 1:
                end_batch_idx = len(sources) - 1
            else:
                end_batch_idx = (i + 1) * BATCH_SIZE

            current_train_indices = np.arange(i * BATCH_SIZE, end_batch_idx)
            graph_data = get_train_data(current_train_indices,
                                        sources=sources,
                                        destinations=destinations,
                                        timestamps=timestamps,
                                        edge_idxs=edge_idxs,
                                        edge_features=edge_features)

            embeddings, embeddings_negative = tgn(graph_data)

            embeddings_source = embeddings[:BATCH_SIZE]
            embeddings_dest = embeddings[BATCH_SIZE:]

            embeddings_source_neg = embeddings_negative[:BATCH_SIZE]
            embeddings_dest_neg = embeddings_negative[BATCH_SIZE:]

            x1, x2 = torch.cat([embeddings_source, embeddings_source_neg], dim=0), torch.cat([embeddings_dest,
                                                                                              embeddings_dest_neg])
            x = torch.cat([x1, x2], dim=1)
            h = act(fc1(x))
            score = fc2(h).squeeze(dim=0)

            pos_score = score[:BATCH_SIZE]
            neg_score = score[BATCH_SIZE:]

            pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()
            with torch.no_grad():
                pos_label = torch.ones(BATCH_SIZE, dtype=torch.float, device=device)
                neg_label = torch.zeros(BATCH_SIZE, dtype=torch.float, device=device)

            loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

            #todo solve a problem of retaining a graph
            loss.backward(retain_graph=True)
            optimizer.step()
            m_loss.append(loss.item())

            #tgn.memory_detach_tensor_grads()

        print("epoch num", epoch_num, " loss", loss, "mean loss,", np.mean(m_loss))

    # print("nodes unnormalized scores", nodes_unnormalized_scores)


def get_args():
    tgn_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "layer_type": TGNLayerType.GraphSumEmbedding,
        "epochs": 2
    }
    config = {**tgn_config, **{"dataset_name": "wikipedia"}}
    print(config)
    return config


def main():
    config = get_args()
    train_tgn(config)


if __name__ == '__main__':
    main()
