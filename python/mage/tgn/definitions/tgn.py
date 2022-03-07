from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from mage.tgn.constants import (
    TGNLayerType,
    MessageFunctionType,
    MemoryUpdaterType,
    MessageAggregatorType,
)
from mage.tgn.definitions.events import Event, InteractionEvent, NodeEvent
from mage.tgn.definitions.memory import Memory
from mage.tgn.definitions.memory_updater import MemoryUpdaterGRU, MemoryUpdaterRNN
from mage.tgn.definitions.message_aggregator import (
    MeanMessageAggregator,
    LastMessageAggregator,
    MessageAggregator,
)
from mage.tgn.definitions.message_function import (
    MessageFunctionMLP,
    MessageFunctionIdentity,
    MessageFunction,
)
from mage.tgn.definitions.messages import (
    RawMessage,
    NodeRawMessage,
    InteractionRawMessage,
)
from mage.tgn.definitions.raw_message_store import RawMessageStore
from mage.tgn.definitions.temporal_neighborhood import TemporalNeighborhood
from mage.tgn.definitions.time_encoding import TimeEncoder

from mage.tgn.helper.simple_mlp import MLP


class TGN(nn.Module):
    def __init__(
        self,
        num_of_layers: int,
        layer_type: TGNLayerType,
        memory_dimension: int,
        time_dimension: int,
        num_edge_features: int,
        num_node_features: int,
        message_dimension: int,
        num_neighbors: int,
        edge_message_function_type: MessageFunctionType,
        message_aggregator_type: MessageAggregatorType,
        memory_updater_type: MemoryUpdaterType,
    ):
        super().__init__()
        self.num_of_layers = num_of_layers
        self.memory_dimension = memory_dimension
        self.time_dimension = time_dimension
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.layer_type = layer_type

        self.num_neighbors = num_neighbors

        self.edge_features: Dict[int, torch.Tensor] = {}
        self.node_features: Dict[int, torch.Tensor] = {}

        self.temporal_neighborhood = TemporalNeighborhood()

        # dimension of raw message for edge
        # m_ij = (s_i, s_j, delta t, e_ij)
        # delta t is only one number :)
        self.edge_raw_message_dimension = (
            2 * self.memory_dimension + 1 + num_edge_features
        )

        # dimension of raw message for node
        # m_i = (s_i, t, v_i)
        self.node_raw_message_dimension = self.memory_dimension + 1 + num_node_features

        self.raw_message_store = RawMessageStore(
            edge_raw_message_dimension=self.edge_raw_message_dimension,
            node_raw_message_dimension=self.node_raw_message_dimension,
        )

        MessageFunctionEdge = get_message_function_type(edge_message_function_type)

        # if edge function is identity, when identity function is applied, it will result with
        # vector of greater dimension then when identity is applied to node raw messsage, so node function must be MLP
        # if edge is MLP, node also will be MLP
        MessageFunctionNode = MessageFunctionMLP

        self.message_dimension = message_dimension

        if edge_message_function_type == MessageFunctionType.Identity:
            # We set message dimension (dimension after message function is applied to raw message) to
            # dimension of raw message of edge interaction event, since message dimension will be
            # just dimension of concatenated vectors in edge interaction event.
            # Also, MLP in node event will then produce vector of same dimension and we can then aggregate these
            # two vectors, with MEAN or LAST
            self.message_dimension = self.edge_raw_message_dimension

        self.edge_message_function = MessageFunctionEdge(
            message_dimension=self.message_dimension,
            raw_message_dimension=self.edge_raw_message_dimension,
        )

        self.node_message_function = MessageFunctionNode(
            message_dimension=self.message_dimension,
            raw_message_dimension=self.node_raw_message_dimension,
        )

        MessageAggregator = get_message_aggregator_type(message_aggregator_type)

        self.message_aggregator = MessageAggregator()

        self.memory = Memory(memory_dimension=memory_dimension)

        MemoryUpdaterType = get_memory_updater_type(memory_updater_type)

        self.memory_updater = MemoryUpdaterType(
            memory_dimension=self.memory_dimension,
            message_dimension=self.message_dimension,
        )

        self.time_encoder = TimeEncoder(out_dimension=self.time_dimension)

    def memory_detach_tensor_grads(self):
        self.memory.detach_tensor_grads()
        # self.raw_message_store.detach_grads()

    def forward(
        self,
        data: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Dict[int, torch.Tensor],
            Dict[int, torch.Tensor],
        ],
    ):
        raise Exception("Should not be implemented")

    def process_current_batch(
        self,
        sources: np.array,
        destinations: np.array,
        node_features: Dict[int, torch.Tensor],
        edge_features: Dict[int, torch.Tensor],
        edge_idxs: np.array,
        timestamps: np.array,
    ):

        self.update_raw_message_store_current_batch(
            sources=sources,
            destinations=destinations,
            node_features=node_features,
            edge_features=edge_features,
            edge_idxs=edge_idxs,
            timestamps=timestamps,
        )

        self.temporal_neighborhood.update_neighborhood(
            sources=sources,
            destinations=destinations,
            timestamps=timestamps,
            edge_idx=edge_idxs,
        )

        for edge_idx, edge_feature in edge_features.items():
            self.edge_features[edge_idx] = edge_feature

        for node_id, node_feature in node_features.items():
            self.node_features[node_id] = node_feature

    def process_previous_batches(self) -> None:

        # dict nodeid -> List[event]
        raw_messages = self.raw_message_store.get_messages()

        processed_messages = self.create_messages(
            node_event_function=self.node_message_function,
            edge_event_function=self.edge_message_function,
            raw_messages=raw_messages,
        )

        aggregated_messages = self.aggregate_messages(
            processed_messages=processed_messages,
            aggregator_function=self.message_aggregator,
        )

        self.update_memory(aggregated_messages, self.memory, self.memory_updater)

    def update_raw_message_store_current_batch(
        self,
        sources: np.array,
        destinations: np.array,
        timestamps: np.array,
        edge_idxs: np.array,
        edge_features: Dict[int, torch.Tensor],
        node_features: Dict[int, torch.Tensor],
    ) -> None:

        # node_events: Dict[int, List[Event]] = create_node_events()
        interaction_events: Dict[int, List[Event]] = self.create_interaction_events(
            sources=sources,
            destinations=destinations,
            timestamps=timestamps,
            edge_indx=edge_idxs,
        )

        # this is what TGN gets
        events: Dict[int, List[Event]] = interaction_events
        # events.sort(key=lambda x:x.get_time()) # sort by time

        raw_messages: Dict[int, List[RawMessage]] = self.create_raw_messages(
            events=events,
            edge_features=edge_features,
            node_features=node_features,
            memory=self.memory,
        )

        self.raw_message_store.update_messages(raw_messages)

    def create_interaction_events(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        edge_indx: np.ndarray,
    ):
        "Every event has two interaction events"
        interaction_events: Dict[int, List[InteractionEvent]] = {
            node: [] for node in set(sources).union(set(destinations))
        }
        for i in range(len(sources)):
            interaction_events[sources[i]].append(
                InteractionEvent(
                    source=sources[i],
                    dest=destinations[i],
                    timestamp=timestamps[i],
                    edge_indx=edge_indx[i],
                )
            )
        return interaction_events

    def create_node_events(
        self,
    ):
        # currently not using this
        return []

    def create_messages(
        self,
        node_event_function: MessageFunction,
        edge_event_function: MessageFunction,
        raw_messages: Dict[int, List[RawMessage]],
    ):
        # change this so that every that dict is of type
        # node_id -> [[],
        #             [],
        #             [],]
        processed_messages_dict = {node: [] for node in raw_messages}
        for node in raw_messages:
            for message in raw_messages[node]:
                if type(message) is NodeRawMessage:
                    node_raw_message = message
                    # torch vstack??
                    processed_messages_dict[node].append(
                        node_event_function(
                            (
                                node_raw_message.source_memory,
                                node_raw_message.timestamp,
                                node_raw_message.node_features,
                            )
                        )
                    )
                elif type(message) is InteractionRawMessage:

                    interaction_raw_message = message

                    # torch vstack??
                    processed_messages_dict[node].append(
                        edge_event_function(
                            (
                                interaction_raw_message.source_memory,
                                interaction_raw_message.dest_memory,
                                interaction_raw_message.delta_time,
                                interaction_raw_message.edge_features,
                            )
                        )
                    )
                else:
                    raise Exception(f"Message Type not supported {type(message)}")
        return processed_messages_dict

    def create_raw_messages(
        self,
        events: Dict[int, List[Event]],
        memory: Memory,
        node_features: Dict[int, torch.Tensor],
        edge_features: Dict[int, torch.Tensor],
    ):
        raw_messages = {node: [] for node in events}
        for node in events:
            node_events = events[node]
            for event in node_events:
                assert node == event.source
                if type(event) is NodeEvent:
                    raw_messages[node].append(
                        NodeRawMessage(
                            source_memory=memory.get_node_memory(node),
                            timestamp=event.timestamp,
                            node_features=node_features[node],
                            source=node,
                        )
                    )
                elif type(event) is InteractionEvent:
                    # every interaction event creates two raw messages
                    raw_messages[event.source].append(
                        InteractionRawMessage(
                            source_memory=memory.get_node_memory(event.source),
                            timestamp=event.timestamp,
                            dest_memory=memory.get_node_memory(event.dest),
                            source=node,
                            edge_features=edge_features[event.edge_indx],
                            delta_time=torch.tensor(
                                np.array(event.timestamp).astype("float"),
                                requires_grad=True,
                            )
                            - memory.get_last_node_update(event.source),
                        )
                    )
                    raw_messages[event.dest].append(
                        InteractionRawMessage(
                            source_memory=memory.get_node_memory(event.dest),
                            timestamp=event.timestamp,
                            dest_memory=memory.get_node_memory(event.source),
                            source=event.dest,
                            edge_features=edge_features[event.edge_indx],
                            delta_time=torch.tensor(
                                np.array(event.timestamp).astype("float"),
                                requires_grad=True,
                            )
                            - memory.get_last_node_update(event.dest),
                        )
                    )
                else:
                    raise Exception(f"Event Type not supported {type(event)}")
        return raw_messages

    def aggregate_messages(
        self,
        processed_messages: Dict[int, List[torch.Tensor]],
        aggregator_function: MessageAggregator,
    ) -> Dict[int, torch.Tensor]:
        # todo change so that it returns for aggregated messages
        #                               [[]
        #                                []], shape=(number_of_nodes, aggregated_length)
        aggregated_messages = {node: None for node in processed_messages}
        for node in processed_messages:
            aggregated_messages[node] = aggregator_function(processed_messages[node])
        return aggregated_messages

    def update_memory(self, messages, memory, memory_updater) -> None:
        # todo change to do all updates at once
        for node in messages:
            updated_memory = memory_updater(
                (messages[node], memory.get_node_memory(node))
            )

            # use flatten to get (memory_dim,)
            updated_memory = torch.flatten(updated_memory)
            memory.set_node_memory(node, updated_memory)

    def _form_computation_graph(
        self, nodes: np.array, timestamps: np.array
    ) -> (
        List[List[Tuple[int, int]]],
        List[Dict[Tuple[int, int], int]],
        List[List[int]],
        List[List[int]],
    ):
        node_layers = [[(n, t) for (n, t) in zip(nodes, timestamps)]]
        edge_layers = []
        timestamp_layers = []

        for _ in range(self.num_of_layers):
            prev = node_layers[-1]
            cur_arr = [(n, v) for (n, v) in prev]

            node_arr = []
            for (v, t) in cur_arr:
                (
                    neighbors,
                    edge_idxs,
                    timestamps,
                ) = self.temporal_neighborhood.get_neighborhood(
                    v, t, self.num_neighbors
                )
                node_arr.extend([(ni, ti) for (ni, ti) in zip(neighbors, timestamps)])
            cur_arr.extend(node_arr)
            node_arr = list(set(cur_arr))
            node_layers.append(node_arr)

        node_layers.reverse()

        # this mapping will be later used to reference node features and edge features and time features
        mappings = [{j: i for (i, j) in enumerate(arr)} for arr in node_layers]

        # todo here we have a potential problem with sampling,
        # because I call self.temporal_neighborhood.get_neighborhood twice, and if I was to use unifrom sampling
        # it could be a problem.
        # fix would be to save in dict what is sampled for which node and then take it here

        edge_arr = []
        timestamp_arr = []
        for (v, t) in node_layers[0]:
            (
                neighbors,
                edge_idxs,
                timestamps,
            ) = self.temporal_neighborhood.get_neighborhood(v, t, self.num_neighbors)
            edge_arr.append(edge_idxs)
            timestamp_arr.append(
                [t - ti for ti in timestamps]
            )  # for neighbors we are always using time diff
        edge_layers.append(edge_arr)
        timestamp_layers.append(timestamp_arr)

        # edge_layers.reverse()
        # timestamp_layers.reverse()

        return node_layers, mappings, edge_layers, timestamp_layers

    def _get_edge_features(self, edge_idx: int) -> torch.Tensor:
        # todo check if edge features don't exist, save them in dict as random or zero
        return (
            self.edge_features[edge_idx]
            if edge_idx in self.edge_features
            else torch.zeros(self.num_edge_features, requires_grad=True)
        )

    def _get_edges_features(self, edge_idxs: List[int]) -> torch.Tensor:
        edges_features = torch.zeros(self.num_neighbors, self.num_edge_features)
        for i, edge_idx in enumerate(edge_idxs):
            edges_features[i, :] = self._get_edge_features(edge_idx)
        return edges_features

    def _get_graph_sum_data(self, nodes: np.array, timestamps: np.array):
        (
            node_layers,
            mappings,
            edge_layers,
            timestamp_layers,
        ) = self._form_computation_graph(nodes, timestamps)

        rows = []
        for v, t in node_layers[0]:
            row = []
            (
                neighbors,
                edge_idxs,
                timestamps,
            ) = self.temporal_neighborhood.get_neighborhood(v, t, self.num_neighbors)
            for vi, ei, ti in zip(neighbors, edge_idxs, timestamps):
                row.append((vi, ti))
            rows.append(row)

        nodes = node_layers[0]

        node_features = torch.zeros(
            len(nodes), self.memory_dimension + self.num_node_features
        )

        for i, (node, _) in enumerate(nodes):
            node_feature = (
                self.node_features[node]
                if node in self.node_features
                else torch.zeros(self.num_node_features, requires_grad=True)
            )
            node_memory = torch.Tensor(
                self.memory.get_node_memory(node).cpu().detach().numpy()
            )
            node_features[i, :] = torch.concat((node_memory, node_feature))

        edge_features = [
            self._get_edges_features(node_neighbors)
            for node_neighbors in edge_layers[0]
        ]

        timestamp_features = [
            self.time_encoder(
                torch.tensor(
                    np.array(time, dtype=np.float32), requires_grad=True
                ).reshape((len(time), 1))
            )
            for time in timestamp_layers[0]
        ]

        return (
            node_layers,
            mappings,
            edge_layers,
            rows,
            node_features,
            edge_features,
            timestamp_features,
        )


############################################

# Instances of TGN

############################################


class TGNEdgesSelfSupervised(TGN):
    def forward(
        self,
        data: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Dict[int, torch.Tensor],
            Dict[int, torch.Tensor],
        ],
    ):
        # source -> np.array(num_of_nodes,)
        # destinations -> np.array(num_of_nodes,)
        # negative_sources -> np.array
        # negative_destinations -> np.array
        # timestamps -> np.array(num_of_nodes,)
        # edge_index -> np.array(num_of_nodes,)

        # edge_features = Dict(int, np.array)
        # node_features -> Dict(int, np.array)
        # tl;dr used to unpack data
        print("forward TGNEdgesSelfSupervised")
        (
            sources,
            destinations,
            negative_sources,
            negative_destinations,
            timestamps,
            edge_idxs,
            edge_features,
            node_features,
        ) = data

        assert (
            sources.shape[0]
            == destinations.shape[0]
            == negative_sources.shape[0]
            == negative_destinations.shape[0]
            == timestamps.shape[0]
            == len(edge_idxs)
            == len(edge_features)
        ), (
            f"Sources, destinations, negative sources, negative destinations, timestamps, edge_indxs and edge_features must be of same dimension, but got "
            f"{sources.shape[0]}, {destinations.shape[0]}, {timestamps.shape[0]}, {len(edge_idxs)}, {len(edge_features)}"
        )

        # part of 1->2->3, all till point 4 in paper from Figure 2
        # we are using this part so that we can get gradients from memory module also and so that they
        # can be included in optimizer
        # By doing this, the computation of the memory-related modules directly influences the loss
        self.process_previous_batches()

        graph_data = self._get_graph_sum_data(
            np.concatenate([sources.copy(), destinations.copy()], dtype=int),
            np.concatenate([timestamps, timestamps]),
        )

        embeddings = self.tgn_net(graph_data)

        graph_data_negative = self._get_graph_sum_data(
            np.concatenate(
                [negative_sources.copy(), negative_destinations.copy()], dtype=int
            ),
            np.concatenate([timestamps, timestamps]),
        )

        embeddings_negative = self.tgn_net(graph_data_negative)

        # here we update raw message store, and this batch will be used in next
        # call of tgn in function self.process_previous_batches
        # the raw messages for this batch interactions are stored in the raw
        # message store  to be used in future batches.
        # in paper on figure 2 this is part 7.
        self.process_current_batch(
            sources, destinations, node_features, edge_features, edge_idxs, timestamps
        )

        return embeddings, embeddings_negative


class TGNGraphAttentionEmbedding(TGN):
    def __init__(
        self,
        num_of_layers: int,
        layer_type: TGNLayerType,
        memory_dimension: int,
        time_dimension: int,
        num_edge_features: int,
        num_node_features: int,
        message_dimension: int,
        num_neighbors: int,
        edge_message_function_type: MessageFunctionType,
        message_aggregator_type: MessageAggregatorType,
        memory_updater_type: MemoryUpdaterType,
        num_attention_heads: int,
    ):
        super().__init__(
            num_of_layers,
            layer_type,
            memory_dimension,
            time_dimension,
            num_edge_features,
            num_node_features,
            message_dimension,
            num_neighbors,
            edge_message_function_type,
            message_aggregator_type,
            memory_updater_type,
        )

        assert layer_type == TGNLayerType.GraphAttentionEmbedding

        self.num_attention_heads = num_attention_heads

        # Initialize TGN layers
        tgn_layers = []

        layer = TGNLayerGraphAttentionEmbedding(
            embedding_dimension=self.memory_dimension + self.num_node_features,
            edge_feature_dim=self.num_edge_features,
            time_encoding_dim=self.time_dimension,
            node_features_dim=self.num_node_features,
            num_neighbors=self.num_neighbors,
            num_attention_heads=num_attention_heads,
            num_of_layers=self.num_of_layers,
        )

        tgn_layers.append(layer)

        self.tgn_net = nn.Sequential(*tgn_layers)


class TGNGraphSumEmbedding(TGN):
    def __init__(
        self,
        num_of_layers: int,
        layer_type: TGNLayerType,
        memory_dimension: int,
        time_dimension: int,
        num_edge_features: int,
        num_node_features: int,
        message_dimension: int,
        num_neighbors: int,
        edge_message_function_type: MessageFunctionType,
        message_aggregator_type: MessageAggregatorType,
        memory_updater_type: MemoryUpdaterType,
    ):
        super().__init__(
            num_of_layers,
            layer_type,
            memory_dimension,
            time_dimension,
            num_edge_features,
            num_node_features,
            message_dimension,
            num_neighbors,
            edge_message_function_type,
            message_aggregator_type,
            memory_updater_type,
        )

        assert layer_type == TGNLayerType.GraphSumEmbedding

        # Initialize TGN layers
        tgn_layers = []

        layer = TGNLayerGraphSumEmbedding(
            embedding_dimension=self.memory_dimension + self.num_node_features,
            edge_feature_dim=self.num_edge_features,
            time_encoding_dim=self.time_dimension,
            node_features_dim=self.num_node_features,
            num_neighbors=self.num_neighbors,
            num_of_layers=self.num_of_layers,
        )

        tgn_layers.append(layer)

        self.tgn_net = nn.Sequential(*tgn_layers)


class TGNGraphAttentionEdgeSelfSupervised(
    TGNGraphAttentionEmbedding, TGNEdgesSelfSupervised
):
    def __init__(
        self,
        num_of_layers: int,
        layer_type: TGNLayerType,
        memory_dimension: int,
        time_dimension: int,
        num_edge_features: int,
        num_node_features: int,
        message_dimension: int,
        num_neighbors: int,
        edge_message_function_type: MessageFunctionType,
        message_aggregator_type: MessageAggregatorType,
        memory_updater_type: MemoryUpdaterType,
        num_attention_heads: int,
    ):
        super().__init__(
            num_of_layers,
            layer_type,
            memory_dimension,
            time_dimension,
            num_edge_features,
            num_node_features,
            message_dimension,
            num_neighbors,
            edge_message_function_type,
            message_aggregator_type,
            memory_updater_type,
            num_attention_heads,
        )


class TGNGraphSumEdgeSelfSupervised(TGNGraphSumEmbedding, TGNEdgesSelfSupervised):
    def __init__(
        self,
        num_of_layers: int,
        layer_type: TGNLayerType,
        memory_dimension: int,
        time_dimension: int,
        num_edge_features: int,
        num_node_features: int,
        message_dimension: int,
        num_neighbors: int,
        edge_message_function_type: MessageFunctionType,
        message_aggregator_type: MessageAggregatorType,
        memory_updater_type: MemoryUpdaterType,
    ):
        super().__init__(
            num_of_layers,
            layer_type,
            memory_dimension,
            time_dimension,
            num_edge_features,
            num_node_features,
            message_dimension,
            num_neighbors,
            edge_message_function_type,
            message_aggregator_type,
            memory_updater_type,
        )


class TGNLayer(nn.Module):
    """
    Base class for all implementations
    """

    def __init__(
        self,
        embedding_dimension: int,
        edge_feature_dim: int,
        time_encoding_dim: int,
        node_features_dim: int,
        num_neighbors: int,
        num_of_layers,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.edge_feature_dim = edge_feature_dim
        self.time_encoding_dim = time_encoding_dim
        self.node_features_dim = node_features_dim
        self.num_neighbors = num_neighbors
        self.num_of_layers = num_of_layers


class TGNLayerGraphSumEmbedding(TGNLayer):
    """
    TGN layer implementation inspired by official TGN implementation
    """

    def __init__(
        self,
        embedding_dimension: int,
        edge_feature_dim: int,
        time_encoding_dim: int,
        node_features_dim: int,
        num_neighbors: int,
        num_of_layers: int,
    ):
        super().__init__(
            embedding_dimension,
            edge_feature_dim,
            time_encoding_dim,
            node_features_dim,
            num_neighbors,
            num_of_layers,
        )
        # Initialize W1 matrix and W2 matrix

        self.linear_1 = torch.nn.Linear(
            embedding_dimension + time_encoding_dim + edge_feature_dim,
            embedding_dimension,
        )

        self.linear_2 = torch.nn.Linear(
            embedding_dimension + embedding_dimension, embedding_dimension
        )
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        (
            node_layers,
            mappings,
            edge_layers,
            neighbors_arr,
            features,
            edge_features,
            time_features,
        ) = data

        out = features

        for k in range(self.num_of_layers):
            mapping = mappings[k]
            nodes = node_layers[k + 1]  # neighbors on next layer
            # represents how we globally gave index to node,timestamp mapping
            global_indexes = np.array([mappings[0][(v, t)] for (v, t) in nodes])
            cur_neighbors = [
                neighbors_arr[index] for index in global_indexes
            ]  # neighbors and timestamps of nodes from next layer
            curr_edges = [edge_features[index] for index in global_indexes]
            curr_time = [time_features[index] for index in global_indexes]

            aggregate = self._aggregate(
                out, cur_neighbors, nodes, mapping, curr_edges, curr_time
            )

            curr_mapped_nodes = np.array([mapping[(v, t)] for (v, t) in nodes])

            concat_neigh_out = torch.cat((out[curr_mapped_nodes], aggregate), dim=1)
            out = self.linear_2(concat_neigh_out)
        return out

    def _aggregate(self, features, rows, nodes, mapping, edge_features, time_features):
        assert len(nodes) == len(rows)
        mapped_rows = [
            np.array([mapping[(vi, ti)] for (vi, ti) in row]) for row in rows
        ]
        out = torch.rand(len(nodes), self.embedding_dimension)

        # row represents list of indexes of "current neighbors" of edge_features on i-th index
        for i, row in enumerate(mapped_rows):
            features_curr = features[row][:]
            edge_feature_curr = edge_features[i][:]
            time_feature_curr = time_features[i][:]
            # shape(num_neighbors, embedding_dim+edge_features_dim+time_encoding_dim)
            # concatenation is done alongside columns, we have only 2 dimension, rows=0 and columns=1
            aggregate = torch.concat(
                (features_curr, edge_feature_curr, time_feature_curr), dim=1
            )

            # sum rows, but keep this dimension
            # shape(1, embedding_dim+edge_features_dim+time_encoding_dim)
            aggregate_sum = torch.sum(aggregate, dim=0, keepdim=True)
            out_linear1 = self.linear_1(aggregate_sum)
            out_relu_linear1 = self.relu(out_linear1)

            out[i, :] = out_relu_linear1

        return out


class TGNLayerGraphAttentionEmbedding(TGNLayer):
    """
    TGN layer implementation inspired by official TGN implementation
    """

    def __init__(
        self,
        embedding_dimension: int,
        edge_feature_dim: int,
        time_encoding_dim: int,
        node_features_dim: int,
        num_neighbors: int,
        num_of_layers: int,
        num_attention_heads: int,
    ):
        super().__init__(
            embedding_dimension,
            edge_feature_dim,
            time_encoding_dim,
            node_features_dim,
            num_neighbors,
            num_of_layers,
        )

        self.query_dim = embedding_dimension + time_encoding_dim
        self.key_dim = embedding_dimension + edge_feature_dim + time_encoding_dim
        self.value_dim = self.key_dim
        self.num_attention_heads = num_attention_heads

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim * self.num_neighbors,
            # set on neighbors num
            vdim=self.value_dim * self.num_neighbors,
            num_heads=num_attention_heads,  # this add as a parameter
            batch_first=True,
        )  # this way no need to do torch.permute later <3

        self.mlp = MLP(
            [
                self.query_dim + embedding_dimension,
                embedding_dimension,
                embedding_dimension,
            ]
        )

    def forward(self, data):
        (
            node_layers,
            mappings,
            edge_layers,
            neighbors_arr,
            features,
            edge_features,
            time_features,
        ) = data

        out = features

        for k in range(self.num_of_layers):
            mapping = mappings[k]
            nodes = node_layers[k + 1]  # neighbors on next layer
            # represents how we globally gave index to node,timestamp mapping
            global_indexes = np.array([mappings[0][(v, t)] for (v, t) in nodes])
            cur_neighbors = [
                neighbors_arr[index] for index in global_indexes
            ]  # neighbors and timestamps of nodes from next layer
            curr_edges = [edge_features[index] for index in global_indexes]
            curr_time = [time_features[index] for index in global_indexes]
            # shape (len(nodes), self.num_neighbors * self.key_dim)
            aggregate = self._aggregate(
                out, cur_neighbors, nodes, mapping, curr_edges, curr_time
            )

            # add third dimension,
            aggregate_unsqueeze = torch.unsqueeze(aggregate, dim=0)

            curr_mapped_nodes = np.array([mapping[(v, t)] for (v, t) in nodes])

            keys = aggregate_unsqueeze
            values = aggregate_unsqueeze
            query_concat = torch.concat(
                (
                    out[curr_mapped_nodes],
                    torch.zeros(len(curr_mapped_nodes), self.time_encoding_dim),
                ),
                dim=1,
            )
            query = torch.unsqueeze(query_concat, dim=0)

            attn_out, _ = self.multi_head_attention(query=query, key=keys, value=values)

            attn_out = torch.squeeze(attn_out)

            concat_neigh_out = torch.cat((out[curr_mapped_nodes], attn_out), dim=1)
            out = self.mlp(concat_neigh_out)
        return out

    def _aggregate(self, features, rows, nodes, mapping, edge_features, time_features):
        assert len(nodes) == len(rows)
        mapped_rows = [
            np.array([mapping[(vi, ti)] for (vi, ti) in row]) for row in rows
        ]

        out = torch.rand(len(nodes), self.num_neighbors * self.key_dim)

        # row represents list of indexes of "current neighbors" of edge_features on i-th index
        for i, row in enumerate(mapped_rows):
            features_curr = features[row][:]
            edge_feature_curr = edge_features[i][:]
            time_feature_curr = time_features[i][:]

            # shape(1, num_neighbors * (embedding_dim + edge_features_dim + time_encoding_dim)
            # after doing concatenation on columns side, reshape to have 1 row
            aggregate = torch.concat(
                (features_curr, edge_feature_curr, time_feature_curr), dim=1
            ).reshape((1, -1))

            out[i, :] = aggregate

        return out


def get_message_function_type(message_function_type: MessageFunctionType):
    if message_function_type == MessageFunctionType.MLP:
        return MessageFunctionMLP
    elif message_function_type == MessageFunctionType.Identity:
        return MessageFunctionIdentity
    else:
        raise Exception(
            f"Message function type {message_function_type} not yet supported."
        )


def get_memory_updater_type(memory_updater_type: MemoryUpdaterType):
    if memory_updater_type == MemoryUpdaterType.GRU:
        return MemoryUpdaterGRU

    elif memory_updater_type == MemoryUpdaterType.RNN:
        return MemoryUpdaterRNN
    else:
        raise Exception(f"Memory updater type {memory_updater_type} not yet supported.")


def get_message_aggregator_type(message_aggregator_type: MessageAggregatorType):
    if message_aggregator_type == MessageAggregatorType.Mean:
        return MeanMessageAggregator

    elif message_aggregator_type == MessageAggregatorType.Last:
        return LastMessageAggregator

    else:
        raise Exception(
            f"Message aggregator type {message_aggregator_type} not yet supported."
        )
